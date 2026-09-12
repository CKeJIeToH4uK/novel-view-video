"""Настоящий Parquet/source/order/record; численный raster проверяется отдельно."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from novel_view.config.job import load_resolved_job
from novel_view.models.moge.request import MogeExecution, MogeRawPrediction, MogeTelemetry
from novel_view.preparation.waymo_ddw import bake, condition, depth, raster
from novel_view.preparation.waymo_ddw.record import read_prepared_record, write_prepared_record
from novel_view.preparation.waymo_ddw.selection import load_selection
from novel_view.preparation.waymo_ddw.warp import WarpResult
from novel_view.workflows.ddw_preparation import parse_job
from novel_view.workflows.runner import execute_job, select_job
from tests.support.stage5 import make_runtime
from tests.support.waymo_v2 import write_reader_dataset


ROOT = Path(__file__).resolve().parents[4]
JOB = ROOT / "jobs/waymo/r4c-ddw-prepared/run.yaml"


def test_parquet_selection_to_ordered_prepared_record(tmp_path, monkeypatch, capsys):
    runtime = make_runtime(tmp_path, image="moge")
    job = load_resolved_job(JOB, ROOT)
    source_selection = ROOT / job.input["selection"]
    selection_path = runtime.roots.selections.parent / job.input["selection"]
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_bytes(source_selection.read_bytes())
    samples = load_selection(selection_path).samples
    expected = {}
    for index, sample in enumerate(samples):
        times = tuple(
            (index + 1) * 10_000_000 + i * 100_000 for i in range(sample.start_frame_index + 123)
        )
        write_reader_dataset(
            runtime.roots.data / "waymo",
            segment=sample.segment_id,
            partition=sample.partition,
            timestamps=times,
        )
        expected[sample.sample_id] = times[
            sample.start_frame_index : sample.start_frame_index + 121
        ]
    assert (select_job(job).image_variant, select_job(job).preset.name) == ("moge", "inference_cp1")
    assert [sample.sample_id for sample in samples] == ["example-front-0002", "example-front-0001"]
    events = []

    def small_raster(source):
        # Spatial stand-in only: consume the actual selected Parquet frame stream.
        frames = tuple(source.frames)
        timestamps = tuple(frame.key.frame_timestamp_micros for frame in frames)
        assert timestamps == expected[source.selected.sample_id]
        assert tuple(frame.key for frame in frames) == source.frame_keys
        assert all(frame.key.segment_id == source.selected.segment_id for frame in frames)
        cameras = [frame.camera("FRONT") for frame in frames]
        events.append(source.selected.sample_id)
        return raster.WaymoFrontRaster(
            source.selected,
            source.frame_keys,
            np.stack([camera.rgb[:2, :7] for camera in cameras]),
            np.ones((2, 7), bool),
            np.stack([camera.calibration.intrinsics for camera in cameras]),
            np.stack([camera.world_to_opencv_camera_at_pose_timestamp for camera in cameras]),
            raster.FrontLidarDepth(
                np.arange(122) * 2,
                np.tile(np.array([[0, 0], [1, 0]], np.float32), (121, 1)),
                np.tile(np.array([999.0, 2.0], np.float32), 121),
                np.tile([True, False], 121),
            ),
        )

    def moge(request, *_):
        assert len(tuple(request.rgb_frames)) == 121
        return MogeExecution(
            MogeRawPrediction(
                np.ones((121, 2, 7), np.float32),
                np.ones((121, 2, 7), bool),
                np.zeros((121, 3, 3), np.float32),
            ),
            MogeTelemetry(1.0, 2.0),
        )

    def warp(request, *_):
        if request.source_rgb_layout == "uint8_thwc":
            rgb = np.moveaxis(request.source_rgb.astype(np.float32) * (2 / 255) - 1, -1, 1)
            events.append("outward")
        else:
            rgb = request.source_rgb
            events.append("return")
        return WarpResult(
            rgb[:, None] + 0.01,
            request.source_depth_z_m[:, None] + 1,
            np.ones((121, 1, 2, 7), bool),
            0,
            0,
            1.0,
        )

    record_path = runtime.roots.prepared / "waymo-ddw" / job.name / "prepared.json"

    def process(command, log, *_):
        assert log.parent == runtime.attempt_root / "logs/ddw-preparation"
        if "--output" in command:
            events.append("prompt")
            Path(command[command.index("--output") + 1]).write_bytes(b"model placeholder")
        else:
            # Model output is a boundary double; real PT contracts have a separate proof.
            assert not record_path.exists()
            rows = json.loads(Path(command[command.index("--tasks") + 1]).read_text())
            events.append(tuple(row["sample_id"] for row in rows))
            for row in rows:
                for name in ("base_latent", "pose_latent", "lidar_depth"):
                    Path(row[name]).write_bytes(b"model placeholder")

    monkeypatch.setattr(
        "novel_view.generation.gen3c.resources.build_generation_resources",
        lambda *_: SimpleNamespace(environment_overrides={}),
    )
    monkeypatch.setattr(raster, "build_front_raster", small_raster)
    monkeypatch.setattr(depth, "run_moge", moge)
    monkeypatch.setattr(condition, "run_forward_warp", warp)
    monkeypatch.setattr(bake, "run_process", process)
    assert execute_job(job, runtime) == 0
    record = read_prepared_record(record_path)
    assert [item.sample_id for item in record.items] == [sample.sample_id for sample in samples]
    assert [item.frame_timestamps_micros for item in record.items] == list(expected.values())
    assert events == [
        "prompt",
        samples[0].sample_id,
        "outward",
        "return",
        samples[1].sample_id,
        "outward",
        "return",
        tuple(expected),
    ]
    assert record.items[0].base_latent == f"items/{samples[0].sample_id}/base.pt"
    assert capsys.readouterr().out.strip() == str(record_path)


def test_external_fields_and_historical_prepared_v1(tmp_path):
    job = load_resolved_job(JOB, ROOT)
    assert parse_job(job).recipe.depth_backend == "moge"
    for changed in (
        replace(job, input={**job.input, "unknown": True}),
        replace(job, parameters={**job.parameters, "unknown": True}),
    ):
        with pytest.raises(ValueError):
            parse_job(changed)
    selection = yaml.safe_load((ROOT / job.input["selection"]).read_text())
    selection["unknown"] = True
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(selection))
    with pytest.raises(ValueError):
        load_selection(path)
    fixture = Path(__file__).parents[1] / "contract/fixtures/waymo_ddw_prepared_v1.json"
    record = read_prepared_record(fixture)
    path = tmp_path / "prepared.json"
    write_prepared_record(path, record)
    assert path.read_bytes() == fixture.read_bytes()
    assert [item.sample_id for item in record.items] == ["b", "a"]
    assert all(len(item.frame_timestamps_micros) == 121 for item in record.items)
    assert (
        record.items[0].base_latent,
        record.items[0].pose_latent,
        record.items[0].lidar_depth,
    ) == ("items/b/base.pt", "items/b/pose.pt", "items/b/lidar-depth.pt")
    document = json.loads(path.read_text())
    document["unknown"] = True
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        read_prepared_record(path)


def test_preparation_imports_stay_cold():
    script = (
        "import sys; "
        "from novel_view.preparation.waymo_ddw import spec, selection, source, depth; "
        "import novel_view.workflows.ddw_preparation, novel_view.workflows.runner; "
        "assert not {'torch', 'cv2', 'pyarrow', 'moge'}.intersection(sys.modules); "
        "assert not any(n.endswith('_worker') or n.startswith('cosmos_predict1') "
        "for n in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

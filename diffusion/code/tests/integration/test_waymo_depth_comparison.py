"""Three ordered camera forwards, retained method outcomes and model-error propagation."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from novel_view.models.moge.request import MogeExecution, MogeRawPrediction, MogeTelemetry
from novel_view.models.vggt.request import (
    VggtOmegaExecution,
    VggtOmegaRawPrediction,
    VggtOmegaTelemetry,
)
from novel_view.preparation.waymo_depth import clip, comparison, moge, vggt
from novel_view.preparation.waymo_depth.candidate_record import read_candidate_record
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.preparation.waymo_depth.selection import (
    DEPTH_BACKENDS,
    DepthCandidateFailure,
    DepthClipMetrics,
)
from novel_view.workflows.runner import select_job
from novel_view.workflows.waymo_depth_comparison import run_v1
from tests.support.stage5 import make_job, make_runtime


@pytest.mark.parametrize("vggt_outcome", ("success", "scientific_refusal", "runtime_error"))
def test_three_camera_comparison_keeps_order_logs_and_prior_outcome(
    tmp_path, monkeypatch, vggt_outcome
):
    runtime = make_runtime(tmp_path, image="moge")
    key = WaymoClipKey("waymo-ddw-lora-v1", "training", "segment", 9, tuple(range(9, 130)))
    selection = runtime.roots.selections / "clip.yaml"
    selection.write_text(yaml.safe_dump({"schema_version": 1, "clip": asdict(key)}))
    names = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")
    K = np.tile(np.diag([1000.0, 1000.0, 1.0]), (121, 1, 1))
    W2C = np.tile(np.eye(4), (121, 1, 1))
    cameras = []
    for index, name in enumerate(names):
        path = tmp_path / f"{name}.npy"
        np.save(path, np.full((121, 1, 1, 3), index, np.uint8))
        cameras.append(
            SimpleNamespace(
                evidence=SimpleNamespace(clip_key=key, camera_name=name),
                raster_plan=SimpleNamespace(rectification_known=np.ones((704, 1280), bool)),
                rgb_path=path,
                K_canvas=K,
                world_to_camera_cv=W2C,
            )
        )
    source = SimpleNamespace(clip_key=key, cameras=tuple(cameras))
    monkeypatch.setattr(clip, "build_depth_comparison_clip", lambda *args: source)
    monkeypatch.setattr("novel_view.inputs.waymo.reader.WaymoV2FrameReader", lambda *args: ())
    calls = []
    raw = np.ones((1, 1, 1), np.float32)

    def run_moge(request, resources, log_path):
        frames = [frame.copy() for frame in request.rgb_frames]
        assert len(frames) == 121
        np.testing.assert_array_equal([f[0, 0, 0] for f in frames], [len(calls)] * 121)
        calls.append(log_path)
        np.testing.assert_allclose(request.fov_x_degrees, np.degrees(2 * np.arctan(1280 / 2000)))
        normalized_K = np.tile(
            [[1000 / 1280, 0, 0.5], [0, 1000 / 704, 0.5], [0, 0, 1]], (121, 1, 1)
        )
        return MogeExecution(
            MogeRawPrediction(raw, raw > 0, normalized_K.astype(np.float32)),
            MogeTelemetry(10.0, 20.0),
        )

    def run_vggt(request, resources, log_path):
        assert (runtime.attempt_root / "moge-v1-lidar-scale.json").is_file()
        frames = [frame.copy() for frame in request.rgb_frames]
        assert len(frames) == 121
        np.testing.assert_array_equal([f[0, 0, 0] for f in frames], [len(calls) - 3] * 121)
        calls.append(log_path)
        if vggt_outcome == "runtime_error":
            raise RuntimeError("unexpected model fault")
        if vggt_outcome == "scientific_refusal":
            raise vggt.VggtDepthUnavailable("camera pack has no metric fit")
        return VggtOmegaExecution(
            VggtOmegaRawPrediction(request.input_plan, raw, raw, raw, raw, raw),
            VggtOmegaTelemetry(30.0, 40.0),
        )

    monkeypatch.setattr(moge, "run_moge", run_moge)
    monkeypatch.setattr(vggt, "run_vggt_omega", run_vggt)
    # Numerical gauge/reprojection and metrics have independent tests, not this tiny model double.
    monkeypatch.setattr(
        moge,
        "build_moge_metric_depth",
        lambda *args: SimpleNamespace(clip=object(), scale_by_frame=np.ones(121)),
    )
    monkeypatch.setattr(
        vggt,
        "build_vggt_metric_depth",
        lambda *args: SimpleNamespace(
            clip=object(), alignment=SimpleNamespace(scale_m_per_model_unit=1.0)
        ),
    )
    monkeypatch.setattr(
        comparison,
        "measure_metric_depth",
        lambda result, evidence, scales, **telemetry: DepthClipMetrics(
            evidence.clip_key,
            evidence.camera_name,
            330_000,
            0.9,
            0.1,
            0.2,
            10.0,
            0.1,
            telemetry["elapsed_seconds"],
            telemetry["peak_ram_mib"],
            telemetry["peak_vram_mib"],
        ),
    )
    job = make_job(
        "waymo_depth_comparison",
        1,
        {"reader": "waymo_v2", "dataset": "waymo", "selection": "selections/clip.yaml"},
        {"moge_checkpoint": "moge/model.pt", "vggt_checkpoint": "vggt/model.pt"},
        preset="inference_cp1",
    )
    assert select_job(job).image_variant == "moge"
    if vggt_outcome == "runtime_error":
        with pytest.raises(RuntimeError):
            run_v1(job, runtime)
        assert not (runtime.attempt_root / "result.json").exists()
        assert not (runtime.attempt_root / "vggt-omega-sim3.json").exists()
    else:
        assert run_v1(job, runtime) == 0
        record = read_candidate_record(runtime.attempt_root / "result.json")
        assert record.clip_key == key
        assert tuple(outcome.backend for outcome in record.outcomes) == DEPTH_BACKENDS
        assert tuple(entry.camera_name for entry in record.outcomes[0].entries) == names
        if vggt_outcome == "scientific_refusal":
            assert isinstance(record.outcomes[1], DepthCandidateFailure)
        else:
            assert tuple(entry.camera_name for entry in record.outcomes[1].entries) == names
            assert [entry.peak_vram_mib for entry in record.outcomes[1].entries] == [40.0] * 3
    expected = [f"{backend}__{name.lower()}" for backend in DEPTH_BACKENDS for name in names]
    assert [path.stem for path in calls] == expected[: 6 if vggt_outcome == "success" else 4]
    assert all(path.parent == runtime.attempt_root / "logs" for path in calls)
    assert not list(runtime.roots.cache.glob("waymo-depth-*"))

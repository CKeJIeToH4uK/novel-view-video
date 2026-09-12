"""Five real Gaussian workflows; only the expensive generation model is doubled."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from novel_view.generation.gaussian.clips import plan_native_clips
from novel_view.generation.gaussian.dense import StandardCameraTable, write_standard_camera_table
from novel_view.generation.gaussian.full import build_gaussian_full_sequence
from novel_view.generation.gaussian.record import read_gaussian_run_record
from novel_view.inputs.gaussian.reader import read_gaussian_export
from novel_view.inputs.gaussian.spec import GaussianFullSequenceSelection
from novel_view.workflows.runner import execute_job, select_job
from tests.support.stage5 import make_job, make_runtime


EXPECTED_RECORDS = json.loads(
    (Path(__file__).parents[1] / "contract/fixtures/gaussian_workflow_records.json").read_text())


def _inputs(runtime, count):
    """Generate producer JSON, native pixels and a real camera-table pair."""
    root = runtime.roots.data / "gaussian"
    root.mkdir()
    payload = json.loads((Path(__file__).parents[1] /
                          "contract/fixtures/gaussian_depth_export_v1.json").read_text())
    k = np.array([[100., 0., 4.], [0., 101., 2.], [0., 0., 1.]])
    source = np.repeat(np.eye(4)[None], count, axis=0)
    for index in range(count):
        angle = np.deg2rad(index * .25)
        rotation = np.array([[np.cos(angle), np.sin(angle), 0.],
                             [-np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        source[index, :3, :3] = rotation
        source[index, :3, 3] = -rotation @ np.array([index, 0., 0.])
    shift = np.eye(4)
    shift[0, 3] = 1.
    timestamps = 1_000_000_000 + np.arange(count) * round(1_000_000_000 / 24)
    template = payload["frames"][0]
    payload["camera"].update(image_size_hw=[4, 8], intrinsics=k.tolist())
    payload["frames"] = [dict(
        template, pose_index=10 + index, timestamp_ns=int(timestamps[index]),
        rgb="rgb.png", depth="depth.npy", mask="mask.png",
        source_w2c=pose.tolist(), target_w2c=(shift @ pose).tolist(),
    ) for index, pose in enumerate(source)]
    (root / "export.json").write_text(json.dumps(payload))
    Image.fromarray(np.full((4, 8, 3), 20, np.uint8)).save(root / "rgb.png")
    Image.fromarray(np.full((4, 8), 255, np.uint8)).save(root / "mask.png")
    np.save(root / "depth.npy", np.full((4, 8), 2, np.float32))
    write_standard_camera_table(StandardCameraTable(
        "world", "car", "/camera/inner/frontal/middle", 7,
        timestamps, source, k, (4, 8),
    ), root / "camera-table")
    return read_gaussian_export(root / "export.json"), source


def _job(runtime, version, count=62):
    kinds = {1: "full_sequence", 2: "independent_clips", 3: "dense_independent",
             4: "dense_overlap21", 5: "dense_handoff"}
    selection = dict(schema_version=1, kind=kinds[version], scene_id=9)
    if version == 2:
        selection["clip_indices"] = [4, 1]
    elif version in (3, 4):
        selection["source_pose_range"] = dict(start=10, stop=10 + count - 1)
    elif version == 5:
        selection["dense_camera_ids"] = [118, 120, 122]
    path = runtime.roots.selections / f"gaussian-{version}.yaml"
    path.write_text(json.dumps(selection))
    inputs = dict(reader="gaussian_depth_export", info_file="gaussian/export.json",
                  selection=f"selections/{path.name}")
    generation = dict(backend="gen3c", checkpoint="gen3c/base", seed=7)
    parameters = dict(frame_rate=24, generation=generation)
    if version in (3, 4):
        inputs.update(source_transforms="gaussian/camera-table/transforms.json",
                      source_intrinsics="gaussian/camera-table/intrinsics.json")
        generation.update(checkpoint="gen3c/tuned.pt", model_id="dense-lora", lora=dict(
            working_manifest="/models/working.json", evidence_path="/models/evidence.json",
            epochs=20, strength=.25,
        ))
        parameters["dense"] = dict(densification_factor=2, bake_stride=2, target_capacity=120)
    if version == 4:
        parameters["dense"]["production_capacity"] = 100
        parameters["context_depth"] = dict(backend="moge", checkpoint="moge/model.pt")
    if version == 5:
        inputs = dict(selection=f"selections/{path.name}",
                      independent_attempt="v3", overlap_attempt="v4")
        parameters = dict(bake_dense_stride=2)
    return make_job("gaussian_generation", version, inputs, parameters,
                    preset="cpu_test" if version == 5 else "inference_cp2")


def _model(monkeypatch):
    seen, sessions = [], []
    # Reduce only spatial payload; the 121/120 windows, tails and 20-row seam stay real.
    monkeypatch.setattr("novel_view.generation.gaussian.full.GEN3C_IMAGE_SIZE_HW", (8, 16))
    monkeypatch.setattr("novel_view.generation.gaussian.dense.GEN3C_IMAGE_SIZE_HW", (8, 16))
    monkeypatch.setattr("novel_view.generation.gaussian.record._IMAGE_SIZE_HW", (8, 16))

    @contextmanager
    def session(model, parameters, resources):
        sessions.append((parameters.window_seed_policy, resources.context_parallel_size))

        def generate(sequence, output):
            conditioning = sequence.build_conditioning()
            seen.append((sequence, conditioning))
            rgb = np.zeros((sequence.model_frame_count, 8, 16, 3), np.uint8)
            chunk = getattr(sequence, "chunk", None)
            slots = sequence.target_output_index
            if hasattr(sequence, "previous_context"):
                ids = np.r_[chunk.production_dense_camera_ids, chunk.context_dense_camera_ids]
                slots = np.r_[slots, chunk.context_target_output_index]
            else:
                ids = np.arange(len(sequence)) if chunk is None else chunk.dense_camera_ids
            delta = 3 if getattr(sequence, "previous_context", None) is not None else 0
            rgb[slots] = ((ids + delta) % 251)[:, None, None, None]
            np.save(output, rgb)
            slots = sequence.context_depth_output_slots
            depth = output.with_name("context-depth.npy") if slots is not None else None
            valid = output.with_name("context-valid.npy") if slots is not None else None
            if slots is not None:
                np.save(depth, np.full((len(slots), 8, 16), 5, np.float32))
                np.save(valid, np.ones((len(slots), 8, 16), np.bool_))
            return SimpleNamespace(
                generated_rgb_path=output, elapsed_seconds=.1,
                per_rank_peak_cuda_allocated_bytes=(1, 2),
                per_rank_peak_cuda_reserved_bytes=(3, 4),
                context_depth_path=depth, context_valid_path=valid,
                context_alignment_scale=tuple(1. for _ in (() if slots is None else slots)),
                context_alignment_bias=tuple(0. for _ in (() if slots is None else slots)),
                context_alignment_mae_m=tuple(0. for _ in (() if slots is None else slots)),
                context_valid_fraction=tuple(1. for _ in (() if slots is None else slots)),
            )

        yield SimpleNamespace(generate=generate)

    monkeypatch.setattr("novel_view.generation.gen3c.session.Gen3cGenerationSession", session)
    return seen, sessions


@pytest.mark.parametrize("version", (1, 2))
def test_full_and_clips_execute_real_order_and_records(tmp_path, monkeypatch, version):
    runtime = make_runtime(tmp_path)
    export, source = _inputs(runtime, 383)
    full = build_gaussian_full_sequence(export, GaussianFullSequenceSelection("full_sequence", 9), 24)
    raster = full.read(0)
    assert raster.rgb.shape == (704, 1280, 3)
    assert raster.depth_z_m.dtype == np.float32 and raster.valid.dtype == np.bool_
    assert np.all(raster.depth_z_m[raster.valid] == 2.)
    assert [(clip.target_start_index, clip.target_stop_index) for clip in
            plan_native_clips(full, (1, 2, 3, 4))] == [(0, 120), (120, 240), (240, 360), (263, 383)]
    seen, sessions = _model(monkeypatch)
    job = _job(runtime, version, 383)
    assert select_job(job).image_variant == "core"
    assert execute_job(job, runtime) == 0
    assert sessions == [("autoregressive/v1", 2)]
    assert [sequence.selected_pose_indices[0] for sequence, _ in seen] == (
        [10] if version == 1 else [273, 10]
    )
    for sequence, conditioning in seen:
        np.testing.assert_allclose(conditioning.anchor_to_source_camera[0], np.eye(4), atol=1e-12)
        np.testing.assert_array_equal(conditioning.source_sequence_index[:4], [0, 0, 1, 2])
        np.testing.assert_allclose(conditioning.anchor_to_query_camera[1, :3, 3], [1, 0, 0], atol=1e-12)
        assert np.all(conditioning.source_sequence_index[len(sequence):] == len(sequence) - 1)
        assert np.all(conditioning.anchor_to_query_camera[len(sequence):] ==
                      conditioning.anchor_to_query_camera[len(sequence)])
    paths = [runtime.attempt_root / "run.json"] if version == 1 else [
        runtime.attempt_root / "clips" / f"clip-{index:03d}/run.json" for index in (3, 0)
    ]
    records = [read_gaussian_run_record(path) for path in paths]
    actual = json.dumps([record.to_mapping() for record in records]).replace(str(tmp_path), "<test-root>")
    assert json.loads(actual) == EXPECTED_RECORDS[str(version)]
    legacy = Path(__file__).parents[1] / "contract/fixtures/gaussian_full_run_v1.json"
    monkeypatch.setattr("novel_view.generation.gaussian.record._IMAGE_SIZE_HW", (704, 1280))
    assert read_gaussian_run_record(legacy).input["selected_pose_indices"] == [2, 5]
    dense_legacy = read_gaussian_run_record(legacy.with_name("gaussian_dense_run_v2.json"))
    assert dense_legacy.input["dense_camera_ids"] == [0, 5, 10]
    assert dense_legacy.input["target_output_index"] == [1, 2, 3]


def test_independent_overlap_and_png_preserve_camera_owners_and_nonzero_seam(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, attempt="v3")
    export, source = _inputs(runtime, 62)
    seen, sessions = _model(monkeypatch)
    for version in (3, 4):
        attempt = runtime.roots.runs / f"v{version}"
        attempt.mkdir(exist_ok=True)
        selected_runtime = replace(runtime, attempt_root=attempt)
        job = _job(selected_runtime, version)
        assert select_job(job).image_variant == ("core" if version == 3 else "moge")
        assert execute_job(job, selected_runtime) == 0
        records = [read_gaussian_run_record(path).to_mapping()
                   for path in sorted((attempt / "chunks").glob("*/run.json"))]
        actual = json.dumps(records).replace(str(tmp_path), "<test-root>")
        assert json.loads(actual) == EXPECTED_RECORDS[str(version)]
    assert sessions == [("autoregressive/v1", 2)] * 2
    independent, tail, overlap, inherited = [item[0] for item in seen]
    assert [sequence.unpadded_frame_count - 1 for sequence, _ in seen] == [120, 3, 120, 23]
    assert [sequence.source_frames[0].pose_index for sequence, _ in seen] == [10, 70, 10, 60]
    np.testing.assert_array_equal(tail.chunk.target_output_index, [1, 2, 3])
    np.testing.assert_array_equal(tail.chunk.model_source_pose_indices[-3:], [61, 61, 61])
    np.testing.assert_allclose(seen[1][1].anchor_to_source_camera[0], np.eye(4), atol=1e-12)
    trajectory = independent.trajectory
    shift = np.eye(4)
    shift[0, 3] = 1.
    np.testing.assert_allclose(trajectory.target_w2c[::2], shift[None] @ source, atol=1e-12)
    # Halfway rotation/centre written independently, not via production interpolation.
    angle = np.deg2rad(.125)
    midpoint = np.array([[np.cos(angle), np.sin(angle), 0., 1. - .5 * np.cos(angle)],
                         [-np.sin(angle), np.cos(angle), 0., .5 * np.sin(angle)],
                         [0., 0., 1., 0.], [0., 0., 0., 1.]])
    np.testing.assert_allclose(trajectory.target_w2c[1], midpoint, atol=1e-12)
    assert overlap.previous_context is None
    np.testing.assert_array_equal(inherited.previous_context.dense_camera_ids, np.arange(100, 120))
    np.testing.assert_array_equal(inherited.previous_context.rgb[:, 0, 0, 0], np.arange(100, 120))
    assert np.all(inherited.previous_context.depth_z_m == 5.)
    for version, count in ((3, 3), (4, 23)):
        record = read_gaussian_run_record(runtime.roots.runs / f"v{version}/chunks/chunk-001/run.json")
        assert record.generation["lora"]["strength"] == .25
        assert len(record.input["dense_camera_ids"]) == count
        if version == 4:
            assert record.chunk["previous_record"] == "../chunk-000/run.json"
            seam = record.seam["previous_context_vs_current_production"]
            assert seam["dense_camera_ids"] == list(range(100, 120))
            assert seam["frame_rgb_mse"] == [9.] * 20 and seam["pooled_rgb_mse"] == 9.
    output = runtime.roots.runs / "png"
    output.mkdir()
    png_runtime = replace(runtime, attempt_root=output)
    job = _job(png_runtime, 5)
    assert (select_job(job).image_variant, select_job(job).preset.gpu_count) == ("core", 0)
    assert execute_job(job, png_runtime) == 0
    assert (output / "dense-camera-table/transforms.json").read_bytes() == (
        runtime.attempt_root / "dense-camera-table/transforms.json"
    ).read_bytes()
    for method, slots, expected in (("independent", [119, 1, 3], [118, 120, 122]),
                                    ("overlap21", [19, 21, 23], [121, 123, 125])):
        directory = output / method / "selected"
        record = json.loads((directory / "prepare.json").read_text())
        assert record["method"] == method and record["bake_dense_stride"] == 2
        assert record["dense_transforms"] == "../../dense-camera-table/transforms.json"
        assert [row["dense_camera_id"] for row in record["frames"]] == [118, 120, 122]
        assert [row["target_output_index"] for row in record["frames"]] == slots
        assert sorted(path.name for path in directory.iterdir()) == [
            "frame_00118.png", "frame_00120.png", "frame_00122.png", "prepare.json",
        ]
        for dense_id, value in zip((118, 120, 122), expected):
            with Image.open(directory / f"frame_{dense_id:05d}.png") as image:
                assert np.all(np.asarray(image) == value)

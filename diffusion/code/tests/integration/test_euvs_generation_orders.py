"""Source-only SQLite/PNG → saved VGGT → real EUVS v1/v2 planning and records."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from novel_view.generation.euvs.record import load_run_record
from novel_view.generation.euvs.source_views import unique_source_views
from novel_view.generation.gen3c.windows import iter_window_calls, materialize_window_cameras
from novel_view.inputs.euvs.spec import EuvsSelection, load_euvs_selection
from novel_view.models.vggt.record import load_saved_vggt_omega_prediction
from novel_view.models.vggt.request import VggtOmegaExecution, VggtOmegaRawPrediction, VggtOmegaTelemetry
from novel_view.workflows.runner import execute_job, select_job
from tests.integration.test_nuplan_euvs_reader import dataset, _write_index
from tests.support.stage5 import make_job, make_runtime


@pytest.mark.parametrize("version,policy", ((1, "autoregressive/v1"), (2, "source-reseed/v1")))
def test_source_only_to_generation_preserves_pixels_slots_and_seed_policy(
    dataset, tmp_path, monkeypatch, version, policy,
):
    rows, arguments = dataset
    # Reuse the generated reader fixture; span two real 121-frame windows.
    for row, timestamp in zip(rows[3:], (1_000_000, 6_000_000)):
        row["timestamp_us"] = timestamp
        with sqlite3.connect(tmp_path / "target.db") as database:
            database.execute("UPDATE ego_pose SET timestamp=? WHERE token=?",
                             (timestamp + 1, bytes.fromhex(row["ego_pose_token"])))
    _write_index(tmp_path, rows)
    dataset_root = tmp_path / "euvs"
    dataset_root.mkdir()
    for filename in ("frames.csv", "first.db", "second.db", "target.db", *(
        row["image_path"] for row in rows[:3]
    )):
        (tmp_path / filename).rename(dataset_root / filename)
    runtime = make_runtime(tmp_path / "runtime", attempt="source")
    runtime = replace(runtime, roots=replace(runtime.roots, data=tmp_path))
    tokens = arguments["source_image_tokens"]
    pairs = [dict(
        name=name, tags=[], location="33", direction="source_to_target", channel="CAM_F0",
        source=dict(traversal="2", image_tokens=source_tokens),
        target=dict(traversal="6", image_tokens=arguments["target_image_tokens"]),
    ) for name, source_tokens in (("first", tokens), ("duplicate", tokens), ("second/pair", tokens[1:]))]
    selection_path = runtime.roots.selections / "generation.yaml"
    selection_path.write_text(json.dumps(dict(schema_version=1, pairs=pairs)))
    input_spec = dict(reader="euvs_nuplan", dataset="euvs",
                      selection="selections/generation.yaml")
    seen_pixels = []

    def vggt(request, _resources, _log):
        rgb = tuple(request.rgb_frames)
        pixel_values = [int(frame[352, 640, 0]) for frame in rgb]
        seen_pixels.append(pixel_values)
        count = len(rgb)
        cameras = np.repeat(np.eye(4, dtype=np.float32)[None, :3], count, axis=0)
        cameras[:, 0, 3] = -np.asarray(pixel_values)
        model_k = request.input_plan.source_to_model_pixels @ np.array(
            [[300., 0., 640.], [0., 300., 352.], [0., 0., 1.]],
        )
        depth = np.ones((count, *request.input_plan.model_size_hw), np.float32)
        return VggtOmegaExecution(VggtOmegaRawPrediction(
            request.input_plan, depth, depth.copy(), np.zeros((count, 9), np.float32),
            cameras, np.repeat(model_k[None], count, axis=0).astype(np.float32),
        ), VggtOmegaTelemetry(1., 2.))

    monkeypatch.setattr("novel_view.models.vggt.backend.run_vggt_omega", vggt)
    source_job = make_job("euvs_source_views", 1, input_spec,
                          dict(geometry=dict(backend="vggt_omega")), preset="inference_cp1")
    assert (select_job(source_job).image_variant, select_job(source_job).preset.gpu_count) == ("core", 1)
    assert execute_job(source_job, runtime) == 0
    assert seen_pixels == [[30, 10, 20], [10, 20]]
    for expected in (tokens, tokens[1:]):
        saved = load_saved_vggt_omega_prediction(runtime.attempt_root / "source-views" / expected[0])
        assert saved.source_tokens == expected
        np.testing.assert_array_equal(saved.prediction.model_w2c[:, 0, 3],
                                      [-10 * int(token, 16) for token in expected])
    selected = load_euvs_selection(selection_path)
    reversed_pair = replace(selected.pairs[0], source=replace(
        selected.pairs[0].source, image_tokens=tokens[::-1],
    ))
    unique = unique_source_views(EuvsSelection((selected.pairs[0], selected.pairs[1], reversed_pair)))
    assert [pair.source.image_tokens for pair in unique] == [tokens, tokens[::-1]]

    observations, sessions = [], []

    @contextmanager
    def session(model, parameters, resources):
        sessions.append((parameters.window_seed_policy, resources.context_parallel_size))

        def generate(request, output):
            conditioning = request.build_conditioning()
            calls = tuple(iter_window_calls(conditioning.source_sequence_index, parameters.window_seed_policy))
            windows = tuple(materialize_window_cameras(
                conditioning.anchor_to_query_camera, conditioning.query_intrinsics,
                conditioning.anchor_to_source_camera, conditioning.source_intrinsics, call,
            ) for call in calls)
            observations.append((request, conditioning, calls, windows))
            # Sparse file retains the actual model shape; write only selected target rows.
            generated = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint8,
                                                  shape=(request.model_frame_count, 704, 1280, 3))
            generated[1] = 41
            generated[121] = 51
            generated.flush()
            del generated
            return SimpleNamespace(generated_rgb_path=output)

        yield SimpleNamespace(generate=generate)

    monkeypatch.setattr("novel_view.generation.gen3c.session.Gen3cGenerationSession", session)
    output = runtime.roots.runs / f"v{version}"
    output.mkdir()
    job = make_job("euvs_generation", version, dict(input_spec, source_views_attempt="source"),
                   dict(geometry=dict(backend="vggt_omega"),
                        generation=dict(backend="gen3c", checkpoint="gen3c/base", seed=42)))
    assert (select_job(job).image_variant, select_job(job).preset.gpu_count) == ("core", 2)
    assert execute_job(job, replace(runtime, attempt_root=output)) == 0
    assert sessions == [(policy, 2)]
    assert [request.source_rgb.sequence_id for request, *_ in observations] == [tokens, tokens, tokens[1:]]
    for name, source_indices, observation in zip(
        ("first", "duplicate", "second%2Fpair"), ((0, 0), (0, 0), (1, 1)), observations,
    ):
        request, conditioning, calls, windows = observation
        record = load_run_record(output / "pairs" / name / "run.json")
        assert record.pair.name == name.replace("%2F", "/")
        assert record.target_output_index == (1, 121)
        assert record.target_source_sequence_index == source_indices
        assert record.sampling.window_seed_policy == policy and record.output.shape == (241, 704, 1280, 3)
        assert [(call.start, call.stop, call.generated_slice_start) for call in calls] == [
            (0, 121, 0), (120, 241, 1),
        ]
        assert [call.source_index for call in calls] == [source_indices[0]] * 2
        assert calls[1].seed_kind == ("previous" if version == 1 else "source")
        if version == 1:
            np.testing.assert_array_equal(windows[0][0][-1], windows[1][0][0])
            np.testing.assert_array_equal(windows[0][1][-1], windows[1][1][0])
        else:
            source_index = calls[1].source_index
            np.testing.assert_array_equal(
                windows[1][0][0], conditioning.anchor_to_source_camera[source_index].astype(np.float32),
            )
            np.testing.assert_array_equal(
                windows[1][1][0], conditioning.source_intrinsics[source_index].astype(np.float32),
            )
            assert not np.array_equal(windows[1][0][0], conditioning.anchor_to_query_camera[120])
    assert not any((dataset_root / row["image_path"]).exists() for row in rows[3:])

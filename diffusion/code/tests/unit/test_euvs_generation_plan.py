from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from novel_view.generation.euvs import plan as plan_module
from novel_view.generation.euvs.plan import (
    EuvsGenerationInput,
    build_euvs_generation_plan,
    build_euvs_source_mapping,
)
from novel_view.generation.euvs.source_views import ResolvedEuvsVggtGeometry
from novel_view.generation.gen3c.conditioning import (
    materialize_gen3c_conditioning_window,
)
from novel_view.generation.gen3c.timeline import Gen3cTimelineError
from novel_view.geometry.depth import PosedDepthSequence
from novel_view.geometry.sim3 import CameraPackAlignment
from novel_view.inputs.nuplan.camera import GlobalCameraGeometry


def _frame(x: float, timestamp_us: int):
    w2c = np.eye(4, dtype=np.float64)
    w2c[0, 3] = -x
    return SimpleNamespace(
        geometry=GlobalCameraGeometry(w2c, 32611),
        ref=SimpleNamespace(timestamp_us=timestamp_us),
    )


def _pair(
    target_x: tuple[float, ...] = (20.0, 4.0),
    target_timestamps: tuple[int, ...] = (1_000_000, 6_000_000),
    source_x: tuple[float, ...] = (0.0, 10.0, 30.0),
):
    source = tuple(_frame(x, index) for index, x in enumerate(source_x))
    target = tuple(
        _frame(x, timestamp)
        for x, timestamp in zip(target_x, target_timestamps, strict=True)
    )
    return SimpleNamespace(
        name="pair-a",
        source=SimpleNamespace(frames=source),
        target=SimpleNamespace(frames=target),
    )


def _resolved_input(pair):
    count = len(pair.source.frames)
    source_w2c = np.repeat(np.eye(4)[None], count, axis=0)
    source_w2c[:, 0, 3] = (-2.0, -12.0, -32.0)[:count]
    intrinsics = np.repeat(np.eye(3)[None], count, axis=0)
    geometry = PosedDepthSequence(
        tuple(str(index) for index in range(count)),
        np.ones((count, 2, 3), np.float32),
        np.ones((count, 2, 3), np.bool_),
        intrinsics,
        source_w2c,
    )
    target_w2c = np.stack(
        [frame.geometry.global_to_camera for frame in pair.target.frames]
    )
    rgb = SimpleNamespace(image_size_hw=(2, 3), frames=())
    return (
        EuvsGenerationInput(
            pair,
            rgb,
            geometry,
            np.repeat(np.eye(3)[None], len(pair.target.frames), axis=0),
            target_w2c,
        ),
        ResolvedEuvsVggtGeometry(
            CameraPackAlignment(
                2.0,
                np.eye(3),
                np.array((2.0, 0.0, 0.0)),
                source_w2c,
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
            ),
            geometry,
            "ordered-source-tokens",
        ),
    )


def test_full_plan_golden_keeps_mapping_timeline_and_window_overlap(
    monkeypatch,
) -> None:
    pair = _pair()
    generation_input, resolved = _resolved_input(pair)
    monkeypatch.setattr(
        plan_module,
        "build_euvs_generation_input",
        lambda *_: generation_input,
    )

    result = build_euvs_generation_plan(pair, generation_input.source_rgb, resolved)

    npt.assert_array_equal(result.mapping.source_sequence_index, (1, 0))
    npt.assert_allclose(result.mapping.target_source_progress, (20.0, 4.0))
    npt.assert_allclose(result.mapping.source_progress_residual, (-10.0, -4.0))
    npt.assert_array_equal(result.timeline.target_output_index, (1, 121))
    assert result.timeline.model_frame_count == 241
    npt.assert_allclose(
        result.conditioning.query_trajectory.anchor_to_target_camera[:, 0, 3],
        (-8.0, 8.0),
    )
    first, second = tuple(result.timeline.iter_windows())
    left = materialize_gen3c_conditioning_window(result.conditioning, first)
    right = materialize_gen3c_conditioning_window(result.conditioning, second)
    npt.assert_array_equal(
        left.anchor_to_query_camera[-1], right.anchor_to_query_camera[0]
    )
    conditioning = result.request.build_conditioning()
    npt.assert_array_equal(conditioning.source_sequence_index[[1, 121]], (1, 0))


def test_endpoint_excess_and_selected_segment_branch_remain_explicit() -> None:
    endpoint = build_euvs_source_mapping(
        _pair(target_x=(-3.0, 14.0), source_x=(0.0, 10.0))
    )
    branch = build_euvs_source_mapping(
        _pair(
            target_x=(49.0,),
            target_timestamps=(1_000_000,),
            source_x=(0.0, 100.0, 101.0, 49.0),
        )
    )

    npt.assert_allclose(endpoint.alignment.source_domain_excess, (-3.0, 4.0))
    npt.assert_allclose(endpoint.target_source_progress, (-3.0, 14.0))
    npt.assert_array_equal(endpoint.source_sequence_index, (0, 1))
    assert int(branch.alignment.projection.segment_index[0]) == 0
    assert int(branch.source_sequence_index[0]) == 0


def test_timestamp_collision_is_rejected(monkeypatch) -> None:
    pair = _pair(target_timestamps=(1_000_000, 1_020_000))
    generation_input, resolved = _resolved_input(pair)
    monkeypatch.setattr(
        plan_module,
        "build_euvs_generation_input",
        lambda *_: generation_input,
    )

    with pytest.raises(Gen3cTimelineError, match=r"target\[0\].*target\[1\].*slot 1"):
        build_euvs_generation_plan(pair, generation_input.source_rgb, resolved)

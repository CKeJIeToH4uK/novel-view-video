"""Historical metric depth: independent scale, camera and heldout oracles."""

from dataclasses import replace

import numpy as np
import pytest

from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.models.moge.request import MogeExecution, MogeRawPrediction, MogeTelemetry
from novel_view.preparation.waymo_depth import depth, moge, vggt
from novel_view.preparation.waymo_depth.depth_samples import ProjectedLidarDepth
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.preparation.waymo_depth.metrics import measure_metric_depth, sample_bilinear_depth


def _array(value, dtype):
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _evidence(targets, xy):
    targets = np.asarray(targets).reshape(121, -1)
    count = targets.size
    per_frame = targets.shape[1]
    key = WaymoClipKey("waymo-ddw-lora-v1", "training", "segment", 0, tuple(range(121)))
    return ProjectedLidarDepth(
        key,
        "FRONT",
        _array(np.arange(122) * per_frame, np.int64),
        _array(np.ones(count), np.uint8),
        _array(np.ones(count), np.uint8),
        _array(np.tile(np.arange(per_frame), 121), np.int32),
        _array(np.tile(xy, (121, 1)), np.float32),
        _array(targets.ravel(), np.float32),
        _array(np.arange(count) % 5 == 0, bool),
    )


@pytest.fixture
def cameras(monkeypatch):
    monkeypatch.setattr(depth, "GEN3C_CANVAS_SIZE_HW", (2, 3))
    K = _array(np.broadcast_to([[8, 0, 1], [0, 9, 0.5], [0, 0, 1]], (121, 3, 3)), np.float64)
    W2C = _array(np.broadcast_to(np.eye(4), (121, 4, 4)), np.float64)
    return K, W2C


def test_moge_per_frame_scale_ignores_heldout_and_preserves_measured_cameras(cameras):
    raw = np.tile(np.arange(1, 7, dtype=np.float32).reshape(1, 2, 3), (121, 1, 1))
    scales = np.linspace(1, 3, 121)
    targets = raw.reshape(121, 6)[:, :5] * scales[:, None]
    targets[:, 0] = 0.01  # This corrupted point is held out in every frame.
    evidence = _evidence(targets, [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]])
    valid = np.ones_like(raw, bool)
    known = np.array([[True, True, True], [True, True, False]])
    result = moge.build_moge_metric_depth(evidence, raw, valid, known, *cameras)
    np.testing.assert_allclose(result.scale_by_frame, scales, rtol=1e-7)
    expected = raw * scales[:, None, None]
    expected[:, 1, 2] = 0
    np.testing.assert_allclose(result.clip.depth_z_m, expected, rtol=1e-6)
    np.testing.assert_array_equal(result.clip.valid, np.broadcast_to(known, raw.shape))
    np.testing.assert_array_equal(result.clip.K_canvas, cameras[0])
    np.testing.assert_array_equal(result.clip.world_to_camera_cv, cameras[1])
    for model_depth, mask in ((raw.copy(), valid), (raw, np.zeros_like(valid))):
        if mask.any():
            model_depth[0, 0, 1] = 0
        with pytest.raises(moge.MogeDepthUnavailable):
            moge.build_moge_metric_depth(evidence, model_depth, mask, known, *cameras)


def test_moge_rejects_model_camera_that_changes_measured_fov(cameras, monkeypatch):
    raw = np.ones((121, 2, 3), np.float32)
    evidence = _evidence(np.ones((121, 5)), [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]])
    execution = MogeExecution(
        MogeRawPrediction(raw, np.ones_like(raw, bool), np.zeros((121, 3, 3), np.float32)),
        MogeTelemetry(1, 1),
    )
    monkeypatch.setattr(moge, "run_moge", lambda *args: execution)
    with pytest.raises(moge.MogeDepthUnavailable):
        moge.run_moge_metric_depth(
            (), evidence, np.ones((2, 3), bool), *cameras, object(), object()
        )


def test_relative_l1_weighted_median_and_four_neighbour_sampling():
    predicted = np.array([1, 1, 100], np.float32)
    targets = np.array([1, 2, 1000], np.float32)
    scale = moge.fit_positive_relative_l1_scale(predicted, targets)
    assert scale == 1  # Ratios 1,2,10 carry unequal weights 1,.5,.1.
    costs = [np.sum(np.abs(s * predicted - targets) / targets) for s in (1, 2, 10)]
    assert costs[0] < min(costs[1:])
    pixels = np.array([[1, 3], [5, 7]], np.float32)
    valid = np.ones((2, 2), bool)
    xy = np.array([[0.5, 0.5], [1, 1], [1.1, 1]], np.float32)
    sampled, supported = sample_bilinear_depth(pixels, valid, xy)
    np.testing.assert_allclose(sampled[:2], [4, 7])
    np.testing.assert_array_equal(supported, [True, True, False])
    for row, column in np.ndindex(2, 2):
        mask = valid.copy()
        mask[row, column] = False
        assert not sample_bilinear_depth(pixels, mask, xy[:1])[1][0]


def test_vggt_camera_sim3_is_independent_of_lidar_values(cameras):
    raw = np.tile(np.array([[[1, 3]]], np.float32), (121, 1, 1))
    model = np.tile(np.eye(4, dtype=np.float32)[:3], (121, 1, 1))
    model[:, 0, 3] = -np.arange(121, dtype=np.float32) / 100
    measured = np.tile(np.eye(4), (121, 1, 1))
    measured[:, 0, 3] = 2 * model[:, 0, 3].astype(np.float64) - 5
    measured[:, 1, 3] = 3
    measured = _array(measured, np.float64)
    model_K = _array(np.tile(np.eye(3), (121, 1, 1)), np.float32)
    target_K = _array(model_K, np.float64)
    known = np.array([[True, False, True], [True, True, True]])
    results = [
        vggt.build_vggt_metric_depth(
            _evidence(np.full(121, value), [[0, 0]]),
            raw,
            model_K,
            model,
            np.ones((1, 2), bool),
            known,
            target_K,
            measured,
        )
        for value in (1, 1_000_000)
    ]
    for result in results:
        assert result.alignment.scale_m_per_model_unit == pytest.approx(2)
        np.testing.assert_allclose(result.alignment.aligned_world_to_camera, measured, atol=2e-7)
        np.testing.assert_allclose(
            result.clip.depth_z_m, np.tile([[[2, 0, 0], [0, 0, 0]]], (121, 1, 1))
        )
        np.testing.assert_array_equal(result.clip.valid, result.clip.depth_z_m > 0)
        np.testing.assert_array_equal(result.clip.K_canvas, target_K)
        np.testing.assert_array_equal(result.clip.world_to_camera_cv, measured)
    raw[0, 0, 0] = 0
    with pytest.raises(WaymoContractError):
        vggt.build_vggt_metric_depth(
            _evidence(np.ones(121), [[0, 0]]),
            raw,
            model_K,
            model,
            np.ones((1, 2), bool),
            known,
            target_K,
            measured,
        )


def test_vggt_reprojection_uses_target_pose_nearest_z_and_source_mask():
    target = np.eye(4)
    target[0, 3] = 0.5
    cases = (
        ([[2, 1]], [[True, True]], 2, [2, 2, 1]),
        ([[2, 1]], [[True, True]], 1, [2, 1, 1]),
        ([[1, 2]], [[False, True]], 1, [0, 2, 2]),
    )
    for pixels, mask, fx, expected in cases:
        K = np.diag([fx, 1, 1])
        actual, valid = vggt._reproject_metric_frame(
            np.array(pixels, np.float32),
            np.array(mask, bool),
            K,
            np.eye(4),
            K,
            target,
            1,
            (1, 3),
        )
        np.testing.assert_allclose(actual, [expected])
        np.testing.assert_array_equal(valid, np.array([expected]) > 0)
    with pytest.raises(WaymoContractError), np.errstate(over="ignore"):
        vggt._reproject_metric_frame(
            np.full((1, 1), np.finfo(np.float32).max, np.float32),
            np.ones((1, 1), bool),
            np.eye(3),
            np.eye(4),
            np.eye(3),
            np.eye(4),
            np.finfo(np.float64).max,
            (1, 1),
        )


def test_metrics_use_valid_heldout_linear_p95_and_median_log_jitter(cameras):
    evidence = _evidence(np.full(121, 10), [[0.25, 0.5]])
    pixels = np.full((121, 2, 3), 10, np.float32)
    pixels[::5] += np.arange(25, dtype=np.float32)[:, None, None] ** 2
    pixels[0] = 0
    clip = depth.MetricDepthClip(
        evidence.clip_key, "FRONT", _array(pixels, np.float32), _array(pixels > 0, bool), *cameras
    )
    scales = 1.0 + np.arange(121) % 2
    scales[61] = 4
    telemetry = dict(elapsed_seconds=3.5, peak_ram_mib=456, peak_vram_mib=123)
    result = measure_metric_depth(clip, evidence, scales, **telemetry)
    assert (result.heldout_count, result.valid_fraction) == (25, 24 / 25)
    assert (result.median_abs_z_m, result.median_abs_rel) == pytest.approx((156.5, 15.65))
    assert result.p95_abs_z_m == pytest.approx(522.25)  # Linear interpolation of squares 1²…24².
    assert result.temporal_scale_jitter == pytest.approx(np.log(2))
    assert (result.elapsed_seconds, result.peak_ram_mib, result.peak_vram_mib) == (3.5, 456, 123)
    pixels[1::5] = 1_000_000
    altered = replace(clip, depth_z_m=_array(pixels, np.float32))
    constant = np.full(121, 7.0)
    first = measure_metric_depth(clip, evidence, constant, **telemetry)
    assert first == measure_metric_depth(altered, evidence, constant, **telemetry)
    assert first.temporal_scale_jitter == 0
    empty = replace(
        clip,
        depth_z_m=_array(np.zeros_like(pixels), np.float32),
        valid=_array(np.zeros_like(pixels), bool),
    )
    with pytest.raises(WaymoContractError):
        measure_metric_depth(empty, evidence, constant, **telemetry)

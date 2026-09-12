"""Независимая математика FRONT raster, metric depth и A→B→A′."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from novel_view.inputs.waymo.camera import (
    OPENCV_FROM_WAYMO_CAMERA,
    WaymoCameraCalibration,
    WaymoCameraFrame,
)
from novel_view.inputs.waymo.frame import WaymoFrameBundle, WaymoSegmentContext
from novel_view.inputs.waymo.lidar import (
    WaymoLidarCalibration,
    WaymoLidarFrame,
    WaymoLidarReturn,
)
from novel_view.inputs.waymo.types import WaymoFrameKey
from novel_view.models.moge.backend import MogeResources
from novel_view.models.moge.request import MogeExecution, MogeRawPrediction, MogeTelemetry
from novel_view.preparation.waymo_ddw import condition, depth
from novel_view.preparation.waymo_ddw.raster import (
    FrontLidarDepth,
    WaymoFrontRaster,
    _rectify_projection_xy,
    build_front_raster,
    build_front_raster_plan,
    filter_lidar_to_rectification_support,
    project_front_lidar,
)
from novel_view.preparation.waymo_ddw.selection import SelectedDdwSample
from novel_view.preparation.waymo_ddw.source import WaymoFrontSource
from novel_view.preparation.waymo_ddw.target_path import build_target_path
from novel_view.preparation.waymo_ddw.warp import WarpResources, WarpResult
from novel_view.preparation.waymo_depth.raster import rectify_waymo_projection_xy


def _measured_frame():
    calibration = WaymoCameraCalibration(
        "FRONT",
        np.array([[145.0, 0, 99], [0, 143, 54], [0, 0, 1]]),
        np.array([-0.12, 0.025, 0.001, -0.002, -0.004]),
        np.eye(4),
        200,
        110,
        4,
    )
    y, x = np.mgrid[:110, :200]
    rgb = np.stack((x % 251, y % 251, (3 * x + y) % 251), -1).astype(np.uint8)
    pose = np.eye(4)
    pose[0, 3] = 3.0
    camera = WaymoCameraFrame(
        calibration,
        b"",
        rgb,
        pose,
        np.zeros(3, np.float32),
        np.zeros(3),
        1.0,
        1.0,
        1.0,
        0.0,
    )
    lidar_calibration = WaymoLidarCalibration("TOP", np.eye(4), 0.0, 0.0, np.array([0.0]))
    ranges = np.zeros((1, 7, 4), np.float32)
    ranges[..., 0] = np.arange(10.0, 17.0)
    projection = np.zeros((1, 7, 6), np.float32)
    projection[..., 0] = 1
    projection[..., 1] = np.arange(96.0, 103.0)
    projection[..., 2] = 54
    pixels = np.zeros((1, 7, 6), np.float32)
    pixels[..., 3] = 100.0
    lidar = WaymoLidarFrame(
        lidar_calibration,
        WaymoLidarReturn(ranges, projection),
        WaymoLidarReturn(np.zeros_like(ranges), np.zeros_like(projection)),
        pixels,
    )
    return WaymoFrameBundle(
        WaymoFrameKey("segment", 1_000_000),
        WaymoSegmentContext("segment", (calibration,), (lidar_calibration,)),
        np.eye(4),
        (camera,),
        (lidar,),
    )


def test_measured_raster_and_holdout_before_support_filter():
    frame = _measured_frame()
    calibration = frame.cameras[0].calibration
    plan = build_front_raster_plan(calibration)
    source = WaymoFrontSource(
        SelectedDdwSample("a", "training", "segment", 0, 1.0, -1),
        (frame.key,),
        iter((frame,)),
    )
    raster = build_front_raster(source)
    K, roi = cv2.getOptimalNewCameraMatrix(
        calibration.intrinsics,
        calibration.distortion_k1_k2_p1_p2_k3,
        (200, 110),
        0.0,
        (200, 110),
        centerPrincipalPoint=False,
    )
    left, top, width, height = plan.crop_xywh
    scale = 1280 / width
    expected_K = K.copy()
    expected_K[:2, :2] *= scale
    expected_K[:2, 2] = scale * (K[:2, 2] - (left, top) + 0.5) - 0.5
    assert plan.valid_roi_xywh == tuple(roi)
    assert width * 11 == height * 20
    assert (left, top) == (roi[0] + (roi[2] - width) // 2, roi[1] + (roi[3] - height) // 2)
    np.testing.assert_allclose(raster.K_canvas[0], expected_K, atol=1e-12)
    camera = OPENCV_FROM_WAYMO_CAMERA.copy()
    camera[2, 3] = -3.0
    np.testing.assert_array_equal(raster.world_to_camera_cv[0], camera)
    expected_rgb = cv2.remap(frame.cameras[0].rgb, plan.map_x, plan.map_y, cv2.INTER_LINEAR)
    expected_rgb[~plan.rectification_known] = 0
    np.testing.assert_array_equal(raster.rgb_thwc[0], expected_rgb)
    assert raster.rgb_thwc.shape == (1, 704, 1280, 3)
    assert raster.rgb_thwc.dtype == np.uint8
    angle = (((7 - np.arange(7) - 0.5) / 7) * 2 - 1) * np.pi
    expected_z = 97.0 + np.cos(angle) * np.arange(10.0, 17.0)
    lidar = project_front_lidar(plan, (frame,))
    np.testing.assert_allclose(lidar.depth_z_m, expected_z, rtol=1e-6)
    np.testing.assert_array_equal(lidar.evaluation_holdout, [1, 0, 0, 0, 0, 1, 0])
    known = plan.rectification_known.copy()
    x, y = np.floor(lidar.canvas_xy[1]).astype(int)
    known[y : y + 2, x : x + 2] = False
    filtered = filter_lidar_to_rectification_support(lidar, known)
    np.testing.assert_array_equal(filtered.frame_offsets, [0, 6])
    np.testing.assert_allclose(filtered.depth_z_m, np.delete(expected_z, 1))
    np.testing.assert_array_equal(filtered.evaluation_holdout, [1, 0, 0, 0, 1, 0])


def test_front_does_not_adopt_legacy_zero_epsilon():
    plan = build_front_raster_plan(_measured_frame().cameras[0].calibration)
    K = np.eye(3)
    K[0, 2] = -2e-5
    plan = replace(
        plan,
        canvas_intrinsics=K,
        calibration=replace(
            plan.calibration,
            intrinsics=np.eye(3),
            distortion_k1_k2_p1_p2_k3=np.zeros(5),
        ),
    )
    points = np.array([[0, 0], [1.5e-5, 0], [2.5e-5, 0]], np.float32)
    legacy, old_valid = rectify_waymo_projection_xy(plan, points)
    front, valid = _rectify_projection_xy(plan, points)
    np.testing.assert_allclose(legacy[:, 0], [-2e-5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(front[:, 0], [-2e-5, -5e-6, 5e-6], atol=1e-12)
    assert old_valid.tolist() == [False, True, True]
    assert valid.tolist() == [False, False, True]


def test_moge_metric_scale_uses_relative_l1_and_only_fit_points(monkeypatch, tmp_path):
    raw = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    target = (raw.reshape(2, 6)[:, :5] * [[2], [3]]).ravel().astype(np.float32)
    target[[0, 5]] = 999.0
    K = np.broadcast_to(np.diag([3.0, 3.0, 1.0]), (2, 3, 3)).copy()
    W2C = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    W2C[1, 0, 3] = 7.0
    front = WaymoFrontRaster(
        SelectedDdwSample("a", "training", "segment", 0, 1.0, -1),
        tuple(WaymoFrameKey("segment", i) for i in (1, 2)),
        np.repeat(np.arange(2, dtype=np.uint8)[:, None, None, None], 18).reshape(2, 2, 3, 3),
        np.array([[1, 1, 1], [1, 1, 0]], bool),
        K,
        W2C,
        FrontLidarDepth(
            np.array([0, 5, 10]),
            np.tile(
                np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]], np.float32),
                (2, 1),
            ),
            target,
            np.array([1, 0, 0, 0, 0] * 2, bool),
        ),
    )

    def model(request, resources, log):
        assert [int(frame[0, 0, 0]) for frame in request.rgb_frames] == [0, 1]
        np.testing.assert_allclose(request.fov_x_degrees, np.degrees(2 * np.arctan(0.5)))
        return MogeExecution(
            MogeRawPrediction(
                raw,
                np.ones_like(raw, bool),
                np.zeros((2, 3, 3), np.float32),
            ),
            MogeTelemetry(12.0, 34.0),
        )

    monkeypatch.setattr(depth, "run_moge", model)
    result = depth.predict_moge_metric_depth(
        front, MogeResources(tmp_path, tmp_path / "m"), tmp_path / "log"
    )
    np.testing.assert_allclose(result.scale_by_frame, [2, 3])
    np.testing.assert_allclose(result.depth_z_m[:, 0], raw[:, 0] * [[2], [3]])
    assert not result.valid[:, 1, 2].any()
    assert not result.depth_z_m[:, 1, 2].any()
    np.testing.assert_array_equal(result.K_canvas, K)
    np.testing.assert_array_equal(result.world_to_camera_cv, W2C)
    assert result.telemetry == MogeTelemetry(12.0, 34.0)
    predicted, target = np.array([1.0, 1.0, 100.0]), np.array([1.0, 2.0, 1000.0])
    scale = depth.fit_positive_relative_l1_scale(predicted, target)
    assert scale == 1.0
    objective = lambda s: np.sum(np.abs(s * predicted - target) / target)
    assert objective(scale) < min(objective(2.0), objective(10.0))


@pytest.mark.parametrize("sign", [-1, 1])
def test_cosine_path_and_float_outward_result_feed_return(monkeypatch, sign):
    measured = np.broadcast_to(np.eye(4), (121, 4, 4)).copy()
    angle = np.deg2rad(35.0)
    measured[:, :2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    measured[:, :3, 3] = np.linspace([1.0, -2.0, 0.5], [2.0, -1.0, 0.5], 121)
    original = measured.copy()
    path = build_target_path(measured, 3.0, sign)
    expected = sign * 3 * (1 - np.cos(np.arange(121) * np.pi / 120)) / 2
    np.testing.assert_allclose(path.displacement_m, expected, atol=1e-14)
    virtual = measured.copy()
    virtual[:, 0, 3] -= expected
    np.testing.assert_allclose(path.virtual_world_to_camera_cv, virtual, atol=1e-14)
    np.testing.assert_array_equal(measured, original)
    np.testing.assert_array_equal(
        build_target_path(measured, 0.0, sign).virtual_world_to_camera_cv, measured
    )
    K = np.broadcast_to(np.diag([3.0, 4.0, 1.0]), (121, 3, 3)).copy()
    front = SimpleNamespace(
        rgb_thwc=np.zeros((121, 2, 3, 3), np.uint8),
        rectification_known=np.ones((2, 3), bool),
        K_canvas=K,
        world_to_camera_cv=measured,
    )
    metric = SimpleNamespace(
        depth_z_m=np.full((121, 2, 3), 4.0, np.float32), valid=np.ones((121, 2, 3), bool)
    )
    outward = WarpResult(
        np.full((121, 1, 3, 2, 3), 0.123456, np.float32),
        np.full((121, 1, 2, 3), 5.5, np.float32),
        np.ones((121, 1, 2, 3), bool),
        10,
        20,
        1.25,
    )
    outward.known[1, 0, 0, 1] = False
    returning = replace(
        outward,
        rgb_minus_one_to_one=outward.rgb_minus_one_to_one + 0.2,
        peak_cuda_allocated_bytes=15,
        peak_cuda_reserved_bytes=18,
        elapsed_seconds=2.5,
    )
    calls = []

    def render(request, *_):
        calls.append(request)
        return outward if len(calls) == 1 else returning

    monkeypatch.setattr(condition, "run_forward_warp", render)
    result = condition.build_condition(
        front, metric, path, WarpResources(Path(".")), Path("out"), Path("back")
    )
    first, second = calls
    assert (first.source_rgb_layout, second.source_rgb_layout) == ("uint8_thwc", "normalized_tchw")
    for actual, expected in (
        (second.source_rgb, outward.rgb_minus_one_to_one[:, 0]),
        (second.source_depth_z_m, outward.depth_z_m[:, 0]),
        (second.source_depth_valid, outward.known[:, 0]),
    ):
        np.testing.assert_array_equal(actual, expected)
    assert second.source_rectification_known.all()
    np.testing.assert_allclose(second.source_world_to_camera_cv, virtual, atol=1e-14)
    np.testing.assert_array_equal(second.target_world_to_camera_cv, measured)
    np.testing.assert_array_equal(second.target_K_canvas, K)
    np.testing.assert_array_equal(result.rgb_minus_one_to_one, returning.rgb_minus_one_to_one[:, 0])
    np.testing.assert_array_equal(result.known, returning.known[:, 0])
    assert (
        result.peak_cuda_allocated_bytes,
        result.peak_cuda_reserved_bytes,
        result.elapsed_seconds,
    ) == (15, 20, 3.75)

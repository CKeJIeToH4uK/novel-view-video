"""Measured three-camera raster and its historical near-zero convention."""

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from novel_view.inputs.waymo.camera import WaymoCameraCalibration
from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.preparation.waymo_depth.raster import (
    build_waymo_gen3c_raster_plan,
    rectify_waymo_projection_xy,
    rectify_waymo_rgb,
)


@pytest.fixture
def plan():
    calibration = WaymoCameraCalibration(
        "FRONT",
        np.array([[145.0, 0, 99], [0, 143, 54], [0, 0, 1]]),
        np.array([-0.12, 0.025, 0.001, -0.002, -0.004]),
        np.eye(4),
        200,
        110,
        4,
    )
    return build_waymo_gen3c_raster_plan(calibration)


def test_alpha_zero_central_crop_and_half_pixel_camera(plan):
    calibration = plan.calibration
    native_K, roi = cv2.getOptimalNewCameraMatrix(
        calibration.intrinsics,
        calibration.distortion_k1_k2_p1_p2_k3,
        (200, 110),
        0.0,
        (200, 110),
        centerPrincipalPoint=False,
    )
    assert plan.valid_roi_xywh == tuple(roi)
    left, top, width, height = plan.crop_xywh
    assert width * 11 == height * 20
    assert (left, top) == (roi[0] + (roi[2] - width) // 2, roi[1] + (roi[3] - height) // 2)
    assert left + width <= roi[0] + roi[2] and top + height <= roi[1] + roi[3]
    scale = 1280 / width
    expected = native_K.copy()
    expected[:2, :2] *= scale
    expected[:2, 2] = scale * (native_K[:2, 2] - [left, top] + 0.5) - 0.5
    np.testing.assert_allclose(plan.canvas_intrinsics, expected, atol=1e-12, rtol=0)
    assert plan.canvas_intrinsics[0, 2] != scale * (native_K[0, 2] - left)
    for name in ("FRONT_LEFT", "FRONT_RIGHT"):
        other = build_waymo_gen3c_raster_plan(replace(calibration, name=name))
        assert other.calibration.name == name
        np.testing.assert_array_equal(other.canvas_intrinsics, plan.canvas_intrinsics)
    with pytest.raises(WaymoContractError):
        build_waymo_gen3c_raster_plan(
            replace(calibration, width=10, height=5, distortion_k1_k2_p1_p2_k3=np.zeros(5))
        )


def test_direct_map_bilinear_known_rgb_and_sparse_projection(plan):
    assert plan.map_x.shape == plan.map_y.shape == plan.rectification_known.shape == (704, 1280)
    coverage = cv2.remap(
        np.ones((110, 200), np.float32),
        plan.map_x,
        plan.map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    np.testing.assert_array_equal(plan.rectification_known, coverage == np.float32(1))
    assert plan.known_fraction > 0.99 and plan.roi_retained_fraction > 0.75
    rows, columns = np.mgrid[:110, :200]
    rgb = np.stack([columns % 251, rows % 251, (3 * columns + 5 * rows) % 251], axis=-1).astype(
        np.uint8
    )
    known = plan.rectification_known.copy()
    known[100, 100] = False
    masked = replace(plan, rectification_known=known)
    actual = rectify_waymo_rgb(masked, SimpleNamespace(rgb=rgb))
    expected = cv2.remap(
        rgb, plan.map_x, plan.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    expected[~known] = 0
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual[..., 0], actual[..., 2])
    output_xy = np.array([[96, 0], [640, 352], [1100, 600]])
    source_xy = np.stack(
        [
            plan.map_x[output_xy[:, 1], output_xy[:, 0]],
            plan.map_y[output_xy[:, 1], output_xy[:, 0]],
        ],
        axis=-1,
    )
    points, valid = rectify_waymo_projection_xy(plan, source_xy)
    np.testing.assert_allclose(points, output_xy, atol=0.05, rtol=0)
    assert valid.tolist() == [True, True, True]
    outside, valid = rectify_waymo_projection_xy(plan, np.array([[-1, 20], [0, 0]], np.float32))
    assert valid.tolist() == [False, False] and (outside[1] < 0).any()


def test_legacy_rounds_tiny_negative_projection_to_zero(plan):
    # Legacy accepts -1e-6 after rounding; FRONT intentionally has no such rounding.
    calibration = replace(
        plan.calibration, intrinsics=np.eye(3), distortion_k1_k2_p1_p2_k3=np.zeros(5)
    )
    K = np.eye(3)
    K[0, 2] = -1e-6
    adjusted = replace(plan, calibration=calibration, canvas_intrinsics=K)
    xy, valid = rectify_waymo_projection_xy(adjusted, np.array([[0, 1]], np.float32))
    np.testing.assert_array_equal(xy, [[0, 1]])
    assert valid.tolist() == [True]

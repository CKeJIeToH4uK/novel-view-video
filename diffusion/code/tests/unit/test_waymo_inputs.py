"""Physical Waymo coordinates and immutable frame-key windows."""

from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from novel_view.inputs.waymo.camera import (
    OPENCV_FROM_WAYMO_CAMERA, WaymoCameraCalibration, WaymoCameraFrame,
)
from novel_view.inputs.waymo.frame import select_waymo_frame_window
from novel_view.inputs.waymo.types import WaymoFrameKey


def test_frame_key_value_order_and_exact_window():
    keys = tuple(WaymoFrameKey("segment", 1_700_000_000_000_000 + i * 100_000)
                 for i in range(198))
    assert keys[0] == WaymoFrameKey("segment", keys[0].frame_timestamp_micros)
    assert keys[0] < keys[1]
    assert select_waymo_frame_window(keys, 51, 121) == keys[51:172]
    with pytest.raises(FrozenInstanceError):
        keys[0].frame_timestamp_micros = 0


def test_camera_axes_and_pose_composition():
    extrinsic = np.eye(4)
    extrinsic[0, 3] = 1
    camera_pose = np.array([[0, -1, 0, 10], [1, 0, 0, 2],
                            [0, 0, 1, 0.5], [0, 0, 0, 1]], dtype=float)
    calibration = WaymoCameraCalibration(
        "FRONT", np.diag([20.0, 21.0, 1.0]), np.zeros(5), extrinsic, 3, 2, 1,
    )
    frame = WaymoCameraFrame(
        calibration, b"jpeg", np.zeros((2, 3, 3), dtype=np.uint8), camera_pose,
        np.zeros(3, dtype=np.float32), np.zeros(3), 1.024, 1.0, 1.044, 0.01,
    )
    assert_array_equal(OPENCV_FROM_WAYMO_CAMERA[:3, :3],
                       [[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    expected_c2w = np.array([[0, -1, 0, 10], [1, 0, 0, 3],
                             [0, 0, 1, 0.5], [0, 0, 0, 1]])
    assert_allclose(frame.world_from_waymo_camera_at_pose_timestamp,
                    expected_c2w, rtol=0, atol=1e-12)
    assert_allclose(frame.world_to_waymo_camera_at_pose_timestamp @ expected_c2w,
                    np.eye(4), rtol=0, atol=1e-12)
    expected_opencv = [[1, 0, 0, -10], [0, 0, -1, 0.5],
                       [0, 1, 0, -3], [0, 0, 0, 1]]
    assert_allclose(frame.world_to_opencv_camera_at_pose_timestamp,
                    expected_opencv, rtol=0, atol=1e-12)
    # A world point two metres along physical camera-forward has OpenCV Z=2.
    assert_allclose(frame.world_to_opencv_camera_at_pose_timestamp @ [10, 5, 0.5, 1],
                    [0, 0, 2, 1], rtol=0, atol=1e-12)

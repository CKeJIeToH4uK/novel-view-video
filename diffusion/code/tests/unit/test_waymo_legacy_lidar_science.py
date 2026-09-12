"""All physical returns, streaming three-camera evidence and literal world points."""

import weakref
from dataclasses import replace

import numpy as np
import pytest

from novel_view.inputs.waymo.camera import (
    OPENCV_FROM_WAYMO_CAMERA,
    WAYMO_CAMERA_ORDER,
    WaymoCameraCalibration,
    WaymoCameraFrame,
)
from novel_view.inputs.waymo.frame import WaymoFrameBundle, WaymoSegmentContext
from novel_view.inputs.waymo.lidar import (
    WAYMO_LIDAR_ORDER,
    WaymoLidarCalibration,
    WaymoLidarFrame,
    WaymoLidarReturn,
)
from novel_view.inputs.waymo.types import WaymoContractError, WaymoFrameKey
from novel_view.preparation.waymo_depth.depth_collection import build_projected_lidar_depths
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.preparation.waymo_depth.lidar_geometry import selected_lidar_points_world
from novel_view.preparation.waymo_depth.raster import WaymoGen3cRasterPlan


def _array(value, dtype=np.float64):
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _rigid(yaw=0.0, xyz=(0.0, 0.0, 0.0)):
    c, s = np.cos(yaw), np.sin(yaw)
    return _array([[c, -s, 0, xyz[0]], [s, c, 0, xyz[1]], [0, 0, 1, xyz[2]], [0, 0, 0, 1]])


def _return(ranges=(10.0, 20.0), cameras=(1, 2), indices=(1, 5)):
    values = np.zeros((2, 4, 4), np.float32)
    values.reshape(-1, 4)[list(indices), 0] = ranges
    projections = np.zeros((2, 4, 6), np.float32)
    projections.reshape(-1, 6)[list(indices)] = [cameras[0], 0, 5.5, cameras[1], 1, 5.5]
    return WaymoLidarReturn(_array(values, np.float32), _array(projections, np.float32))


@pytest.fixture
def scene():
    K = _array([[10, 0, 10], [0, 10, 5.5], [0, 0, 1]])
    calibrations = tuple(
        WaymoCameraCalibration(
            name,
            K,
            _array(np.zeros(5)),
            _rigid(),
            20,
            11,
            0,
        )
        for name in WAYMO_CAMERA_ORDER
    )
    camera_frames = tuple(
        WaymoCameraFrame(
            calibration,
            b"jpeg",
            _array(np.zeros((11, 20, 3)), np.uint8),
            _rigid(),
            _array(np.zeros(3), np.float32),
            _array(np.zeros(3)),
            1.0,
            1.0,
            1.01,
            0.005,
        )
        for calibration in calibrations
    )
    lidars = []
    for index, name in enumerate(WAYMO_LIDAR_ORDER, 1):
        calibration = WaymoLidarCalibration(
            name,
            _rigid(0.03 * index, (index, -0.1 * index, 0.05 * index)),
            -0.2,
            0.1,
            _array([-0.2, 0.1]) if index % 2 else None,
        )
        pose = _array(np.zeros((2, 4, 6)), np.float32) if name == "TOP" else None
        lidars.append(
            WaymoLidarFrame(
                calibration,
                _return((10 + index, 20 + index), (1, 2)),
                _return((30 + index, 40 + index), (2, 3)),
                pose,
            )
        )
    context = WaymoSegmentContext("segment", calibrations, tuple(l.calibration for l in lidars))
    frame = WaymoFrameBundle(
        WaymoFrameKey("segment", 100), context, _rigid(), camera_frames, tuple(lidars)
    )
    key = WaymoClipKey("waymo-ddw-lora-v1", "training", "segment", 0, tuple(range(100, 221)))
    canvas_K = _array([[640, 0, 671.5], [0, 640, 383.5], [0, 0, 1]])
    plans = tuple(
        WaymoGen3cRasterPlan(
            calibration,
            (0, 0, 20, 11),
            (0, 0, 20, 11),
            canvas_K,
            _array(np.zeros((704, 1280)), np.float32),
            _array(np.zeros((704, 1280)), np.float32),
            _array(np.ones((704, 1280)), bool),
        )
        for calibration in calibrations[:3]
    )
    return key, frame, plans


def test_three_cameras_stream_all_five_lidars_and_both_returns_in_order(scene):
    key, frame, plans = scene

    class Frames:
        index = 0
        previous = None

        def __iter__(self):
            return self

        def __next__(self):
            assert self.previous is None or self.previous() is None
            if self.index == 121:
                raise StopIteration
            pose = _rigid()
            self.previous = weakref.ref(pose)
            result = replace(
                frame,
                key=WaymoFrameKey("segment", 100 + self.index),
                context=replace(frame.context),
                world_from_vehicle_for_frame=pose,
            )
            self.index += 1
            return result

    stream = Frames()
    results = build_projected_lidar_depths(key, plans, stream)
    assert stream.index == 121
    assert [r.camera_name for r in results] == ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"]
    for result, per_frame, return_ids, xy in zip(
        results,
        (10, 20, 10),
        ((1, 1), (1, 1, 2, 2), (2, 2)),
        ((31.5, 383.5), (95.5, 383.5), (95.5, 383.5)),
        strict=True,
    ):
        np.testing.assert_array_equal(result.frame_offsets, np.arange(122) * per_frame)
        np.testing.assert_array_equal(
            result.lidar_ids, np.tile(np.repeat(np.arange(1, 6), per_frame // 5), 121)
        )
        np.testing.assert_array_equal(result.return_ids, np.tile(return_ids, 5 * 121))
        np.testing.assert_array_equal(
            result.flat_pixel_indices, np.tile([1, 5], 121 * per_frame // 2)
        )
        np.testing.assert_array_equal(result.heldout, np.arange(121 * per_frame) % 5 == 0)
        np.testing.assert_allclose(result.canvas_xy[0], xy)
        assert result.frame_slice(3) == slice(3 * per_frame, 4 * per_frame)
        assert np.isfinite(result.depth_z_m).all() and (result.depth_z_m > 0).all()
    # TOP row0 inclination is .1 after reversing; lidar yaw cancels azimuth correction.
    assert results[0].depth_z_m[0] == pytest.approx(11 * np.cos(0.1) / np.sqrt(2) + 1, rel=1e-6)
    # One real input mismatch, not the old internal constructor/ownership matrix.
    wrong = replace(frame, key=WaymoFrameKey("segment", 999))
    with pytest.raises(WaymoContractError):
        build_projected_lidar_depths(key, plans, iter([wrong]))


def test_top_pixel_yaw_and_non_top_frame_pose_use_distinct_physical_axes(scene):
    _, frame, _ = scene
    calibration = WaymoLidarCalibration("TOP", _rigid(), 0, 0, _array([0.0, 0.0]))
    pixel_pose = np.zeros((2, 4, 6), np.float32)
    pixel_pose.reshape(-1, 6)[1] = [0, 0, np.pi / 2, 1, 0, 0]
    top = WaymoLidarFrame(calibration, _return(), _return(), _array(pixel_pose, np.float32))
    front = replace(
        top,
        calibration=replace(calibration, name="FRONT"),
        pixel_pose_vehicle_to_global_rpy_xyz=None,
    )
    frame = replace(frame, world_from_vehicle_for_frame=_rigid(xyz=(2, 0, 0)))
    indices = np.array([1], np.int32)
    radius = 10 / np.sqrt(2)
    world = selected_lidar_points_world(frame, top, top.return1, indices)
    np.testing.assert_allclose(world, [[1 - radius, radius, 0]], atol=1e-6)
    np.testing.assert_allclose(
        selected_lidar_points_world(frame, front, front.return1, indices),
        [[2 + radius, radius, 0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        world @ OPENCV_FROM_WAYMO_CAMERA[:3, :3].T, [[-radius, 0, 1 - radius]], atol=1e-6
    )


def test_reversed_explicit_beams_uniform_midpoints_and_extrinsics_have_literal_oracle(scene):
    _, frame, _ = scene
    top_calibration = WaymoLidarCalibration(
        "TOP", _rigid(np.pi / 2, (1, 2, 3)), -1, 1, _array([0.1, 0.3])
    )
    front_calibration = WaymoLidarCalibration(
        "FRONT", _rigid(np.pi / 6, (0.5, -1, 2)), -0.4, 0.2, None
    )
    pose = np.zeros((2, 4, 6), np.float32)
    pose.reshape(-1, 6)[[0, 7], 3:] = [[4, 5, 6], [-1, -2, -3]]
    top = WaymoLidarFrame(
        top_calibration,
        _return((10, 20), indices=(0, 7)),
        _return((0, 0)),
        _array(pose, np.float32),
    )
    front = WaymoLidarFrame(
        front_calibration, _return((12, 18), indices=(0, 7)), _return((0, 0)), None
    )
    frame = replace(frame, world_from_vehicle_for_frame=_rigid(xyz=(3, 4, 5)))
    indices = np.array([0, 7], np.int32)
    expected = (
        [
            [-1.7552490977566437, 13.755249097756645, 11.955202066613396],
            [-14.071483851539046, -14.07148385153905, 1.9966683329365633],
        ],
        [
            [-4.974676982045331, 11.47467698204533, 7.59975003124814],
            [-8.832241787814578, -9.332241787814583, 2.546728733418587],
        ],
    )
    for lidar, points in zip((top, front), expected, strict=True):
        np.testing.assert_allclose(
            selected_lidar_points_world(frame, lidar, lidar.return1, indices),
            points,
            rtol=1e-12,
            atol=1e-12,
        )

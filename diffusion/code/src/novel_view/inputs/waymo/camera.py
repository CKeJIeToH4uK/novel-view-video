"""Physical camera calibration, payload, time, and axes for Waymo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from novel_view.inputs.waymo._arrays import readonly_owned


WaymoCameraName: TypeAlias = Literal[
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
]
WaymoRollingShutterDirection: TypeAlias = Literal[0, 1, 2, 3, 4, 5]
WAYMO_CAMERA_ORDER: tuple[WaymoCameraName, ...] = (
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
)

OPENCV_FROM_WAYMO_CAMERA: npt.NDArray[np.float64] = readonly_owned(
    np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
)
"""Homogeneous transform from Waymo camera axes to OpenCV camera axes."""


@dataclass(frozen=True, slots=True, eq=False)
class WaymoCameraCalibration:
    """One camera calibration shared by a physical Waymo segment.

    ``vehicle_from_waymo_camera`` maps native Waymo camera axes (forward,
    left, up) into vehicle coordinates. ``rolling_shutter_direction`` keeps
    the official integer enum ID ``UNKNOWN=0`` through ``GLOBAL_SHUTTER=5``.
    """

    name: WaymoCameraName
    intrinsics: npt.NDArray[np.float64]
    distortion_k1_k2_p1_p2_k3: npt.NDArray[np.float64]
    vehicle_from_waymo_camera: npt.NDArray[np.float64]
    width: int
    height: int
    rolling_shutter_direction: WaymoRollingShutterDirection


@dataclass(frozen=True, slots=True, eq=False)
class WaymoCameraFrame:
    """One decoded camera payload and nominal pose in a Waymo frame.

    The pose, trigger, readout completion and shutter fields are seconds.
    Linear velocity uses global axes; angular velocity uses vehicle axes.
    """

    calibration: WaymoCameraCalibration
    jpeg_bytes: bytes
    rgb: npt.NDArray[np.uint8]
    world_from_vehicle_at_camera_pose_timestamp: npt.NDArray[np.float64]
    global_linear_velocity_mps: npt.NDArray[np.float32]
    vehicle_angular_velocity_radps: npt.NDArray[np.float64]
    pose_timestamp_seconds: float
    camera_trigger_time_seconds: float
    camera_readout_done_time_seconds: float
    shutter_seconds: float

    @property
    def name(self) -> WaymoCameraName:
        """Return the physical camera name owned by the calibration."""
        return self.calibration.name

    @property
    def trigger_to_readout_done_interval_seconds(self) -> float:
        """Return trigger-to-readout-done, not pure sensor readout time."""
        return self.camera_readout_done_time_seconds - self.camera_trigger_time_seconds

    @property
    def world_from_waymo_camera_at_pose_timestamp(self) -> npt.NDArray[np.float64]:
        """Compose camera-to-vehicle with vehicle-to-world at camera time."""
        return readonly_owned(
            self.world_from_vehicle_at_camera_pose_timestamp
            @ self.calibration.vehicle_from_waymo_camera
        )

    @property
    def world_to_waymo_camera_at_pose_timestamp(self) -> npt.NDArray[np.float64]:
        """Return nominal world-to-camera in native Waymo camera axes."""
        world_from_camera = self.world_from_waymo_camera_at_pose_timestamp
        rotation = world_from_camera[:3, :3]
        translation = world_from_camera[:3, 3]
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation.T
        result[:3, 3] = -(rotation.T @ translation)
        return readonly_owned(result)

    @property
    def world_to_opencv_camera_at_pose_timestamp(self) -> npt.NDArray[np.float64]:
        """Return nominal world-to-camera in OpenCV right/down/forward axes."""
        return readonly_owned(
            OPENCV_FROM_WAYMO_CAMERA
            @ self.world_to_waymo_camera_at_pose_timestamp
        )


__all__ = [
    "OPENCV_FROM_WAYMO_CAMERA",
    "WAYMO_CAMERA_ORDER",
    "WaymoCameraCalibration",
    "WaymoCameraFrame",
    "WaymoCameraName",
    "WaymoRollingShutterDirection",
]

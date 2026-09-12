"""Physical Waymo segment context, frame bundle, and key windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from novel_view.inputs.waymo.camera import (
    WaymoCameraCalibration,
    WaymoCameraFrame,
    WaymoCameraName,
)
from novel_view.inputs.waymo.lidar import (
    WaymoLidarCalibration,
    WaymoLidarFrame,
    WaymoLidarName,
)
from novel_view.inputs.waymo.types import WaymoContractError, WaymoFrameKey


@dataclass(frozen=True, slots=True, eq=False)
class WaymoSegmentContext:
    """Shared physical calibrations for one Waymo segment."""

    segment_id: str
    camera_calibrations: tuple[WaymoCameraCalibration, ...]
    lidar_calibrations: tuple[WaymoLidarCalibration, ...]

    def camera_calibration(self, name: WaymoCameraName) -> WaymoCameraCalibration:
        """Return one calibration by physical camera name."""
        for calibration in self.camera_calibrations:
            if calibration.name == name:
                return calibration
        raise WaymoContractError(f"unknown Waymo camera {name!r}")

    def lidar_calibration(self, name: WaymoLidarName) -> WaymoLidarCalibration:
        """Return one calibration by physical LiDAR name."""
        for calibration in self.lidar_calibrations:
            if calibration.name == name:
                return calibration
        raise WaymoContractError(f"unknown Waymo lidar {name!r}")


@dataclass(frozen=True, slots=True, eq=False)
class WaymoFrameBundle:
    """All decoded physical sensor records sharing one Waymo frame key.

    The key joins component rows but does not claim simultaneous camera
    exposure. ``world_from_vehicle_for_frame`` is Waymo's nominal frame pose;
    camera-specific pose and exposure times remain on each camera frame.
    """

    key: WaymoFrameKey
    context: WaymoSegmentContext
    world_from_vehicle_for_frame: npt.NDArray[np.float64]
    cameras: tuple[WaymoCameraFrame, ...]
    lidars: tuple[WaymoLidarFrame, ...]

    def camera(self, name: WaymoCameraName) -> WaymoCameraFrame:
        """Return one frame by physical camera name."""
        for frame in self.cameras:
            if frame.name == name:
                return frame
        raise WaymoContractError(f"unknown Waymo camera {name!r}")

    def lidar(self, name: WaymoLidarName) -> WaymoLidarFrame:
        """Return one frame by physical LiDAR name."""
        for frame in self.lidars:
            if frame.name == name:
                return frame
        raise WaymoContractError(f"unknown Waymo lidar {name!r}")


def select_waymo_frame_window(
    timeline: Sequence[WaymoFrameKey],
    start_frame_index: int,
    count: int,
) -> tuple[WaymoFrameKey, ...]:
    """Select one exact neighbouring window from a canonical timeline."""
    keys = tuple(timeline)
    if not keys:
        raise WaymoContractError("timeline must not be empty")
    if any(key.segment_id != keys[0].segment_id for key in keys[1:]):
        raise WaymoContractError("timeline must contain one segment_id")
    if any(
        current.frame_timestamp_micros <= previous.frame_timestamp_micros
        for previous, current in zip(keys, keys[1:], strict=False)
    ):
        raise WaymoContractError("timeline timestamps must be strictly increasing")
    if start_frame_index < 0 or count <= 0:
        raise WaymoContractError(
            "window start must be non-negative and count must be positive"
        )
    selected = keys[start_frame_index : start_frame_index + count]
    if len(selected) != count:
        raise WaymoContractError(
            f"requested {count} keys from index {start_frame_index}, "
            f"but timeline contains {len(keys)}"
        )
    return selected


__all__ = [
    "WaymoFrameBundle",
    "WaymoSegmentContext",
    "select_waymo_frame_window",
]

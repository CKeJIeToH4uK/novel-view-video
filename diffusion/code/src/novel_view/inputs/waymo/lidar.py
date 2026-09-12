"""Physical LiDAR calibration and native returns for Waymo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt


WaymoLidarName: TypeAlias = Literal[
    "TOP",
    "FRONT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "REAR",
]
WAYMO_LIDAR_ORDER: tuple[WaymoLidarName, ...] = (
    "TOP",
    "FRONT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "REAR",
)


@dataclass(frozen=True, slots=True, eq=False)
class WaymoLidarCalibration:
    """One LiDAR calibration shared by a physical Waymo segment.

    Explicit beam inclinations may be absent for a uniform sensor; the
    minimum and maximum angles then define the native vertical grid.
    """

    name: WaymoLidarName
    vehicle_from_lidar: npt.NDArray[np.float64]
    beam_inclination_min_rad: float
    beam_inclination_max_rad: float
    beam_inclination_values_rad: npt.NDArray[np.float64] | None


@dataclass(frozen=True, slots=True, eq=False)
class WaymoLidarReturn:
    """One native range-image return and its camera projections.

    Range channels are ``range, intensity, elongation, is_in_nlz``.
    Projection channels are ``camera1, x1, y1, camera2, x2, y2``.
    """

    range_image: npt.NDArray[np.float32]
    camera_projection: npt.NDArray[np.float32]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return the native range-image grid as ``(height, width)``."""
        return cast(tuple[int, int], self.range_image.shape[:2])


@dataclass(frozen=True, slots=True, eq=False)
class WaymoLidarFrame:
    """Both returns for one LiDAR and the optional TOP pixel pose.

    TOP pose channels are ``roll, pitch, yaw, x, y, z``. Waymo stores this
    pose once; both returns deliberately reference the same physical grid.
    """

    calibration: WaymoLidarCalibration
    return1: WaymoLidarReturn
    return2: WaymoLidarReturn
    pixel_pose_vehicle_to_global_rpy_xyz: npt.NDArray[np.float32] | None

    @property
    def name(self) -> WaymoLidarName:
        """Return the physical LiDAR name owned by the calibration."""
        return self.calibration.name


__all__ = [
    "WAYMO_LIDAR_ORDER",
    "WaymoLidarCalibration",
    "WaymoLidarFrame",
    "WaymoLidarName",
    "WaymoLidarReturn",
]

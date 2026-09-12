"""Official Waymo range-image geometry shared by depth and FRONT evidence."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from novel_view.inputs.waymo.frame import WaymoFrameBundle
from novel_view.inputs.waymo.lidar import WaymoLidarFrame, WaymoLidarReturn
from novel_view.inputs.waymo.types import WaymoContractError


def selected_lidar_points_world(
    frame: WaymoFrameBundle,
    lidar: WaymoLidarFrame,
    lidar_return: WaymoLidarReturn,
    flat_indices: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Convert selected native ranges to world, with TOP per-pixel motion."""
    height, width = lidar_return.spatial_shape
    rows, columns = np.divmod(flat_indices, width)
    ranges = lidar_return.range_image[..., 0].reshape(-1)[
        flat_indices
    ].astype(np.float64)
    inclination = _beam_inclinations(lidar, height)[rows]
    extrinsic = lidar.calibration.vehicle_from_lidar
    correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratios = (width - columns.astype(np.float64) - 0.5) / width
    azimuth = (ratios * 2.0 - 1.0) * np.pi - correction
    cos_inclination = np.cos(inclination)
    points_lidar = np.stack(
        (
            np.cos(azimuth) * cos_inclination * ranges,
            np.sin(azimuth) * cos_inclination * ranges,
            np.sin(inclination) * ranges,
        ),
        axis=1,
    )
    points_vehicle = _transform(extrinsic, points_lidar)
    if lidar.name != "TOP":
        return _transform(frame.world_from_vehicle_for_frame, points_vehicle)
    pose = lidar.pixel_pose_vehicle_to_global_rpy_xyz
    if pose is None:
        raise WaymoContractError("TOP lidar requires per-pixel pose")
    selected_pose = pose.reshape(-1, 6)[flat_indices].astype(np.float64)
    return _rotate_rpy(points_vehicle, selected_pose[:, :3]) + selected_pose[:, 3:]


def _beam_inclinations(
    lidar: WaymoLidarFrame,
    height: int,
) -> npt.NDArray[np.float64]:
    values = lidar.calibration.beam_inclination_values_rad
    if values is None:
        step = (
            lidar.calibration.beam_inclination_max_rad
            - lidar.calibration.beam_inclination_min_rad
        ) / height
        values = (
            lidar.calibration.beam_inclination_min_rad
            + (np.arange(height, dtype=np.float64) + 0.5) * step
        )
    return np.asarray(values[::-1], dtype=np.float64)


def _rotate_rpy(
    points: npt.NDArray[np.float64],
    rpy: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply official intrinsic roll, pitch, yaw as Rz @ Ry @ Rx."""
    roll, pitch, yaw = rpy.T
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    x, y, z = points.T
    rolled_y, rolled_z = cr * y - sr * z, sr * y + cr * z
    pitched_x = cp * x + sp * rolled_z
    pitched_z = -sp * x + cp * rolled_z
    return np.stack(
        (
            cy * pitched_x - sy * rolled_y,
            sy * pitched_x + cy * rolled_y,
            pitched_z,
        ),
        axis=1,
    )


def _transform(
    transform: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return points @ transform[:3, :3].T + transform[:3, 3]

"""Canonical FRONT raster and measured LiDAR evidence for Waymo DDW."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

import cv2
import numpy as np
import numpy.typing as npt

from novel_view.inputs.waymo.frame import WaymoFrameBundle
from novel_view.inputs.waymo.lidar import WaymoLidarFrame, WaymoLidarReturn
from novel_view.inputs.waymo.types import WaymoContractError, WaymoFrameKey
from novel_view.preparation.waymo_depth.lidar_geometry import (
    selected_lidar_points_world,
)
from novel_view.preparation.waymo_depth.raster import (
    GEN3C_CANVAS_SIZE_HW as RASTER_SIZE_HW,
    WaymoGen3cRasterPlan as WaymoFrontRasterPlan,
    build_waymo_gen3c_raster_plan as build_front_raster_plan,
    rectify_waymo_rgb as rectify_front_rgb,
)
from novel_view.preparation.waymo_ddw.selection import SelectedDdwSample
from novel_view.preparation.waymo_ddw.source import WaymoFrontSource


_FRONT_CAMERA_ID = 1


@dataclass(frozen=True, slots=True, eq=False)
class FrontLidarDepth:
    """Ordered sparse FRONT camera-Z with a fixed evaluation split."""

    frame_offsets: npt.NDArray[np.int64]
    canvas_xy: npt.NDArray[np.float32]
    depth_z_m: npt.NDArray[np.float32]
    evaluation_holdout: npt.NDArray[np.bool_]

    def frame_slice(self, frame_index: int) -> slice:
        return slice(
            int(self.frame_offsets[frame_index]),
            int(self.frame_offsets[frame_index + 1]),
        )


@dataclass(frozen=True, slots=True, eq=False)
class WaymoFrontRaster:
    """One selected clip on the fixed measured FRONT camera canvas."""

    selected: SelectedDdwSample
    frame_keys: tuple[WaymoFrameKey, ...]
    rgb_thwc: npt.NDArray[np.uint8]
    rectification_known: npt.NDArray[np.bool_]
    K_canvas: npt.NDArray[np.float64]
    world_to_camera_cv: npt.NDArray[np.float64]
    lidar: FrontLidarDepth


def build_front_raster(source: WaymoFrontSource) -> WaymoFrontRaster:
    """Consume one exact source stream into FRONT RGB and LiDAR evidence."""
    frames = iter(source.frames)
    first = next(frames)
    first_front = first.camera("FRONT")
    plan = build_front_raster_plan(first_front.calibration)
    frame_count = len(source.frame_keys)
    height, width = RASTER_SIZE_HW
    rgb = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    world_to_camera = np.empty((frame_count, 4, 4), dtype=np.float64)
    lidar_collector = _FrontLidarCollector(plan)

    for index, frame in enumerate(chain((first,), frames)):
        front = frame.camera("FRONT")
        rgb[index] = rectify_front_rgb(plan, front)
        world_to_camera[index] = front.world_to_opencv_camera_at_pose_timestamp
        lidar_collector.append(frame)

    lidar = filter_lidar_to_rectification_support(
        lidar_collector.finish(),
        plan.rectification_known,
    )
    K_canvas = np.repeat(plan.K_canvas[None], frame_count, axis=0)
    return WaymoFrontRaster(
        selected=source.selected,
        frame_keys=source.frame_keys,
        rgb_thwc=rgb,
        rectification_known=plan.rectification_known,
        K_canvas=K_canvas,
        world_to_camera_cv=world_to_camera,
        lidar=lidar,
    )


def project_front_lidar(
    plan: WaymoFrontRasterPlan,
    frames: Iterable[WaymoFrameBundle],
) -> FrontLidarDepth:
    """Project every LiDAR return in canonical reader order to FRONT camera-Z."""
    collector = _FrontLidarCollector(plan)
    for frame in frames:
        collector.append(frame)
    return collector.finish()


class _FrontLidarCollector:
    """Mutable columns for one sequential pass over physical frames."""

    def __init__(self, plan: WaymoFrontRasterPlan) -> None:
        self.plan = plan
        self.offsets = [0]
        self.row_index = 0
        self.xy: list[npt.NDArray[np.float32]] = []
        self.depth: list[npt.NDArray[np.float32]] = []
        self.holdout: list[npt.NDArray[np.bool_]] = []

    def append(self, frame: WaymoFrameBundle) -> None:
        frame_count = 0
        world_to_camera = frame.camera("FRONT").world_to_opencv_camera_at_pose_timestamp
        for lidar in frame.lidars:
            for lidar_return in (lidar.return1, lidar.return2):
                projected = _project_return(
                    frame,
                    lidar,
                    lidar_return,
                    self.plan,
                    world_to_camera,
                )
                if projected is None:
                    continue
                canvas_xy, depth_z_m = projected
                count = len(depth_z_m)
                evaluation_holdout = (
                    np.arange(
                        self.row_index,
                        self.row_index + count,
                        dtype=np.int64,
                    )
                    % 5
                    == 0
                )
                self.row_index += count
                frame_count += count
                self.xy.append(canvas_xy)
                self.depth.append(depth_z_m)
                self.holdout.append(evaluation_holdout)
        self.offsets.append(self.offsets[-1] + frame_count)

    def finish(self) -> FrontLidarDepth:
        if not self.depth:
            raise WaymoContractError("FRONT clip has no projected LiDAR depth")
        return FrontLidarDepth(
            frame_offsets=np.asarray(self.offsets, dtype=np.int64),
            canvas_xy=np.ascontiguousarray(
                np.concatenate(self.xy),
                dtype=np.float32,
            ),
            depth_z_m=np.ascontiguousarray(
                np.concatenate(self.depth),
                dtype=np.float32,
            ),
            evaluation_holdout=np.ascontiguousarray(np.concatenate(self.holdout)),
        )


def filter_lidar_to_rectification_support(
    lidar: FrontLidarDepth,
    rectification_known: npt.NDArray[np.bool_],
) -> FrontLidarDepth:
    """Drop unsupported rows while preserving their original split labels."""
    keep = _bilinear_support(rectification_known, lidar.canvas_xy)
    offsets = [0]
    for frame_index in range(len(lidar.frame_offsets) - 1):
        frame_keep = keep[lidar.frame_slice(frame_index)]
        offsets.append(offsets[-1] + int(np.count_nonzero(frame_keep)))
    return FrontLidarDepth(
        frame_offsets=np.asarray(offsets, dtype=np.int64),
        canvas_xy=np.ascontiguousarray(lidar.canvas_xy[keep]),
        depth_z_m=np.ascontiguousarray(lidar.depth_z_m[keep]),
        evaluation_holdout=np.ascontiguousarray(lidar.evaluation_holdout[keep]),
    )


def _project_return(
    frame: WaymoFrameBundle,
    lidar: WaymoLidarFrame,
    lidar_return: WaymoLidarReturn,
    plan: WaymoFrontRasterPlan,
    world_to_camera: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None:
    ranges = lidar_return.range_image[..., 0]
    projections = lidar_return.camera_projection
    first = projections[..., 0] == _FRONT_CAMERA_ID
    second = (~first) & (projections[..., 3] == _FRONT_CAMERA_ID)
    flat = np.flatnonzero((ranges > 0.0) & (first | second)).astype(np.int32)
    if not flat.size:
        return None
    projection_rows = projections.reshape(-1, 6)[flat]
    use_first = first.reshape(-1)[flat]
    source_xy = np.where(
        use_first[:, None],
        projection_rows[:, 1:3],
        projection_rows[:, 4:6],
    ).astype(np.float32, copy=False)
    canvas_xy, on_canvas = _rectify_projection_xy(plan, source_xy)
    points_world = selected_lidar_points_world(frame, lidar, lidar_return, flat)
    depth = points_world @ world_to_camera[2, :3] + world_to_camera[2, 3]
    keep = on_canvas & np.isfinite(depth) & (depth > 0.0)
    if not np.any(keep):
        return None
    return (
        np.ascontiguousarray(canvas_xy[keep], dtype=np.float32),
        np.ascontiguousarray(depth[keep], dtype=np.float32),
    )


def _rectify_projection_xy(
    plan: WaymoFrontRasterPlan,
    source_xy: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    canvas_xy = cv2.undistortPoints(
        source_xy.reshape(-1, 1, 2),
        plan.calibration.intrinsics,
        plan.calibration.distortion_k1_k2_p1_p2_k3,
        R=np.eye(3, dtype=np.float64),
        P=plan.K_canvas,
    ).reshape(-1, 2)
    canvas_xy = np.asarray(canvas_xy, dtype=np.float32)
    height, width = RASTER_SIZE_HW
    valid = (
        (source_xy[:, 0] >= 0.0)
        & (source_xy[:, 0] < plan.calibration.width)
        & (source_xy[:, 1] >= 0.0)
        & (source_xy[:, 1] < plan.calibration.height)
        & (canvas_xy[:, 0] >= 0.0)
        & (canvas_xy[:, 0] < width)
        & (canvas_xy[:, 1] >= 0.0)
        & (canvas_xy[:, 1] < height)
    )
    return canvas_xy, valid


def _bilinear_support(
    known: npt.NDArray[np.bool_],
    canvas_xy: npt.NDArray[np.float32],
) -> npt.NDArray[np.bool_]:
    height, width = known.shape
    x = canvas_xy[:, 0].astype(np.float64)
    y = canvas_xy[:, 1].astype(np.float64)
    inside = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)
    x0 = np.floor(np.clip(x, 0.0, width - 1)).astype(np.intp)
    y0 = np.floor(np.clip(y, 0.0, height - 1)).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    return inside & known[y0, x0] & known[y0, x1] & known[y1, x0] & known[y1, x1]


__all__ = [
    "FrontLidarDepth",
    "RASTER_SIZE_HW",
    "WaymoFrontRaster",
    "WaymoFrontRasterPlan",
    "build_front_raster",
    "build_front_raster_plan",
    "filter_lidar_to_rectification_support",
    "project_front_lidar",
    "rectify_front_rgb",
]

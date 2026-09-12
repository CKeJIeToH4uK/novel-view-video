"""Single-pass collection of Waymo LiDAR evidence for selected cameras."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from novel_view.preparation.waymo_depth.depth import (
    DEPTH_CAMERA_ORDER,
    DepthCameraName,
)
from novel_view.preparation.waymo_depth.depth_samples import ProjectedLidarDepth
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.preparation.waymo_depth.lidar_geometry import (
    selected_lidar_points_world,
)
from novel_view.preparation.waymo_depth.raster import (
    WaymoGen3cRasterPlan,
    rectify_waymo_projection_xy,
)
from novel_view.inputs.waymo._arrays import readonly_owned
from novel_view.inputs.waymo.frame import WaymoFrameBundle
from novel_view.inputs.waymo.lidar import WaymoLidarFrame, WaymoLidarReturn
from novel_view.inputs.waymo.types import WaymoContractError


_CAMERA_IDS = dict(zip(DEPTH_CAMERA_ORDER, (1, 2, 3), strict=True))


def build_projected_lidar_depth(
    clip_key: WaymoClipKey,
    raster_plan: WaymoGen3cRasterPlan,
    frames: Iterable[WaymoFrameBundle],
) -> ProjectedLidarDepth:
    """Stream one clip into evidence for one selected camera."""
    return build_projected_lidar_depths(clip_key, (raster_plan,), frames)[0]


def build_projected_lidar_depths(
    clip_key: WaymoClipKey,
    raster_plans: tuple[WaymoGen3cRasterPlan, ...],
    frames: Iterable[WaymoFrameBundle],
) -> tuple[ProjectedLidarDepth, ...]:
    """Collect ordered evidence for selected cameras in one frame pass."""
    if not isinstance(clip_key, WaymoClipKey):
        raise WaymoContractError("clip_key must be WaymoClipKey")
    if (
        type(raster_plans) is not tuple
        or not 1 <= len(raster_plans) <= len(DEPTH_CAMERA_ORDER)
        or any(not isinstance(plan, WaymoGen3cRasterPlan) for plan in raster_plans)
    ):
        raise WaymoContractError("raster_plans must contain one to three plans")
    camera_names = tuple(plan.calibration.name for plan in raster_plans)
    expected_order = tuple(name for name in DEPTH_CAMERA_ORDER if name in camera_names)
    if camera_names != expected_order:
        raise WaymoContractError(
            "raster_plans must use unique selected cameras in canonical order"
        )

    columns_by_plan = tuple(_empty_columns() for _ in raster_plans)
    offsets_by_plan = tuple([0] for _ in raster_plans)
    iterator = iter(frames)
    for frame_index, timestamp in enumerate(clip_key.frame_timestamps_micros):
        try:
            frame = next(iterator)
        except StopIteration as exc:
            raise WaymoContractError("frame stream ended before clip key") from exc
        if (
            not isinstance(frame, WaymoFrameBundle)
            or frame.key.segment_id != clip_key.segment_id
            or frame.key.frame_timestamp_micros != timestamp
        ):
            raise WaymoContractError(
                f"frame[{frame_index}] does not match the clip key"
            )
        for plan, columns, offsets in zip(
            raster_plans, columns_by_plan, offsets_by_plan, strict=True
        ):
            offsets.append(
                offsets[-1] + _append_frame_samples(frame, plan, columns)
            )
        del frame
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise WaymoContractError("frame stream contains more than the clip key")
    return tuple(
        _finish_projected_lidar_depth(clip_key, plan, columns, offsets)
        for plan, columns, offsets in zip(
            raster_plans, columns_by_plan, offsets_by_plan, strict=True
        )
    )


def _empty_columns() -> dict[str, list[npt.NDArray[np.generic]]]:
    return {name: [] for name in ("lidar", "return", "flat", "xy", "z")}


def _append_frame_samples(
    frame: WaymoFrameBundle,
    raster_plan: WaymoGen3cRasterPlan,
    columns: dict[str, list[npt.NDArray[np.generic]]],
) -> int:
    frame_sample_count = 0
    for lidar_id, lidar in enumerate(frame.lidars, start=1):
        for return_id, lidar_return in enumerate(
            (lidar.return1, lidar.return2), start=1
        ):
            result = _project_return(frame, lidar, lidar_return, raster_plan)
            if result is None:
                continue
            flat, xy, depth = result
            count = len(flat)
            columns["lidar"].append(np.full(count, lidar_id, dtype=np.uint8))
            columns["return"].append(np.full(count, return_id, dtype=np.uint8))
            columns["flat"].append(flat)
            columns["xy"].append(xy)
            columns["z"].append(depth)
            frame_sample_count += count
    return frame_sample_count


def _finish_projected_lidar_depth(
    clip_key: WaymoClipKey,
    raster_plan: WaymoGen3cRasterPlan,
    columns: dict[str, list[npt.NDArray[np.generic]]],
    offsets: list[int],
) -> ProjectedLidarDepth:
    camera_name = cast(DepthCameraName, raster_plan.calibration.name)
    if not columns["flat"]:
        raise WaymoContractError(
            f"clip has no projected LiDAR depth samples for {camera_name}"
        )
    sample_count = offsets[-1]
    return ProjectedLidarDepth(
        clip_key=clip_key,
        camera_name=camera_name,
        frame_offsets=readonly_owned(np.asarray(offsets, dtype=np.int64)),
        lidar_ids=_concatenate(columns["lidar"], np.uint8),
        return_ids=_concatenate(columns["return"], np.uint8),
        flat_pixel_indices=_concatenate(columns["flat"], np.int32),
        canvas_xy=_concatenate(columns["xy"], np.float32),
        depth_z_m=_concatenate(columns["z"], np.float32),
        heldout=readonly_owned(np.arange(sample_count) % 5 == 0),
    )


def _project_return(
    frame: WaymoFrameBundle,
    lidar: WaymoLidarFrame,
    lidar_return: WaymoLidarReturn,
    raster_plan: WaymoGen3cRasterPlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    camera_id = _CAMERA_IDS[cast(DepthCameraName, raster_plan.calibration.name)]
    ranges = lidar_return.range_image[..., 0]
    projections = lidar_return.camera_projection
    first = projections[..., 0] == camera_id
    second = (~first) & (projections[..., 3] == camera_id)
    selected = (ranges > 0.0) & (first | second)
    flat = np.flatnonzero(selected).astype(np.int32, copy=False)
    if flat.size == 0:
        return None
    projection_flat = projections.reshape(-1, 6)[flat]
    use_first = first.reshape(-1)[flat]
    source_xy = np.where(
        use_first[:, None],
        projection_flat[:, 1:3],
        projection_flat[:, 4:6],
    ).astype(np.float32, copy=False)
    canvas_xy, on_canvas = rectify_waymo_projection_xy(
        raster_plan, np.ascontiguousarray(source_xy)
    )
    points_world = selected_lidar_points_world(frame, lidar, lidar_return, flat)
    world_to_camera = frame.camera(
        cast(DepthCameraName, raster_plan.calibration.name)
    ).world_to_opencv_camera_at_pose_timestamp
    depth = points_world @ world_to_camera[2, :3] + world_to_camera[2, 3]
    keep = on_canvas & np.isfinite(depth) & (depth > 0.0)
    if not np.any(keep):
        return None
    return (
        np.ascontiguousarray(flat[keep], dtype=np.int32),
        np.ascontiguousarray(canvas_xy[keep], dtype=np.float32),
        np.ascontiguousarray(depth[keep], dtype=np.float32),
    )


def _concatenate(
    values: list[npt.NDArray[np.generic]],
    dtype: type[np.generic],
) -> np.ndarray:
    return readonly_owned(np.ascontiguousarray(np.concatenate(values), dtype=dtype))

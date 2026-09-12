"""Build one shared three-camera clip for historical depth comparison."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from novel_view.inputs.waymo._arrays import readonly_owned
from novel_view.inputs.waymo.frame import WaymoFrameBundle
from novel_view.preparation.waymo_depth.depth import (
    DEPTH_CAMERA_ORDER,
    DepthCameraName,
)
from novel_view.preparation.waymo_depth.depth_collection import (
    build_projected_lidar_depths,
)
from novel_view.preparation.waymo_depth.depth_samples import ProjectedLidarDepth
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.preparation.waymo_depth.raster import (
    GEN3C_CANVAS_SIZE_HW,
    WaymoGen3cRasterPlan,
    build_waymo_gen3c_raster_plan,
    rectify_waymo_rgb,
)


@dataclass(frozen=True, slots=True, eq=False)
class DepthComparisonCamera:
    """Measured camera geometry and one temporary canonical RGB sequence."""

    evidence: ProjectedLidarDepth
    raster_plan: WaymoGen3cRasterPlan
    rgb_path: Path
    K_canvas: npt.NDArray[np.float64]
    world_to_camera_cv: npt.NDArray[np.float64]

    __hash__: ClassVar[None] = None


@dataclass(frozen=True, slots=True, eq=False)
class DepthComparisonClip:
    """The same ordered FRONT/LEFT/RIGHT evidence used by both methods."""

    clip_key: WaymoClipKey
    cameras: tuple[DepthComparisonCamera, ...]

    __hash__: ClassVar[None] = None

    def camera(self, name: DepthCameraName) -> DepthComparisonCamera:
        return next(camera for camera in self.cameras if camera.evidence.camera_name == name)


def build_depth_comparison_clip(
    clip_key: WaymoClipKey,
    frames: Iterable[WaymoFrameBundle],
    scratch_directory: Path,
) -> DepthComparisonClip:
    """Rectify three RGB streams and collect shared LiDAR in one frame pass."""
    iterator = iter(frames)
    first_frame = next(iterator)
    plans = tuple(
        build_waymo_gen3c_raster_plan(first_frame.camera(name).calibration)
        for name in DEPTH_CAMERA_ORDER
    )
    paths = tuple(
        scratch_directory / f"{name.lower()}-rgb.npy" for name in DEPTH_CAMERA_ORDER
    )
    frame_count = len(clip_key.frame_timestamps_micros)
    writers = tuple(
        np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.uint8,
            shape=(frame_count, *GEN3C_CANVAS_SIZE_HW, 3),
        )
        for path in paths
    )
    cameras_w2c = tuple(
        np.empty((frame_count, 4, 4), dtype=np.float64)
        for _ in DEPTH_CAMERA_ORDER
    )
    try:
        staged_frames = _stage_frames(
            frame_count,
            first_frame,
            iterator,
            plans,
            writers,
            cameras_w2c,
        )
        evidence = build_projected_lidar_depths(clip_key, plans, staged_frames)
        for writer in writers:
            writer.flush()
    finally:
        for writer in writers:
            writer._mmap.close()
    cameras = tuple(
        DepthComparisonCamera(
            evidence=item,
            raster_plan=plan,
            rgb_path=path,
            K_canvas=readonly_owned(
                np.repeat(plan.canvas_intrinsics[None], frame_count, axis=0)
            ),
            world_to_camera_cv=readonly_owned(world_to_camera),
        )
        for item, plan, path, world_to_camera in zip(
            evidence, plans, paths, cameras_w2c, strict=True
        )
    )
    return DepthComparisonClip(clip_key, cameras)


def iter_depth_comparison_rgb(
    camera: DepthComparisonCamera,
) -> Iterator[npt.NDArray[np.uint8]]:
    """Yield the temporary RGB sequence and always close its mmap."""
    rgb = np.load(camera.rgb_path, mmap_mode="r", allow_pickle=False)
    try:
        yield from rgb
    finally:
        if isinstance(rgb, np.memmap):
            rgb._mmap.close()


def _stage_frames(
    frame_count: int,
    first_frame: WaymoFrameBundle,
    iterator: Iterator[WaymoFrameBundle],
    plans: tuple[WaymoGen3cRasterPlan, ...],
    writers: tuple[np.memmap, ...],
    cameras_w2c: tuple[npt.NDArray[np.float64], ...],
) -> Iterator[WaymoFrameBundle]:
    for index, frame in enumerate(chain((first_frame,), iterator)):
        if index < frame_count:
            for plan, writer, world_to_camera in zip(
                plans, writers, cameras_w2c, strict=True
            ):
                camera = frame.camera(plan.calibration.name)
                writer[index] = rectify_waymo_rgb(plan, camera)
                world_to_camera[index] = camera.world_to_opencv_camera_at_pose_timestamp
        yield frame


__all__ = [
    "DepthComparisonCamera",
    "DepthComparisonClip",
    "build_depth_comparison_clip",
    "iter_depth_comparison_rgb",
]

"""Borrow one historical FRONT RGB/depth source for legacy DDW."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from novel_view.preparation.waymo_depth.keyset import WaymoClipKey

if TYPE_CHECKING:
    from novel_view.preparation.waymo_depth.clip import DepthComparisonCamera
    from novel_view.preparation.waymo_depth.depth import MetricDepthClip


@dataclass(frozen=True, slots=True, eq=False)
class LegacyDdwSource:
    """One staged uint8 FRONT sequence with its selected metric geometry."""

    clip_key: WaymoClipKey
    rgb_thwc: npt.NDArray[np.uint8]
    depth_z_m: npt.NDArray[np.float32]
    depth_valid: npt.NDArray[np.bool_]
    rectification_known: npt.NDArray[np.bool_]
    K_canvas: npt.NDArray[np.float64]
    world_to_camera_cv: npt.NDArray[np.float64]


@contextmanager
def open_legacy_ddw_source(
    camera: DepthComparisonCamera,
    metric_depth: MetricDepthClip,
) -> Iterator[LegacyDdwSource]:
    """Open the existing staged RGB mmap and close only this borrowed mapping."""
    rgb = np.load(camera.rgb_path, mmap_mode="r", allow_pickle=False)
    try:
        yield LegacyDdwSource(
            clip_key=metric_depth.clip_key,
            rgb_thwc=cast(npt.NDArray[np.uint8], rgb),
            depth_z_m=metric_depth.depth_z_m,
            depth_valid=metric_depth.valid,
            rectification_known=camera.raster_plan.rectification_known,
            K_canvas=metric_depth.K_canvas,
            world_to_camera_cv=metric_depth.world_to_camera_cv,
        )
    finally:
        if isinstance(rgb, np.memmap):
            cast(Any, rgb)._mmap.close()


__all__ = ["LegacyDdwSource", "open_legacy_ddw_source"]

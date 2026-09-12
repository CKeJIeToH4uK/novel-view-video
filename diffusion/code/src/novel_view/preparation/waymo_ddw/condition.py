"""Exact A to B to A-prime condition for Waymo DDW preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from novel_view.preparation.waymo_ddw.warp import (
    WarpRequest,
    WarpResources,
    run_forward_warp,
)

if TYPE_CHECKING:
    from novel_view.preparation.waymo_ddw.depth import MetricDepth
    from novel_view.preparation.waymo_ddw.raster import WaymoFrontRaster
    from novel_view.preparation.waymo_ddw.target_path import DdwPath


@dataclass(frozen=True, slots=True, eq=False)
class DdwCondition:
    """Returned A-prime pixels, known mask and two-pass telemetry."""

    rgb_minus_one_to_one: npt.NDArray[np.float32]
    known: npt.NDArray[np.bool_]
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    elapsed_seconds: float


def build_condition(
    front: WaymoFrontRaster,
    depth: MetricDepth,
    path: DdwPath,
    resources: WarpResources,
    outward_log_path: Path,
    return_log_path: Path,
) -> DdwCondition:
    """Warp measured A to virtual B, then the actual B back to measured A."""
    outward = run_forward_warp(
        WarpRequest(
            source_rgb=front.rgb_thwc,
            source_rgb_layout="uint8_thwc",
            source_depth_z_m=depth.depth_z_m,
            source_depth_valid=depth.valid,
            source_rectification_known=front.rectification_known,
            source_K_canvas=front.K_canvas,
            source_world_to_camera_cv=front.world_to_camera_cv,
            target_K_canvas=front.K_canvas,
            target_world_to_camera_cv=path.virtual_world_to_camera_cv,
        ),
        resources,
        outward_log_path,
    )
    returning = run_forward_warp(
        WarpRequest(
            source_rgb=outward.rgb_minus_one_to_one[:, 0],
            source_rgb_layout="normalized_tchw",
            source_depth_z_m=outward.depth_z_m[:, 0],
            source_depth_valid=outward.known[:, 0],
            source_rectification_known=np.ones(
                front.rectification_known.shape,
                dtype=np.bool_,
            ),
            source_K_canvas=front.K_canvas,
            source_world_to_camera_cv=path.virtual_world_to_camera_cv,
            target_K_canvas=front.K_canvas,
            target_world_to_camera_cv=front.world_to_camera_cv,
        ),
        resources,
        return_log_path,
    )
    return DdwCondition(
        rgb_minus_one_to_one=returning.rgb_minus_one_to_one[:, 0],
        known=returning.known[:, 0],
        peak_cuda_allocated_bytes=max(
            outward.peak_cuda_allocated_bytes,
            returning.peak_cuda_allocated_bytes,
        ),
        peak_cuda_reserved_bytes=max(
            outward.peak_cuda_reserved_bytes,
            returning.peak_cuda_reserved_bytes,
        ),
        elapsed_seconds=float(outward.elapsed_seconds + returning.elapsed_seconds),
    )


__all__ = ["DdwCondition", "build_condition"]

"""Compose the historical one-source A to B to A-prime DDW render."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from novel_view.preparation.waymo_ddw.condition import DdwCondition
from novel_view.preparation.waymo_ddw.legacy.path import LegacyDdwPath
from novel_view.preparation.waymo_ddw.legacy.source import LegacyDdwSource
from novel_view.preparation.waymo_ddw.warp import (
    WarpRequest,
    WarpResources,
    WarpResult,
    run_forward_warp,
)


@dataclass(frozen=True, slots=True, eq=False)
class LegacyDdwRenderResult:
    """Historical path and the final dense A-prime condition."""

    path: LegacyDdwPath
    condition: DdwCondition


def render_legacy_ddw(
    source: LegacyDdwSource,
    path: LegacyDdwPath,
    resources: WarpResources,
    outward_log_path: Path,
    return_log_path: Path,
) -> LegacyDdwRenderResult:
    """Warp measured A to virtual B and the actual B back to measured A."""
    outward: WarpResult | None = None
    returning: WarpResult | None = None
    try:
        outward = run_forward_warp(
            WarpRequest(
                source_rgb=source.rgb_thwc,
                source_rgb_layout="uint8_thwc",
                source_depth_z_m=source.depth_z_m,
                source_depth_valid=source.depth_valid,
                source_rectification_known=source.rectification_known,
                source_K_canvas=source.K_canvas,
                source_world_to_camera_cv=source.world_to_camera_cv,
                target_K_canvas=source.K_canvas,
                target_world_to_camera_cv=path.target.virtual_world_to_camera_cv,
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
                    source.rectification_known.shape,
                    dtype=np.bool_,
                ),
                source_K_canvas=source.K_canvas,
                source_world_to_camera_cv=path.target.virtual_world_to_camera_cv,
                target_K_canvas=source.K_canvas,
                target_world_to_camera_cv=source.world_to_camera_cv,
            ),
            resources,
            return_log_path,
        )
        condition = DdwCondition(
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
            elapsed_seconds=float(
                outward.elapsed_seconds + returning.elapsed_seconds
            ),
        )
        return LegacyDdwRenderResult(path=path, condition=condition)
    finally:
        del outward, returning


__all__ = ["LegacyDdwRenderResult", "render_legacy_ddw"]

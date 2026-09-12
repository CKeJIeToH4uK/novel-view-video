"""Reduce one historical local DDW render before Cache4D reference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from novel_view.preparation.waymo_ddw.legacy.axis import DdwVariant
from novel_view.preparation.waymo_ddw.legacy.headroom import (
    DdwHeadroomScore,
    measure_ddw_headroom_uint8_target,
    score_ddw_headroom,
)
from novel_view.preparation.waymo_ddw.legacy.masks import (
    DdwMaskDescriptors,
    measure_ddw_masks,
)
from novel_view.preparation.waymo_ddw.legacy.path import (
    LegacyDdwPath,
    build_legacy_ddw_path,
)
from novel_view.preparation.waymo_ddw.legacy.render import (
    LegacyDdwRenderResult,
    render_legacy_ddw,
)
from novel_view.preparation.waymo_ddw.legacy.source import LegacyDdwSource
from novel_view.preparation.waymo_ddw.warp import WarpResources


@dataclass(frozen=True, slots=True)
class DdwLocalMeasurement:
    """Small local evidence retained after releasing the dense condition."""

    path: LegacyDdwPath
    observed: DdwMaskDescriptors
    headroom: DdwHeadroomScore
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    elapsed_seconds: float


def measure_legacy_ddw_local(
    source: LegacyDdwSource,
    variant: DdwVariant,
    resources: WarpResources,
    outward_log_path: Path,
    return_log_path: Path,
) -> DdwLocalMeasurement:
    """Render and immediately reduce one variant without running reference."""
    path = build_legacy_ddw_path(source, variant)
    rendered: LegacyDdwRenderResult | None = None
    try:
        rendered = render_legacy_ddw(
            source,
            path,
            resources,
            outward_log_path,
            return_log_path,
        )
        observed, headroom = measure_legacy_ddw_rendered(source, rendered)
        return DdwLocalMeasurement(
            path=path,
            observed=observed,
            headroom=headroom,
            peak_cuda_allocated_bytes=(
                rendered.condition.peak_cuda_allocated_bytes
            ),
            peak_cuda_reserved_bytes=(
                rendered.condition.peak_cuda_reserved_bytes
            ),
            elapsed_seconds=rendered.condition.elapsed_seconds,
        )
    finally:
        del rendered


def measure_legacy_ddw_rendered(
    source: LegacyDdwSource,
    rendered: LegacyDdwRenderResult,
) -> tuple[DdwMaskDescriptors, DdwHeadroomScore]:
    """Reduce one retained A-prime render using the frozen local metrics."""
    condition = rendered.condition
    observed = measure_ddw_masks(
        condition.known,
        np.broadcast_to(
            source.rectification_known,
            condition.known.shape,
        ),
    )
    headroom = score_ddw_headroom(
        measure_ddw_headroom_uint8_target(
            source.rgb_thwc,
            source.rectification_known,
            condition.rgb_minus_one_to_one,
            condition.known,
        )
    )
    return observed, headroom


__all__ = [
    "DdwLocalMeasurement",
    "measure_legacy_ddw_local",
    "measure_legacy_ddw_rendered",
]

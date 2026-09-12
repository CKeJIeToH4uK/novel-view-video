"""Global Cache4D reference for one historical Waymo DDW path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from novel_view.models.gen3c.cache4d.backend import (
    Gen3cCache4dResources,
    Gen3cCache4dResult,
    render_gen3c_cache4d,
)
from novel_view.models.gen3c.cache4d.request import (
    Gen3cConditioning,
    Gen3cSourceFrame,
)
from novel_view.preparation.waymo_ddw.legacy.masks import (
    DdwMaskDescriptors,
    measure_ddw_masks,
)
from novel_view.preparation.waymo_ddw.legacy.path import LegacyDdwPath
from novel_view.preparation.waymo_ddw.legacy.source import LegacyDdwSource


_FRAME_COUNT = 121


@dataclass(frozen=True, slots=True, eq=False)
class LegacyDdwSourceRows:
    """One measured FRONT row stream for the shared Cache4D request."""

    source: LegacyDdwSource

    def __len__(self) -> int:
        return _FRAME_COUNT

    def read(self, index: int) -> Gen3cSourceFrame:
        return Gen3cSourceFrame(
            rgb=self.source.rgb_thwc[index],
            depth_z_m=self.source.depth_z_m[index],
            valid=(
                self.source.depth_valid[index]
                & self.source.rectification_known
            ),
        )


@dataclass(frozen=True, slots=True)
class DdwReferenceMeasurement:
    """Small descriptors and telemetry after releasing the Cache4D output."""

    path: LegacyDdwPath
    descriptors: DdwMaskDescriptors
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    elapsed_seconds: float


def measure_legacy_ddw_reference(
    source: LegacyDdwSource,
    path: LegacyDdwPath,
    resources: Gen3cCache4dResources,
    log_path: Path,
) -> DdwReferenceMeasurement:
    """Render the full one-way Cache4D mask and immediately reduce it."""
    rows = LegacyDdwSourceRows(source)
    selected = np.arange(_FRAME_COUNT, dtype=np.int64)
    selected.setflags(write=False)
    conditioning = Gen3cConditioning(
        source_rows=rows,
        source_intrinsics=source.K_canvas,
        anchor_to_source_camera=source.world_to_camera_cv,
        source_sequence_index=selected,
        anchor_to_query_camera=path.target.virtual_world_to_camera_cv,
        query_intrinsics=source.K_canvas,
        selected_global_slots=selected,
    )
    result: Gen3cCache4dResult | None = None
    try:
        result = render_gen3c_cache4d(conditioning, resources, log_path)
        descriptors = measure_ddw_masks(
            result.selected_coverage_mask,
            np.broadcast_to(
                source.rectification_known,
                result.selected_coverage_mask.shape,
            ),
        )
        return DdwReferenceMeasurement(
            path=path,
            descriptors=descriptors,
            peak_cuda_allocated_bytes=result.peak_cuda_allocated_bytes,
            peak_cuda_reserved_bytes=result.peak_cuda_reserved_bytes,
            elapsed_seconds=result.elapsed_seconds,
        )
    finally:
        del result, conditioning, rows


__all__ = [
    "DdwReferenceMeasurement",
    "LegacyDdwSourceRows",
    "measure_legacy_ddw_reference",
]

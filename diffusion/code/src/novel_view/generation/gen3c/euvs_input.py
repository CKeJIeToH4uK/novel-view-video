"""EUVS adapter for the dataset-neutral Gen3C worker input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from novel_view.generation.gen3c.conditioning import (
    Gen3cProjectedSingleSourcePlan,
    materialize_gen3c_conditioning_window,
)
from novel_view.generation.gen3c.request import (
    gen3c_diagnostic_slots,
)
from novel_view.models.gen3c.cache4d.request import (
    Gen3cConditioning,
    Gen3cSourceFrame,
)

if TYPE_CHECKING:
    from novel_view.geometry.depth import PosedDepthSequence
    from novel_view.inputs.euvs.rgb import EuvsRasterizedSequence


@dataclass(frozen=True, slots=True, eq=False)
class EuvsGen3cConditioningInput:
    """Materialize one existing EUVS plan without changing its policy."""

    plan: Gen3cProjectedSingleSourcePlan
    source_rgb: EuvsRasterizedSequence
    source_geometry: PosedDepthSequence

    @property
    def image_size_hw(self) -> tuple[int, int]:
        return self.source_rgb.image_size_hw

    @property
    def model_frame_count(self) -> int:
        return self.plan.query_trajectory.timeline.model_frame_count

    @property
    def unpadded_frame_count(self) -> int:
        return self.plan.query_trajectory.timeline.unpadded_frame_count

    @property
    def context_depth_output_slots(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self.source_rgb.frames)

    def read(self, index: int) -> Gen3cSourceFrame:
        """Return one already loaded EUVS source row."""
        return Gen3cSourceFrame(
            rgb=self.source_rgb.frames[index].raster.rgb,
            depth_z_m=self.source_geometry.depth_z_m[index],
            valid=self.source_geometry.geometry_valid_mask[index],
        )

    def build_conditioning(self) -> Gen3cConditioning:
        """Build the exact pre-existing EUVS camera and source schedule."""
        timeline = self.plan.query_trajectory.timeline
        arrays = (
            np.empty(timeline.model_frame_count, dtype=np.int64),
            np.empty((timeline.model_frame_count, 4, 4), dtype=np.float64),
            np.empty((timeline.model_frame_count, 3, 3), dtype=np.float64),
        )
        for number, window in enumerate(timeline.iter_windows()):
            current = materialize_gen3c_conditioning_window(self.plan, window)
            values = (
                current.source_sequence_index,
                current.anchor_to_query_camera,
                current.query_intrinsics,
            )
            offset = 0
            if number:
                if any(
                    not np.array_equal(array[window.start], value[0])
                    for array, value in zip(arrays, values, strict=True)
                ):
                    raise ValueError(
                        f"conditioning overlap {window.start} is unstable"
                    )
                offset = 1
            for array, value in zip(arrays, values, strict=True):
                array[window.start + offset : window.stop] = value[offset:]

        selected = gen3c_diagnostic_slots(
            timeline.model_frame_count,
            timeline.unpadded_frame_count,
        )
        return Gen3cConditioning(
            source_rows=self,
            source_intrinsics=self.source_geometry.intrinsics,
            anchor_to_source_camera=(
                self.plan.query_trajectory.anchor_to_source_camera
            ),
            source_sequence_index=arrays[0],
            anchor_to_query_camera=arrays[1],
            query_intrinsics=arrays[2],
            selected_global_slots=selected,
        )


__all__ = ["EuvsGen3cConditioningInput"]

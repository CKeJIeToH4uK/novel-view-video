"""Build the ordinary Gaussian full-sequence Gen3C conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from novel_view.generation.gen3c.request import gen3c_diagnostic_slots
from novel_view.generation.gen3c.timeline import GEN3C_WINDOW_STEP
from novel_view.geometry.raster import CoverCropTransform
from novel_view.inputs.gaussian.reader import (
    GaussianExport,
    GaussianFrame,
    read_gaussian_raster,
)
from novel_view.inputs.gaussian.spec import (
    GaussianFullSequenceSelection,
    GaussianIndependentClipsSelection,
)
from novel_view.models.gen3c.cache4d.request import (
    Gen3cConditioning,
    Gen3cSourceFrame,
)
from novel_view.models.gen3c.spec import GEN3C_IMAGE_SIZE_HW


@dataclass(frozen=True, slots=True, eq=False)
class GaussianFullSequence:
    """Producer-ordered rows and one padded autoregressive camera schedule."""

    export: GaussianExport
    frame_rate: int
    raster_transform: CoverCropTransform
    output_intrinsics: npt.NDArray[np.float64]
    frames: tuple[GaussianFrame, ...]

    @property
    def image_size_hw(self) -> tuple[int, int]:
        return GEN3C_IMAGE_SIZE_HW

    @property
    def unpadded_frame_count(self) -> int:
        return len(self.frames) + 1

    @property
    def model_frame_count(self) -> int:
        windows = (len(self.frames) + GEN3C_WINDOW_STEP - 1) // GEN3C_WINDOW_STEP
        return windows * GEN3C_WINDOW_STEP + 1

    @property
    def target_output_index(self) -> npt.NDArray[np.int64]:
        return np.arange(1, len(self.frames) + 1, dtype=np.int64)

    @property
    def selected_pose_indices(self) -> tuple[int, ...]:
        return tuple(frame.pose_index for frame in self.frames)

    @property
    def context_depth_output_slots(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self.frames)

    def read(self, index: int) -> Gen3cSourceFrame:
        raster = read_gaussian_raster(
            self.frames[index],
            self.export.source_size_hw,
            self.raster_transform,
        )
        return Gen3cSourceFrame(
            raster.rgb,
            raster.depth_z_m,
            raster.valid,
        )

    def build_conditioning(self) -> Gen3cConditioning:
        """Rebase source/target W2C once and hold the short tail."""
        count = len(self.frames)
        anchor_inverse = np.linalg.inv(self.frames[0].source_w2c)
        source_w2c = np.stack(
            [frame.source_w2c @ anchor_inverse for frame in self.frames]
        )
        target_w2c = np.stack(
            [frame.target_w2c @ anchor_inverse for frame in self.frames]
        )
        source_index = np.arange(self.model_frame_count, dtype=np.int64) - 1
        np.clip(source_index, 0, count - 1, out=source_index)
        query_w2c = np.empty((self.model_frame_count, 4, 4), dtype=np.float64)
        query_w2c[0] = source_w2c[0]
        query_w2c[1 : count + 1] = target_w2c
        query_w2c[count + 1 :] = target_w2c[-1]
        return Gen3cConditioning(
            source_rows=self,
            source_intrinsics=np.repeat(
                self.output_intrinsics[None], count, axis=0
            ),
            anchor_to_source_camera=source_w2c,
            source_sequence_index=source_index,
            anchor_to_query_camera=query_w2c,
            query_intrinsics=np.repeat(
                self.output_intrinsics[None], self.model_frame_count, axis=0
            ),
            selected_global_slots=gen3c_diagnostic_slots(
                self.model_frame_count,
                self.unpadded_frame_count,
            ),
        )


def build_gaussian_full_sequence(
    export: GaussianExport,
    selection: GaussianFullSequenceSelection | GaussianIndependentClipsSelection,
    frame_rate: int,
) -> GaussianFullSequence:
    """Bind one external export to its selected scene and nearest FPS rows."""
    if export.scene_id != selection.scene_id:
        raise ValueError("Gaussian export disagrees with selected scene")
    transform = CoverCropTransform(export.source_size_hw, GEN3C_IMAGE_SIZE_HW)
    return GaussianFullSequence(
        export=export,
        frame_rate=frame_rate,
        raster_transform=transform,
        output_intrinsics=transform.transform_intrinsics(export.intrinsics),
        frames=_select_continuous_frames(export.frames, frame_rate),
    )


def _select_continuous_frames(
    frames: tuple[GaussianFrame, ...],
    frame_rate: int,
) -> tuple[GaussianFrame, ...]:
    origin = frames[0].timestamp_ns
    by_slot: dict[int, tuple[int, GaussianFrame]] = {}
    for frame in frames:
        delta = frame.timestamp_ns - origin
        slot = (delta * frame_rate + 500_000_000) // 1_000_000_000
        residual = abs(delta * frame_rate - slot * 1_000_000_000)
        previous = by_slot.get(slot)
        if previous is None or residual < previous[0]:
            by_slot[slot] = residual, frame
    expected = tuple(range(max(by_slot) + 1))
    if tuple(sorted(by_slot)) != expected:
        raise ValueError("Gaussian export does not cover a continuous timeline")
    return tuple(by_slot[slot][1] for slot in expected)


__all__ = [
    "GaussianFullSequence",
    "build_gaussian_full_sequence",
]

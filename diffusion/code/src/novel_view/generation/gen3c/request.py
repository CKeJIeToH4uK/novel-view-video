"""Thin generation request around the shared Cache4D conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from novel_view.generation.gen3c.windows import GEN3C_WINDOW_STEP
from novel_view.models.gen3c.cache4d.request import Gen3cConditioning


class Gen3cConditioningInput(Protocol):
    """Science object that can build one dataset-free conditioning request."""

    @property
    def image_size_hw(self) -> tuple[int, int]: ...

    @property
    def model_frame_count(self) -> int: ...

    @property
    def unpadded_frame_count(self) -> int: ...

    def build_conditioning(self) -> Gen3cConditioning: ...

    @property
    def context_depth_output_slots(self) -> npt.NDArray[np.int64] | None: ...


@dataclass(frozen=True, slots=True, eq=False)
class Gen3cGenerationRequest:
    """Conditioning, optional context slots, and caller-owned RGB output."""

    conditioning: Gen3cConditioning
    output_rgb_path: Path
    context_depth_output_slots: npt.NDArray[np.int64] | None = None


def gen3c_diagnostic_slots(
    model_frame_count: int,
    unpadded_frame_count: int,
) -> npt.NDArray[np.int64]:
    """Return a bounded preview set covering the first seam and final data."""
    slots = {
        0,
        1,
        GEN3C_WINDOW_STEP - 1,
        GEN3C_WINDOW_STEP,
        GEN3C_WINDOW_STEP + 1,
        unpadded_frame_count - 1,
        model_frame_count - 1,
    }
    return np.asarray(
        sorted(slot for slot in slots if 0 <= slot < model_frame_count),
        dtype=np.int64,
    )


__all__ = [
    "Gen3cConditioningInput",
    "Gen3cGenerationRequest",
    "gen3c_diagnostic_slots",
]

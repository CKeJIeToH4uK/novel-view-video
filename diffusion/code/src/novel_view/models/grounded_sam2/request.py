"""Source-neutral values crossing the Grounded-SAM2 boundary."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, eq=False)
class GroundedSam2Request:
    """Ordered native RGB plus the exact composite model recipe."""

    rgb_frames: Iterable[npt.NDArray[np.uint8]]
    frame_count: int
    image_size_hw: tuple[int, int]
    sam2_model_config: str
    prompt: str
    box_threshold: float
    text_threshold: float


@dataclass(frozen=True, slots=True, eq=False)
class GroundedSam2Prediction:
    """One union dynamic mask per input frame on the native grid."""

    dynamic_mask: npt.NDArray[np.bool_]


__all__ = ["GroundedSam2Prediction", "GroundedSam2Request"]

"""Dataset-neutral dense camera-Z depth on one ordered image grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, eq=False)
class PosedDepthSequence:
    """Numeric depth, pinholes and W2C cameras for one ordered sequence.

    Concrete model adapters own array validation, physical semantics and
    immutable storage. This lightweight contract carries their result without
    importing dataset inputs or copying the heavy arrays again.
    """

    sequence_id: tuple[str, ...]
    depth_z_m: npt.NDArray[np.float32]
    geometry_valid_mask: npt.NDArray[np.bool_]
    intrinsics: npt.NDArray[np.float64]
    reference_to_camera: npt.NDArray[np.float64]
    backend_confidence: npt.NDArray[np.float32] | None = None

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    @property
    def frame_count(self) -> int:
        """Return the numeric leading dimension."""
        return int(self.depth_z_m.shape[0])

    @property
    def image_size_hw(self) -> tuple[int, int]:
        """Return the dense depth grid as ``(height, width)``."""
        return int(self.depth_z_m.shape[1]), int(self.depth_z_m.shape[2])

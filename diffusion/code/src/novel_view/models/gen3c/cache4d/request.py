"""Light numeric request types shared by every Cache4D consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, eq=False)
class Gen3cSourceFrame:
    """One already prepared RGB/depth/valid source row."""

    rgb: npt.NDArray[np.uint8]
    depth_z_m: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]


class Gen3cSourceRows(Protocol):
    """Static row reader for one concrete Gen3C source pack."""

    def __len__(self) -> int: ...

    def read(self, index: int) -> Gen3cSourceFrame: ...


@dataclass(frozen=True, slots=True, eq=False)
class Gen3cContextConditioning:
    """Optional second Cache4D layer built from generated context."""

    rows: Gen3cSourceRows
    anchor_to_context_camera: npt.NDArray[np.float64]
    context_intrinsics: npt.NDArray[np.float64]
    context_sequence_index: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True, eq=False)
class Gen3cConditioning:
    """Dataset-free source evidence and global camera schedule."""

    source_rows: Gen3cSourceRows
    source_intrinsics: npt.NDArray[np.float64]
    anchor_to_source_camera: npt.NDArray[np.float64]
    source_sequence_index: npt.NDArray[np.int64]
    anchor_to_query_camera: npt.NDArray[np.float64]
    query_intrinsics: npt.NDArray[np.float64]
    selected_global_slots: npt.NDArray[np.int64]
    context: Gen3cContextConditioning | None = None


__all__ = [
    "Gen3cConditioning",
    "Gen3cContextConditioning",
    "Gen3cSourceFrame",
    "Gen3cSourceRows",
]

"""Describe exact Gen3C window, seed, and seam decisions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from novel_view.models.gen3c.spec import GEN3C_WINDOW_SIZE

GEN3C_WINDOW_STEP = 120
WINDOW_SEED_AUTOREGRESSIVE = "autoregressive/v1"
WINDOW_SEED_SOURCE_RESEED = "source-reseed/v1"
WINDOW_SEED_POLICIES = frozenset(
    (WINDOW_SEED_AUTOREGRESSIVE, WINDOW_SEED_SOURCE_RESEED)
)

Gen3cWindowSeedKind = Literal["source", "previous"]


@dataclass(frozen=True, slots=True)
class Gen3cWindowCall:
    """One ordered model call within a shared Gen3C sequence."""

    start: int
    stop: int
    seed_kind: Gen3cWindowSeedKind
    source_index: int
    replace_query_row_zero: bool
    generated_slice_start: int


def iter_window_calls(
    source_sequence_index: npt.NDArray[np.int64],
    window_seed_policy: str,
) -> Iterator[Gen3cWindowCall]:
    """Yield the pinned 121-frame calls without running Cache4D or Gen3C."""
    for start in range(
        0,
        int(source_sequence_index.size) - 1,
        GEN3C_WINDOW_STEP,
    ):
        first = start == 0
        source_reseed = window_seed_policy == WINDOW_SEED_SOURCE_RESEED
        yield Gen3cWindowCall(
            start=start,
            stop=start + GEN3C_WINDOW_SIZE,
            seed_kind=(
                "source" if first or source_reseed else "previous"
            ),
            source_index=int(source_sequence_index[start]),
            replace_query_row_zero=source_reseed,
            generated_slice_start=0 if first else 1,
        )


def materialize_window_cameras(
    query_w2c: npt.NDArray[np.float64],
    query_intrinsics: npt.NDArray[np.float64],
    source_w2c: npt.NDArray[np.float64],
    source_intrinsics: npt.NDArray[np.float64],
    call: Gen3cWindowCall,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Copy one query window and apply its coherent source-camera reseed."""
    window_w2c = np.array(
        query_w2c[call.start : call.stop],
        dtype=np.float32,
        copy=True,
    )
    window_intrinsics = np.array(
        query_intrinsics[call.start : call.stop],
        dtype=np.float32,
        copy=True,
    )
    if call.replace_query_row_zero:
        window_w2c[0] = source_w2c[call.source_index]
        window_intrinsics[0] = source_intrinsics[call.source_index]
    return window_w2c, window_intrinsics


__all__ = [
    "GEN3C_WINDOW_SIZE",
    "GEN3C_WINDOW_STEP",
    "Gen3cWindowCall",
    "WINDOW_SEED_AUTOREGRESSIVE",
    "WINDOW_SEED_POLICIES",
    "WINDOW_SEED_SOURCE_RESEED",
    "iter_window_calls",
    "materialize_window_cameras",
]

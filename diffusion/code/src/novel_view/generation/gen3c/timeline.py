"""Place exact targets on the pinned Gen3C temporal grid."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

GEN3C_FRAMES_PER_SECOND = 24
GEN3C_WINDOW_SIZE = 121
GEN3C_WINDOW_OVERLAP = 1
GEN3C_WINDOW_STEP = GEN3C_WINDOW_SIZE - GEN3C_WINDOW_OVERLAP

_MICROSECONDS_PER_SECOND = 1_000_000
_HALF_SECOND_IN_MICROSECONDS = _MICROSECONDS_PER_SECOND // 2


class Gen3cTimelineError(ValueError):
    """Exact targets collide on the pinned Gen3C timeline."""


@dataclass(frozen=True, slots=True)
class Gen3cWindow:
    """One half-open 121-slot view of a shared persistent trajectory."""

    start: int
    stop: int


@dataclass(frozen=True, slots=True, eq=False)
class Gen3cTimeline:
    """Exact target anchors, source choices and derived model windows."""

    target_output_index: npt.NDArray[np.int64]
    source_sequence_index: npt.NDArray[np.int64]

    @property
    def unpadded_frame_count(self) -> int:
        """Return the prefix ending immediately after the last target."""
        return int(self.target_output_index[-1]) + 1

    @property
    def model_frame_count(self) -> int:
        """Return the smallest compatible ``120*k+1`` model length."""
        return _compatible_frame_count(self.unpadded_frame_count)

    @property
    def padding_frame_count(self) -> int:
        """Return model slots after the final exact target."""
        return self.model_frame_count - self.unpadded_frame_count

    @property
    def window_count(self) -> int:
        """Return the number of overlapping persistent-model windows."""
        return (self.model_frame_count - 1) // GEN3C_WINDOW_STEP

    def iter_windows(self) -> Iterator[Gen3cWindow]:
        """Yield canonical windows over one shared global trajectory."""
        for number in range(self.window_count):
            start = number * GEN3C_WINDOW_STEP
            yield Gen3cWindow(start, start + GEN3C_WINDOW_SIZE)


def build_direct_gen3c_timeline(
    target_timestamps_us: Sequence[int],
    source_sequence_index: npt.ArrayLike,
) -> Gen3cTimeline:
    """Quantize target-relative time and retain exact target source choices.

    Target order is never sorted or repaired. Slot ``0`` is the source seed,
    the first target occupies slot ``1``, and later targets use exact integer
    round-half-up arithmetic at 24 fps. Two targets in one output slot are a
    scientifically ambiguous request and therefore fail here.
    """
    timestamps = tuple(target_timestamps_us)
    mapping = np.asarray(source_sequence_index, dtype=np.int64)
    indices = [1]
    first_timestamp = timestamps[0]
    for target_index, timestamp in enumerate(timestamps[1:], start=1):
        elapsed_us = timestamp - first_timestamp
        output_index = 1 + _round_half_up_frames(elapsed_us)
        if output_index <= indices[-1]:
            raise Gen3cTimelineError(
                f"target[{target_index - 1}] and target[{target_index}] "
                f"collide at Gen3C output slot {output_index}"
            )
        indices.append(output_index)
    return Gen3cTimeline(
        target_output_index=_readonly_int64(indices),
        source_sequence_index=_readonly_int64(mapping),
    )


def _round_half_up_frames(elapsed_us: int) -> int:
    numerator = elapsed_us * GEN3C_FRAMES_PER_SECOND
    return (numerator + _HALF_SECOND_IN_MICROSECONDS) // _MICROSECONDS_PER_SECOND


def _compatible_frame_count(unpadded_frame_count: int) -> int:
    window_count = max(
        1,
        (unpadded_frame_count - 1 + GEN3C_WINDOW_STEP - 1)
        // GEN3C_WINDOW_STEP,
    )
    return window_count * GEN3C_WINDOW_STEP + GEN3C_WINDOW_OVERLAP


def _readonly_int64(value: object) -> npt.NDArray[np.int64]:
    array = np.array(value, dtype=np.int64, copy=True, order="C")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.int64).reshape(
        array.shape
    )

"""Plan selected source-seeded clips from one Gaussian native sequence."""

from __future__ import annotations

from dataclasses import dataclass, replace

from novel_view.generation.gaussian.full import GaussianFullSequence
from novel_view.generation.gen3c.timeline import GEN3C_WINDOW_STEP


@dataclass(frozen=True, slots=True)
class GaussianClip:
    """One 1-based independent clip and its fresh source-seeded sequence."""

    index: int
    count: int
    target_start_index: int
    full_target_count: int
    sequence: GaussianFullSequence

    @property
    def target_stop_index(self) -> int:
        return self.target_start_index + len(self.sequence)


def plan_native_clips(
    sequence: GaussianFullSequence,
    clip_indices: tuple[int, ...],
) -> tuple[GaussianClip, ...]:
    """Return exact selected clips, right-aligning the final full window."""
    total = len(sequence)
    size = GEN3C_WINDOW_STEP
    if total < size:
        raise ValueError(
            f"independent Gaussian clips require at least {size} target frames"
        )
    starts = list(range(0, total - size + 1, size))
    final_start = total - size
    if starts[-1] != final_start:
        starts.append(final_start)
    count = len(starts)
    if any(index > count for index in clip_indices):
        raise ValueError(f"Gaussian clip index must be within 1..{count}")
    return tuple(
        GaussianClip(
            index=index,
            count=count,
            target_start_index=starts[index - 1],
            full_target_count=total,
            sequence=replace(
                sequence,
                frames=sequence.frames[
                    starts[index - 1] : starts[index - 1] + size
                ],
            ),
        )
        for index in clip_indices
    )


__all__ = ["GaussianClip", "plan_native_clips"]

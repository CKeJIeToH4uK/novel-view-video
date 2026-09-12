"""Apply equal-camera, equal-pair and equal-location reductions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from novel_view.metrics.image import MetricError, MetricStatus, MetricValue


@dataclass(frozen=True, slots=True)
class FrameMacroMetric:
    """Equal-camera mean plus counts of deliberately omitted states."""

    finite_mean: float | None
    finite_frame_count: int
    positive_infinity_frame_count: int
    undefined_frame_count: int

    def __post_init__(self) -> None:
        if self.finite_mean is not None and not math.isfinite(self.finite_mean):
            raise MetricError("finite_mean must be finite or None")
        counts = (
            self.finite_frame_count,
            self.positive_infinity_frame_count,
            self.undefined_frame_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise MetricError("metric-state counts must be non-negative integers")
        if (self.finite_mean is None) != (self.finite_frame_count == 0):
            raise MetricError("finite_mean presence must match finite count")


def summarize_cameras(values: Sequence[MetricValue]) -> FrameMacroMetric:
    """Average finite camera rows and retain every exceptional-state count."""
    finite = tuple(
        value.value
        for value in values
        if value.status is MetricStatus.FINITE and value.value is not None
    )
    return FrameMacroMetric(
        finite_mean=float(np.mean(finite)) if finite else None,
        finite_frame_count=len(finite),
        positive_infinity_frame_count=sum(
            value.status is MetricStatus.POSITIVE_INFINITY for value in values
        ),
        undefined_frame_count=sum(
            value.status is MetricStatus.UNDEFINED for value in values
        ),
    )


def mean_columns(
    rows: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Give every complete row equal weight in each numeric column."""
    width = len(rows[0])
    return tuple(
        sum(row[index] for row in rows) / len(rows)
        for index in range(width)
    )


__all__ = ["FrameMacroMetric", "mean_columns", "summarize_cameras"]

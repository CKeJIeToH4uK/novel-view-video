"""Проекция планарных точек на ориентированную полилинию."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class PolylineError(ValueError):
    """Исходная полилиния математически не определяет проекцию."""


@dataclass(frozen=True, slots=True, eq=False)
class PolylineProjection:
    """Ближайшие проекции на зажатые сегменты в порядке запросов."""

    projected_xy: npt.NDArray[np.float64]
    segment_index: npt.NDArray[np.int64]
    segment_fraction: npt.NDArray[np.float64]
    source_progress: npt.NDArray[np.float64]
    distance: npt.NDArray[np.float64]
    signed_cross_track: npt.NDArray[np.float64]

    def __eq__(self, other: object) -> bool:
        """Сохранить точное сравнение действующего EUVS-контракта."""
        if not isinstance(other, PolylineProjection):
            return NotImplemented
        return all(
            bool(np.array_equal(left, right))
            for left, right in (
                (self.projected_xy, other.projected_xy),
                (self.segment_index, other.segment_index),
                (self.segment_fraction, other.segment_fraction),
                (self.source_progress, other.source_progress),
                (self.distance, other.distance),
                (self.signed_cross_track, other.signed_cross_track),
            )
        )


def project_points_to_polyline(
    source_xy: npt.ArrayLike,
    query_xy: npt.ArrayLike,
) -> PolylineProjection:
    """Спроецировать точки на ближайшие зажатые сегменты полилинии."""
    source = np.asarray(source_xy, dtype=np.float64)
    queries = np.asarray(query_xy, dtype=np.float64)
    if source.shape[0] < 2:
        raise PolylineError("source polyline must contain at least two vertices")

    starts = source[:-1]
    deltas = source[1:] - starts
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    zero_segments = np.flatnonzero(lengths == 0.0)
    if zero_segments.size:
        raise PolylineError(
            "source polyline contains a zero-length segment at index "
            f"{int(zero_segments[0])}"
        )

    unit_directions = deltas / lengths[:, None]
    segment_progress = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(lengths[:-1]))
    )
    row_count = queries.shape[0]
    projected_xy = np.empty((row_count, 2), dtype=np.float64)
    segment_index = np.empty(row_count, dtype=np.int64)
    segment_fraction = np.empty(row_count, dtype=np.float64)
    source_progress = np.empty(row_count, dtype=np.float64)
    distance = np.empty(row_count, dtype=np.float64)
    signed_cross_track = np.empty(row_count, dtype=np.float64)

    for query_index, point in enumerate(queries):
        offsets = point - starts
        longitudinal = np.einsum("ij,ij->i", offsets, unit_directions)
        fractions = np.clip(longitudinal / lengths, 0.0, 1.0)
        candidates = starts + fractions[:, None] * deltas
        candidates[fractions == 0.0] = starts[fractions == 0.0]
        candidates[fractions == 1.0] = source[1:][fractions == 1.0]
        residuals = point - candidates
        candidate_distances = np.hypot(residuals[:, 0], residuals[:, 1])
        selected = int(np.argmin(candidate_distances))
        fraction = fractions[selected]

        projected_xy[query_index] = candidates[selected]
        segment_index[query_index] = selected
        segment_fraction[query_index] = fraction
        source_progress[query_index] = (
            segment_progress[selected] + fraction * lengths[selected]
        )
        distance[query_index] = candidate_distances[selected]
        signed_cross_track[query_index] = (
            unit_directions[selected, 0] * offsets[selected, 1]
            - unit_directions[selected, 1] * offsets[selected, 0]
        )

    for values in (
        projected_xy,
        segment_index,
        segment_fraction,
        source_progress,
        distance,
        signed_cross_track,
    ):
        values.flags.writeable = False

    return PolylineProjection(
        projected_xy=projected_xy,
        segment_index=segment_index,
        segment_fraction=segment_fraction,
        source_progress=source_progress,
        distance=distance,
        signed_cross_track=signed_cross_track,
    )


def select_nearest_polyline_vertex_indices(
    projection: PolylineProjection,
) -> npt.NDArray[np.int64]:
    """Выбрать ближайший конец каждого уже найденного сегмента."""
    selected = np.array(projection.segment_index, dtype=np.int64, copy=True)
    selected += projection.segment_fraction > 0.5
    return selected

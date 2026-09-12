"""Clamped planar projection, oriented distance, arc progress and deterministic ties."""

from numpy.testing import assert_array_equal as equal

from novel_view.geometry.polyline import (
    project_points_to_polyline, select_nearest_polyline_vertex_indices,
)


def test_projection_clamp_preserves_order_and_signed_normal():
    result = project_points_to_polyline(
        [[0, 0], [10, 0]], [[7, -4], [2, 3], [-3, 4], [13, -4]],
    )
    equal(result.projected_xy, [[7, 0], [2, 0], [0, 0], [10, 0]])
    equal(result.segment_fraction, [.7, .2, 0, 1])
    equal(result.source_progress, [7, 2, 0, 10])
    equal(result.distance, [4, 3, 5, 5])
    equal(result.signed_cross_track, [-4, 3, 4, -4])


def test_polyline_progress_accumulates_unequal_segment_lengths():
    result = project_points_to_polyline([[0, 0], [3, 0], [3, 4]], [[1, 1], [4, 2]])
    equal(result.segment_index, [0, 1])
    equal(result.segment_fraction, [1 / 3, .5])
    equal(result.source_progress, [1, 5])
    equal(result.projected_xy, [[1, 0], [3, 2]])
    equal(result.signed_cross_track, [1, -1])
    equal(result.distance, [1, 1])


def test_projection_ties_keep_first_segment_and_lower_vertex():
    shared = project_points_to_polyline([[0, 0], [2, 0], [2, 2]], [[2, 0]])
    equal(shared.segment_index, [0])
    equal(shared.segment_fraction, [1])
    middle = project_points_to_polyline([[0, 0], [2, 0]], [[1, 1], [1.000002, -1]])
    equal(select_nearest_polyline_vertex_indices(middle), [0, 1])

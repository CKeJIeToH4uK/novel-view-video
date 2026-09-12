"""Compact scientific proofs for the migrated historical DDW policies."""

from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from novel_view.preparation.waymo_ddw.legacy.axis import DdwMaskCandidate, DdwVariant
from novel_view.preparation.waymo_ddw.legacy.calibration import calibrate_ddw_mask_gate
from novel_view.preparation.waymo_ddw.legacy.gate import (
    DDW_MASK_GATE_ID, DdwMaskGateSpec, DdwMaskReferenceTolerance,
    score_ddw_masks,
)
from novel_view.preparation.waymo_ddw.legacy.headroom import (
    DDW_MINIMUM_CHANGED_PIXEL_FRACTION, DDW_MINIMUM_MEAN_ABSOLUTE_CORRUPTION,
    DDW_NORMALIZED_FLOAT32_ATOL, DdwHeadroomMetrics,
    measure_ddw_headroom_arrays, measure_ddw_headroom_uint8_target,
    score_ddw_headroom,
)
from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors, measure_ddw_masks
from novel_view.preparation.waymo_ddw.legacy.selection import DDW_VARIANTS, select_ddw_variants
from novel_view.preparation.waymo_ddw.legacy.v2 import DDW_V2_VARIANTS
from novel_view.preparation.waymo_ddw.legacy.v2_calibration import calibrate_ddw_v2_mask_gate
from novel_view.preparation.waymo_ddw.legacy.v2_gate import (
    DDW_V2_MASK_GATE_ID, DdwV2MaskGateSpec, DdwV2MaskReferenceTolerance,
    score_ddw_v2_masks,
)
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.inputs.waymo.types import WaymoContractError


def _key(index: int) -> WaymoClipKey:
    timestamps = tuple(index * 1_000_000 + frame for frame in range(121))
    return WaymoClipKey("synthetic-ddw", "validation", f"segment-{index}", index, timestamps)


def _mask_fixture() -> tuple[np.ndarray, np.ndarray]:
    known = np.ones((121, 7, 8), dtype=np.bool_)
    for index in range(121):
        known[index, 1:5, 1 : 1 + index % 5] = False
        known[index, 5, 5] = index % 3 != 0
        known[index, 6, 6] = index % 4 != 0
    rectification = np.ones((121, 7, 8), dtype=np.bool_)
    rectification[:, 0, 0] = False
    return known, rectification


def _component_areas(mask: np.ndarray) -> tuple[int, ...]:
    remaining = set(map(tuple, np.argwhere(mask)))
    areas: list[int] = []
    while remaining:
        queue = deque((remaining.pop(),))
        area = 0
        while queue:
            y, x = queue.popleft()
            area += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbour = (y + dy, x + dx)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        queue.append(neighbour)
        areas.append(area)
    return tuple(areas)


def _independent_masks(known: np.ndarray, rect: np.ndarray) -> tuple[float, ...]:
    coverage, largest, components, boundary = [], [], [], []
    for frame_known, frame_rect in zip(known, rect, strict=True):
        rect_count = int(np.count_nonzero(frame_rect))
        coverage.append(np.count_nonzero(frame_known & frame_rect) / rect_count)
        areas = _component_areas((~frame_known) & frame_rect)
        largest.append(max(areas, default=0) / rect_count)
        components.append(sum(area >= 16 for area in areas))
        horizontal = frame_rect[:, :-1] & frame_rect[:, 1:]
        vertical = frame_rect[:-1] & frame_rect[1:]
        pair_count = np.count_nonzero(horizontal) + np.count_nonzero(vertical)
        changed = np.count_nonzero(
            horizontal & (frame_known[:, :-1] != frame_known[:, 1:])
        ) + np.count_nonzero(vertical & (frame_known[:-1] != frame_known[1:]))
        boundary.append(changed / pair_count if pair_count else 0.0)
    temporal_iou, flicker = [], []
    for index in range(120):
        support = rect[index] & rect[index + 1]
        left, right = known[index] & support, known[index + 1] & support
        union = int(np.count_nonzero(left | right))
        temporal_iou.append(1.0 if union == 0 else np.count_nonzero(left & right) / union)
        flicker.append(np.count_nonzero(left ^ right) / np.count_nonzero(support))
    values = coverage, largest, components, boundary, temporal_iou, flicker
    probabilities = 0.05, 0.95, 0.95, 0.95, 0.05, 0.95
    return tuple(
        float(np.quantile(column, probability, method="linear"))
        for column, probability in zip(values, probabilities, strict=True)
    )


def test_mask_descriptors_match_independent_six_metric_reference() -> None:
    known, rectification = _mask_fixture()
    actual = measure_ddw_masks(known, rectification)
    actual_values = tuple(
        getattr(actual, name) for name in actual.__dataclass_fields__
    )
    np.testing.assert_allclose(actual_values, _independent_masks(known, rectification))


def test_masks_keep_eight_connectivity_area_sixteen_and_empty_union() -> None:
    known = np.ones((121, 12, 12), dtype=np.bool_)
    known[:, 1:4, 1:6] = False
    known[:, 6:10, 7:11] = False
    result = measure_ddw_masks(known, np.ones_like(known))
    assert result.hole_component_count_p95 == 1.0

    empty = np.zeros((121, 2, 2), dtype=np.bool_)
    result = measure_ddw_masks(empty, np.ones_like(empty))
    assert (result.temporal_iou_q05, result.temporal_flicker_p95) == (1.0, 0.0)


def test_uint8_headroom_matches_old_float32_path_for_every_code() -> None:
    values = np.arange(256, dtype=np.uint8).reshape(16, 16)
    frame = np.stack((values, np.flipud(values), np.fliplr(values)), axis=-1)
    target_uint8 = np.broadcast_to(frame, (121, *frame.shape)).copy()
    target_normalized = np.asarray(target_uint8, dtype=np.float32)
    target_normalized = target_normalized * (2.0 / 255.0) - 1.0
    target_normalized = np.moveaxis(target_normalized, -1, 1)
    condition = target_normalized.copy()
    condition[1:, :, ::2, 1::2] += np.float32(4.0 / 255.0)
    known = np.ones((121, 16, 16), dtype=np.bool_)
    known[1:, 0, :] = False
    rectification = np.ones((16, 16), dtype=np.bool_)
    rectification[:, 0] = False

    old = measure_ddw_headroom_arrays(
        target_normalized, rectification, condition[:, None], known[:, None]
    )
    new = measure_ddw_headroom_uint8_target(
        target_uint8, rectification, condition, known
    )
    assert new == old
    assert new.restoration_pixel_count == 120 * 15 * 15
    with pytest.raises(WaymoContractError, match="restoration region is empty"):
        measure_ddw_headroom_arrays(target_normalized, rectification, condition[:, None], np.zeros_like(known)[:, None])


def test_headroom_thresholds_are_inclusive_and_failures_are_ordered() -> None:
    passing = DdwHeadroomMetrics(
        1,
        DDW_MINIMUM_MEAN_ABSOLUTE_CORRUPTION - DDW_NORMALIZED_FLOAT32_ATOL,
        DDW_MINIMUM_CHANGED_PIXEL_FRACTION,
    )
    assert score_ddw_headroom(passing).passed
    below = np.nextafter(passing.mean_absolute_corruption, -np.inf)
    assert score_ddw_headroom(replace(passing, mean_absolute_corruption=below)).failures == ("mean_absolute_corruption",)
    assert score_ddw_headroom(DdwHeadroomMetrics(1, 0.0, 0.0)).failures == ("mean_absolute_corruption", "changed_pixel_fraction")


def _gate_specs() -> tuple[DdwMaskGateSpec, DdwV2MaskGateSpec]:
    zero_v1 = DdwMaskReferenceTolerance(*(0.0 for _ in range(6)))
    zero_v2 = DdwV2MaskReferenceTolerance(*(0.0 for _ in range(6)))
    values = (0.8, 0.2, 3.0, 0.1, 0.7, 0.05)
    return (
        DdwMaskGateSpec(DDW_MASK_GATE_ID, *values, zero_v1, zero_v1),
        DdwV2MaskGateSpec(DDW_V2_MASK_GATE_ID, *values, zero_v2, zero_v2),
    )


def test_versioned_gates_keep_inclusive_bounds_and_failure_order() -> None:
    boundary = DdwMaskDescriptors(0.8, 0.2, 3.0, 0.1, 0.7, 0.05)
    v1, v2 = _gate_specs()
    assert v1.gate_id != v2.gate_id
    assert score_ddw_masks(v1, boundary, boundary).passed
    assert score_ddw_v2_masks(v2, boundary, boundary).passed
    failed = DdwMaskDescriptors(0.7, 0.3, 4.0, 0.2, 0.6, 0.15)
    expected = (
        "coverage_q05", "temporal_iou_q05", "largest_hole_fraction_p95",
        "hole_component_count_p95", "boundary_fraction_p95", "temporal_flicker_p95",
    )
    assert score_ddw_masks(v1, failed, failed).failures == expected
    assert score_ddw_v2_masks(v2, failed, failed).failures == expected
    tolerance = replace(
        v1,
        reference_atol=DdwMaskReferenceTolerance(.01, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert score_ddw_masks(tolerance, replace(boundary, coverage_q05=.805), boundary).passed
    outside = replace(boundary, coverage_q05=.811)
    assert score_ddw_masks(tolerance, outside, boundary).failures == (
        "coverage_q05.reference",
    )


def _v1_candidates() -> tuple[DdwMaskCandidate, ...]:
    observed = (
        (.82, .10, 10.0, .05, .90, .02), (.61, .11, 11.0, .06, .89, .03),
        (.80, .31, 12.0, .07, .88, .04), (.79, .13, 24.0, .08, .87, .05),
        (.78, .14, 14.0, .41, .86, .06), (.77, .15, 15.0, .10, .52, .07),
        (.76, .16, 16.0, .11, .84, .46), (.75, .17, 17.0, .12, .83, .09),
    )
    candidates = []
    for index, (variant, values) in enumerate(zip(DDW_VARIANTS, observed, strict=True)):
        reference = (
            values[0] + (0.08 if index == 0 else 0.01),
            values[1] + (0.09 if index == 1 else -0.01),
            values[2] - (9.0 if index == 2 else 1.0),
            values[3] + (0.12 if index == 3 else 0.01),
            values[4] - (0.14 if index == 4 else -0.01),
            values[5] + (0.18 if index == 5 else -0.01),
        )
        candidates.append(
            DdwMaskCandidate(_key(99), variant, DdwMaskDescriptors(*values), DdwMaskDescriptors(*reference))
        )
    return tuple(candidates)


def test_calibrations_keep_separate_axes_and_frozen_margin_math() -> None:
    v1_candidates = _v1_candidates()
    v1 = calibrate_ddw_mask_gate(v1_candidates).spec
    assert v1.minimum_coverage_q05 == pytest.approx(0.56)
    assert v1.maximum_hole_component_count_p95 == pytest.approx(31.0)
    assert v1.reference_atol.hole_component_count_p95 == pytest.approx(15.75)
    low_v1 = tuple(
        replace(item, observed=replace(item.observed, coverage_q05=.49))
        for item in v1_candidates
    )
    with pytest.raises(WaymoContractError, match="coverage is too low"):
        calibrate_ddw_mask_gate(low_v1)
    extreme = replace(v1_candidates[0], observed=replace(
        v1_candidates[0].observed, largest_hole_fraction_p95=.99,
        boundary_fraction_p95=.99, temporal_iou_q05=.01,
        temporal_flicker_p95=.99,
    ))
    clamped = calibrate_ddw_mask_gate((extreme, *v1_candidates[1:])).spec
    assert (
        clamped.maximum_largest_hole_fraction_p95,
        clamped.maximum_boundary_fraction_p95,
        clamped.minimum_temporal_iou_q05,
        clamped.maximum_temporal_flicker_p95,
    ) == (1.0, 1.0, 0.0, 1.0)

    reference = DdwMaskDescriptors(.75, .1, 2.0, .1, .85, .1)
    v2_candidates = tuple(
        DdwMaskCandidate(
            _key(98), variant,
            DdwMaskDescriptors(.8, .1 + .01 * index, 2.0 + index, .1 + .01 * index, .85 - .01 * index, .1 + .01 * index),
            reference,
        )
        for index, variant in enumerate(DDW_V2_VARIANTS)
    )
    v2 = calibrate_ddw_v2_mask_gate(v2_candidates).spec
    assert v2.minimum_coverage_q05 == 0.75
    assert v2.maximum_hole_component_count_p95 == 9.75
    low_v2 = tuple(
        replace(item, observed=replace(item.observed, coverage_q05=.49))
        for item in v2_candidates
    )
    with pytest.raises(WaymoContractError, match="coverage is too low"):
        calibrate_ddw_v2_mask_gate(low_v2)


def test_selection_requires_both_signs_on_all_eight_clips() -> None:
    descriptor = DdwMaskDescriptors(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    spec = calibrate_ddw_mask_gate(
        tuple(DdwMaskCandidate(_key(99), variant, descriptor, descriptor) for variant in DDW_VARIANTS)
    ).spec
    keys = tuple(_key(index) for index in range(8))
    candidates = tuple(
        DdwMaskCandidate(key, variant, descriptor, descriptor)
        for key in keys for variant in DDW_VARIANTS
    )
    assert select_ddw_variants(spec, keys, candidates).accepted_variants == DDW_VARIANTS
    failed = replace(descriptor, coverage_q05=0.0)
    changed = tuple(
        replace(item, observed=failed)
        if item.clip_key == keys[3] and item.variant == DdwVariant(2, 1)
        else item
        for item in candidates
    )
    accepted = select_ddw_variants(spec, keys, changed).accepted_variants
    assert DdwVariant(2, -1) not in accepted and DdwVariant(2, 1) not in accepted
    no_pairs = tuple(
        replace(item, observed=failed) if item.variant.sign == -1 else item
        for item in candidates
    )
    with pytest.raises(WaymoContractError, match="no DDW magnitude"):
        select_ddw_variants(spec, keys, no_pairs)

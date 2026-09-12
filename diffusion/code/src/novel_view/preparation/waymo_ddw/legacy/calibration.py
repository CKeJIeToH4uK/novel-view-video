"""Historical v1 calibration of the Waymo DDW mask gate from one canary."""

from __future__ import annotations

from dataclasses import dataclass

from novel_view.preparation.waymo_ddw.legacy.axis import DdwMaskCandidate
from novel_view.preparation.waymo_ddw.legacy.gate import (
    DDW_MASK_GATE_ID,
    DDW_MASK_METRIC_NAMES,
    DdwMaskGateSpec,
    DdwMaskReferenceTolerance,
)
from novel_view.preparation.waymo_ddw.legacy.selection import DDW_VARIANTS
from novel_view.inputs.waymo.types import WaymoContractError


DDW_MASK_CALIBRATION_ID = "waymo-ddw-mask-calibration-v1"
DDW_CALIBRATION_MINIMUM_COVERAGE_Q05 = 0.5
DDW_CALIBRATION_FRACTION_MARGIN = 0.05
DDW_CALIBRATION_COMPONENT_MARGIN_RATIO = 0.25
DDW_CALIBRATION_COMPONENT_MARGIN_COUNT = 1.0


@dataclass(frozen=True, slots=True)
class DdwMaskCalibration:
    """The exact engineering-canary axis and the derived immutable gate."""

    calibration_id: str
    candidates: tuple[DdwMaskCandidate, ...]
    spec: DdwMaskGateSpec


def calibrate_ddw_mask_gate(
    candidates: tuple[DdwMaskCandidate, ...],
) -> DdwMaskCalibration:
    """Apply the frozen margin rule to one selected clip and the v1 axis."""
    _require_canary_axis(candidates)
    return DdwMaskCalibration(
        DDW_MASK_CALIBRATION_ID,
        candidates,
        _derive_spec(candidates),
    )


def _derive_spec(
    candidates: tuple[DdwMaskCandidate, ...],
) -> DdwMaskGateSpec:
    observed = tuple(candidate.observed for candidate in candidates)
    minimum_coverage = min(value.coverage_q05 for value in observed)
    if minimum_coverage < DDW_CALIBRATION_MINIMUM_COVERAGE_Q05:
        raise WaymoContractError(
            "engineering canary coverage is too low to calibrate DDW"
        )
    fraction = DDW_CALIBRATION_FRACTION_MARGIN
    component_maximum = max(
        value.hole_component_count_p95 for value in observed
    )
    reference_maximum = max(
        candidate.reference.hole_component_count_p95
        for candidate in candidates
    )
    absolute_differences = {
        name: max(
            abs(getattr(candidate.observed, name) - getattr(candidate.reference, name))
            for candidate in candidates
        )
        for name in DDW_MASK_METRIC_NAMES
    }
    reference_atol = DdwMaskReferenceTolerance(
        coverage_q05=absolute_differences["coverage_q05"] + fraction,
        largest_hole_fraction_p95=(
            absolute_differences["largest_hole_fraction_p95"] + fraction
        ),
        hole_component_count_p95=(
            absolute_differences["hole_component_count_p95"]
            + DDW_CALIBRATION_COMPONENT_MARGIN_COUNT
            + DDW_CALIBRATION_COMPONENT_MARGIN_RATIO * reference_maximum
        ),
        boundary_fraction_p95=(
            absolute_differences["boundary_fraction_p95"] + fraction
        ),
        temporal_iou_q05=absolute_differences["temporal_iou_q05"] + fraction,
        temporal_flicker_p95=(
            absolute_differences["temporal_flicker_p95"] + fraction
        ),
    )
    zero = DdwMaskReferenceTolerance(*(0.0 for _ in DDW_MASK_METRIC_NAMES))
    return DdwMaskGateSpec(
        gate_id=DDW_MASK_GATE_ID,
        minimum_coverage_q05=max(0.0, minimum_coverage - fraction),
        maximum_largest_hole_fraction_p95=min(
            1.0,
            max(value.largest_hole_fraction_p95 for value in observed) + fraction,
        ),
        maximum_hole_component_count_p95=(
            component_maximum
            * (1.0 + DDW_CALIBRATION_COMPONENT_MARGIN_RATIO)
            + DDW_CALIBRATION_COMPONENT_MARGIN_COUNT
        ),
        maximum_boundary_fraction_p95=min(
            1.0,
            max(value.boundary_fraction_p95 for value in observed) + fraction,
        ),
        minimum_temporal_iou_q05=max(
            0.0,
            min(value.temporal_iou_q05 for value in observed) - fraction,
        ),
        maximum_temporal_flicker_p95=min(
            1.0,
            max(value.temporal_flicker_p95 for value in observed) + fraction,
        ),
        reference_atol=reference_atol,
        reference_rtol=zero,
    )


def _require_canary_axis(candidates: tuple[DdwMaskCandidate, ...]) -> None:
    first_key = candidates[0].clip_key
    if tuple((candidate.clip_key, candidate.variant) for candidate in candidates) != tuple(
        (first_key, variant) for variant in DDW_VARIANTS
    ):
        raise WaymoContractError(
            "canary candidates must contain one clip x ordered DDW variants"
        )


__all__ = [
    "DDW_CALIBRATION_COMPONENT_MARGIN_COUNT",
    "DDW_CALIBRATION_COMPONENT_MARGIN_RATIO",
    "DDW_CALIBRATION_FRACTION_MARGIN",
    "DDW_CALIBRATION_MINIMUM_COVERAGE_Q05",
    "DDW_MASK_CALIBRATION_ID",
    "DdwMaskCalibration",
    "calibrate_ddw_mask_gate",
]

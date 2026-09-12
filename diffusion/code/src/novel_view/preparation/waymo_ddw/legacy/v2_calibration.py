"""Pure preregistered DDW-v2 mask calibration from one six-axis canary."""

from __future__ import annotations

from dataclasses import dataclass

from novel_view.preparation.waymo_ddw.legacy._mask_score import DDW_MASK_METRIC_NAMES
from novel_view.preparation.waymo_ddw.legacy.axis import DdwMaskCandidate
from novel_view.preparation.waymo_ddw.legacy.v2 import DDW_V2_VARIANTS
from novel_view.preparation.waymo_ddw.legacy.v2_gate import (
    DDW_V2_MASK_GATE_ID,
    DdwV2MaskGateSpec,
    DdwV2MaskReferenceTolerance,
)
from novel_view.inputs.waymo.types import WaymoContractError


DDW_V2_MASK_CALIBRATION_ID = "waymo-ddw-mask-calibration-v2"
DDW_V2_MINIMUM_COVERAGE_Q05 = 0.5
DDW_V2_FRACTION_MARGIN = 0.05
DDW_V2_COMPONENT_MARGIN_RATIO = 0.25
DDW_V2_COMPONENT_MARGIN_COUNT = 1.0


@dataclass(frozen=True, slots=True)
class DdwV2MaskCalibration:
    calibration_id: str
    candidates: tuple[DdwMaskCandidate, ...]
    spec: DdwV2MaskGateSpec


def calibrate_ddw_v2_mask_gate(
    candidates: tuple[DdwMaskCandidate, ...],
) -> DdwV2MaskCalibration:
    _require_axis(candidates)
    return DdwV2MaskCalibration(
        DDW_V2_MASK_CALIBRATION_ID,
        candidates,
        _derive_spec(candidates),
    )


def _derive_spec(candidates: tuple[DdwMaskCandidate, ...]) -> DdwV2MaskGateSpec:
    observed = tuple(candidate.observed for candidate in candidates)
    minimum_coverage = min(item.coverage_q05 for item in observed)
    if minimum_coverage < DDW_V2_MINIMUM_COVERAGE_Q05:
        raise WaymoContractError("DDW-v2 canary coverage is too low")
    margin = DDW_V2_FRACTION_MARGIN
    reference_maximum = max(
        candidate.reference.hole_component_count_p95 for candidate in candidates
    )
    differences = {
        name: max(
            abs(getattr(item.observed, name) - getattr(item.reference, name))
            for item in candidates
        )
        for name in DDW_MASK_METRIC_NAMES
    }
    atol = DdwV2MaskReferenceTolerance(
        differences["coverage_q05"] + margin,
        differences["largest_hole_fraction_p95"] + margin,
        differences["hole_component_count_p95"]
        + DDW_V2_COMPONENT_MARGIN_COUNT
        + DDW_V2_COMPONENT_MARGIN_RATIO * reference_maximum,
        differences["boundary_fraction_p95"] + margin,
        differences["temporal_iou_q05"] + margin,
        differences["temporal_flicker_p95"] + margin,
    )
    zero = DdwV2MaskReferenceTolerance(*(0.0 for _ in DDW_MASK_METRIC_NAMES))
    return DdwV2MaskGateSpec(
        DDW_V2_MASK_GATE_ID,
        max(0.0, minimum_coverage - margin),
        min(1.0, max(item.largest_hole_fraction_p95 for item in observed) + margin),
        max(item.hole_component_count_p95 for item in observed)
        * (1.0 + DDW_V2_COMPONENT_MARGIN_RATIO)
        + DDW_V2_COMPONENT_MARGIN_COUNT,
        min(1.0, max(item.boundary_fraction_p95 for item in observed) + margin),
        max(0.0, min(item.temporal_iou_q05 for item in observed) - margin),
        min(1.0, max(item.temporal_flicker_p95 for item in observed) + margin),
        atol,
        zero,
    )


def _require_axis(candidates: tuple[DdwMaskCandidate, ...]) -> None:
    key = candidates[0].clip_key
    if tuple((item.clip_key, item.variant) for item in candidates) != tuple(
        (key, variant) for variant in DDW_V2_VARIANTS
    ):
        raise WaymoContractError("candidates must contain one exact DDW-v2 axis")


__all__ = [
    "DDW_V2_MASK_CALIBRATION_ID",
    "DdwV2MaskCalibration",
    "calibrate_ddw_v2_mask_gate",
]

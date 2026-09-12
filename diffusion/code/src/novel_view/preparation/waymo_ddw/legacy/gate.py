"""Versioned pure-data gate for one Waymo DDW mask report."""

from __future__ import annotations

from dataclasses import dataclass

from novel_view.preparation.waymo_ddw.legacy._mask_score import (
    DDW_MASK_METRIC_NAMES,
    score_ddw_mask_failures,
)
from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors


DDW_MASK_GATE_ID = "waymo-ddw-mask-gate-v1"


@dataclass(frozen=True, slots=True)
class DdwMaskReferenceTolerance:
    coverage_q05: float
    largest_hole_fraction_p95: float
    hole_component_count_p95: float
    boundary_fraction_p95: float
    temporal_iou_q05: float
    temporal_flicker_p95: float


@dataclass(frozen=True, slots=True)
class DdwMaskGateSpec:
    gate_id: str
    minimum_coverage_q05: float
    maximum_largest_hole_fraction_p95: float
    maximum_hole_component_count_p95: float
    maximum_boundary_fraction_p95: float
    minimum_temporal_iou_q05: float
    maximum_temporal_flicker_p95: float
    reference_atol: DdwMaskReferenceTolerance
    reference_rtol: DdwMaskReferenceTolerance


@dataclass(frozen=True, slots=True)
class DdwMaskGateScore:
    observed: DdwMaskDescriptors
    reference: DdwMaskDescriptors
    passed: bool
    failures: tuple[str, ...]


def score_ddw_masks(
    spec: DdwMaskGateSpec,
    observed: DdwMaskDescriptors,
    reference: DdwMaskDescriptors,
) -> DdwMaskGateScore:
    """Apply inclusive hard bounds and exact reference tolerances."""
    failures = score_ddw_mask_failures(
        observed,
        reference,
        (
            ("coverage_q05", spec.minimum_coverage_q05),
            ("temporal_iou_q05", spec.minimum_temporal_iou_q05),
        ),
        (
            (
                "largest_hole_fraction_p95",
                spec.maximum_largest_hole_fraction_p95,
            ),
            (
                "hole_component_count_p95",
                spec.maximum_hole_component_count_p95,
            ),
            ("boundary_fraction_p95", spec.maximum_boundary_fraction_p95),
            ("temporal_flicker_p95", spec.maximum_temporal_flicker_p95),
        ),
        tuple(getattr(spec.reference_atol, name) for name in DDW_MASK_METRIC_NAMES),
        tuple(getattr(spec.reference_rtol, name) for name in DDW_MASK_METRIC_NAMES),
    )
    return DdwMaskGateScore(
        observed,
        reference,
        not failures,
        failures,
    )


__all__ = [
    "DDW_MASK_GATE_ID",
    "DDW_MASK_METRIC_NAMES",
    "DdwMaskGateScore",
    "DdwMaskGateSpec",
    "DdwMaskReferenceTolerance",
    "score_ddw_masks",
]

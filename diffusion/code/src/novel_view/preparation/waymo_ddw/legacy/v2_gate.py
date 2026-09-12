"""Separate pure-data mask gate contract for Waymo DDW-v2."""

from __future__ import annotations

from dataclasses import dataclass

from novel_view.preparation.waymo_ddw.legacy._mask_score import (
    DDW_MASK_METRIC_NAMES,
    score_ddw_mask_failures,
)
from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors


DDW_V2_MASK_GATE_ID = "waymo-ddw-mask-gate-v2"


@dataclass(frozen=True, slots=True)
class DdwV2MaskReferenceTolerance:
    coverage_q05: float
    largest_hole_fraction_p95: float
    hole_component_count_p95: float
    boundary_fraction_p95: float
    temporal_iou_q05: float
    temporal_flicker_p95: float


@dataclass(frozen=True, slots=True)
class DdwV2MaskGateSpec:
    gate_id: str
    minimum_coverage_q05: float
    maximum_largest_hole_fraction_p95: float
    maximum_hole_component_count_p95: float
    maximum_boundary_fraction_p95: float
    minimum_temporal_iou_q05: float
    maximum_temporal_flicker_p95: float
    reference_atol: DdwV2MaskReferenceTolerance
    reference_rtol: DdwV2MaskReferenceTolerance


@dataclass(frozen=True, slots=True)
class DdwV2MaskGateScore:
    observed: DdwMaskDescriptors
    reference: DdwMaskDescriptors
    passed: bool
    failures: tuple[str, ...]


def score_ddw_v2_masks(
    spec: DdwV2MaskGateSpec,
    observed: DdwMaskDescriptors,
    reference: DdwMaskDescriptors,
) -> DdwV2MaskGateScore:
    """Apply v2 typed bounds through the shared six-metric arithmetic."""
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
    return DdwV2MaskGateScore(observed, reference, not failures, failures)


__all__ = [
    "DDW_V2_MASK_GATE_ID",
    "DdwV2MaskGateScore",
    "DdwV2MaskGateSpec",
    "DdwV2MaskReferenceTolerance",
    "score_ddw_v2_masks",
]

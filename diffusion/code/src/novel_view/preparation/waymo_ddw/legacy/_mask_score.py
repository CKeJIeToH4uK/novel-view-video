"""Private numeric scoring shared by versioned Waymo DDW mask gates."""

from __future__ import annotations

from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors


DDW_MASK_METRIC_NAMES = (
    "coverage_q05",
    "largest_hole_fraction_p95",
    "hole_component_count_p95",
    "boundary_fraction_p95",
    "temporal_iou_q05",
    "temporal_flicker_p95",
)


def score_ddw_mask_failures(
    observed: DdwMaskDescriptors,
    reference: DdwMaskDescriptors,
    minimums: tuple[tuple[str, float], ...],
    maximums: tuple[tuple[str, float], ...],
    reference_atol: tuple[float, ...],
    reference_rtol: tuple[float, ...],
) -> tuple[str, ...]:
    """Return ordered failures for already validated versioned inputs."""
    failures: list[str] = []
    for metric, minimum in minimums:
        if getattr(observed, metric) < minimum:
            failures.append(metric)
    for metric, maximum in maximums:
        if getattr(observed, metric) > maximum:
            failures.append(metric)
    for index, metric in enumerate(DDW_MASK_METRIC_NAMES):
        value = getattr(observed, metric)
        reference_value = getattr(reference, metric)
        tolerance = reference_atol[index] + reference_rtol[index] * abs(
            reference_value
        )
        if abs(value - reference_value) > tolerance:
            failures.append(f"{metric}.reference")
    return tuple(failures)

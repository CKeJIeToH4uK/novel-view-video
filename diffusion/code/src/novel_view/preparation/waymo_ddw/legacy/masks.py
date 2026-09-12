"""Pure mask descriptors for the frozen Waymo DDW validity gate."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from novel_view.inputs.waymo.types import WaymoContractError


_FRAME_COUNT = 121
_MIN_COMPONENT_AREA = 16


@dataclass(frozen=True, slots=True)
class DdwMaskDescriptors:
    coverage_q05: float
    largest_hole_fraction_p95: float
    hole_component_count_p95: float
    boundary_fraction_p95: float
    temporal_iou_q05: float
    temporal_flicker_p95: float


def measure_ddw_masks(
    known: npt.NDArray[np.bool_],
    rectification_known: npt.NDArray[np.bool_],
) -> DdwMaskDescriptors:
    """Reduce one exact clip without allocating a full-clip temporary."""
    _mask_cube(known, "known")
    _mask_cube(rectification_known, "rectification_known")
    if rectification_known.shape != known.shape:
        raise WaymoContractError("rectification_known must match known")

    coverage: list[float] = []
    largest_hole: list[float] = []
    component_count: list[float] = []
    boundary: list[float] = []
    for frame_index in range(_FRAME_COUNT):
        frame_known = known[frame_index]
        frame_rect = rectification_known[frame_index]
        rect_count = int(np.count_nonzero(frame_rect))
        if rect_count == 0:
            raise WaymoContractError(
                f"rectification_known[{frame_index}] must be non-empty"
            )
        coverage.append(
            float(np.count_nonzero(frame_known & frame_rect) / rect_count)
        )
        hole = (~frame_known) & frame_rect
        areas = _component_areas(hole)
        largest_hole.append(
            0.0 if areas.size == 0 else float(areas.max() / rect_count)
        )
        component_count.append(
            float(np.count_nonzero(areas >= _MIN_COMPONENT_AREA))
        )
        boundary.append(_boundary_fraction(frame_known, frame_rect))

    temporal_iou: list[float] = []
    temporal_flicker: list[float] = []
    for frame_index in range(_FRAME_COUNT - 1):
        support = (
            rectification_known[frame_index]
            & rectification_known[frame_index + 1]
        )
        support_count = int(np.count_nonzero(support))
        if support_count == 0:
            raise WaymoContractError(
                "adjacent rectification masks must overlap"
            )
        left = known[frame_index] & support
        right = known[frame_index + 1] & support
        union_count = int(np.count_nonzero(left | right))
        temporal_iou.append(
            1.0
            if union_count == 0
            else float(np.count_nonzero(left & right) / union_count)
        )
        temporal_flicker.append(
            float(np.count_nonzero(left ^ right) / support_count)
        )

    return DdwMaskDescriptors(
        coverage_q05=_quantile(coverage, 0.05),
        largest_hole_fraction_p95=_quantile(largest_hole, 0.95),
        hole_component_count_p95=_quantile(component_count, 0.95),
        boundary_fraction_p95=_quantile(boundary, 0.95),
        temporal_iou_q05=_quantile(temporal_iou, 0.05),
        temporal_flicker_p95=_quantile(temporal_flicker, 0.95),
    )


def _component_areas(hole: npt.NDArray[np.bool_]) -> npt.NDArray[np.int32]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        hole.astype(np.uint8),
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    if count == 1:
        return np.empty(0, dtype=np.int32)
    return np.asarray(stats[1:, cv2.CC_STAT_AREA], dtype=np.int32)


def _boundary_fraction(
    known: npt.NDArray[np.bool_],
    rectification_known: npt.NDArray[np.bool_],
) -> float:
    horizontal = rectification_known[:, :-1] & rectification_known[:, 1:]
    vertical = rectification_known[:-1] & rectification_known[1:]
    pair_count = int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))
    if pair_count == 0:
        return 0.0
    changed = np.count_nonzero(
        horizontal & (known[:, :-1] != known[:, 1:])
    ) + np.count_nonzero(vertical & (known[:-1] != known[1:]))
    return float(changed / pair_count)


def _mask_cube(value: object, name: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.bool_)
        or value.ndim != 3
        or value.shape[0] != _FRAME_COUNT
        or value.shape[1] <= 0
        or value.shape[2] <= 0
    ):
        raise WaymoContractError(f"{name} must be bool [121,H,W]")


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


__all__ = ["DdwMaskDescriptors", "measure_ddw_masks"]

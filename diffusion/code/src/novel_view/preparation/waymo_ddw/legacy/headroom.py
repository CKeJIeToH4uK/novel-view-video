"""Pure evidence that a DDW condition retains visible repair headroom."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from novel_view.inputs.waymo.types import WaymoContractError


DDW_HEADROOM_GATE_ID = "waymo-ddw-headroom-gate-v1"
DDW_CHANGED_PIXEL_THRESHOLD = 4.0 / 255.0
DDW_MINIMUM_MEAN_ABSOLUTE_CORRUPTION = 2.0 / 255.0
DDW_MINIMUM_CHANGED_PIXEL_FRACTION = 0.01
DDW_NORMALIZED_FLOAT32_ATOL = 2.0 * float(np.finfo(np.float32).eps)
_FIRST_RESTORATION_FRAME = 1


@dataclass(frozen=True, slots=True)
class DdwHeadroomMetrics:
    """Two RGB corruption measures over the fixed known restoration region."""

    restoration_pixel_count: int
    mean_absolute_corruption: float
    changed_pixel_fraction: float


@dataclass(frozen=True, slots=True)
class DdwHeadroomScore:
    """Frozen pre-canary STOP decision without variant ranking."""

    gate_id: str
    metrics: DdwHeadroomMetrics
    passed: bool
    failures: tuple[str, ...]


def measure_ddw_headroom_arrays(
    target_rgb: npt.NDArray[np.float32],
    target_rectification_known: npt.NDArray[np.bool_],
    condition_rgb: npt.NDArray[np.float32],
    condition_known: npt.NDArray[np.bool_],
) -> DdwHeadroomMetrics:
    """Reduce already validated A′/A arrays with frame-bounded temporaries."""
    absolute_sum = 0.0
    changed_count = 0
    restoration_count = 0
    for index in range(_FIRST_RESTORATION_FRAME, 121):
        frame_sum, frame_changed, frame_count = _measure_headroom_frame(
            target_rgb[index],
            target_rectification_known,
            condition_rgb[index, 0],
            condition_known[index, 0],
        )
        absolute_sum += frame_sum
        changed_count += frame_changed
        restoration_count += frame_count
    return _headroom_metrics(absolute_sum, changed_count, restoration_count)


def measure_ddw_headroom_uint8_target(
    target_rgb_thwc: npt.NDArray[np.uint8],
    target_rectification_known: npt.NDArray[np.bool_],
    condition_rgb: npt.NDArray[np.float32],
    condition_known: npt.NDArray[np.bool_],
) -> DdwHeadroomMetrics:
    """Measure a uint8 THWC target against normalized TCHW condition frames."""
    absolute_sum = 0.0
    changed_count = 0
    restoration_count = 0
    for index in range(_FIRST_RESTORATION_FRAME, 121):
        target_frame = np.asarray(target_rgb_thwc[index], dtype=np.float32)
        target_frame = target_frame * (2.0 / 255.0) - 1.0
        target_frame = np.moveaxis(target_frame, -1, 0)
        frame_sum, frame_changed, frame_count = _measure_headroom_frame(
            target_frame,
            target_rectification_known,
            condition_rgb[index],
            condition_known[index],
        )
        absolute_sum += frame_sum
        changed_count += frame_changed
        restoration_count += frame_count
    return _headroom_metrics(absolute_sum, changed_count, restoration_count)


def _measure_headroom_frame(
    target_rgb: npt.NDArray[np.float32],
    target_rectification_known: npt.NDArray[np.bool_],
    condition_rgb: npt.NDArray[np.float32],
    condition_known: npt.NDArray[np.bool_],
) -> tuple[float, int, int]:
    support = condition_known & target_rectification_known
    support_count = int(np.count_nonzero(support))
    if support_count == 0:
        return 0.0, 0, 0
    difference = np.abs(condition_rgb - target_rgb)
    absolute_sum = float(np.sum(difference[:, support], dtype=np.float64))
    changed_count = int(
        np.count_nonzero(
            np.max(difference, axis=0)[support]
            + DDW_NORMALIZED_FLOAT32_ATOL
            >= DDW_CHANGED_PIXEL_THRESHOLD
        )
    )
    return absolute_sum, changed_count, support_count


def _headroom_metrics(
    absolute_sum: float,
    changed_count: int,
    restoration_count: int,
) -> DdwHeadroomMetrics:
    if restoration_count == 0:
        raise WaymoContractError("DDW restoration region is empty")
    return DdwHeadroomMetrics(
        restoration_count,
        float(absolute_sum / (3 * restoration_count)),
        float(changed_count / restoration_count),
    )


def score_ddw_headroom(metrics: DdwHeadroomMetrics) -> DdwHeadroomScore:
    """Apply the frozen minimum visible-magnitude and spatial-mass rule."""
    failures = _failures(metrics)
    return DdwHeadroomScore(
        DDW_HEADROOM_GATE_ID,
        metrics,
        not failures,
        failures,
    )


def _failures(metrics: DdwHeadroomMetrics) -> tuple[str, ...]:
    failures: list[str] = []
    if (
        metrics.mean_absolute_corruption + DDW_NORMALIZED_FLOAT32_ATOL
        < DDW_MINIMUM_MEAN_ABSOLUTE_CORRUPTION
    ):
        failures.append("mean_absolute_corruption")
    if metrics.changed_pixel_fraction < DDW_MINIMUM_CHANGED_PIXEL_FRACTION:
        failures.append("changed_pixel_fraction")
    return tuple(failures)


__all__ = [
    "DDW_CHANGED_PIXEL_THRESHOLD",
    "DDW_HEADROOM_GATE_ID",
    "DDW_MINIMUM_CHANGED_PIXEL_FRACTION",
    "DDW_MINIMUM_MEAN_ABSOLUTE_CORRUPTION",
    "DDW_NORMALIZED_FLOAT32_ATOL",
    "DdwHeadroomMetrics",
    "DdwHeadroomScore",
    "measure_ddw_headroom_arrays",
    "measure_ddw_headroom_uint8_target",
    "score_ddw_headroom",
]

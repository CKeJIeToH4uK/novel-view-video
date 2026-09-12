"""Compute the numeric parts of the audited-static/v1 image recipe."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import cv2
import numpy as np
import numpy.typing as npt

_SSIM_WINDOW_SIZE = 11
_SSIM_SIGMA = 1.5
_SSIM_RADIUS = _SSIM_WINDOW_SIZE // 2
IMAGE_METRIC_PROTOCOL = "audited-static/v1"


class MetricError(RuntimeError):
    """One metric value or scientific support is inconsistent."""


class MetricStatus(str, Enum):
    """Explicit state of one scalar measurement."""

    FINITE = "finite"
    POSITIVE_INFINITY = "positive_infinity"
    UNDEFINED = "undefined"


class UndefinedReason(str, Enum):
    """Scientific reason why a metric has no value."""

    EMPTY_SUPPORT = "empty_support"
    NO_FULL_WINDOWS = "no_full_windows"


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One scalar without hidden NaN or infinity in its numeric field."""

    status: MetricStatus
    value: float | None
    undefined_reason: UndefinedReason | None = None

    def __post_init__(self) -> None:
        if self.status is MetricStatus.FINITE:
            valid = (
                self.value is not None
                and math.isfinite(self.value)
                and self.undefined_reason is None
            )
        elif self.status is MetricStatus.POSITIVE_INFINITY:
            valid = self.value is None and self.undefined_reason is None
        elif self.status is MetricStatus.UNDEFINED:
            valid = (
                self.value is None
                and isinstance(self.undefined_reason, UndefinedReason)
            )
        else:
            valid = False
        if not valid:
            raise MetricError("metric state, value and reason disagree")

    @classmethod
    def finite(cls, value: float) -> MetricValue:
        return cls(MetricStatus.FINITE, float(value))

    @classmethod
    def positive_infinity(cls) -> MetricValue:
        return cls(MetricStatus.POSITIVE_INFINITY, None)

    @classmethod
    def undefined(cls, reason: UndefinedReason) -> MetricValue:
        return cls(MetricStatus.UNDEFINED, None, reason)


def masked_psnr(
    prediction: npt.NDArray[np.uint8],
    target: npt.NDArray[np.uint8],
    support: npt.NDArray[np.bool_],
) -> MetricValue:
    """Return exact RGB PSNR over selected uint8 pixels."""
    pixel_count = int(np.count_nonzero(support))
    if pixel_count == 0:
        return MetricValue.undefined(UndefinedReason.EMPTY_SUPPORT)
    difference = prediction[support].astype(np.int32) - target[support]
    squared_error = int(np.sum(np.square(difference, dtype=np.int64)))
    if squared_error == 0:
        return MetricValue.positive_infinity()
    return MetricValue.finite(
        _psnr_from_mse(squared_error / (pixel_count * 3))
    )


def reduce_rgb_mse_psnr(
    prediction: npt.NDArray[np.uint8],
    target: npt.NDArray[np.uint8],
) -> tuple[float, float | None]:
    """Reduce equal uint8 RGB arrays; ``None`` denotes infinite PSNR."""
    if (
        prediction.dtype != np.uint8
        or target.dtype != np.uint8
        or prediction.shape != target.shape
        or prediction.ndim < 3
        or prediction.shape[-1] != 3
        or prediction.size == 0
    ):
        raise MetricError(
            "RGB reduction requires equal non-empty uint8 [...,H,W,3] arrays"
        )
    difference = prediction.astype(np.int16) - target.astype(np.int16)
    mean_squared_error = float(
        np.mean(np.square(difference, dtype=np.int32))
    )
    return (
        mean_squared_error,
        None
        if mean_squared_error == 0.0
        else _psnr_from_mse(mean_squared_error),
    )


def masked_ssim(
    prediction: npt.NDArray[np.uint8],
    target: npt.NDArray[np.uint8],
    support: npt.NDArray[np.bool_],
) -> tuple[MetricValue, int]:
    """Average RGB SSIM only at fully supported 11×11 window centres."""
    return reduce_ssim_map(ssim_map(prediction, target), support)


def reduce_ssim_map(
    values: npt.NDArray[np.float64],
    support: npt.NDArray[np.bool_],
) -> tuple[MetricValue, int]:
    """Reduce one shared SSIM map over fully supported window centres."""
    height, width = support.shape
    if height < _SSIM_WINDOW_SIZE or width < _SSIM_WINDOW_SIZE:
        return MetricValue.undefined(UndefinedReason.NO_FULL_WINDOWS), 0
    valid_centres = cv2.erode(
        support.astype(np.uint8),
        np.ones((_SSIM_WINDOW_SIZE, _SSIM_WINDOW_SIZE), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.bool_)
    valid_centres[:_SSIM_RADIUS] = False
    valid_centres[-_SSIM_RADIUS:] = False
    valid_centres[:, :_SSIM_RADIUS] = False
    valid_centres[:, -_SSIM_RADIUS:] = False
    window_count = int(np.count_nonzero(valid_centres))
    if window_count == 0:
        return MetricValue.undefined(UndefinedReason.NO_FULL_WINDOWS), 0
    value = float(np.mean(values[valid_centres]))
    return MetricValue.finite(value), window_count


def masked_average(
    values: npt.NDArray[np.floating],
    support: npt.NDArray[np.bool_],
) -> tuple[float | None, float]:
    """Average a spatial metric over supported pixels."""
    weight = float(np.count_nonzero(support))
    if weight == 0.0:
        return None, 0.0
    return float(np.mean(values[support], dtype=np.float64)), weight


def weighted_average(
    values: npt.NDArray[np.floating],
    weights: npt.NDArray[np.float64],
) -> tuple[float | None, float]:
    """Average raw feature metrics with their exact effective mass."""
    weight = float(np.sum(weights, dtype=np.float64))
    if weight == 0.0:
        return None, 0.0
    value = float(
        np.sum(values * weights, dtype=np.float64) / weight
    )
    return value, weight


def fractional_patch_support(
    support: npt.NDArray[np.bool_],
    grid_size_hw: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Integrate binary pixel cells over uniform feature-patch preimages."""
    height, width = support.shape
    grid_height, grid_width = grid_size_hw
    prefix = np.pad(
        np.cumsum(
            np.cumsum(support.astype(np.float64), axis=0),
            axis=1,
        ),
        ((1, 0), (1, 0)),
    )
    y_edges = np.linspace(0.0, float(height), grid_height + 1)
    x_edges = np.linspace(0.0, float(width), grid_width + 1)
    cumulative = _continuous_prefix(prefix, support, y_edges, x_edges)
    area = (
        cumulative[1:, 1:]
        - cumulative[:-1, 1:]
        - cumulative[1:, :-1]
        + cumulative[:-1, :-1]
    )
    patch_area = (height / grid_height) * (width / grid_width)
    return np.clip(area / patch_area, 0.0, 1.0).astype(
        np.float64,
        copy=False,
    )


def _psnr_from_mse(mean_squared_error: float) -> float:
    return 10.0 * math.log10((255.0 * 255.0) / mean_squared_error)


def ssim_map(
    prediction: npt.NDArray[np.uint8],
    target: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float64]:
    prediction_f64 = prediction.astype(np.float64) / 255.0
    target_f64 = target.astype(np.float64) / 255.0
    kernel = cv2.getGaussianKernel(
        _SSIM_WINDOW_SIZE,
        _SSIM_SIGMA,
        cv2.CV_64F,
    )

    def blur(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return cv2.sepFilter2D(
            value,
            cv2.CV_64F,
            kernel,
            kernel,
            borderType=cv2.BORDER_REFLECT,
        )

    mean_prediction = blur(prediction_f64)
    mean_target = blur(target_f64)
    variance_prediction = blur(prediction_f64 * prediction_f64) - (
        mean_prediction * mean_prediction
    )
    variance_target = blur(target_f64 * target_f64) - (
        mean_target * mean_target
    )
    covariance = blur(prediction_f64 * target_f64) - (
        mean_prediction * mean_target
    )
    c1 = 0.01**2
    c2 = 0.03**2
    return (
        (2.0 * mean_prediction * mean_target + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_prediction * mean_prediction + mean_target * mean_target + c1)
            * (variance_prediction + variance_target + c2)
        )
    )


def _continuous_prefix(
    prefix: npt.NDArray[np.float64],
    support: npt.NDArray[np.bool_],
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    height, width = support.shape
    y_base = np.minimum(np.floor(y).astype(np.int64), height - 1)
    x_base = np.minimum(np.floor(x).astype(np.int64), width - 1)
    y_fraction = y - y_base
    x_fraction = x - x_base
    rows = y_base[:, None]
    columns = x_base[None, :]
    base = prefix[rows, columns]
    column_strip = prefix[rows, columns + 1] - base
    row_strip = prefix[rows + 1, columns] - base
    cell = support[rows, columns]
    return (
        base
        + x_fraction[None, :] * column_strip
        + y_fraction[:, None] * row_strip
        + y_fraction[:, None] * x_fraction[None, :] * cell
    )


__all__ = [
    "IMAGE_METRIC_PROTOCOL",
    "MetricError",
    "MetricStatus",
    "MetricValue",
    "UndefinedReason",
    "fractional_patch_support",
    "masked_average",
    "masked_psnr",
    "masked_ssim",
    "reduce_rgb_mse_psnr",
    "reduce_ssim_map",
    "ssim_map",
    "weighted_average",
]

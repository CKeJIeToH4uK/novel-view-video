"""Run and metric-align historical MoGe-v1 depth in measured Waymo cameras."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from novel_view.preparation.waymo_depth import depth as waymo_depth_contract
from novel_view.preparation.waymo_depth.depth import MetricDepthClip
from novel_view.preparation.waymo_depth.metrics import sample_bilinear_depth
from novel_view.preparation.waymo_depth.depth_samples import ProjectedLidarDepth
from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.inputs.waymo._arrays import readonly_owned, require_array
from novel_view.models.moge.backend import MogeResources, run_moge
from novel_view.models.moge.request import MogeRequest
from novel_view.models.moge.spec import MogeError


class MogeDepthUnavailable(WaymoContractError):
    """MoGe ran, but its output cannot form this scientific candidate."""


@dataclass(frozen=True, slots=True, eq=False)
class MogeMetricDepth:
    """One metric clip and its independently fitted positive frame scales."""

    clip: MetricDepthClip
    scale_by_frame: npt.NDArray[np.float64]

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.clip, MetricDepthClip):
            raise WaymoContractError("clip must be MetricDepthClip")
        scales = require_array(
            self.scale_by_frame,
            "scale_by_frame",
            dtype=np.dtype(np.float64),
            shape=(len(self.clip.clip_key.frame_timestamps_micros),),
        )
        if not np.isfinite(scales).all() or np.any(scales <= 0.0):
            raise WaymoContractError("scale_by_frame must be finite and positive")


def build_moge_metric_depth(
    evidence: ProjectedLidarDepth,
    depth_model_units: npt.NDArray[np.float32],
    model_valid: npt.NDArray[np.bool_],
    rectification_known: npt.NDArray[np.bool_],
    K_canvas: npt.NDArray[np.float64],
    world_to_camera_cv: npt.NDArray[np.float64],
) -> MogeMetricDepth:
    """Fit one positive scale per frame using only non-heldout LiDAR.

    The fit exactly minimizes the relative L1 objective used by MoGe's
    scale-invariant depth evaluation. Dense depth remains on the model's
    input raster, but the returned camera and metric gauge are the measured
    Waymo camera, never a camera predicted by MoGe.
    """
    if not isinstance(evidence, ProjectedLidarDepth):
        raise WaymoContractError("evidence must be ProjectedLidarDepth")
    frame_count = len(evidence.clip_key.frame_timestamps_micros)
    depth = _model_array(
        depth_model_units,
        "depth_model_units",
        np.dtype(np.float32),
        frame_count,
    )
    if depth.shape[1:] != waymo_depth_contract.GEN3C_CANVAS_SIZE_HW:
        raise WaymoContractError(
            "depth_model_units must use the canonical Gen3C canvas"
        )
    valid = _model_array(
        model_valid,
        "model_valid",
        np.dtype(np.bool_),
        frame_count,
        spatial_shape=depth.shape[1:],
    )
    known = _canvas_mask(rectification_known, depth.shape[1:])

    metric_depth = np.zeros(depth.shape, dtype=np.float32)
    metric_valid = np.empty(depth.shape, dtype=np.bool_)
    scales = np.empty(frame_count, dtype=np.float64)
    for frame_index in range(frame_count):
        frame_valid = valid[frame_index] & known
        frame_depth = depth[frame_index]
        if np.any(frame_valid & (~np.isfinite(frame_depth) | (frame_depth <= 0.0))):
            raise MogeDepthUnavailable(
                f"valid depth_model_units[{frame_index}] must be finite and positive"
            )
        evidence_slice = evidence.frame_slice(frame_index)
        sampled, sampled_valid = sample_bilinear_depth(
            frame_depth,
            frame_valid,
            evidence.canvas_xy[evidence_slice],
        )
        fit = sampled_valid & ~evidence.heldout[evidence_slice]
        if not np.any(fit):
            raise MogeDepthUnavailable(
                f"frame[{frame_index}] has no non-heldout valid MoGe samples"
            )
        scale = fit_positive_relative_l1_scale(
            sampled[fit],
            evidence.depth_z_m[evidence_slice][fit],
        )
        scaled = np.asarray(frame_depth * np.float32(scale), dtype=np.float32)
        if np.any(frame_valid & ~np.isfinite(scaled)):
            raise MogeDepthUnavailable(f"scaled depth[{frame_index}] overflowed")
        metric_valid[frame_index] = frame_valid
        metric_depth[frame_index, frame_valid] = scaled[frame_valid]
        scales[frame_index] = scale

    metric_depth.setflags(write=False)
    metric_valid.setflags(write=False)
    clip = MetricDepthClip(
        clip_key=evidence.clip_key,
        camera_name=evidence.camera_name,
        depth_z_m=metric_depth,
        valid=metric_valid,
        K_canvas=K_canvas,
        world_to_camera_cv=world_to_camera_cv,
    )
    return MogeMetricDepth(clip, readonly_owned(scales))


def fit_positive_relative_l1_scale(
    predicted: npt.NDArray[np.floating],
    target_m: npt.NDArray[np.floating],
) -> float:
    """Minimize ``sum(abs(scale * predicted - target) / target)``."""
    if (
        not isinstance(predicted, np.ndarray)
        or not isinstance(target_m, np.ndarray)
        or predicted.ndim != 1
        or target_m.shape != predicted.shape
        or predicted.size == 0
    ):
        raise WaymoContractError("scale inputs must be matching non-empty vectors")
    predicted64 = np.asarray(predicted, dtype=np.float64)
    target64 = np.asarray(target_m, dtype=np.float64)
    if (
        not np.isfinite(predicted64).all()
        or not np.isfinite(target64).all()
        or np.any(predicted64 <= 0.0)
        or np.any(target64 <= 0.0)
    ):
        raise WaymoContractError("scale inputs must be finite and positive")
    ratios = target64 / predicted64
    weights = predicted64 / target64
    order = np.argsort(ratios, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left"))
    return float(ratios[order[index]])


def _model_array(
    value: object,
    name: str,
    dtype: np.dtype[np.generic],
    frame_count: int,
    *,
    spatial_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.ndim != 3
        or value.shape[0] != frame_count
        or (spatial_shape is not None and value.shape[1:] != spatial_shape)
        or not value.flags.c_contiguous
    ):
        raise WaymoContractError(
            f"{name} must be C-contiguous {dtype} [frames,height,width]"
        )
    return value


def _canvas_mask(value: object, shape: tuple[int, int]) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.bool_)
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise WaymoContractError("rectification_known must match the model raster")
    return value


@dataclass(frozen=True, slots=True)
class MogeDepthResult:
    metric_depth: MogeMetricDepth
    elapsed_seconds: float
    worker_peak_ram_mib: float
    worker_peak_vram_mib: float

    def __post_init__(self) -> None:
        for name in (
            "elapsed_seconds",
            "worker_peak_ram_mib",
            "worker_peak_vram_mib",
        ):
            value = getattr(self, name)
            if not isinstance(value, float) or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def run_moge_metric_depth(
    rgb_frames: Iterable[npt.NDArray[np.uint8]],
    evidence: ProjectedLidarDepth,
    rectification_known: npt.NDArray[np.bool_],
    K_canvas: npt.NDArray[np.float64],
    world_to_camera_cv: npt.NDArray[np.float64],
    resources: MogeResources,
    log_path: Path,
) -> MogeDepthResult:
    """Run one clip and fit its measured-camera metric scale per frame."""
    frame_count = len(evidence.clip_key.frame_timestamps_micros)
    fov_x_degrees = np.degrees(
        2.0 * np.arctan(rectification_known.shape[1] / (2.0 * K_canvas[:, 0, 0]))
    )
    started = time.perf_counter()
    execution = run_moge(
        MogeRequest(
            rgb_frames=rgb_frames,
            fov_x_degrees=fov_x_degrees,
            frame_count=frame_count,
            image_size_hw=rectification_known.shape,
        ),
        resources,
        log_path,
    )
    _require_model_intrinsics(
        execution.prediction.intrinsics,
        fov_x_degrees,
        rectification_known.shape,
    )
    metric_depth = build_moge_metric_depth(
        evidence,
        execution.prediction.depth_model_units,
        execution.prediction.valid,
        rectification_known,
        K_canvas,
        world_to_camera_cv,
    )
    return MogeDepthResult(
        metric_depth=metric_depth,
        elapsed_seconds=float(time.perf_counter() - started),
        worker_peak_ram_mib=execution.telemetry.peak_ram_mib,
        worker_peak_vram_mib=execution.telemetry.peak_vram_mib,
    )


def _require_model_intrinsics(
    value: object,
    fov_x_degrees: npt.NDArray[np.float64],
    size_hw: tuple[int, int],
) -> None:
    """Require MoGe depth to preserve the measured pinhole FOV request."""
    expected_fx = 0.5 / np.tan(np.radians(fov_x_degrees) / 2.0)
    aspect = size_hw[1] / size_hw[0]
    expected = np.zeros((len(fov_x_degrees), 3, 3), dtype=np.float32)
    expected[:, 0, 0] = expected_fx
    expected[:, 1, 1] = expected_fx * aspect
    expected[:, 0, 2] = expected[:, 1, 2] = 0.5
    expected[:, 2, 2] = 1.0
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.float32)
        or value.shape != expected.shape
        or not np.allclose(value, expected, rtol=1e-5, atol=1e-6)
    ):
        raise MogeDepthUnavailable("MoGe intrinsics do not preserve measured fov_x")


__all__ = [
    "MogeDepthResult",
    "MogeDepthUnavailable",
    "MogeError",
    "MogeMetricDepth",
    "MogeResources",
    "build_moge_metric_depth",
    "fit_positive_relative_l1_scale",
    "run_moge_metric_depth",
]

"""MoGe-v1 depth aligned to measured Waymo FRONT camera-Z."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from novel_view.inputs.waymo.types import WaymoFrameKey
from novel_view.models.moge.backend import MogeResources, run_moge
from novel_view.models.moge.request import (
    MogeRawPrediction,
    MogeRequest,
    MogeTelemetry,
)

if TYPE_CHECKING:
    from novel_view.preparation.waymo_ddw.raster import WaymoFrontRaster
    from novel_view.preparation.waymo_ddw.selection import SelectedDdwSample


@dataclass(frozen=True, slots=True, eq=False)
class MetricDepth:
    """Dense metric camera-Z in the original measured FRONT camera pack."""

    selected: SelectedDdwSample
    frame_keys: tuple[WaymoFrameKey, ...]
    depth_z_m: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]
    scale_by_frame: npt.NDArray[np.float64]
    K_canvas: npt.NDArray[np.float64]
    world_to_camera_cv: npt.NDArray[np.float64]
    telemetry: MogeTelemetry


def predict_moge_metric_depth(
    front: WaymoFrontRaster,
    resources: MogeResources,
    log_path: Path,
) -> MetricDepth:
    """Run the concrete MoGe worker, then align each frame to measured LiDAR."""
    frame_count = len(front.frame_keys)
    height, width = front.rgb_thwc.shape[1:3]
    fov_x_degrees = np.degrees(2.0 * np.arctan(width / (2.0 * front.K_canvas[:, 0, 0])))
    execution = run_moge(
        MogeRequest(
            rgb_frames=iter(front.rgb_thwc),
            fov_x_degrees=fov_x_degrees,
            frame_count=frame_count,
            image_size_hw=(height, width),
        ),
        resources,
        log_path,
    )
    return align_moge_metric_depth(front, execution.prediction, execution.telemetry)


def align_moge_metric_depth(
    front: WaymoFrontRaster,
    prediction: MogeRawPrediction,
    telemetry: MogeTelemetry,
) -> MetricDepth:
    """Fit positive per-frame relative-L1 scale without evaluation points."""
    depth = prediction.depth_model_units
    valid = prediction.valid
    expected_shape = front.rgb_thwc.shape[:3]
    if depth.dtype != np.float32 or depth.shape != expected_shape:
        raise ValueError("MoGe depth has the wrong float32 frame shape")
    if valid.dtype != np.bool_ or valid.shape != expected_shape:
        raise ValueError("MoGe validity has the wrong boolean frame shape")

    metric = np.zeros(expected_shape, dtype=np.float32)
    metric_valid = np.empty(expected_shape, dtype=np.bool_)
    scales = np.empty(len(front.frame_keys), dtype=np.float64)
    for frame_index in range(len(front.frame_keys)):
        frame_depth = depth[frame_index]
        frame_valid = valid[frame_index] & front.rectification_known
        if np.any(frame_valid & (~np.isfinite(frame_depth) | (frame_depth <= 0.0))):
            raise ValueError(f"MoGe depth frame {frame_index} is not positive finite")
        lidar_slice = front.lidar.frame_slice(frame_index)
        sampled, sampled_valid = _sample_bilinear(
            frame_depth,
            frame_valid,
            front.lidar.canvas_xy[lidar_slice],
        )
        fit = sampled_valid & ~front.lidar.evaluation_holdout[lidar_slice]
        if not np.any(fit):
            raise ValueError(
                f"MoGe frame {frame_index} has no non-heldout LiDAR support"
            )
        scale = fit_positive_relative_l1_scale(
            sampled[fit],
            front.lidar.depth_z_m[lidar_slice][fit],
        )
        scaled = np.asarray(frame_depth * np.float32(scale), dtype=np.float32)
        if np.any(frame_valid & ~np.isfinite(scaled)):
            raise ValueError(f"MoGe metric depth frame {frame_index} overflowed")
        metric_valid[frame_index] = frame_valid
        metric[frame_index, frame_valid] = scaled[frame_valid]
        scales[frame_index] = scale

    return MetricDepth(
        selected=front.selected,
        frame_keys=front.frame_keys,
        depth_z_m=metric,
        valid=metric_valid,
        scale_by_frame=scales,
        K_canvas=front.K_canvas,
        world_to_camera_cv=front.world_to_camera_cv,
        telemetry=telemetry,
    )


def fit_positive_relative_l1_scale(
    predicted: npt.NDArray[np.float64],
    target_m: npt.NDArray[np.float32],
) -> float:
    """Return the exact weighted median minimizing relative L1 error."""
    predicted64 = np.asarray(predicted, dtype=np.float64)
    target64 = np.asarray(target_m, dtype=np.float64)
    ratios = target64 / predicted64
    weights = predicted64 / target64
    order = np.argsort(ratios, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left"))
    return float(ratios[order[index]])


def _sample_bilinear(
    depth: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_],
    canvas_xy: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    height, width = depth.shape
    x = canvas_xy[:, 0].astype(np.float64)
    y = canvas_xy[:, 1].astype(np.float64)
    inside = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)
    x0 = np.floor(np.clip(x, 0.0, width - 1)).astype(np.intp)
    y0 = np.floor(np.clip(y, 0.0, height - 1)).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    sampled = (
        depth[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + depth[y0, x1] * wx * (1.0 - wy)
        + depth[y1, x0] * (1.0 - wx) * wy
        + depth[y1, x1] * wx * wy
    )
    support = (
        inside
        & valid[y0, x0]
        & valid[y0, x1]
        & valid[y1, x0]
        & valid[y1, x1]
        & np.isfinite(sampled)
        & (sampled > 0.0)
    )
    return sampled, support


__all__ = [
    "MetricDepth",
    "align_moge_metric_depth",
    "fit_positive_relative_l1_scale",
    "predict_moge_metric_depth",
]

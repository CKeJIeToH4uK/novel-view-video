"""Held-out LiDAR metrics for one historical Waymo depth candidate."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from novel_view.preparation.waymo_depth.depth import MetricDepthClip
from novel_view.preparation.waymo_depth.selection import DepthClipMetrics
from novel_view.preparation.waymo_depth.depth_samples import ProjectedLidarDepth
from novel_view.inputs.waymo.types import WaymoContractError


def measure_metric_depth(
    clip: MetricDepthClip,
    evidence: ProjectedLidarDepth,
    scale_by_frame: npt.NDArray[np.float64],
    *,
    elapsed_seconds: float,
    peak_ram_mib: float,
    peak_vram_mib: float,
) -> DepthClipMetrics:
    """Measure one complete candidate only on the frozen held-out samples."""
    if not isinstance(clip, MetricDepthClip):
        raise WaymoContractError("clip must be MetricDepthClip")
    if not isinstance(evidence, ProjectedLidarDepth):
        raise WaymoContractError("evidence must be ProjectedLidarDepth")
    if clip.clip_key != evidence.clip_key or clip.camera_name != evidence.camera_name:
        raise WaymoContractError("clip and evidence must use the same key and camera")
    frame_count = len(clip.clip_key.frame_timestamps_micros)
    scales = _scale_sequence(scale_by_frame, frame_count)

    predicted: list[npt.NDArray[np.float64]] = []
    measured: list[npt.NDArray[np.float64]] = []
    heldout_count = int(np.count_nonzero(evidence.heldout))
    for frame_index in range(frame_count):
        sample_slice = evidence.frame_slice(frame_index)
        sampled, sampled_valid = sample_bilinear_depth(
            clip.depth_z_m[frame_index],
            clip.valid[frame_index],
            evidence.canvas_xy[sample_slice],
        )
        selected = sampled_valid & evidence.heldout[sample_slice]
        if np.any(selected):
            predicted.append(sampled[selected])
            measured.append(
                evidence.depth_z_m[sample_slice][selected].astype(np.float64)
            )
    if heldout_count == 0 or not predicted:
        raise WaymoContractError("held-out LiDAR has no valid predicted depth")

    predicted_z = np.concatenate(predicted)
    measured_z = np.concatenate(measured)
    absolute_z = np.abs(predicted_z - measured_z)
    absolute_relative = absolute_z / measured_z
    return DepthClipMetrics(
        clip_key=clip.clip_key,
        camera_name=clip.camera_name,
        heldout_count=heldout_count,
        valid_fraction=float(predicted_z.size / heldout_count),
        median_abs_rel=float(np.median(absolute_relative)),
        median_abs_z_m=float(np.median(absolute_z)),
        p95_abs_z_m=float(np.quantile(absolute_z, 0.95, method="linear")),
        temporal_scale_jitter=_temporal_scale_jitter(scales),
        elapsed_seconds=elapsed_seconds,
        peak_ram_mib=peak_ram_mib,
        peak_vram_mib=peak_vram_mib,
    )


def sample_bilinear_depth(
    depth: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_],
    canvas_xy: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Sample depth at canvas coordinates with an exact four-pixel support."""
    if (
        not isinstance(depth, np.ndarray)
        or depth.dtype != np.dtype(np.float32)
        or depth.ndim != 2
        or not depth.flags.c_contiguous
    ):
        raise WaymoContractError("depth must be a C-contiguous float32 image")
    if (
        not isinstance(valid, np.ndarray)
        or valid.dtype != np.dtype(np.bool_)
        or valid.shape != depth.shape
        or not valid.flags.c_contiguous
    ):
        raise WaymoContractError("valid must be a matching C-contiguous bool image")
    coordinates = cast(npt.NDArray[np.float32], canvas_xy)
    if (
        not isinstance(coordinates, np.ndarray)
        or coordinates.dtype != np.dtype(np.float32)
        or coordinates.ndim != 2
        or coordinates.shape[1:] != (2,)
        or not coordinates.flags.c_contiguous
        or not np.isfinite(coordinates).all()
    ):
        raise WaymoContractError(
            "canvas_xy must be finite C-contiguous float32 [N,2]"
        )

    height, width = depth.shape
    x = coordinates[:, 0].astype(np.float64)
    y = coordinates[:, 1].astype(np.float64)
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
    support_valid = (
        inside
        & valid[y0, x0]
        & valid[y0, x1]
        & valid[y1, x0]
        & valid[y1, x1]
        & np.isfinite(sampled)
        & (sampled > 0.0)
    )
    return sampled, support_valid


def _scale_sequence(value: object, frame_count: int) -> npt.NDArray[np.float64]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.float64)
        or value.shape != (frame_count,)
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
        or np.any(value <= 0.0)
    ):
        raise WaymoContractError(
            "scale_by_frame must be finite positive C-contiguous float64 [frames]"
        )
    return value


def _temporal_scale_jitter(scales: npt.NDArray[np.float64]) -> float:
    if len(scales) < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(np.log(scales)))))

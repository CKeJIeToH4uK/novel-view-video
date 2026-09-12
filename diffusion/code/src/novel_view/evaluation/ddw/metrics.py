"""Heldout LiDAR, RGB, and four-panel evidence for DDW evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import numpy.typing as npt

from novel_view.evaluation.ddw.sampling import DdwVariant
from novel_view.preparation.waymo_ddw.depth import (
    fit_positive_relative_l1_scale,
)


VARIANTS: tuple[DdwVariant, ...] = ("base", "v1", "v2")


@dataclass(frozen=True, slots=True, eq=False)
class SparseEvaluationLidar:
    """NumPy view of the strict Stage 6 sparse LiDAR artifact."""

    frame_offsets: npt.NDArray[np.int64]
    canvas_xy: npt.NDArray[np.float32]
    depth_z_m: npt.NDArray[np.float32]
    evaluation_holdout: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FrameDepthEvidence:
    """Scale support and heldout coverage for one primary frame."""

    frame_index: int
    scale: float
    nonheldout_fit_count: int
    heldout_count: int
    valid_count: int


@dataclass(frozen=True, slots=True)
class DepthMetrics:
    """Pooled heldout-point measurements for one item and variant."""

    median_abs_rel: float
    median_abs_z_m: float
    q95_abs_z_m: float
    valid_fraction: float
    temporal_scale_jitter: float
    frames: tuple[FrameDepthEvidence, ...]


@dataclass(frozen=True, slots=True)
class RgbMetrics:
    """All-121-frame target parity and change relative to base."""

    mae: float
    mae_improvement_vs_base: float | None
    improved_frame_fraction_vs_base: float | None
    mean_abs_change_vs_base: float | None


@dataclass(frozen=True, slots=True)
class VariantItemMetrics:
    """Depth and RGB evidence for one named output variant."""

    variant: DdwVariant
    depth: DepthMetrics
    rgb: RgbMetrics


@dataclass(frozen=True, slots=True)
class DdwItemMetrics:
    """All three matched outputs for one validation sample."""

    sample_id: str
    variants: tuple[VariantItemMetrics, ...]


@dataclass(frozen=True, slots=True)
class VariantAggregate:
    """Median depth and mean RGB item-level aggregates."""

    variant: DdwVariant
    median_abs_rel: float
    median_abs_z_m: float
    q95_abs_z_m: float
    valid_fraction: float
    temporal_scale_jitter: float
    rgb_mae: float
    mae_improvement_vs_base: float | None
    improved_frame_fraction_vs_base: float | None
    mean_abs_change_vs_base: float | None


def evaluate_ddw_item(
    sample_id: str,
    target: npt.NDArray[np.uint8],
    decoded: Mapping[DdwVariant, npt.NDArray[np.uint8]],
    relative_depth: Mapping[DdwVariant, npt.NDArray[np.float32]],
    model_valid: Mapping[DdwVariant, npt.NDArray[np.bool_]],
    lidar: SparseEvaluationLidar,
) -> DdwItemMetrics:
    """Compute exact per-item metrics without pooling different samples."""
    _require_variant_arrays(target, decoded, relative_depth, model_valid)
    frame_mae = {variant: _frame_mae(decoded[variant], target) for variant in VARIANTS}
    base_mae = frame_mae["base"]
    variants = []
    for variant in VARIANTS:
        current = frame_mae[variant]
        if variant == "base":
            rgb = RgbMetrics(float(current.mean()), None, None, None)
        else:
            rgb = RgbMetrics(
                mae=float(current.mean()),
                mae_improvement_vs_base=float(base_mae.mean() - current.mean()),
                improved_frame_fraction_vs_base=float(np.mean(current < base_mae)),
                mean_abs_change_vs_base=float(
                    _frame_mae(decoded[variant], decoded["base"]).mean()
                ),
            )
        variants.append(
            VariantItemMetrics(
                variant,
                _depth_metrics(relative_depth[variant], model_valid[variant], lidar),
                rgb,
            )
        )
    return DdwItemMetrics(sample_id, tuple(variants))


def aggregate_ddw_items(
    items: tuple[DdwItemMetrics, ...],
) -> tuple[VariantAggregate, ...]:
    """Aggregate eight item-level measurements without pooling their points."""
    if not items:
        raise ValueError("DDW evaluation requires validation items")
    aggregates = []
    for index, variant in enumerate(VARIANTS):
        rows = [item.variants[index] for item in items]
        if any(row.variant != variant for row in rows):
            raise ValueError("DDW item variant order differs")
        rgb_values = [row.rgb for row in rows]
        aggregates.append(
            VariantAggregate(
                variant=variant,
                median_abs_rel=float(
                    np.median([row.depth.median_abs_rel for row in rows])
                ),
                median_abs_z_m=float(
                    np.median([row.depth.median_abs_z_m for row in rows])
                ),
                q95_abs_z_m=float(np.median([row.depth.q95_abs_z_m for row in rows])),
                valid_fraction=float(
                    np.median([row.depth.valid_fraction for row in rows])
                ),
                temporal_scale_jitter=float(
                    np.median([row.depth.temporal_scale_jitter for row in rows])
                ),
                rgb_mae=float(np.mean([row.mae for row in rgb_values])),
                mae_improvement_vs_base=_mean_optional(
                    [row.mae_improvement_vs_base for row in rgb_values]
                ),
                improved_frame_fraction_vs_base=_mean_optional(
                    [row.improved_frame_fraction_vs_base for row in rgb_values]
                ),
                mean_abs_change_vs_base=_mean_optional(
                    [row.mean_abs_change_vs_base for row in rgb_values]
                ),
            )
        )
    return tuple(aggregates)


def write_four_panel_video(
    path: Path,
    target: npt.NDArray[np.uint8],
    decoded: Mapping[DdwVariant, npt.NDArray[np.uint8]],
    lidar: SparseEvaluationLidar,
    *,
    fps: int = 10,
) -> None:
    """Write target | base | v1 | v2+LiDAR directly as one MP4."""
    import cv2

    height, width = target.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 4, height),
    )
    if not writer.isOpened():
        raise RuntimeError("cannot open DDW four-panel video writer")
    try:
        for frame_index in range(target.shape[0]):
            v2_lidar = decoded["v2"][frame_index].copy()
            frame_slice = slice(
                int(lidar.frame_offsets[frame_index]),
                int(lidar.frame_offsets[frame_index + 1]),
            )
            _draw_lidar(
                v2_lidar,
                lidar.canvas_xy[frame_slice],
                lidar.depth_z_m[frame_slice],
            )
            canvas = np.concatenate(
                (
                    target[frame_index],
                    decoded["base"][frame_index],
                    decoded["v1"][frame_index],
                    v2_lidar,
                ),
                axis=1,
            )
            for label, x in (
                ("target", 20),
                ("base", width + 20),
                ("v1 Kendall EDM", 2 * width + 20),
                ("v2 Kendall EDM + LiDAR", 3 * width + 20),
            ):
                cv2.putText(
                    canvas,
                    label,
                    (x, min(44, height - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _depth_metrics(
    depth: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_],
    lidar: SparseEvaluationLidar,
) -> DepthMetrics:
    frame_rows = []
    abs_rel_parts = []
    abs_z_parts = []
    scales = []
    heldout_total = 0
    heldout_valid_total = 0
    for frame_index in range(1, 121):
        frame_slice = slice(
            int(lidar.frame_offsets[frame_index]),
            int(lidar.frame_offsets[frame_index + 1]),
        )
        points = lidar.canvas_xy[frame_slice]
        target_z = lidar.depth_z_m[frame_slice]
        heldout = lidar.evaluation_holdout[frame_slice]
        sampled, sampled_valid = _sample_bilinear(
            depth[frame_index],
            valid[frame_index],
            points,
        )
        fit = sampled_valid & ~heldout
        if not np.any(fit):
            raise ValueError(
                f"DDW depth frame {frame_index} has no nonheldout fit support"
            )
        scale = fit_positive_relative_l1_scale(sampled[fit], target_z[fit])
        score = sampled_valid & heldout
        residual = np.abs(sampled[score] * scale - target_z[score])
        abs_z_parts.append(residual)
        abs_rel_parts.append(residual / target_z[score])
        heldout_count = int(np.count_nonzero(heldout))
        valid_count = int(np.count_nonzero(score))
        heldout_total += heldout_count
        heldout_valid_total += valid_count
        scales.append(scale)
        frame_rows.append(
            FrameDepthEvidence(
                frame_index,
                scale,
                int(np.count_nonzero(fit)),
                heldout_count,
                valid_count,
            )
        )
    if heldout_total == 0 or heldout_valid_total == 0:
        raise ValueError("DDW item has no numerically defined heldout depth score")
    abs_rel = np.concatenate(abs_rel_parts)
    abs_z = np.concatenate(abs_z_parts)
    scales_array = np.asarray(scales, dtype=np.float64)
    jitter = np.abs(np.diff(np.log(scales_array)))
    return DepthMetrics(
        median_abs_rel=float(np.median(abs_rel)),
        median_abs_z_m=float(np.median(abs_z)),
        q95_abs_z_m=float(np.quantile(abs_z, 0.95, method="linear")),
        valid_fraction=heldout_valid_total / heldout_total,
        temporal_scale_jitter=float(np.median(jitter)),
        frames=tuple(frame_rows),
    )


def _sample_bilinear(
    depth: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_],
    canvas_xy: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    height, width = depth.shape
    x = canvas_xy[:, 0].astype(np.float64)
    y = canvas_xy[:, 1].astype(np.float64)
    inside = (x >= 0.0) & (x < width - 1) & (y >= 0.0) & (y < height - 1)
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


def _frame_mae(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(left.shape[0], dtype=np.float64)
    for index in range(left.shape[0]):
        delta = left[index].astype(np.int16) - right[index].astype(np.int16)
        result[index] = float(np.abs(delta).mean())
    return result


def _mean_optional(values: list[float | None]) -> float | None:
    if values[0] is None:
        if any(value is not None for value in values):
            raise ValueError("DDW aggregate optional metrics differ")
        return None
    return float(np.mean([value for value in values if value is not None]))


def _require_variant_arrays(
    target: np.ndarray,
    decoded: Mapping[DdwVariant, np.ndarray],
    depth: Mapping[DdwVariant, np.ndarray],
    valid: Mapping[DdwVariant, np.ndarray],
) -> None:
    if target.dtype != np.uint8 or target.ndim != 4 or target.shape[0] != 121:
        raise ValueError("DDW target must be uint8 THWC with 121 frames")
    if (
        tuple(decoded) != VARIANTS
        or tuple(depth) != VARIANTS
        or tuple(valid) != VARIANTS
    ):
        raise ValueError("DDW evaluation variants must be ordered base, v1, v2")
    expected_depth_shape = target.shape[:3]
    for variant in VARIANTS:
        if decoded[variant].dtype != np.uint8 or decoded[variant].shape != target.shape:
            raise ValueError(f"{variant} decoded video differs from target shape")
        if (
            depth[variant].dtype != np.float32
            or depth[variant].shape != expected_depth_shape
        ):
            raise ValueError(f"{variant} relative depth has the wrong contract")
        if (
            valid[variant].dtype != np.bool_
            or valid[variant].shape != expected_depth_shape
        ):
            raise ValueError(f"{variant} model-valid mask has the wrong contract")


def _draw_lidar(
    image: npt.NDArray[np.uint8],
    xy: npt.NDArray[np.float32],
    depth_z_m: npt.NDArray[np.float32],
) -> None:
    import cv2

    if len(xy) == 0:
        return
    height, width = image.shape[:2]
    pixels = np.rint(xy).astype(np.intp)
    x = np.clip(pixels[:, 0], 0, width - 1)
    y = np.clip(pixels[:, 1], 0, height - 1)
    normalized = np.clip(depth_z_m / np.float32(80.0), 0.0, 1.0)
    colors_bgr = cv2.applyColorMap(
        np.asarray(np.rint(normalized * 255.0), dtype=np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)
    colors_rgb = colors_bgr[:, ::-1]
    for dy, dx in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)):
        image[np.clip(y + dy, 0, height - 1), np.clip(x + dx, 0, width - 1)] = (
            colors_rgb
        )


__all__ = [
    "DdwItemMetrics",
    "DepthMetrics",
    "FrameDepthEvidence",
    "RgbMetrics",
    "SparseEvaluationLidar",
    "VARIANTS",
    "VariantAggregate",
    "VariantItemMetrics",
    "aggregate_ddw_items",
    "evaluate_ddw_item",
    "write_four_panel_video",
]

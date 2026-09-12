"""Run and reproject historical VGGT depth into measured Waymo cameras."""

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
from novel_view.preparation.waymo_depth.depth_samples import ProjectedLidarDepth
from novel_view.geometry.sim3 import (
    CameraPackAlignment,
    CameraPackAlignmentError,
    align_camera_packs,
)
from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.inputs.waymo._arrays import require_array
from novel_view.models.vggt.backend import VggtOmegaResources, run_vggt_omega
from novel_view.models.vggt.request import VggtOmegaRequest
from novel_view.models.vggt.spec import (
    VggtOmegaError,
    VggtOmegaInputMode,
    build_vggt_omega_input_plan,
)

_ROW_CHUNK = 64
_BICUBIC_SUPPORT = 2.0


class VggtDepthUnavailable(WaymoContractError):
    """VGGT ran, but its predicted geometry cannot form this candidate."""


@dataclass(frozen=True, slots=True, eq=False)
class VggtMetricDepth:
    """Metric Waymo depth and the one clip-wide VGGT Sim(3) fit."""

    clip: MetricDepthClip
    alignment: CameraPackAlignment

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.clip, MetricDepthClip):
            raise WaymoContractError("clip must be MetricDepthClip")
        if len(self.alignment.aligned_world_to_camera) != len(
            self.clip.clip_key.frame_timestamps_micros
        ):
            raise WaymoContractError("clip and alignment must have the same frames")


def build_vggt_metric_depth(
    evidence: ProjectedLidarDepth,
    depth_model_units: npt.NDArray[np.float32],
    model_intrinsics: npt.NDArray[np.float32],
    model_world_to_camera: npt.NDArray[np.float32],
    model_rectification_known: npt.NDArray[np.bool_],
    target_rectification_known: npt.NDArray[np.bool_],
    K_canvas: npt.NDArray[np.float64],
    world_to_camera_cv: npt.NDArray[np.float64],
) -> VggtMetricDepth:
    """Use one camera-pack Sim(3), then reproject every predicted 3D point.

    LiDAR evidence binds the exact clip and camera but never fits VGGT: its
    metric gauge comes only from the joint predicted/measured camera packs.
    Native depth stays attached to predicted cameras until the full 3D
    reprojection into measured Waymo ``K_canvas``/W2C.
    """
    if not isinstance(evidence, ProjectedLidarDepth):
        raise WaymoContractError("evidence must be ProjectedLidarDepth")
    frame_count = len(evidence.clip_key.frame_timestamps_micros)
    depth = _depth_pack(depth_model_units, frame_count)
    model_known = _known_mask(
        model_rectification_known,
        depth.shape[1:],
        "model_rectification_known",
    )
    model_K = _intrinsics_pack(
        model_intrinsics,
        "model_intrinsics",
        np.dtype(np.float32),
        frame_count,
    )
    target_K = _intrinsics_pack(
        require_array(K_canvas, "K_canvas", dtype=np.dtype(np.float64),
                      shape=(frame_count, 3, 3)),
        "K_canvas",
        np.dtype(np.float64),
        frame_count,
        atol=1e-12,
    )
    target_w2c = require_array(
        world_to_camera_cv,
        "world_to_camera_cv",
        dtype=np.dtype(np.float64),
        shape=(frame_count, 4, 4),
    )
    target_size = waymo_depth_contract.GEN3C_CANVAS_SIZE_HW
    target_known = _known_mask(target_rectification_known, target_size,
                               "target_rectification_known")
    try:
        alignment = align_camera_packs(
            model_world_to_camera,
            target_w2c,
        )
    except CameraPackAlignmentError as error:
        raise VggtDepthUnavailable(
            f"VGGT camera alignment failed: {error}"
        ) from error

    metric_depth = np.zeros((frame_count, *target_size), dtype=np.float32)
    metric_valid = np.zeros((frame_count, *target_size), dtype=np.bool_)
    scale = alignment.scale_m_per_model_unit
    for frame_index in range(frame_count):
        frame_depth, frame_valid = _reproject_metric_frame(
            depth[frame_index],
            model_known,
            model_K[frame_index],
            alignment.aligned_world_to_camera[frame_index],
            target_K[frame_index],
            target_w2c[frame_index],
            scale,
            target_size,
        )
        frame_valid &= target_known
        metric_valid[frame_index] = frame_valid
        metric_depth[frame_index, frame_valid] = frame_depth[frame_valid]

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
    return VggtMetricDepth(clip, alignment)


def _reproject_metric_frame(
    depth_model_units: npt.NDArray[np.float32],
    model_valid: npt.NDArray[np.bool_],
    model_K: npt.NDArray[np.floating],
    aligned_model_w2c: npt.NDArray[np.float64],
    target_K: npt.NDArray[np.float64],
    target_w2c: npt.NDArray[np.float64],
    scale_m_per_model_unit: float,
    target_size_hw: tuple[int, int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Project one dense predicted camera into one measured camera."""
    source_rotation = aligned_model_w2c[:3, :3]
    target_rotation = target_w2c[:3, :3]
    source_to_target_rotation = target_rotation @ source_rotation.T
    source_to_target_translation = target_w2c[:3, 3] - (
        source_to_target_rotation @ aligned_model_w2c[:3, 3]
    )
    try:
        inverse_model_K = np.linalg.inv(model_K.astype(np.float64))
    except np.linalg.LinAlgError as error:
        raise VggtDepthUnavailable("model_intrinsics must be invertible") from error

    minimum_z = np.full(np.prod(target_size_hw), np.inf, dtype=np.float64)
    height, width = depth_model_units.shape
    x = np.arange(width, dtype=np.float64)
    for row_start in range(0, height, _ROW_CHUNK):
        row_stop = min(row_start + _ROW_CHUNK, height)
        y = np.arange(row_start, row_stop, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(x, y)
        selected = model_valid[row_start:row_stop].reshape(-1)
        if not np.any(selected):
            continue
        pixels = np.stack(
            (
                grid_x.ravel()[selected],
                grid_y.ravel()[selected],
                np.ones(np.count_nonzero(selected)),
            ),
        )
        rays = inverse_model_K @ pixels
        source_z = depth_model_units[row_start:row_stop].reshape(-1)[selected]
        with np.errstate(over="ignore", invalid="ignore"):
            source_z = source_z.astype(np.float64) * scale_m_per_model_unit
        if not np.isfinite(source_z).all() or np.any(source_z <= 0.0):
            raise VggtDepthUnavailable("scaled VGGT depth left the float64 range")
        source_points = rays * source_z
        target_points = source_to_target_rotation @ source_points
        target_points += source_to_target_translation[:, None]
        if not np.isfinite(target_points).all():
            raise VggtDepthUnavailable("VGGT camera reprojection overflowed")
        target_z = target_points[2]
        projected = target_K @ target_points
        if not np.isfinite(projected).all():
            raise VggtDepthUnavailable("VGGT pinhole projection overflowed")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            target_x = projected[0] / target_z
            target_y = projected[1] / target_z
        _splat_minimum_z(minimum_z, target_x, target_y, target_z, target_size_hw)

    valid = np.isfinite(minimum_z).reshape(target_size_hw)
    depth = np.zeros(target_size_hw, dtype=np.float32)
    if np.any(valid):
        visible_z = minimum_z.reshape(target_size_hw)[valid]
        cast_depth = visible_z.astype(np.float32)
        if not np.isfinite(cast_depth).all() or np.any(cast_depth <= 0.0):
            raise VggtDepthUnavailable(
                "reprojected VGGT depth left the positive float32 range"
            )
        depth[valid] = cast_depth
    return depth, valid


def _splat_minimum_z(
    minimum_z: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    size_hw: tuple[int, int],
) -> None:
    """Apply the established deterministic four-neighbour depth protocol."""
    height, width = size_hw
    inside = (
        (z > 0.0)
        & np.isfinite(z)
        & np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= width - 1)
        & (y >= 0.0)
        & (y <= height - 1)
    )
    if not np.any(inside):
        return
    x0 = np.floor(x[inside]).astype(np.int64)
    x1 = np.ceil(x[inside]).astype(np.int64)
    y0 = np.floor(y[inside]).astype(np.int64)
    y1 = np.ceil(y[inside]).astype(np.int64)
    visible_z = z[inside]
    for target_x, target_y, selected in (
        (x0, y0, np.ones(x0.shape, dtype=np.bool_)),
        (x1, y0, x1 != x0),
        (x0, y1, y1 != y0),
        (x1, y1, (x1 != x0) & (y1 != y0)),
    ):
        if np.any(selected):
            pixel = target_y[selected] * width + target_x[selected]
            np.minimum.at(minimum_z, pixel, visible_z[selected])


def _depth_pack(value: object, frame_count: int) -> npt.NDArray[np.float32]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.float32)
        or value.ndim != 3
        or value.shape[0] != frame_count
        or value.shape[1] <= 0
        or value.shape[2] <= 0
        or not value.flags.c_contiguous
    ):
        raise WaymoContractError(
            "depth_model_units must be C-contiguous float32 [frames,height,width]"
        )
    for index, frame in enumerate(value):
        if not np.isfinite(frame).all() or np.any(frame <= 0.0):
            raise VggtDepthUnavailable(
                f"depth_model_units[{index}] must be finite and positive"
            )
    return value


def _intrinsics_pack(
    value: object,
    name: str,
    dtype: np.dtype[np.generic],
    frame_count: int,
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.shape != (frame_count, 3, 3)
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
        or np.any(value[:, 0, 0] <= 0.0)
        or np.any(value[:, 1, 1] <= 0.0)
        or not np.allclose(value[:, 2], (0.0, 0.0, 1.0), rtol=0.0, atol=atol)
        or not np.allclose(value[:, 1, 0], 0.0, rtol=0.0, atol=atol)
    ):
        raise WaymoContractError(
            f"{name} must be finite C-contiguous {dtype} pinhole [frames,3,3]"
        )
    return value


def _known_mask(value: object, shape: tuple[int, int],
                name: str) -> npt.NDArray[np.bool_]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.bool_)
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise WaymoContractError(
            f"{name} must be a matching C-contiguous bool mask"
        )
    return value


def build_conservative_model_mask(
    source_known: npt.NDArray[np.bool_],
    model_size_hw: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Keep only model pixels whose official bicubic support is known."""
    source_height, source_width = source_known.shape
    model_height, model_width = model_size_hw
    y0, y1 = _support_bounds(source_height, model_height)
    x0, x1 = _support_bounds(source_width, model_width)
    integral = np.pad(~source_known, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    unknown_count = (
        integral[y1[:, None], x1]
        - integral[y0[:, None], x1]
        - integral[y1[:, None], x0]
        + integral[y0[:, None], x0]
    )
    result = np.asarray(unknown_count == 0, dtype=np.bool_, order="C")
    result.setflags(write=False)
    return result


def _support_bounds(source_size: int, model_size: int) -> tuple[np.ndarray, np.ndarray]:
    scale = source_size / model_size
    centre = (np.arange(model_size, dtype=np.float64) + 0.5) * scale - 0.5
    radius = int(np.ceil(_BICUBIC_SUPPORT * max(1.0, scale))) + 1
    anchor = np.floor(centre).astype(np.int64)
    return (
        np.maximum(anchor - radius, 0),
        np.minimum(anchor + radius + 1, source_size),
    )


@dataclass(frozen=True, slots=True)
class VggtDepthResult:
    metric_depth: VggtMetricDepth
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


def run_vggt_metric_depth(
    rgb_frames: Iterable[npt.NDArray[np.uint8]],
    evidence: ProjectedLidarDepth,
    rectification_known: npt.NDArray[np.bool_],
    K_canvas: npt.NDArray[np.float64],
    world_to_camera_cv: npt.NDArray[np.float64],
    resources: VggtOmegaResources,
    log_path: Path,
) -> VggtDepthResult:
    """Run one camera pack, align one Sim(3), then reproject to measured cameras."""
    frame_count = len(evidence.clip_key.frame_timestamps_micros)
    plan = build_vggt_omega_input_plan(
        waymo_depth_contract.GEN3C_CANVAS_SIZE_HW,
        VggtOmegaInputMode.BALANCED_512,
    )
    model_known = build_conservative_model_mask(
        rectification_known,
        plan.model_size_hw,
    )
    started = time.perf_counter()
    execution = run_vggt_omega(
        VggtOmegaRequest(
            rgb_frames=rgb_frames,
            frame_count=frame_count,
            input_plan=plan,
        ),
        resources,
        log_path,
    )
    raw = execution.prediction
    metric_depth = build_vggt_metric_depth(
        evidence,
        raw.depth_model_units,
        raw.model_intrinsics,
        raw.model_w2c,
        model_known,
        rectification_known,
        K_canvas,
        world_to_camera_cv,
    )
    return VggtDepthResult(
        metric_depth=metric_depth,
        elapsed_seconds=float(time.perf_counter() - started),
        worker_peak_ram_mib=execution.telemetry.peak_ram_mib,
        worker_peak_vram_mib=execution.telemetry.peak_vram_mib,
    )


__all__ = [
    "VggtDepthResult",
    "VggtDepthUnavailable",
    "VggtMetricDepth",
    "VggtOmegaError",
    "VggtOmegaResources",
    "build_conservative_model_mask",
    "build_vggt_metric_depth",
    "run_vggt_metric_depth",
]

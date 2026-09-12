"""Align and filter metric context depth for a Gen3C generation request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

_MAX_DEPTH_M = 100.0
_LOCAL_WINDOW_SIZE = 5
_LOCAL_DEPTH_RATIO = 0.05
_NEIGHBOUR_ABSOLUTE_M = 0.10
_NEIGHBOUR_RELATIVE = 0.05
_MIN_ALIGNMENT_PIXELS = 4096
_MAX_CONTEXT_FRAME_COUNT = 20


class ContextDepthError(RuntimeError):
    """Generated context depth cannot be aligned to original evidence."""


@dataclass(frozen=True, slots=True, eq=False)
class ContextDepthResult:
    """Filtered metric camera-Z depth and per-frame alignment diagnostics."""

    depth_z_m: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]
    alignment_scale: npt.NDArray[np.float64]
    alignment_bias: npt.NDArray[np.float64]
    alignment_mae_m: npt.NDArray[np.float64]


def prepare_context_depth(
    moge_depth: npt.NDArray[np.float32],
    moge_valid: npt.NDArray[np.bool_],
    reference_depth_z_m: npt.NDArray[np.float32],
    reference_valid: npt.NDArray[np.bool_],
    w2c: npt.NDArray[np.float64],
    intrinsics: npt.NDArray[np.float64],
    *,
    torch: Any,
    align_depth: Callable[..., Any],
    reliable_depth_mask_range_batch: Callable[..., Any],
    forward_warp: Callable[..., Any],
    min_alignment_pixels: int = _MIN_ALIGNMENT_PIXELS,
) -> ContextDepthResult:
    """Align MoGe to original metric Z and retain neighbour-consistent depth."""
    shape = _validate_inputs(
        moge_depth,
        moge_valid,
        reference_depth_z_m,
        reference_valid,
        w2c,
        intrinsics,
        min_alignment_pixels,
    )
    frame_count, height, width = shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.from_numpy(np.array(moge_depth, copy=True)).to(device)
    reference = torch.from_numpy(
        np.array(reference_depth_z_m, copy=True)
    ).to(device)
    common = torch.from_numpy(
        moge_valid
        & reference_valid
        & np.isfinite(moge_depth)
        & (moge_depth > 0.0)
        & np.isfinite(reference_depth_z_m)
        & (reference_depth_z_m > 0.0)
    ).to(device)
    infinity = torch.full((), float("inf"), device=device)
    source_masked = torch.where(common, source, infinity)
    reference_masked = torch.where(common, reference, infinity)

    aligned = torch.empty_like(source)
    scales = np.empty(frame_count, dtype=np.float64)
    biases = np.empty(frame_count, dtype=np.float64)
    residuals = np.empty(frame_count, dtype=np.float64)
    with torch.inference_mode():
        for index in range(frame_count):
            scale, bias = _alignment_fit(
                source_masked[index],
                reference_masked[index],
                common[index],
                torch,
                min_alignment_pixels,
                index,
            )
            value = align_depth(
                source_masked[index],
                reference_masked[index],
                common[index],
                alignment_method="rigid",
            )
            aligned[index] = value
            scales[index] = scale
            biases[index] = bias
            residual_support = (
                common[index] & torch.isfinite(value) & (value > 0.0)
            )
            if not bool(residual_support.any().item()):
                raise ContextDepthError(
                    f"frame {index} has no finite positive aligned depth"
                )
            residuals[index] = float(
                torch.mean(
                    torch.abs(
                        value[residual_support]
                        - reference[index][residual_support]
                    )
                ).item()
            )

        finite = torch.isfinite(aligned) & (aligned > 0.0)
        valid = common & finite & (aligned <= _MAX_DEPTH_M)
        filtered_depth = torch.where(valid, aligned, torch.zeros_like(aligned))
        local = reliable_depth_mask_range_batch(
            filtered_depth[:, None],
            window_size=_LOCAL_WINDOW_SIZE,
            ratio_thresh=_LOCAL_DEPTH_RATIO,
        )[:, 0].bool()
        valid &= local
        filtered_depth = torch.where(valid, aligned, torch.zeros_like(aligned))
        valid &= _neighbour_consistency(
            filtered_depth,
            valid,
            torch.from_numpy(np.array(w2c, copy=True)).to(
                device=device, dtype=torch.float32
            ),
            torch.from_numpy(np.array(intrinsics, copy=True)).to(
                device=device, dtype=torch.float32
            ),
            torch,
            forward_warp,
        )
        filtered_depth = torch.where(valid, aligned, torch.zeros_like(aligned))

    depth_numpy = filtered_depth.cpu().numpy().astype(np.float32, copy=False)
    valid_numpy = valid.cpu().numpy().astype(np.bool_, copy=False)
    return ContextDepthResult(
        depth_z_m=_readonly(depth_numpy),
        valid=_readonly(valid_numpy),
        alignment_scale=_readonly(scales),
        alignment_bias=_readonly(biases),
        alignment_mae_m=_readonly(residuals),
    )


def _alignment_fit(
    source_depth: Any,
    target_depth: Any,
    common: Any,
    torch: Any,
    min_pixels: int,
    frame_index: int,
) -> tuple[float, float]:
    """Reproduce the official rigid fit support before calling it."""
    source_inv = 1.0 / source_depth
    target_inv = 1.0 / target_depth
    quantiles = torch.tensor((0.1, 0.9), device=source_depth.device)
    source_low, source_high = torch.quantile(
        source_inv[source_inv > 0.0], quantiles
    )
    target_low, target_high = torch.quantile(target_inv[common], quantiles)
    fit = (
        (source_inv > source_low)
        & (source_inv < source_high)
        & (target_inv > target_low)
        & (target_inv < target_high)
    )
    count = int(fit.sum().item())
    if count < min_pixels:
        raise ContextDepthError(
            f"frame {frame_index} has {count} rigid-alignment pixels; "
            f"need at least {min_pixels}"
        )
    source_data = source_inv[fit].reshape(-1, 1)
    design = torch.cat((source_data, torch.ones_like(source_data)), dim=1)
    if int(torch.linalg.matrix_rank(design).item()) != 2:
        raise ContextDepthError(
            f"frame {frame_index} rigid-alignment design has rank below 2"
        )
    target_data = target_inv[fit].reshape(-1, 1)
    solution = torch.linalg.lstsq(design, target_data).solution
    scale = float(solution[0, 0].item())
    bias = float(solution[1, 0].item())
    if not np.isfinite(scale) or not np.isfinite(bias) or scale <= 0.0:
        raise ContextDepthError(
            f"frame {frame_index} rigid-alignment scale/bias is invalid"
        )
    return scale, bias


def _neighbour_consistency(
    depth: Any,
    valid: Any,
    w2c: Any,
    intrinsics: Any,
    torch: Any,
    forward_warp: Callable[..., Any],
) -> Any:
    """Require every frame to agree with all of its temporal neighbours."""
    frame_count, height, width = depth.shape
    if frame_count == 1:
        return torch.ones_like(valid)
    source_indices: list[int] = []
    target_indices: list[int] = []
    for left in range(frame_count - 1):
        source_indices.extend((left, left + 1))
        target_indices.extend((left + 1, left))
    source_index = torch.tensor(source_indices, device=depth.device)
    target_index = torch.tensor(target_indices, device=depth.device)
    dummy = torch.zeros(
        (source_index.numel(), 1, height, width),
        device=depth.device,
        dtype=depth.dtype,
    )
    _, warped_mask, warped_depth, _ = forward_warp(
        dummy,
        valid[source_index, None].to(depth.dtype),
        depth[source_index, None],
        w2c[source_index],
        w2c[target_index],
        intrinsics[source_index],
        intrinsics[target_index],
        is_image=False,
        is_depth=True,
        render_depth=True,
    )
    target_depth = depth[target_index]
    tolerance = torch.maximum(
        torch.full_like(target_depth, _NEIGHBOUR_ABSOLUTE_M),
        _NEIGHBOUR_RELATIVE * target_depth,
    )
    pair_agreement = (
        (warped_mask[:, 0] > 0.5)
        & valid[target_index]
        & torch.isfinite(warped_depth)
        & (torch.abs(warped_depth - target_depth) <= tolerance)
    )
    agreement = torch.ones_like(valid)
    for row, target in enumerate(target_indices):
        agreement[target] &= pair_agreement[row]
    return agreement


def _validate_inputs(
    moge_depth: np.ndarray,
    moge_valid: np.ndarray,
    reference_depth: np.ndarray,
    reference_valid: np.ndarray,
    w2c: np.ndarray,
    intrinsics: np.ndarray,
    min_alignment_pixels: int,
) -> tuple[int, int, int]:
    if moge_depth.dtype != np.float32 or moge_depth.ndim != 3:
        raise ContextDepthError("moge_depth must be float32 [F,H,W]")
    shape = moge_depth.shape
    for name, value, dtype, expected in (
        ("moge_valid", moge_valid, np.bool_, shape),
        ("reference_depth_z_m", reference_depth, np.float32, shape),
        ("reference_valid", reference_valid, np.bool_, shape),
        ("w2c", w2c, np.float64, (shape[0], 4, 4)),
        ("intrinsics", intrinsics, np.float64, (shape[0], 3, 3)),
    ):
        if value.dtype != np.dtype(dtype) or value.shape != expected:
            raise ContextDepthError(
                f"{name} must be {np.dtype(dtype).name} {expected}"
            )
    if not 1 <= shape[0] <= _MAX_CONTEXT_FRAME_COUNT:
        raise ContextDepthError("overlap context depth needs 1..20 frames")
    if min_alignment_pixels <= 0:
        raise ContextDepthError("min_alignment_pixels must be positive")
    if not np.all(np.isfinite(w2c)) or not np.all(np.isfinite(intrinsics)):
        raise ContextDepthError("context cameras must be finite")
    return shape


def _readonly(value: npt.NDArray[Any]) -> npt.NDArray[Any]:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


__all__ = ["ContextDepthError", "ContextDepthResult", "prepare_context_depth"]

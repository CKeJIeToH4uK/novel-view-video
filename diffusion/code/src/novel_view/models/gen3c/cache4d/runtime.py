"""Heavy in-process Cache4D construction and raw rendering."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_MAX_CACHE_DEPTH_M = 100.0
_FILTER_POINTS_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class _DenseCacheInputs:
    """Heavy source evidence gathered in exact global-slot order."""

    normalized_rgb: np.ndarray
    depth_z_m: np.ndarray
    cache_support: np.ndarray
    source_w2c: np.ndarray
    source_intrinsics: np.ndarray
    far_depth_pixel_fraction: np.ndarray


def import_cache4d(upstream_root: Path | None) -> Any:
    """Import installed Cache4D or an explicitly selected pinned checkout."""
    if upstream_root is not None:
        sys.path.insert(0, str(upstream_root))
    return importlib.import_module(
        "cosmos_predict1.diffusion.inference.cache_3d"
    )


def dense_cache_inputs(
    values: dict[str, np.ndarray],
    chunk_size: int,
) -> _DenseCacheInputs:
    """Gather and normalize one source row per global slot."""
    if chunk_size < 1:
        raise ValueError("unprojection chunk size must be positive")
    indices = values["source_index"]
    frame_count = indices.size
    _, height, width, _ = values["source_rgb"].shape
    image = np.empty((frame_count, 3, height, width), dtype=np.float32)
    depth = np.zeros((frame_count, 1, height, width), dtype=np.float32)
    cache_support = np.zeros_like(depth)
    far_fraction = np.empty(frame_count, dtype=np.float64)
    for start in range(0, frame_count, chunk_size):
        stop = min(start + chunk_size, frame_count)
        selected = indices[start:stop]
        rgb = np.asarray(values["source_rgb"][selected], dtype=np.float32)
        image[start:stop] = np.moveaxis(
            rgb * (2.0 / 255.0) - 1.0,
            -1,
            1,
        )
        source_depth = np.asarray(values["source_depth"][selected])
        source_valid = np.asarray(values["source_valid"][selected])
        support = source_valid & (source_depth > 0.0) & (
            source_depth <= _MAX_CACHE_DEPTH_M
        )
        np.copyto(depth[start:stop, 0], source_depth, where=support)
        cache_support[start:stop, 0] = support
        far_fraction[start:stop] = np.mean(
            source_valid & (source_depth > _MAX_CACHE_DEPTH_M),
            axis=(1, 2),
        )
    base = _DenseCacheInputs(
        normalized_rgb=image,
        depth_z_m=depth,
        cache_support=cache_support,
        source_w2c=np.asarray(
            values["source_w2c"][indices],
            dtype=np.float32,
        ),
        source_intrinsics=np.asarray(
            values["source_intrinsics"][indices],
            dtype=np.float32,
        ),
        far_depth_pixel_fraction=far_fraction,
    )
    if "context_rgb" not in values:
        return base

    context_index = values["context_index"]
    context_rgb = np.asarray(
        values["context_rgb"][context_index],
        dtype=np.float32,
    )
    context_depth = np.asarray(values["context_depth"][context_index])
    context_valid = np.asarray(values["context_valid"][context_index])
    context_support = context_valid & (context_depth > 0.0) & (
        context_depth <= _MAX_CACHE_DEPTH_M
    )
    context_depth_filtered = np.zeros(
        (frame_count, 1, height, width),
        dtype=np.float32,
    )
    np.copyto(
        context_depth_filtered[:, 0],
        context_depth,
        where=context_support,
    )
    return _DenseCacheInputs(
        normalized_rgb=np.stack(
            (
                base.normalized_rgb,
                np.moveaxis(
                    context_rgb * (2.0 / 255.0) - 1.0,
                    -1,
                    1,
                ),
            ),
            axis=1,
        ),
        depth_z_m=np.stack(
            (base.depth_z_m, context_depth_filtered),
            axis=1,
        ),
        cache_support=np.stack(
            (base.cache_support, context_support[:, None]),
            axis=1,
        ),
        source_w2c=np.stack(
            (
                base.source_w2c,
                np.asarray(
                    values["context_w2c"][context_index],
                    dtype=np.float32,
                ),
            ),
            axis=1,
        ),
        source_intrinsics=np.stack(
            (
                base.source_intrinsics,
                np.asarray(
                    values["context_intrinsics"][context_index],
                    dtype=np.float32,
                ),
            ),
            axis=1,
        ),
        far_depth_pixel_fraction=base.far_depth_pixel_fraction,
    )


def _unproject_points(
    module: Any,
    torch: Any,
    depth: np.ndarray,
    source_w2c: np.ndarray,
    source_intrinsics: np.ndarray,
    chunk_size: int,
    device: Any,
) -> Any:
    """Apply official unprojection in bounded GPU chunks."""
    height, width = depth.shape[-2:]
    leading_shape = depth.shape[:-3]
    flat_depth = depth.reshape(-1, 1, height, width)
    flat_w2c = source_w2c.reshape(-1, 4, 4)
    flat_intrinsics = source_intrinsics.reshape(-1, 3, 3)
    row_count = flat_depth.shape[0]
    result = torch.empty(
        (row_count, height, width, 3),
        dtype=torch.float32,
    )
    for start in range(0, row_count, chunk_size):
        stop = min(start + chunk_size, row_count)
        depth_chunk = torch.from_numpy(flat_depth[start:stop]).to(device)
        w2c_chunk = torch.from_numpy(flat_w2c[start:stop]).to(device)
        k_chunk = torch.from_numpy(flat_intrinsics[start:stop]).to(device)
        with torch.inference_mode():
            points = module.unproject_points(
                depth_chunk,
                w2c_chunk,
                k_chunk,
                is_depth=True,
            )
        result[start:stop].copy_(points.cpu())
        del depth_chunk, w2c_chunk, k_chunk, points
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return result.reshape(*leading_shape, height, width, 3)


def create_cache4d(
    module: Any,
    torch: Any,
    dense: _DenseCacheInputs,
    chunk_size: int,
    device: Any,
) -> Any:
    """Construct the pinned Cache4D with precomputed world points."""
    points = _unproject_points(
        module,
        torch,
        dense.depth_z_m,
        dense.source_w2c,
        dense.source_intrinsics,
        chunk_size,
        device,
    )
    depth_tensor = torch.from_numpy(dense.depth_z_m).to(device)
    input_format = ["F", "C", "H", "W"]
    if dense.normalized_rgb.ndim == 5:
        input_format = ["F", "N", "C", "H", "W"]
    return module.Cache4D(
        input_image=torch.from_numpy(dense.normalized_rgb),
        input_depth=depth_tensor,
        input_mask=torch.from_numpy(dense.cache_support),
        input_w2c=torch.from_numpy(dense.source_w2c),
        input_intrinsics=torch.from_numpy(dense.source_intrinsics),
        input_points=points,
        input_format=input_format,
        weight_dtype=torch.float32,
        is_depth=True,
        device=str(device),
        filter_points_threshold=_FILTER_POINTS_THRESHOLD,
        foreground_masking=False,
    )


def render_cache4d(
    cache: Any,
    torch: Any,
    query_w2c: np.ndarray,
    query_intrinsics: np.ndarray,
    start_frame_idx: int,
    device: Any,
) -> tuple[Any, Any]:
    """Render one caller-selected camera window through the raw model API."""
    w2c = torch.from_numpy(
        np.asarray(query_w2c, dtype=np.float32)
    ).unsqueeze(0).to(device)
    intrinsics = torch.from_numpy(
        np.asarray(query_intrinsics, dtype=np.float32)
    ).unsqueeze(0).to(device)
    with torch.inference_mode():
        rgb, mask = cache.render_cache(
            w2c,
            intrinsics,
            start_frame_idx=start_frame_idx,
        )
    frame_count = query_w2c.shape[0]
    buffer_count = int(cache.input_image.shape[2])
    height, width = cache.input_image.shape[-2:]
    expected_rgb = (1, frame_count, buffer_count, 3, height, width)
    expected_mask = (1, frame_count, buffer_count, 1, height, width)
    if tuple(rgb.shape) != expected_rgb or tuple(mask.shape) != expected_mask:
        raise RuntimeError(
            f"unexpected Cache4D shapes: {tuple(rgb.shape)}, "
            f"{tuple(mask.shape)}"
        )
    return rgb, mask


__all__ = [
    "create_cache4d",
    "dense_cache_inputs",
    "import_cache4d",
    "render_cache4d",
]

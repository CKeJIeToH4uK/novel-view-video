"""The single transient NPY protocol for Cache4D conditioning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from novel_view.models.gen3c.cache4d.request import (
    Gen3cConditioning,
    Gen3cSourceRows,
)

_BASE_FILES = {
    "source_rgb": "source_rgb.npy",
    "source_depth": "source_depth_z_m.npy",
    "source_valid": "source_geometry_valid.npy",
    "source_intrinsics": "source_intrinsics.npy",
    "source_w2c": "anchor_to_source_camera.npy",
    "source_index": "source_sequence_index.npy",
    "query_w2c": "anchor_to_query_camera.npy",
    "query_intrinsics": "query_intrinsics.npy",
    "selected_slots": "selected_global_slots.npy",
}
_CONTEXT_FILES = {
    "context_rgb": "context_rgb.npy",
    "context_depth": "context_depth_z_m.npy",
    "context_valid": "context_valid.npy",
    "context_w2c": "anchor_to_context_camera.npy",
    "context_intrinsics": "context_intrinsics.npy",
    "context_index": "context_sequence_index.npy",
}
_DIAGNOSTIC_FILES = {
    "coverage": "coverage_fraction.npy",
    "far_depth": "far_depth_pixel_fraction.npy",
    "rgb": "selected_rgb.npy",
    "mask": "selected_coverage_mask.npy",
    "overlap": "overlap_errors.npy",
    "peaks": "peak_cuda_bytes.npy",
    "elapsed": "elapsed_seconds.npy",
}


def write_cache4d_conditioning(
    root: Path,
    conditioning: Gen3cConditioning,
) -> npt.NDArray[np.int64]:
    """Stream one source pack and save its lightweight camera schedule."""
    _write_rows(
        root,
        conditioning.source_rows,
        _BASE_FILES["source_rgb"],
        _BASE_FILES["source_depth"],
        _BASE_FILES["source_valid"],
    )
    for name, value in (
        ("source_intrinsics", conditioning.source_intrinsics),
        ("source_w2c", conditioning.anchor_to_source_camera),
        ("source_index", conditioning.source_sequence_index),
        ("query_w2c", conditioning.anchor_to_query_camera),
        ("query_intrinsics", conditioning.query_intrinsics),
        ("selected_slots", conditioning.selected_global_slots),
    ):
        np.save(root / _BASE_FILES[name], value, allow_pickle=False)

    context = conditioning.context
    if context is not None:
        _write_rows(
            root,
            context.rows,
            _CONTEXT_FILES["context_rgb"],
            _CONTEXT_FILES["context_depth"],
            _CONTEXT_FILES["context_valid"],
        )
        for name, value in (
            ("context_w2c", context.anchor_to_context_camera),
            ("context_intrinsics", context.context_intrinsics),
            ("context_index", context.context_sequence_index),
        ):
            np.save(root / _CONTEXT_FILES[name], value, allow_pickle=False)
    return conditioning.selected_global_slots


def read_cache4d_conditioning(root: Path) -> dict[str, np.ndarray]:
    """Open the trusted base exchange and its optional context layer."""
    values = {
        name: np.load(root / filename, mmap_mode="r", allow_pickle=False)
        for name, filename in _BASE_FILES.items()
    }
    if (root / _CONTEXT_FILES["context_rgb"]).is_file():
        values.update(
            {
                name: np.load(
                    root / filename,
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for name, filename in _CONTEXT_FILES.items()
            }
        )
    return values


def write_cache4d_diagnostic_result(
    root: Path,
    values: dict[str, np.ndarray],
) -> None:
    """Write the standalone worker's numeric result."""
    for name, filename in _DIAGNOSTIC_FILES.items():
        np.save(root / filename, values[name], allow_pickle=False)


def read_cache4d_diagnostic_result(
    root: Path,
) -> dict[str, np.ndarray]:
    """Open the trusted standalone worker result without copying it."""
    return {
        name: np.load(root / filename, mmap_mode="r", allow_pickle=False)
        for name, filename in _DIAGNOSTIC_FILES.items()
    }


def _write_rows(
    root: Path,
    rows: Gen3cSourceRows,
    rgb_filename: str,
    depth_filename: str,
    valid_filename: str,
) -> None:
    first = rows.read(0)
    height, width = first.depth_z_m.shape
    arrays = tuple(
        np.lib.format.open_memmap(
            root / filename,
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        for filename, dtype, shape in (
            (rgb_filename, np.uint8, (len(rows), height, width, 3)),
            (depth_filename, np.float32, (len(rows), height, width)),
            (valid_filename, np.bool_, (len(rows), height, width)),
        )
    )
    try:
        arrays[0][0] = first.rgb
        arrays[1][0] = first.depth_z_m
        arrays[2][0] = first.valid
        del first
        for index in range(1, len(rows)):
            frame = rows.read(index)
            arrays[0][index] = frame.rgb
            arrays[1][index] = frame.depth_z_m
            arrays[2][index] = frame.valid
            del frame
        for array in arrays:
            array.flush()
    finally:
        for array in arrays:
            array._mmap.close()


__all__ = [
    "read_cache4d_conditioning",
    "read_cache4d_diagnostic_result",
    "write_cache4d_conditioning",
    "write_cache4d_diagnostic_result",
]

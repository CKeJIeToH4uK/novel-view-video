"""The one transient NPY protocol used by standalone MoGe-v1."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from novel_view.models.moge.request import (
    MogeExecution,
    MogeRawPrediction,
    MogeRequest,
    MogeTelemetry,
)


def write_moge_request(root: Path, request: MogeRequest) -> None:
    """Write the ordered RGB stream and measured FOV once."""
    height, width = request.image_size_hw
    output = np.lib.format.open_memmap(
        root / "rgb.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(request.frame_count, height, width, 3),
    )
    try:
        for index, frame in zip(range(request.frame_count), request.rgb_frames, strict=True):
            output[index] = frame
        output.flush()
    finally:
        output._mmap.close()
    np.save(root / "fov_x_degrees.npy", request.fov_x_degrees, allow_pickle=False)


def read_moge_execution(root: Path) -> MogeExecution:
    """Copy raw worker values out of their transient memory maps."""
    values = {
        name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "depth_model_units",
            "model_valid",
            "model_intrinsics",
            "peak_ram_mib",
            "peak_vram_mib",
        )
    }
    try:
        return MogeExecution(
            prediction=MogeRawPrediction(
                depth_model_units=cast(
                    npt.NDArray[np.float32], _owned(values["depth_model_units"])
                ),
                valid=cast(
                    npt.NDArray[np.bool_], _owned(values["model_valid"])
                ),
                intrinsics=cast(
                    npt.NDArray[np.float32], _owned(values["model_intrinsics"])
                ),
            ),
            telemetry=MogeTelemetry(
                peak_ram_mib=float(values["peak_ram_mib"]),
                peak_vram_mib=float(values["peak_vram_mib"]),
            ),
        )
    finally:
        for value in values.values():
            if isinstance(value, np.memmap):
                value._mmap.close()


def _owned(value: npt.NDArray[np.generic]) -> npt.NDArray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


__all__ = ["read_moge_execution", "write_moge_request"]

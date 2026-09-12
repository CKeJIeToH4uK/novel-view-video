"""The single transient NPY protocol for raw VGGT-Omega inference."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from novel_view.models.vggt.request import (
    VggtOmegaExecution,
    VggtOmegaRawPrediction,
    VggtOmegaRequest,
    VggtOmegaTelemetry,
)


INPUT_RGB_FILE = "rgb.npy"
RAW_OUTPUT_FILES = {
    "depth_model_units": "depth_model_units.npy",
    "depth_confidence": "depth_confidence.npy",
    "pose_encoding": "pose_encoding.npy",
    "model_w2c": "model_w2c.npy",
    "model_intrinsics": "model_intrinsics.npy",
    "peak_ram_mib": "peak_ram_mib.npy",
    "peak_vram_mib": "peak_vram_mib.npy",
}


def write_vggt_omega_request(root: Path, request: VggtOmegaRequest) -> None:
    """Stream the ordered RGB sequence once into the model exchange."""
    height, width = request.input_plan.source_size_hw
    output = np.lib.format.open_memmap(
        root / INPUT_RGB_FILE,
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


def read_vggt_omega_execution(
    root: Path,
    request: VggtOmegaRequest,
) -> VggtOmegaExecution:
    """Copy the seven trusted worker outputs out of transient memory maps."""
    values = {
        name: np.load(root / filename, mmap_mode="r", allow_pickle=False)
        for name, filename in RAW_OUTPUT_FILES.items()
    }
    try:
        prediction = VggtOmegaRawPrediction(
            input_plan=request.input_plan,
            depth_model_units=cast(
                npt.NDArray[np.float32], _owned(values["depth_model_units"])
            ),
            depth_confidence=cast(
                npt.NDArray[np.float32], _owned(values["depth_confidence"])
            ),
            pose_encoding=cast(
                npt.NDArray[np.float32], _owned(values["pose_encoding"])
            ),
            model_w2c=cast(
                npt.NDArray[np.float32], _owned(values["model_w2c"])
            ),
            model_intrinsics=cast(
                npt.NDArray[np.float32], _owned(values["model_intrinsics"])
            ),
        )
        return VggtOmegaExecution(
            prediction=prediction,
            telemetry=VggtOmegaTelemetry(
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


__all__ = [
    "INPUT_RGB_FILE",
    "RAW_OUTPUT_FILES",
    "read_vggt_omega_execution",
    "write_vggt_omega_request",
]

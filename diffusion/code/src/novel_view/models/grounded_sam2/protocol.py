"""Transient native RGB and union-mask protocol for Grounded-SAM2."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from novel_view.models.grounded_sam2.request import (
    GroundedSam2Prediction,
    GroundedSam2Request,
)


def write_grounded_sam2_request(
    root: Path,
    request: GroundedSam2Request,
) -> None:
    """Stream ordered native RGB once into the model exchange."""
    height, width = request.image_size_hw
    output = np.lib.format.open_memmap(
        root / "native_rgb.npy",
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


def read_grounded_sam2_prediction(root: Path) -> GroundedSam2Prediction:
    """Detach the trusted worker result from its transient memory map."""
    value = np.load(root / "dynamic_mask.npy", mmap_mode="r", allow_pickle=False)
    try:
        result = np.array(value, copy=True, order="C")
        result.setflags(write=False)
        return GroundedSam2Prediction(
            cast(npt.NDArray[np.bool_], result)
        )
    finally:
        if isinstance(value, np.memmap):
            value._mmap.close()


__all__ = [
    "read_grounded_sam2_prediction",
    "write_grounded_sam2_request",
]

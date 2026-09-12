"""Source-neutral values crossing the standalone MoGe-v1 boundary."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, eq=False)
class MogeRequest:
    """One ordered RGB sequence with its already measured horizontal FOV."""

    rgb_frames: Iterable[npt.NDArray[np.uint8]]
    fov_x_degrees: npt.NDArray[np.float64]
    frame_count: int
    image_size_hw: tuple[int, int]


@dataclass(frozen=True, slots=True, eq=False)
class MogeRawPrediction:
    """Raw model-unit depth, validity, and normalized intrinsics."""

    depth_model_units: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]
    intrinsics: npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class MogeTelemetry:
    """Peak process memory reported by one standalone invocation."""

    peak_ram_mib: float
    peak_vram_mib: float


@dataclass(frozen=True, slots=True, eq=False)
class MogeExecution:
    """A raw prediction and its process telemetry."""

    prediction: MogeRawPrediction
    telemetry: MogeTelemetry


__all__ = [
    "MogeExecution",
    "MogeRawPrediction",
    "MogeRequest",
    "MogeTelemetry",
]

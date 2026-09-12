"""Light source-neutral request and result types for VGGT-Omega."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from novel_view.models.vggt.spec import VggtOmegaInputPlan


@dataclass(frozen=True, slots=True, eq=False)
class VggtOmegaRequest:
    """One ordered RGB sequence and its already selected model grid."""

    rgb_frames: Iterable[npt.NDArray[np.uint8]]
    frame_count: int
    input_plan: VggtOmegaInputPlan


@dataclass(frozen=True, slots=True, eq=False)
class VggtOmegaRawPrediction:
    """Owned raw arrays in VGGT's camera frame and unknown depth gauge."""

    input_plan: VggtOmegaInputPlan
    depth_model_units: npt.NDArray[np.float32]
    depth_confidence: npt.NDArray[np.float32]
    pose_encoding: npt.NDArray[np.float32]
    model_w2c: npt.NDArray[np.float32]
    model_intrinsics: npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class VggtOmegaTelemetry:
    """Peak process memory reported by one model invocation."""

    peak_ram_mib: float
    peak_vram_mib: float


@dataclass(frozen=True, slots=True, eq=False)
class VggtOmegaExecution:
    """The raw model result and process telemetry kept as separate facts."""

    prediction: VggtOmegaRawPrediction
    telemetry: VggtOmegaTelemetry


__all__ = [
    "VggtOmegaExecution",
    "VggtOmegaRawPrediction",
    "VggtOmegaRequest",
    "VggtOmegaTelemetry",
]

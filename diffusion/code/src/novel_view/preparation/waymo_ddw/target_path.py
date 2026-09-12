"""Cosine target-camera path for Waymo DDW preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


_FRAME_COUNT = 121


@dataclass(frozen=True, slots=True, eq=False)
class DdwPath:
    """Lateral displacement and the corresponding virtual W2C cameras."""

    displacement_m: npt.NDArray[np.float64]
    virtual_world_to_camera_cv: npt.NDArray[np.float64]


def build_target_path(
    source_world_to_camera_cv: npt.NDArray[np.float64],
    magnitude_m: float,
    sign: int,
) -> DdwPath:
    """Move the measured camera smoothly along its local X axis."""
    phase = np.arange(_FRAME_COUNT, dtype=np.float64) * (np.pi / 120.0)
    displacement = sign * magnitude_m * (1.0 - np.cos(phase)) / 2.0
    virtual = source_world_to_camera_cv.copy()
    virtual[:, 0, 3] -= displacement
    return DdwPath(displacement, virtual)


__all__ = ["DdwPath", "build_target_path"]

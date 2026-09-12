"""VGGT camera-pack alignment over explicit predicted and measured W2C."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from novel_view.geometry.sim3 import CameraPackAlignment, align_camera_packs


def align_vggt_cameras(
    predicted_w2c: npt.NDArray[np.float32],
    measured_w2c: npt.NDArray[np.float64],
) -> CameraPackAlignment:
    """Fit VGGT's predicted camera world to one measured metric world."""
    return align_camera_packs(predicted_w2c, measured_w2c)


__all__ = ["align_vggt_cameras"]

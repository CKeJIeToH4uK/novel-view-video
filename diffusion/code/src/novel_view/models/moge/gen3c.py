"""Pinned in-process MoGe seam used by resident Gen3C generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from novel_view.models.moge._inference import load_moge_model


@dataclass(frozen=True, slots=True, eq=False)
class MogeContextDepth:
    """One raw Cosmos-helper depth and validity plane."""

    depth_model_units: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]


def load_gen3c_moge_predictor() -> Callable:
    """Import the exact helper shipped by the pinned Cosmos checkout."""
    from cosmos_predict1.diffusion.inference.gen3c_single_image import (
        _predict_moge_depth,
    )

    return _predict_moge_depth


def predict_gen3c_context_depth(
    predictor: Callable,
    rgb: npt.NDArray[np.uint8],
    height: int,
    width: int,
    device: Any,
    model: Any,
) -> MogeContextDepth:
    """Run one slot and preserve Cosmos' leading-dimension extraction."""
    _, depth, valid, _, _ = predictor(rgb, height, width, device, model)
    return MogeContextDepth(
        depth_model_units=depth[0, 0, 0].float().cpu().numpy(),
        valid=valid[0, 0, 0].cpu().numpy(),
    )


__all__ = [
    "MogeContextDepth",
    "load_gen3c_moge_predictor",
    "load_moge_model",
    "predict_gen3c_context_depth",
]

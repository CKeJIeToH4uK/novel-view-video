"""Fixed identity and three proven image-grid recipes for VGGT-Omega."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


VGGT_OMEGA_BACKEND = "vggt-omega"
_PATCH_SIZE = 16
_DIRECT_SIZE_HW = (704, 1_280)


def vggt_usage_note(checkpoint: str) -> str:
    """Информация о конкретном VGGT-входе, без проверки прав или файлов."""
    return (
        f"VGGT-Omega ({checkpoint}): код — FAIR Noncommercial Research License, "
        "только некоммерческое исследование. Условия доступа к выбранным весам "
        "проверяются отдельно; подробности — docs/licensing.md."
    )


class VggtOmegaError(RuntimeError):
    """One VGGT-Omega request cannot be prepared or executed."""


class VggtOmegaInputMode(str, Enum):
    """The three image-grid recipes already used by the project."""

    BALANCED_512 = "balanced-512"
    BALANCED_896 = "balanced-896"
    DIRECT_704_1280 = "direct-704x1280"


@dataclass(frozen=True, slots=True, eq=False)
class VggtOmegaInputPlan:
    """Exact relation between the caller raster and VGGT input grid."""

    mode: VggtOmegaInputMode
    source_size_hw: tuple[int, int]
    model_size_hw: tuple[int, int]
    source_to_model_pixels: npt.NDArray[np.float64]


def build_vggt_omega_input_plan(
    source_size_hw: tuple[int, int],
    mode: VggtOmegaInputMode,
) -> VggtOmegaInputPlan:
    """Resolve one explicit recipe without hiding the official crop."""
    aspect_ratio = source_size_hw[0] / source_size_hw[1]
    if not 0.5 <= aspect_ratio <= 2.0:
        raise VggtOmegaError(
            "source aspect ratio must be within [0.5, 2.0]; this adapter "
            "does not hide the official loader's center crop"
        )
    if mode is VggtOmegaInputMode.DIRECT_704_1280:
        if source_size_hw != _DIRECT_SIZE_HW:
            raise VggtOmegaError(
                f"{mode.value} requires source_size_hw {_DIRECT_SIZE_HW}"
            )
        model_size_hw = source_size_hw
    else:
        resolution = 512 if mode is VggtOmegaInputMode.BALANCED_512 else 896
        token_count = (resolution // _PATCH_SIZE) ** 2
        width_patches_float = math.sqrt(token_count / aspect_ratio)
        height_patches_float = token_count / width_patches_float
        width_patches = max(1, int(np.round(width_patches_float)))
        height_patches = max(1, int(np.round(height_patches_float)))
        model_size_hw = (
            height_patches * _PATCH_SIZE,
            width_patches * _PATCH_SIZE,
        )

    source_height, source_width = source_size_hw
    model_height, model_width = model_size_hw
    transform = np.array(
        [
            [model_width / source_width, 0.0, 0.0],
            [0.0, model_height / source_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform.setflags(write=False)
    return VggtOmegaInputPlan(
        mode=mode,
        source_size_hw=source_size_hw,
        model_size_hw=model_size_hw,
        source_to_model_pixels=transform,
    )


__all__ = [
    "VGGT_OMEGA_BACKEND",
    "VggtOmegaError",
    "VggtOmegaInputMode",
    "VggtOmegaInputPlan",
    "build_vggt_omega_input_plan",
    "vggt_usage_note",
]

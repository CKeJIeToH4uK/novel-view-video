"""Exact in-process DINOv2 ViT-B/14 load and patch features."""

from __future__ import annotations

import sys
from pathlib import Path

DINOV2_INPUT_SIZE_HW = (462, 840)
DINOV2_GRID_SIZE_HW = (33, 60)


class Dinov2Vitb14:
    """One resident pinned DINOv2 model returning raw patch tokens."""

    def __init__(self, repository: Path, checkpoint: Path) -> None:
        import torch

        repository_entry = str(repository)
        sys.path.insert(0, repository_entry)
        try:
            from dinov2.hub.backbones import dinov2_vitb14
        finally:
            sys.path.remove(repository_entry)

        model = dinov2_vitb14(pretrained=False)
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        self._model = model.eval().cuda()
        self._torch = torch

    def patch_features(self, pair_01):
        """Return normalized row-major patch tokens for one RGB pair."""
        resized = self._torch.nn.functional.interpolate(
            pair_01,
            size=DINOV2_INPUT_SIZE_HW,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        mean = resized.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        standard_deviation = resized.new_tensor(
            (0.229, 0.224, 0.225)
        )[None, :, None, None]
        normalized = (resized - mean) / standard_deviation
        return self._model.get_intermediate_layers(
            normalized,
            n=1,
            norm=True,
        )[0]


__all__ = [
    "DINOV2_GRID_SIZE_HW",
    "DINOV2_INPUT_SIZE_HW",
    "Dinov2Vitb14",
]

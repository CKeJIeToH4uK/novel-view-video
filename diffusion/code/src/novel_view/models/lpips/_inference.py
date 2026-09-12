"""Exact in-process LPIPS AlexNet construction and spatial forward."""

from __future__ import annotations

import os
from pathlib import Path


class LpipsAlexNet:
    """One resident LPIPS 0.1 AlexNet model for spatial comparisons."""

    def __init__(self, torch_home: Path) -> None:
        os.environ["TORCH_HOME"] = str(torch_home)

        import lpips

        self._model = lpips.LPIPS(
            net="alex",
            version="0.1",
            spatial=True,
            eval_mode=True,
        ).eval().cuda()

    def spatial_map(self, pair_01):
        """Compare one ordered RGB pair supplied in the ``[0, 1]`` range."""
        return self._model(
            pair_01[0:1] * 2.0 - 1.0,
            pair_01[1:2] * 2.0 - 1.0,
            normalize=False,
        )


__all__ = ["LpipsAlexNet"]

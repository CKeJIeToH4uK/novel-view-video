"""Heavy MoGe-v1 imports and direct raw model inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from novel_view.models.moge.spec import NUM_TOKENS


def load_moge_model(checkpoint: Path, device: Any) -> Any:
    """Load one resident MoGe-v1 model on the caller-owned device."""
    from moge.model.v1 import MoGeModel

    return MoGeModel.from_pretrained(checkpoint).to(device).eval()


def infer_moge_frame(model: Any, image: Any, fov_x_degrees: float) -> dict:
    """Run the pinned standalone recipe and return the three raw keys."""
    return model.infer(
        image,
        fov_x=float(fov_x_degrees),
        num_tokens=NUM_TOKENS,
        apply_mask=False,
        force_projection=True,
        use_fp16=True,
    )


__all__ = ["infer_moge_frame", "load_moge_model"]

"""Private heavy VGGT-Omega process; core modules never import this file."""

from __future__ import annotations

import argparse
import math
import resource
from pathlib import Path

import numpy as np

from novel_view.models.vggt.protocol import INPUT_RGB_FILE, RAW_OUTPUT_FILES


_PATCH_SIZE = 16


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    return parser.parse_args()


def _target_size(
    source_size_hw: tuple[int, int],
    mode: str,
) -> tuple[int, int]:
    height, width = source_size_hw
    if mode == "direct-704x1280":
        return source_size_hw
    resolution = 512 if mode == "balanced-512" else 896
    token_count = (resolution // _PATCH_SIZE) ** 2
    width_patches_float = math.sqrt(token_count / (height / width))
    height_patches_float = token_count / width_patches_float
    width_patches = max(1, int(np.round(width_patches_float)))
    height_patches = max(1, int(np.round(height_patches_float)))
    return height_patches * _PATCH_SIZE, width_patches * _PATCH_SIZE


def _preprocess_images(rgb: np.ndarray, target_size_hw: tuple[int, int]):
    """Apply the pinned official Pillow bicubic and ToTensor operations."""
    import torch
    from PIL import Image
    from torchvision import transforms

    target_height, target_width = target_size_hw
    resize = rgb.shape[1:3] != target_size_hw
    to_tensor = transforms.ToTensor()
    images = []
    for frame in rgb:
        image = Image.fromarray(np.asarray(frame), mode="RGB")
        if resize:
            image = image.resize(
                (target_width, target_height),
                Image.Resampling.BICUBIC,
            )
        images.append(to_tensor(image))
    return torch.stack(images)


def _run(arguments: argparse.Namespace) -> None:
    import torch

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    rgb = np.load(
        arguments.exchange / INPUT_RGB_FILE,
        mmap_mode="r",
        allow_pickle=False,
    )
    images = _preprocess_images(rgb, _target_size(rgb.shape[1:3], arguments.mode))

    model = VGGTOmega()
    state_dict = torch.load(
        arguments.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model = model.to("cuda").eval()
    images = images.to("cuda")
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        predictions = model(images)
        pose_encoding = predictions["pose_enc"]
        model_w2c, model_intrinsics = encoding_to_camera(
            pose_encoding,
            predictions["images"].shape[-2:],
        )
    torch.cuda.synchronize()

    values = {
        "depth_model_units": predictions["depth"][0, ..., 0].cpu().numpy(),
        "depth_confidence": predictions["depth_conf"][0].cpu().numpy(),
        "pose_encoding": pose_encoding[0].cpu().numpy(),
        "model_w2c": model_w2c[0].cpu().numpy(),
        "model_intrinsics": model_intrinsics[0].cpu().numpy(),
        "peak_vram_mib": np.asarray(
            torch.cuda.max_memory_reserved() / (1024.0**2),
            dtype=np.float64,
        ),
        "peak_ram_mib": np.asarray(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            dtype=np.float64,
        ),
    }
    for name, filename in RAW_OUTPUT_FILES.items():
        np.save(arguments.exchange / filename, values[name], allow_pickle=False)
    print(
        f"processed={len(rgb)} "
        f"peak_ram_mib={float(values['peak_ram_mib']):.3f} "
        f"peak_vram_mib={float(values['peak_vram_mib']):.3f}",
        flush=True,
    )


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()

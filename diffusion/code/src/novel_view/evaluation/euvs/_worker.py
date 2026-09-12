"""Private one-request LPIPS AlexNet and DINOv2 metric worker."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from novel_view.metrics.image import (
    fractional_patch_support,
    masked_average,
    weighted_average,
)
from novel_view.models.dinov2._inference import (
    DINOV2_GRID_SIZE_HW,
    Dinov2Vitb14,
)
from novel_view.models.lpips._inference import LpipsAlexNet


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-rgb", type=Path, required=True)
    parser.add_argument("--input-support", type=Path, required=True)
    parser.add_argument("--output-perceptual", type=Path, required=True)
    parser.add_argument("--dinov2-repository", type=Path, required=True)
    parser.add_argument("--dinov2-checkpoint", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    return parser.parse_args()


def _run(arguments: argparse.Namespace) -> None:
    import torch

    rgb = np.load(arguments.input_rgb, mmap_mode="r", allow_pickle=False)
    support = np.load(
        arguments.input_support,
        mmap_mode="r",
        allow_pickle=False,
    )
    dino = Dinov2Vitb14(
        arguments.dinov2_repository,
        arguments.dinov2_checkpoint,
    )
    lpips_model = LpipsAlexNet(arguments.torch_home)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output = np.lib.format.open_memmap(
        arguments.output_perceptual,
        mode="w+",
        dtype=np.float64,
        shape=(rgb.shape[0], 2, 3),
    )
    output.fill(np.nan)
    try:
        for frame_index in range(rgb.shape[0]):
            pair = torch.from_numpy(
                np.array(rgb[frame_index], copy=True, order="C")
            ).permute(0, 3, 1, 2).float().cuda() / 255.0
            with torch.inference_mode():
                lpips_map = lpips_model.spatial_map(pair)[0, 0]
                features = dino.patch_features(pair)
                cosine = torch.nn.functional.cosine_similarity(
                    features[0],
                    features[1],
                    dim=1,
                    eps=1e-8,
                ).clamp(-1.0, 1.0)
            lpips_values = lpips_map.float().cpu().numpy()
            cosine_values = cosine.float().cpu().numpy()
            for track_index in range(2):
                track_support = np.asarray(
                    support[frame_index, track_index],
                    dtype=np.bool_,
                )
                lpips, _ = masked_average(lpips_values, track_support)
                if lpips is not None:
                    output[frame_index, track_index, 0] = lpips
                patch_weights = fractional_patch_support(
                    track_support,
                    DINOV2_GRID_SIZE_HW,
                )
                dino_value, mass = weighted_average(
                    cosine_values,
                    patch_weights.ravel(),
                )
                output[frame_index, track_index, 2] = mass
                if dino_value is not None:
                    output[frame_index, track_index, 1] = dino_value
        output.flush()
    finally:
        output._mmap.close()
        if isinstance(rgb, np.memmap):
            rgb._mmap.close()
        if isinstance(support, np.memmap):
            support._mmap.close()


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()

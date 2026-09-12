"""Internal worker for the frozen DA3 Nested research prototype."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _arguments() -> argparse.Namespace:
    """Parse the private one-request worker protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-rgb", type=Path, required=True)
    parser.add_argument("--input-extrinsics", type=Path, required=True)
    parser.add_argument("--input-intrinsics", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--process-resolution", type=int, required=True)
    parser.add_argument("--expected-height", type=int, required=True)
    parser.add_argument("--expected-width", type=int, required=True)
    return parser.parse_args()


def _load_inputs(
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the numeric request without pickle."""
    rgb = np.load(
        arguments.input_rgb,
        mmap_mode="r",
        allow_pickle=False,
    )
    extrinsics = np.load(
        arguments.input_extrinsics,
        allow_pickle=False,
    )
    intrinsics = np.load(
        arguments.input_intrinsics,
        allow_pickle=False,
    )
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.dtype(np.uint8)
        or rgb.ndim != 4
        or rgb.shape[0] == 0
        or rgb.shape[-1] != 3
    ):
        raise ValueError(
            "input RGB must be non-empty uint8 [N,H,W,3]"
        )
    frame_count = rgb.shape[0]
    if (
        not isinstance(extrinsics, np.ndarray)
        or extrinsics.dtype != np.dtype(np.float64)
        or extrinsics.shape != (frame_count, 4, 4)
        or not np.all(np.isfinite(extrinsics))
    ):
        raise ValueError(
            "input extrinsics must be finite float64 [N,4,4]"
        )
    if (
        not isinstance(intrinsics, np.ndarray)
        or intrinsics.dtype != np.dtype(np.float64)
        or intrinsics.shape != (frame_count, 3, 3)
        or not np.all(np.isfinite(intrinsics))
    ):
        raise ValueError(
            "input intrinsics must be finite float64 [N,3,3]"
        )
    return rgb, extrinsics, intrinsics


def _run(arguments: argparse.Namespace) -> None:
    """Load the local checkpoint, run one sequence, and save raw arrays."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from depth_anything_3.api import DepthAnything3

    if not torch.cuda.is_available():
        raise RuntimeError("DA3 Nested worker requires CUDA")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    rgb, extrinsics, intrinsics = _load_inputs(arguments)
    model = DepthAnything3.from_pretrained(
        str(arguments.model_directory)
    )
    model = model.to(device="cuda").eval()

    prediction = model.inference(
        image=[np.asarray(frame) for frame in rgb],
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=True,
        infer_gs=False,
        use_ray_pose=False,
        process_res=arguments.process_resolution,
        process_res_method="upper_bound_resize",
        export_dir=None,
    )

    expected_shape = (
        rgb.shape[0],
        arguments.expected_height,
        arguments.expected_width,
    )
    if prediction.depth.shape != expected_shape:
        raise RuntimeError(
            f"DA3 depth shape {prediction.depth.shape} differs from "
            f"{expected_shape}"
        )
    if (
        prediction.processed_images is None
        or prediction.processed_images.shape
        != (*expected_shape, 3)
    ):
        raise RuntimeError(
            "DA3 processed image grid differs from the declared model grid"
        )
    if prediction.conf is None:
        raise RuntimeError("DA3 Nested did not return confidence")
    if prediction.extrinsics is None:
        raise RuntimeError(
            "DA3 Nested did not return conditioned extrinsics"
        )
    if prediction.intrinsics is None:
        raise RuntimeError(
            "DA3 Nested did not return conditioned intrinsics"
        )

    sky = prediction.sky
    has_sky = sky is not None
    if sky is None:
        sky = np.zeros(expected_shape, dtype=np.bool_)

    output_directory = arguments.output_directory
    np.save(
        output_directory / "depth_z_m.npy",
        np.asarray(prediction.depth, dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        output_directory / "confidence.npy",
        np.asarray(prediction.conf, dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        output_directory / "sky_mask.npy",
        np.asarray(sky, dtype=np.bool_),
        allow_pickle=False,
    )
    np.save(
        output_directory / "has_sky.npy",
        np.asarray(has_sky, dtype=np.bool_),
        allow_pickle=False,
    )
    np.save(
        output_directory / "conditioned_w2c.npy",
        np.asarray(prediction.extrinsics, dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        output_directory / "conditioned_intrinsics.npy",
        np.asarray(prediction.intrinsics, dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        output_directory / "is_metric.npy",
        np.asarray(bool(prediction.is_metric), dtype=np.bool_),
        allow_pickle=False,
    )


def main() -> None:
    """Execute exactly one isolated DA3 Nested inference request."""
    _run(_arguments())


if __name__ == "__main__":
    main()

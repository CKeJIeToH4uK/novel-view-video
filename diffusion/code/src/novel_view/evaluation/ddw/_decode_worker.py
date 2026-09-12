"""Private one-GPU worker for VAE decode, MoGe, metrics, and video."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from novel_view.evaluation.ddw.metrics import (
    SparseEvaluationLidar,
    VARIANTS,
    evaluate_ddw_item,
    write_four_panel_video,
)
from novel_view.evaluation.ddw.record import DdwItemResult, write_item_result
from novel_view.evaluation.ddw.sampling import read_temporary_latent
from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT
from novel_view.preparation.waymo_ddw.artifacts import read_lidar_depth


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--lidar-depth", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--moge-checkpoint", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-locator", required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _run(arguments: argparse.Namespace) -> None:
    import torch

    started = time.monotonic()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    target = np.load(arguments.target, mmap_mode="r", allow_pickle=False)
    if (
        target.dtype != np.uint8
        or target.shape != R4C_GEN3C_MODEL_CONTRACT.rgb_thwc_shape
    ):
        raise ValueError("DDW evaluation target has the wrong array contract")
    latents = {
        variant: read_temporary_latent(
            arguments.latent_root / f"{variant}.pt",
            arguments.sample_id,
            variant,
        )
        for variant in VARIANTS
    }
    vae = _load_vae(arguments.tokenizer)
    decoded_paths = {}
    with torch.inference_mode():
        for variant in VARIANTS:
            decoded = _decode_latent(torch, vae, latents[variant], device)
            path = arguments.scratch / f"{variant}-rgb.npy"
            np.save(path, decoded, allow_pickle=False)
            decoded_paths[variant] = path
            del decoded
    del vae, latents
    gc.collect()
    torch.cuda.empty_cache()

    lidar_artifact = read_lidar_depth(arguments.lidar_depth, arguments.sample_id)
    lidar = SparseEvaluationLidar(
        lidar_artifact.frame_offsets.numpy(),
        lidar_artifact.canvas_xy.numpy(),
        lidar_artifact.depth_z_m.numpy(),
        lidar_artifact.evaluation_holdout.numpy(),
    )
    intrinsics = lidar_artifact.K_canvas.numpy()
    width = target.shape[2]
    fov_x_degrees = np.degrees(2.0 * np.arctan(width / (2.0 * intrinsics[:, 0, 0])))
    from novel_view.models.moge._inference import (
        infer_moge_frame,
        load_moge_model,
    )

    moge = load_moge_model(arguments.moge_checkpoint, device)
    depth_paths = {}
    valid_paths = {}
    for variant in VARIANTS:
        rgb = np.load(decoded_paths[variant], mmap_mode="r", allow_pickle=False)
        depth_path = arguments.scratch / f"{variant}-depth.npy"
        valid_path = arguments.scratch / f"{variant}-valid.npy"
        depth = np.lib.format.open_memmap(
            depth_path,
            mode="w+",
            dtype=np.float32,
            shape=target.shape[:3],
        )
        valid = np.lib.format.open_memmap(
            valid_path,
            mode="w+",
            dtype=np.bool_,
            shape=target.shape[:3],
        )
        for frame_index in range(target.shape[0]):
            image = (
                torch.from_numpy(np.array(rgb[frame_index], copy=True))
                .permute(2, 0, 1)
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
            )
            prediction = infer_moge_frame(
                moge,
                image,
                float(fov_x_degrees[frame_index]),
            )
            depth[frame_index] = prediction["depth"].detach().cpu().numpy()
            valid[frame_index] = prediction["mask"].detach().cpu().numpy()
        depth.flush()
        valid.flush()
        depth_paths[variant] = depth_path
        valid_paths[variant] = valid_path
        del depth, valid, rgb
    torch.cuda.synchronize()
    del moge
    gc.collect()
    torch.cuda.empty_cache()

    decoded = {
        variant: np.load(decoded_paths[variant], mmap_mode="r", allow_pickle=False)
        for variant in VARIANTS
    }
    depth = {
        variant: np.load(depth_paths[variant], mmap_mode="r", allow_pickle=False)
        for variant in VARIANTS
    }
    valid = {
        variant: np.load(valid_paths[variant], mmap_mode="r", allow_pickle=False)
        for variant in VARIANTS
    }
    metrics = evaluate_ddw_item(
        arguments.sample_id,
        target,
        decoded,
        depth,
        valid,
        lidar,
    )
    write_four_panel_video(arguments.video, target, decoded, lidar, fps=10)
    write_item_result(
        arguments.result,
        DdwItemResult(arguments.sample_id, arguments.video_locator, metrics),
    )
    print(
        json.dumps(
            {
                "event": "ddw_evaluation_item_complete",
                "sample_id": arguments.sample_id,
                "video": arguments.video_locator,
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


def _load_vae(tokenizer: Path) -> Any:
    """Load the installed image-pinned Gen3C tokenizer/VAE once."""
    import importlib

    config_module = importlib.import_module("cosmos_predict1.diffusion.config.config")
    helper_module = importlib.import_module("cosmos_predict1.utils.config_helper")
    model_module = importlib.import_module(
        "cosmos_predict1.diffusion.model.model_gen3c"
    )
    inference = importlib.import_module(
        "cosmos_predict1.diffusion.inference.inference_utils"
    )
    config = helper_module.override(
        config_module.make_config(),
        ["--", "experiment=GEN3C_Cosmos_7B"],
    )
    config.validate()
    config.freeze()
    model = model_module.DiffusionGen3CModel(config.model)
    inference.load_tokenizer_model(model, str(tokenizer))
    return model


def _decode_latent(torch: Any, model: Any, latent: Any, device: Any) -> np.ndarray:
    decoded = model.decode(latent.to(device))
    if (
        not torch.is_tensor(decoded)
        or tuple(decoded.shape) != R4C_GEN3C_MODEL_CONTRACT.target_bcthw_shape
        or not bool(torch.isfinite(decoded).all())
    ):
        raise RuntimeError("Gen3C decoder returned the wrong R4c video contract")
    return (
        ((decoded + 1.0).clamp(0.0, 2.0) / 2.0)
        .squeeze(0)
        .permute(1, 2, 3, 0)
        .mul(255)
        .to(torch.uint8)
        .contiguous()
        .cpu()
        .numpy()
    )


if __name__ == "__main__":
    _run(_arguments())

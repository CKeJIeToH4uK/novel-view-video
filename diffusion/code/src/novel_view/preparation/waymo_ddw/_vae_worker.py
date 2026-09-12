"""Private bounded Gen3C tokenizer/VAE worker for DDW preparation."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from novel_view.preparation.waymo_ddw.artifacts import (
    BASE_LATENT_SHAPE,
    POSE_LATENT_SHAPE,
    write_base_latents,
    write_lidar_depth_arrays,
    write_pose_latent,
)
from novel_view.preparation.waymo_ddw.bake import read_condition_pixels


_MODEL_NAME = "GEN3C_Cosmos_7B"
_FRAME_COUNT = 121
_SIZE_HW = (704, 1280)
_RGB_SHAPE = (_FRAME_COUNT, *_SIZE_HW, 3)


@dataclass(frozen=True, slots=True)
class _Task:
    sample_id: str
    magnitude_m: float
    sign: int
    target_rgb: Path
    condition_rgb: Path
    condition_known: Path
    lidar_input: Path
    base_latent: Path
    pose_latent: Path
    lidar_depth: Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    return parser.parse_args()


def _load_tasks(path: Path) -> tuple[_Task, ...]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        _Task(
            sample_id=row["sample_id"],
            magnitude_m=row["magnitude_m"],
            sign=row["sign"],
            target_rgb=Path(row["target_rgb"]),
            condition_rgb=Path(row["condition_rgb"]),
            condition_known=Path(row["condition_known"]),
            lidar_input=Path(row["lidar_input"]),
            base_latent=Path(row["base_latent"]),
            pose_latent=Path(row["pose_latent"]),
            lidar_depth=Path(row["lidar_depth"]),
        )
        for row in rows
    )


def _load_model(tokenizer_path: Path) -> tuple[Any, Any, Any]:
    """Load the installed image-pinned tokenizer/VAE on the assigned GPU."""
    torch = importlib.import_module("torch")
    torch.cuda.set_device(0)
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
        ["--", f"experiment={_MODEL_NAME}"],
    )
    config.validate()
    config.freeze()
    model = model_module.DiffusionGen3CModel(config.model)
    inference.load_tokenizer_model(model, str(tokenizer_path))
    return torch, model, inference.create_condition_latent_from_input_frames


def _normalize_target(torch: Any, rgb_bcthw: Any, device: Any) -> Any:
    """Preserve the established BF16-first clean-target arithmetic."""
    return (
        rgb_bcthw.to(
            device=device,
            dtype=torch.bfloat16,
            memory_format=torch.contiguous_format,
        )
        .div_(127.5)
        .sub_(1.0)
    )


def _normalize_condition(
    torch: Any,
    rgb_bcthw: Any,
    known_b1thw: Any,
    device: Any,
) -> tuple[Any, Any]:
    """Preserve float32-first DDW pixels and their BTNCHW layouts."""
    rgb = (
        rgb_bcthw.to(
            device=device,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        )
        .div_(127.5)
        .sub_(1.0)
        .to(dtype=torch.bfloat16)
    )
    known = known_b1thw.to(
        device=device,
        dtype=torch.bfloat16,
        memory_format=torch.contiguous_format,
    )
    return (
        rgb.permute(0, 2, 1, 3, 4).unsqueeze(2),
        known.permute(0, 2, 1, 3, 4).unsqueeze(2),
    )


def _read_target(path: Path) -> np.ndarray:
    target = np.load(path, allow_pickle=False)
    if target.dtype != np.uint8 or target.shape != _RGB_SHAPE:
        raise ValueError("VAE target has the wrong array contract")
    return target


def _read_condition(task: _Task) -> tuple[np.ndarray, np.ndarray]:
    return read_condition_pixels(
        task.condition_rgb,
        task.condition_known,
        _SIZE_HW,
    )


def _cpu_latent(torch: Any, value: Any, shape: tuple[int, ...], name: str) -> Any:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} is not a tensor")
    result = value.to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(result.shape) != shape or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} has the wrong tensor contract")
    return result


def _write_lidar_for_task(task: _Task) -> None:
    """Publish measured LiDAR from the parent-staged batch transport."""
    with np.load(task.lidar_input, allow_pickle=False) as values:
        write_lidar_depth_arrays(
            task.lidar_depth,
            task.sample_id,
            values["K_canvas"],
            values["frame_offsets"],
            values["canvas_xy"],
            values["depth_z_m"],
            values["evaluation_holdout"],
        )


def _bake_task(
    task: _Task,
    torch: Any,
    model: Any,
    create_source_latent: Any,
    device: Any,
) -> None:
    target_rgb = _read_target(task.target_rgb)
    target = _normalize_target(
        torch,
        torch.from_numpy(target_rgb).permute(3, 0, 1, 2).unsqueeze(0),
        device,
    )
    del target_rgb
    with torch.inference_mode():
        clean = _cpu_latent(
            torch,
            model.encode(target),
            BASE_LATENT_SHAPE,
            "clean_latent",
        )
        source_raw, padded = create_source_latent(
            model,
            target[:, :, :1].contiguous(),
            num_frames_condition=1,
        )
        del padded, target
        source = _cpu_latent(
            torch,
            source_raw,
            BASE_LATENT_SHAPE,
            "source_latent",
        )

        condition_rgb, condition_known = _read_condition(task)
        condition = torch.from_numpy(condition_rgb).permute(3, 0, 1, 2).unsqueeze(0)
        known = torch.from_numpy(condition_known).unsqueeze(0).unsqueeze(0)
        condition, known = _normalize_condition(torch, condition, known, device)
        del condition_rgb, condition_known
        pose = _cpu_latent(
            torch,
            model.encode_warped_frames(condition, known, torch.bfloat16),
            POSE_LATENT_SHAPE,
            "pose_latent",
        )
        del condition, known

    write_base_latents(task.base_latent, task.sample_id, clean, source)
    write_pose_latent(
        task.pose_latent,
        task.sample_id,
        task.magnitude_m,
        task.sign,
        pose,
    )


def _run(arguments: argparse.Namespace) -> None:
    tasks = _load_tasks(arguments.tasks)
    for task in tasks:
        _write_lidar_for_task(task)
    torch, model, create_source_latent = _load_model(arguments.tokenizer)
    device = torch.device("cuda", 0)
    for task in tasks:
        _bake_task(task, torch, model, create_source_latent, device)


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()

"""Scratch pixel transport and model-ready baking for Waymo DDW."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np
import numpy.typing as npt

from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import run_process

if TYPE_CHECKING:
    from novel_view.preparation.waymo_ddw.condition import DdwCondition
    from novel_view.preparation.waymo_ddw.raster import WaymoFrontRaster


_T = TypeVar("_T")
_VAE_BATCH_SIZE = 8


@dataclass(frozen=True, slots=True)
class VaeBakeTask:
    """Small path-only description of one staged DDW VAE item."""

    sample_id: str
    magnitude_m: float
    sign: int
    target_rgb_path: Path
    condition_rgb_path: Path
    condition_known_path: Path
    lidar_input_path: Path
    base_latent_path: Path
    pose_latent_path: Path
    lidar_depth_path: Path


def write_condition_pixels(
    condition: DdwCondition,
    rgb_path: Path,
    known_path: Path,
) -> None:
    """Write uint8 THWC RGB and little-endian packed known frame by frame."""
    frame_count, height, width = condition.known.shape
    rgb = np.lib.format.open_memmap(
        rgb_path,
        mode="w+",
        dtype=np.uint8,
        shape=(frame_count, height, width, 3),
    )
    known = np.lib.format.open_memmap(
        known_path,
        mode="w+",
        dtype=np.uint8,
        shape=(frame_count, (height * width + 7) // 8),
    )
    try:
        for index in range(frame_count):
            rgb[index] = np.rint(
                np.clip(
                    np.moveaxis(condition.rgb_minus_one_to_one[index], 0, -1) + 1.0,
                    0.0,
                    2.0,
                )
                * 127.5
            ).astype(np.uint8)
            known[index] = np.packbits(
                condition.known[index].reshape(-1),
                bitorder="little",
            )
        rgb.flush()
        known.flush()
    finally:
        rgb._mmap.close()
        known._mmap.close()


def read_condition_pixels(
    rgb_path: Path,
    known_path: Path,
    size_hw: tuple[int, int],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
    """Read one staged RGB/mask pair for its immediate VAE consumer."""
    rgb = cast(npt.NDArray[np.uint8], np.load(rgb_path, allow_pickle=False))
    packed = cast(npt.NDArray[np.uint8], np.load(known_path, allow_pickle=False))
    height, width = size_hw
    known = np.unpackbits(
        packed,
        axis=1,
        count=height * width,
        bitorder="little",
    ).reshape(packed.shape[0], height, width)
    return rgb, known.astype(np.bool_, copy=False)


def bake_prompt(
    text_encoder_path: Path,
    output_path: Path,
    log_path: Path,
    environment_overrides: Mapping[str, str] | None = None,
) -> Path:
    """Bake the dataset-wide empty prompt in one installed Gen3C process."""
    run_process(
        [
            str(GEN3C_PYTHON),
            str(Path(__file__).with_name("_prompt_worker.py")),
            "--text-encoder",
            str(text_encoder_path),
            "--output",
            str(output_path),
        ],
        log_path,
        "Gen3C empty-prompt bake",
        environment_overrides,
    )
    return output_path


def iter_vae_batches(values: tuple[_T, ...]) -> Iterator[tuple[_T, ...]]:
    """Yield ordered groups that fit one resident VAE model load."""
    for start in range(0, len(values), _VAE_BATCH_SIZE):
        yield values[start : start + _VAE_BATCH_SIZE]


def stage_bake_input(
    front: WaymoFrontRaster,
    condition: DdwCondition,
    scratch_root: Path,
    output_root: Path,
) -> VaeBakeTask:
    """Move target, condition, and sparse LiDAR to immediate batch scratch."""
    sample_id = front.selected.sample_id
    prefix = scratch_root / sample_id
    target_path = prefix.with_name(f"{prefix.name}-target.npy")
    condition_rgb_path = prefix.with_name(f"{prefix.name}-condition-rgb.npy")
    condition_known_path = prefix.with_name(f"{prefix.name}-condition-known.npy")
    lidar_input_path = prefix.with_name(f"{prefix.name}-lidar.npz")
    np.save(target_path, front.rgb_thwc, allow_pickle=False)
    write_condition_pixels(condition, condition_rgb_path, condition_known_path)
    np.savez(
        lidar_input_path,
        K_canvas=front.K_canvas,
        frame_offsets=front.lidar.frame_offsets,
        canvas_xy=front.lidar.canvas_xy,
        depth_z_m=front.lidar.depth_z_m,
        evaluation_holdout=front.lidar.evaluation_holdout,
    )

    leaf = output_root / "items" / sample_id
    leaf.mkdir(parents=True, exist_ok=True)
    return VaeBakeTask(
        sample_id=sample_id,
        magnitude_m=front.selected.magnitude_m,
        sign=front.selected.sign,
        target_rgb_path=target_path,
        condition_rgb_path=condition_rgb_path,
        condition_known_path=condition_known_path,
        lidar_input_path=lidar_input_path,
        base_latent_path=leaf / "base.pt",
        pose_latent_path=leaf / "pose.pt",
        lidar_depth_path=leaf / "lidar-depth.pt",
    )


def bake_items(
    tasks: tuple[VaeBakeTask, ...],
    tokenizer_path: Path,
    task_path: Path,
    log_path: Path,
    environment_overrides: Mapping[str, str] | None = None,
) -> tuple[VaeBakeTask, ...]:
    """Bake one bounded group with one installed Gen3C model load."""
    task_path.write_text(
        json.dumps([_task_document(task) for task in tasks]),
        encoding="utf-8",
    )
    run_process(
        [
            str(GEN3C_PYTHON),
            str(Path(__file__).with_name("_vae_worker.py")),
            "--tasks",
            str(task_path),
            "--tokenizer",
            str(tokenizer_path),
        ],
        log_path,
        "Gen3C DDW VAE bake",
        environment_overrides,
    )
    return tasks


def _task_document(task: VaeBakeTask) -> dict[str, object]:
    return {
        "sample_id": task.sample_id,
        "magnitude_m": task.magnitude_m,
        "sign": task.sign,
        "target_rgb": str(task.target_rgb_path),
        "condition_rgb": str(task.condition_rgb_path),
        "condition_known": str(task.condition_known_path),
        "lidar_input": str(task.lidar_input_path),
        "base_latent": str(task.base_latent_path),
        "pose_latent": str(task.pose_latent_path),
        "lidar_depth": str(task.lidar_depth_path),
    }


__all__ = [
    "VaeBakeTask",
    "bake_items",
    "bake_prompt",
    "iter_vae_batches",
    "read_condition_pixels",
    "stage_bake_input",
    "write_condition_pixels",
]

"""Versioned model-ready tensor artifacts for Waymo DDW preparation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from torch import Tensor

    from novel_view.preparation.waymo_ddw.raster import FrontLidarDepth


BASE_FORMAT = "novel-view/gen3c-base-latents/v1"
POSE_FORMAT = "novel-view/gen3c-ddw-pose-latent/v1"
PROMPT_FORMAT = "novel-view/gen3c-empty-prompt/v1"
LIDAR_DEPTH_FORMAT = "novel-view/waymo-front-lidar-depth/v1"

BASE_LATENT_SHAPE = (1, 16, 16, 88, 160)
POSE_LATENT_SHAPE = (1, 64, 16, 88, 160)
PROMPT_EMBEDDING_SHAPE = (1, 512, 1024)
RASTER_SIZE_HW = (704, 1280)
FRAME_COUNT = 121


@dataclass(frozen=True, slots=True, eq=False)
class BaseLatents:
    sample_id: str
    clean_latent: Tensor
    source_latent: Tensor


@dataclass(frozen=True, slots=True, eq=False)
class DdwPoseLatent:
    sample_id: str
    magnitude_m: float
    sign: int
    pose_latent: Tensor


@dataclass(frozen=True, slots=True, eq=False)
class EmptyPrompt:
    t5_text_embeddings: Tensor


@dataclass(frozen=True, slots=True, eq=False)
class LidarDepth:
    sample_id: str
    camera_name: str
    raster_size_hw: tuple[int, int]
    K_canvas: Tensor
    frame_offsets: Tensor
    canvas_xy: Tensor
    depth_z_m: Tensor
    evaluation_holdout: Tensor


def write_base_latents(
    path: Path,
    sample_id: str,
    clean_latent: Tensor,
    source_latent: Tensor,
) -> None:
    """Write one direct base-latent payload."""
    _torch().save(
        {
            "format": BASE_FORMAT,
            "sample_id": sample_id,
            "clean_latent": clean_latent,
            "source_latent": source_latent,
        },
        path,
    )


def read_base_latents(path: Path, expected_sample_id: str) -> BaseLatents:
    payload = _load(path, BASE_FORMAT, {"sample_id", "clean_latent", "source_latent"})
    _require_sample(payload, expected_sample_id)
    torch = _torch()
    clean = _tensor(
        torch,
        payload["clean_latent"],
        torch.bfloat16,
        BASE_LATENT_SHAPE,
        "clean_latent",
    )
    source = _tensor(
        torch,
        payload["source_latent"],
        torch.bfloat16,
        BASE_LATENT_SHAPE,
        "source_latent",
    )
    return BaseLatents(expected_sample_id, clean, source)


def write_pose_latent(
    path: Path,
    sample_id: str,
    magnitude_m: float,
    sign: int,
    pose_latent: Tensor,
) -> None:
    """Write one direct DDW pose-latent payload."""
    _torch().save(
        {
            "format": POSE_FORMAT,
            "sample_id": sample_id,
            "magnitude_m": magnitude_m,
            "sign": sign,
            "pose_latent": pose_latent,
        },
        path,
    )


def read_pose_latent(
    path: Path,
    expected_sample_id: str,
    expected_magnitude_m: float,
    expected_sign: int,
) -> DdwPoseLatent:
    payload = _load(
        path,
        POSE_FORMAT,
        {"sample_id", "magnitude_m", "sign", "pose_latent"},
    )
    _require_sample(payload, expected_sample_id)
    if (payload["magnitude_m"], payload["sign"]) != (
        expected_magnitude_m,
        expected_sign,
    ):
        raise ValueError("pose latent has the wrong DDW variant")
    torch = _torch()
    pose = _tensor(
        torch,
        payload["pose_latent"],
        torch.bfloat16,
        POSE_LATENT_SHAPE,
        "pose_latent",
    )
    return DdwPoseLatent(
        expected_sample_id,
        expected_magnitude_m,
        expected_sign,
        pose,
    )


def write_empty_prompt(path: Path, t5_text_embeddings: Tensor) -> None:
    """Write the one dataset-wide empty-prompt embedding."""
    _torch().save(
        {
            "format": PROMPT_FORMAT,
            "t5_text_embeddings": t5_text_embeddings,
        },
        path,
    )


def read_empty_prompt(path: Path) -> EmptyPrompt:
    payload = _load(path, PROMPT_FORMAT, {"t5_text_embeddings"})
    torch = _torch()
    embeddings = _tensor(
        torch,
        payload["t5_text_embeddings"],
        torch.bfloat16,
        PROMPT_EMBEDDING_SHAPE,
        "t5_text_embeddings",
    )
    return EmptyPrompt(embeddings)


def write_lidar_depth(
    path: Path,
    sample_id: str,
    K_canvas: npt.NDArray[np.float64],
    lidar: FrontLidarDepth,
) -> None:
    """Write measured FRONT-A sparse camera-Z and its fixed holdout split."""
    write_lidar_depth_arrays(
        path,
        sample_id,
        K_canvas,
        lidar.frame_offsets,
        lidar.canvas_xy,
        lidar.depth_z_m,
        lidar.evaluation_holdout,
    )


def write_lidar_depth_arrays(
    path: Path,
    sample_id: str,
    K_canvas: npt.NDArray[np.float64],
    frame_offsets: npt.NDArray[np.int64],
    canvas_xy: npt.NDArray[np.float32],
    depth_z_m: npt.NDArray[np.float32],
    evaluation_holdout: npt.NDArray[np.bool_],
) -> None:
    """Write the same owner-specific payload from its staged NumPy arrays."""
    torch = _torch()
    torch.save(
        {
            "format": LIDAR_DEPTH_FORMAT,
            "sample_id": sample_id,
            "camera_name": "FRONT",
            "raster_size_hw": RASTER_SIZE_HW,
            "K_canvas": _from_numpy(torch, K_canvas),
            "frame_offsets": _from_numpy(torch, frame_offsets),
            "canvas_xy": _from_numpy(torch, canvas_xy),
            "depth_z_m": _from_numpy(torch, depth_z_m),
            "evaluation_holdout": _from_numpy(torch, evaluation_holdout),
        },
        path,
    )


def read_lidar_depth(path: Path, expected_sample_id: str) -> LidarDepth:
    fields = {
        "sample_id",
        "camera_name",
        "raster_size_hw",
        "K_canvas",
        "frame_offsets",
        "canvas_xy",
        "depth_z_m",
        "evaluation_holdout",
    }
    payload = _load(path, LIDAR_DEPTH_FORMAT, fields)
    _require_sample(payload, expected_sample_id)
    if (
        payload["camera_name"] != "FRONT"
        or tuple(payload["raster_size_hw"]) != RASTER_SIZE_HW
    ):
        raise ValueError("LiDAR depth has the wrong measured camera")
    torch = _torch()
    K = _tensor(
        torch, payload["K_canvas"], torch.float64, (FRAME_COUNT, 3, 3), "K_canvas"
    )
    offsets = _tensor(
        torch,
        payload["frame_offsets"],
        torch.int64,
        (FRAME_COUNT + 1,),
        "frame_offsets",
        finite=False,
    )
    xy_value = payload["canvas_xy"]
    point_count = int(xy_value.shape[0]) if torch.is_tensor(xy_value) else -1
    xy = _tensor(torch, xy_value, torch.float32, (point_count, 2), "canvas_xy")
    depth = _tensor(
        torch, payload["depth_z_m"], torch.float32, (point_count,), "depth_z_m"
    )
    holdout = _tensor(
        torch,
        payload["evaluation_holdout"],
        torch.bool,
        (point_count,),
        "evaluation_holdout",
        finite=False,
    )
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != point_count
        or bool(torch.any(offsets[1:] < offsets[:-1]))
        or bool(torch.any(depth <= 0.0))
        or bool(torch.any(xy[:, 0] < 0.0))
        or bool(torch.any(xy[:, 0] >= RASTER_SIZE_HW[1]))
        or bool(torch.any(xy[:, 1] < 0.0))
        or bool(torch.any(xy[:, 1] >= RASTER_SIZE_HW[0]))
        or not bool(torch.any(~holdout))
    ):
        raise ValueError("LiDAR depth violates sparse measured-camera invariants")
    return LidarDepth(
        expected_sample_id,
        "FRONT",
        RASTER_SIZE_HW,
        K,
        offsets,
        xy,
        depth,
        holdout,
    )


def _torch() -> Any:
    return importlib.import_module("torch")


def _load(path: Path, format_name: str, fields: set[str]) -> dict[str, Any]:
    payload = _torch().load(path, weights_only=True, map_location="cpu")
    if type(payload) is not dict or set(payload) != {"format", *fields}:
        raise ValueError("artifact has unexpected fields")
    if payload["format"] != format_name:
        raise ValueError("artifact has the wrong format")
    return cast(dict[str, Any], payload)


def _require_sample(payload: dict[str, Any], expected_sample_id: str) -> None:
    if payload["sample_id"] != expected_sample_id:
        raise ValueError("artifact has the wrong sample_id")


def _tensor(
    torch: Any,
    value: Any,
    dtype: Any,
    shape: tuple[int, ...],
    name: str,
    *,
    finite: bool = True,
) -> Tensor:
    if (
        not torch.is_tensor(value)
        or value.device.type != "cpu"
        or value.dtype != dtype
        or tuple(value.shape) != shape
        or not value.is_contiguous()
        or (finite and not bool(torch.isfinite(value).all()))
    ):
        raise ValueError(f"{name} has the wrong tensor contract")
    return cast("Tensor", value)


def _from_numpy(torch: Any, value: npt.NDArray[np.generic]) -> Tensor:
    return cast("Tensor", torch.from_numpy(np.array(value, copy=True, order="C")))


__all__ = [
    "BASE_FORMAT",
    "BASE_LATENT_SHAPE",
    "BaseLatents",
    "DdwPoseLatent",
    "EmptyPrompt",
    "LIDAR_DEPTH_FORMAT",
    "LidarDepth",
    "POSE_FORMAT",
    "POSE_LATENT_SHAPE",
    "PROMPT_EMBEDDING_SHAPE",
    "PROMPT_FORMAT",
    "read_base_latents",
    "read_empty_prompt",
    "read_lidar_depth",
    "read_pose_latent",
    "write_base_latents",
    "write_empty_prompt",
    "write_lidar_depth",
    "write_lidar_depth_arrays",
    "write_pose_latent",
]

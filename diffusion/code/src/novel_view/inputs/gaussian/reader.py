"""Decode the existing ``gaussian_depth_export`` version 1 JSON."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import cv2

from novel_view.geometry.raster import CoverCropTransform


class GaussianExportError(ValueError):
    """A Gaussian depth export cannot be decoded."""


@dataclass(frozen=True, slots=True, eq=False)
class GaussianFrame:
    """One native Gaussian source row in producer order."""

    pose_index: int
    timestamp_ns: int
    rgb_path: Path
    depth_path: Path
    mask_path: Path
    source_w2c: npt.NDArray[np.float64]
    target_w2c: npt.NDArray[np.float64]

    __hash__: ClassVar[None] = None


@dataclass(frozen=True, slots=True, eq=False)
class GaussianExport:
    """Decoded fields consumed from one exact Gaussian depth export."""

    info_file: Path
    scene_id: int
    splats_variant: str
    camera_name: str
    target_name: str
    target_shift_camera_xyz_m: tuple[float, float, float]
    source_size_hw: tuple[int, int]
    intrinsics: npt.NDArray[np.float64]
    frames: tuple[GaussianFrame, ...]

    __hash__: ClassVar[None] = None


@dataclass(frozen=True, slots=True, eq=False)
class GaussianRaster:
    """One transformed RGB/depth/valid producer row."""

    rgb: npt.NDArray[np.uint8]
    depth_z_m: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]

    __hash__: ClassVar[None] = None  # type: ignore[assignment]


def read_gaussian_export(info_file: Path) -> GaussianExport:
    """Decode one exact producer-v1 JSON without opening frame payloads."""
    try:
        export_info = json.loads(info_file.read_text(encoding="utf-8"))
        if (
            export_info["format"] != "gaussian_depth_export"
            or export_info["format_version"] != 1
        ):
            raise GaussianExportError("unsupported Gaussian depth export")

        camera_info = export_info["camera"]
        if camera_info["axes"] != "opencv_x_right_y_down_z_forward" or (
            camera_info["pose"] != "world_to_camera"
        ):
            raise GaussianExportError("Gaussian cameras must use OpenCV W2C")

        source_size_hw = tuple(map(int, camera_info["image_size_hw"]))
        if len(source_size_hw) != 2:
            raise GaussianExportError("camera image_size_hw must have two values")

        intrinsics = np.asarray(camera_info["intrinsics"], dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            raise GaussianExportError("camera intrinsics must be finite 3x3")

        shift = tuple(map(float, export_info["target"]["shift_camera_xyz_m"]))
        if len(shift) != 3 or not np.all(np.isfinite(shift)):
            raise GaussianExportError(
                "target shift_camera_xyz_m must contain three finite values"
            )

        export_root = info_file.parent
        frames = tuple(
            _frame(export_root, frame_info)
            for frame_info in export_info["frames"]
        )
        if not frames:
            raise GaussianExportError("Gaussian export contains no frames")
        if any(
            current.timestamp_ns <= previous.timestamp_ns
            or current.pose_index <= previous.pose_index
            for previous, current in zip(frames, frames[1:])
        ):
            raise GaussianExportError(
                "Gaussian frames must have increasing poses and timestamps"
            )

        return GaussianExport(
            info_file=info_file,
            scene_id=int(export_info["inputs"]["scene_id"]),
            splats_variant=str(export_info["inputs"]["splats_variant"]),
            camera_name=str(camera_info["name"]),
            target_name=str(export_info["target"]["name"]),
            target_shift_camera_xyz_m=shift,
            source_size_hw=source_size_hw,
            intrinsics=_readonly(intrinsics),
            frames=frames,
        )
    except GaussianExportError:
        raise
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise GaussianExportError(
            f"invalid Gaussian depth export {info_file}: {error}"
        ) from error


def read_gaussian_raster(
    frame: GaussianFrame,
    source_size_hw: tuple[int, int],
    transform: CoverCropTransform,
) -> GaussianRaster:
    """Open and transform one exact producer RGB/depth/mask triplet."""
    rgb_bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(frame.mask_path), cv2.IMREAD_GRAYSCALE)
    depth = np.load(frame.depth_path, allow_pickle=False)
    if (
        rgb_bgr is None
        or mask is None
        or rgb_bgr.shape[:2] != source_size_hw
        or mask.shape != source_size_hw
        or depth.shape != source_size_hw
        or not np.issubdtype(depth.dtype, np.floating)
    ):
        raise GaussianExportError(
            f"invalid raster triplet for pose {frame.pose_index}"
        )
    native_valid = (mask == 255) & np.isfinite(depth) & (depth > 0.0)
    rgb = _resize_crop(
        cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        transform,
        cv2.INTER_LINEAR,
    )
    depth_z_m = np.asarray(
        _resize_crop(
            np.where(native_valid, depth, 0.0).astype(np.float32),
            transform,
            cv2.INTER_LINEAR,
        ),
        dtype=np.float32,
    )
    valid = _resize_crop(
        native_valid.astype(np.uint8),
        transform,
        cv2.INTER_NEAREST,
    ).astype(bool)
    valid &= np.isfinite(depth_z_m) & (depth_z_m > 0.0)
    return GaussianRaster(
        rgb=np.ascontiguousarray(rgb),
        depth_z_m=np.where(valid, depth_z_m, 0.0).astype(np.float32),
        valid=np.ascontiguousarray(valid),
    )


def _resize_crop(
    value: npt.NDArray[np.generic],
    transform: CoverCropTransform,
    interpolation: int,
) -> npt.NDArray[np.generic]:
    resized_height, resized_width = transform.resized_size_hw
    resized = cv2.resize(
        value,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    top, left = transform.crop_top, transform.crop_left
    height, width = transform.output_size_hw
    return resized[top : top + height, left : left + width]


def _frame(export_root: Path, frame_info: object) -> GaussianFrame:
    if not isinstance(frame_info, dict):
        raise GaussianExportError("Gaussian frame must be an object")
    return GaussianFrame(
        pose_index=int(frame_info["pose_index"]),
        timestamp_ns=int(frame_info["timestamp_ns"]),
        rgb_path=export_root / str(frame_info["rgb"]),
        depth_path=export_root / str(frame_info["depth"]),
        mask_path=export_root / str(frame_info["mask"]),
        source_w2c=_w2c(frame_info["source_w2c"]),
        target_w2c=_w2c(frame_info["target_w2c"]),
    )


def _w2c(value: object) -> npt.NDArray[np.float64]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise GaussianExportError("camera matrix must be finite 4x4")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=1e-6,
    ) or np.linalg.det(rotation) <= 0.0 or not np.allclose(
        matrix[3],
        (0.0, 0.0, 0.0, 1.0),
        atol=1e-9,
    ):
        raise GaussianExportError("camera matrix must be a rigid W2C")
    return _readonly(matrix)


def _readonly(
    value: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    owned = np.array(value, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


__all__ = [
    "GaussianExport",
    "GaussianExportError",
    "GaussianFrame",
    "GaussianRaster",
    "read_gaussian_export",
    "read_gaussian_raster",
]

"""Backend-neutral metric camera-Z depth on measured Waymo cameras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.inputs.waymo._arrays import require_array
from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT


DepthCameraName: TypeAlias = Literal["FRONT", "FRONT_LEFT", "FRONT_RIGHT"]
DEPTH_CAMERA_ORDER: tuple[DepthCameraName, ...] = (
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
)
GEN3C_CANVAS_SIZE_HW = R4C_GEN3C_MODEL_CONTRACT.raster_size_hw
_BOTTOM_ROW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False)
class MetricDepthClip:
    """Dense camera-Z metres on exact measured Waymo canvas cameras.

    The candidate adapter must already reproject model geometry into the
    measured ``K_canvas`` and ``world_to_camera_cv`` sequence. Invalid depth
    is represented by zero. Arrays are accepted without copying and must
    therefore already be owned, C-contiguous and read-only.
    """

    clip_key: WaymoClipKey
    camera_name: DepthCameraName
    depth_z_m: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]
    K_canvas: npt.NDArray[np.float64]
    world_to_camera_cv: npt.NDArray[np.float64]

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.clip_key, WaymoClipKey):
            raise WaymoContractError("clip_key must be WaymoClipKey")
        if type(self.camera_name) is not str or self.camera_name not in DEPTH_CAMERA_ORDER:
            raise WaymoContractError("camera_name must be a selected Waymo camera")
        frame_count = len(self.clip_key.frame_timestamps_micros)
        dense_shape = (frame_count, *GEN3C_CANVAS_SIZE_HW)
        depth = require_array(
            self.depth_z_m,
            "depth_z_m",
            dtype=np.dtype(np.float32),
            shape=dense_shape,
        )
        valid = require_array(
            self.valid,
            "valid",
            dtype=np.dtype(np.bool_),
            shape=dense_shape,
        )
        require_metric_camera_pack(
            self.K_canvas,
            self.world_to_camera_cv,
            frame_count,
        )
        _validate_depth(depth, valid)


def require_metric_camera_pack(
    K_canvas: object,
    world_to_camera_cv: object,
    frame_count: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate one measured camera pack before an expensive model call."""
    if type(frame_count) is not int or frame_count <= 0:
        raise WaymoContractError("frame_count must be a positive int")
    intrinsics = cast(
        npt.NDArray[np.float64],
        require_array(K_canvas, "K_canvas", dtype=np.dtype(np.float64),
                      shape=(frame_count, 3, 3)),
    )
    cameras = cast(
        npt.NDArray[np.float64],
        require_array(world_to_camera_cv, "world_to_camera_cv",
                      dtype=np.dtype(np.float64), shape=(frame_count, 4, 4)),
    )
    _validate_intrinsics(intrinsics)
    _validate_cameras(cameras)
    return intrinsics, cameras


def _validate_depth(
    depth: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_],
) -> None:
    """Validate per frame so a 121-frame temporary mask is never allocated."""
    for index, (frame_depth, frame_valid) in enumerate(
        zip(depth, valid, strict=True)
    ):
        if not np.isfinite(frame_depth).all() or np.any(frame_depth < 0.0):
            raise WaymoContractError(f"depth_z_m[{index}] must be finite and non-negative")
        if np.any(frame_valid & (frame_depth <= 0.0)):
            raise WaymoContractError(f"valid depth_z_m[{index}] must be positive")
        if np.any((~frame_valid) & (frame_depth != 0.0)):
            raise WaymoContractError(f"invalid depth_z_m[{index}] must be zero")


def _validate_intrinsics(intrinsics: npt.NDArray[np.float64]) -> None:
    if not np.isfinite(intrinsics).all():
        raise WaymoContractError("K_canvas must be finite")
    if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
        raise WaymoContractError("K_canvas must have positive focal lengths")
    if not np.allclose(
        intrinsics[:, 2],
        (0.0, 0.0, 1.0),
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(intrinsics[:, 1, 0], 0.0, rtol=0.0, atol=1e-12):
        raise WaymoContractError("K_canvas must contain pinhole matrices")


def _validate_cameras(cameras: npt.NDArray[np.float64]) -> None:
    if not np.isfinite(cameras).all() or not np.array_equal(
        cameras[:, 3],
        np.broadcast_to(_BOTTOM_ROW, cameras[:, 3].shape),
    ):
        raise WaymoContractError("world_to_camera_cv must be finite homogeneous W2C")
    for index, rotation in enumerate(cameras[:, :3, :3]):
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            rtol=0.0,
            atol=1e-6,
        ) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6):
            raise WaymoContractError(
                f"world_to_camera_cv[{index}] must contain a proper rotation"
            )

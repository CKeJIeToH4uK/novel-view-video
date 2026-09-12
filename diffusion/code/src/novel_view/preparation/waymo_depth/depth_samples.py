"""Waymo LiDAR camera-Z evidence shared by depth candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from novel_view.preparation.waymo_depth.depth import (
    DEPTH_CAMERA_ORDER,
    DepthCameraName,
)
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.inputs.waymo._arrays import require_array


@dataclass(frozen=True, slots=True, eq=False)
class ProjectedLidarDepth:
    """Ordered sparse camera-Z evidence on one canonical camera canvas.

    ``frame_offsets`` partitions the remaining arrays into the exact 121
    frames. Within each frame samples are ordered by LiDAR, return and native
    flat pixel index. Every fifth sample in that total order is held out;
    candidate-specific fitting may use only the complement.
    """

    clip_key: WaymoClipKey
    camera_name: DepthCameraName
    frame_offsets: npt.NDArray[np.int64]
    lidar_ids: npt.NDArray[np.uint8]
    return_ids: npt.NDArray[np.uint8]
    flat_pixel_indices: npt.NDArray[np.int32]
    canvas_xy: npt.NDArray[np.float32]
    depth_z_m: npt.NDArray[np.float32]
    heldout: npt.NDArray[np.bool_]

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.clip_key, WaymoClipKey):
            raise WaymoContractError("clip_key must be WaymoClipKey")
        if type(self.camera_name) is not str or self.camera_name not in (
            DEPTH_CAMERA_ORDER
        ):
            raise WaymoContractError("camera_name must be selected")
        frame_count = len(self.clip_key.frame_timestamps_micros)
        offsets = require_array(
            self.frame_offsets,
            "frame_offsets",
            dtype=np.dtype(np.int64),
            shape=(frame_count + 1,),
        )
        sample_count = int(offsets[-1]) if offsets.size else 0
        if (
            offsets[0] != 0
            or sample_count <= 0
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise WaymoContractError(
                "frame_offsets must delimit non-empty ordered samples"
            )
        for name, value, dtype, shape in (
            ("lidar_ids", self.lidar_ids, np.uint8, (sample_count,)),
            ("return_ids", self.return_ids, np.uint8, (sample_count,)),
            (
                "flat_pixel_indices",
                self.flat_pixel_indices,
                np.int32,
                (sample_count,),
            ),
            ("canvas_xy", self.canvas_xy, np.float32, (sample_count, 2)),
            ("depth_z_m", self.depth_z_m, np.float32, (sample_count,)),
            ("heldout", self.heldout, np.bool_, (sample_count,)),
        ):
            require_array(value, name, dtype=np.dtype(dtype), shape=shape)
        if np.any((self.lidar_ids < 1) | (self.lidar_ids > 5)):
            raise WaymoContractError("lidar_ids must contain IDs 1..5")
        if np.any((self.return_ids < 1) | (self.return_ids > 2)):
            raise WaymoContractError("return_ids must contain IDs 1..2")
        _require_canonical_sample_order(
            offsets,
            self.lidar_ids,
            self.return_ids,
            self.flat_pixel_indices,
        )
        if not np.isfinite(self.canvas_xy).all():
            raise WaymoContractError("canvas_xy must be finite")
        if not np.isfinite(self.depth_z_m).all() or np.any(
            self.depth_z_m <= 0.0
        ):
            raise WaymoContractError("depth_z_m must be finite and positive")
        expected_heldout = np.arange(sample_count) % 5 == 0
        if not np.array_equal(self.heldout, expected_heldout):
            raise WaymoContractError("heldout must select every fifth sample")

    def frame_slice(self, frame_index: int) -> slice:
        """Return the sparse sample slice for one exact frame."""
        frame_count = len(self.frame_offsets) - 1
        if type(frame_index) is not int or not 0 <= frame_index < frame_count:
            raise WaymoContractError("frame_index is outside the clip")
        return slice(
            int(self.frame_offsets[frame_index]),
            int(self.frame_offsets[frame_index + 1]),
        )


def _require_canonical_sample_order(
    offsets: npt.NDArray[np.int64],
    lidar_ids: npt.NDArray[np.uint8],
    return_ids: npt.NDArray[np.uint8],
    flat_indices: npt.NDArray[np.int32],
) -> None:
    """Require strict ``(lidar, return, flat)`` order inside every frame."""
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        if stop - start < 2:
            continue
        left = slice(int(start), int(stop) - 1)
        right = slice(int(start) + 1, int(stop))
        same_lidar = lidar_ids[right] == lidar_ids[left]
        same_return = return_ids[right] == return_ids[left]
        increasing = (
            (lidar_ids[right] > lidar_ids[left])
            | (same_lidar & (return_ids[right] > return_ids[left]))
            | (
                same_lidar
                & same_return
                & (flat_indices[right] > flat_indices[left])
            )
        )
        if not np.all(increasing):
            raise WaymoContractError(
                "samples must be strictly ordered by lidar, return and flat index"
            )

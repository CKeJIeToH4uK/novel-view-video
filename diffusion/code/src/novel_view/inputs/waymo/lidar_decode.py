"""Exact native LiDAR-array decoders for modular Waymo Perception v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .decode import (
    _fixed_array,
    _frame_key,
    _list_array,
    _sensor_id,
)
from .types import WaymoContractError, WaymoFrameKey


_SEGMENT = "key.segment_context_name"
_TIMESTAMP = "key.frame_timestamp_micros"
_LIDAR = "key.laser_name"
_RANGE = "[LiDARComponent]"
_PROJECTION = "[LiDARCameraProjectionComponent]"
_POSE = "[LiDARPoseComponent]"

LIDAR_RANGE_COLUMNS = (
    _SEGMENT,
    _TIMESTAMP,
    _LIDAR,
    f"{_RANGE}.range_image_return1.values",
    f"{_RANGE}.range_image_return1.shape",
    f"{_RANGE}.range_image_return2.values",
    f"{_RANGE}.range_image_return2.shape",
)
LIDAR_PROJECTION_COLUMNS = (
    _SEGMENT,
    _TIMESTAMP,
    _LIDAR,
    f"{_PROJECTION}.range_image_return1.values",
    f"{_PROJECTION}.range_image_return1.shape",
    f"{_PROJECTION}.range_image_return2.values",
    f"{_PROJECTION}.range_image_return2.shape",
)
LIDAR_POSE_COLUMNS = (
    _SEGMENT,
    _TIMESTAMP,
    _LIDAR,
    f"{_POSE}.range_image_return1.values",
    f"{_POSE}.range_image_return1.shape",
)


@dataclass(frozen=True, slots=True, eq=False)
class DecodedLidarPair:
    """Two owned native arrays decoded from one keyed LiDAR row."""

    key: WaymoFrameKey
    sensor_id: int
    return1: np.ndarray
    return2: np.ndarray


def decode_v2_lidar_ranges(batch: Any, row: int) -> DecodedLidarPair:
    """Decode both `[H,W,4]` float32 range-image returns."""
    return _decode_pair(batch, row, _RANGE, channels=4)


def decode_v2_lidar_projections(batch: Any, row: int) -> DecodedLidarPair:
    """Decode both `[H,W,6]` float32 camera-projection returns."""
    return _decode_pair(batch, row, _PROJECTION, channels=6)


def decode_v2_lidar_pose(
    batch: Any,
    row: int,
) -> tuple[WaymoFrameKey, int, np.ndarray]:
    """Decode the single TOP `[H,W,6]` vehicle-to-global pose grid."""
    key = _frame_key(batch, row)
    sensor_id = _sensor_id(batch, _LIDAR, row)
    if sensor_id != 1:
        raise WaymoContractError("lidar_pose is defined only for TOP sensor ID 1")
    return (
        key,
        sensor_id,
        _shaped_array(
            batch,
            f"{_POSE}.range_image_return1.values",
            f"{_POSE}.range_image_return1.shape",
            row,
            channels=6,
        ),
    )


def _decode_pair(
    batch: Any,
    row: int,
    prefix: str,
    *,
    channels: int,
) -> DecodedLidarPair:
    key = _frame_key(batch, row)
    sensor_id = _sensor_id(batch, _LIDAR, row)
    arrays = tuple(
        _shaped_array(
            batch,
            f"{prefix}.range_image_return{return_index}.values",
            f"{prefix}.range_image_return{return_index}.shape",
            row,
            channels=channels,
        )
        for return_index in (1, 2)
    )
    return DecodedLidarPair(key, sensor_id, arrays[0], arrays[1])


def _shaped_array(
    batch: Any,
    values_column: str,
    shape_column: str,
    row: int,
    *,
    channels: int,
) -> np.ndarray:
    shape_values = _fixed_array(batch, shape_column, row, np.int32, (3,))
    shape = tuple(int(value) for value in shape_values)
    if any(size <= 0 for size in shape) or shape[-1] != channels:
        raise WaymoContractError(f"{values_column} has invalid declared shape {shape}")
    return _list_array(batch, values_column, row, np.float32, shape=shape)

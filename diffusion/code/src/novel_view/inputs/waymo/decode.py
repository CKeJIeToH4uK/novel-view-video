"""Exact PyArrow-row decoders for modular Waymo Perception v2."""

from __future__ import annotations

from typing import Any

import numpy as np

from .camera import (
    WAYMO_CAMERA_ORDER,
    WaymoCameraCalibration,
    WaymoCameraFrame,
)
from .frame import WaymoSegmentContext
from .index import WaymoV2Files, _import_parquet
from .lidar import WAYMO_LIDAR_ORDER, WaymoLidarCalibration
from .types import WaymoContractError, WaymoFrameKey


_SEGMENT = "key.segment_context_name"
_TIMESTAMP = "key.frame_timestamp_micros"
_CAMERA = "key.camera_name"
_LIDAR = "key.laser_name"
_CAM = "[CameraCalibrationComponent]"
_IMAGE = "[CameraImageComponent]"
_LIDAR_CAL = "[LiDARCalibrationComponent]"
_VEHICLE = "[VehiclePoseComponent]"

_PRIMITIVE_DTYPES = {
    _TIMESTAMP: np.dtype(np.int64),
    _CAMERA: np.dtype(np.int8),
    _LIDAR: np.dtype(np.int8),
    f"{_CAM}.width": np.dtype(np.int32),
    f"{_CAM}.height": np.dtype(np.int32),
    f"{_CAM}.rolling_shutter_direction": np.dtype(np.int8),
    f"{_IMAGE}.velocity.linear_velocity.x": np.dtype(np.float32),
    f"{_IMAGE}.velocity.linear_velocity.y": np.dtype(np.float32),
    f"{_IMAGE}.velocity.linear_velocity.z": np.dtype(np.float32),
}
_ARROW_PRIMITIVE_NAMES = {
    np.dtype(np.int8): "int8",
    np.dtype(np.int32): "int32",
    np.dtype(np.int64): "int64",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}

_CAMERA_CALIBRATION_COLUMNS = (
    _SEGMENT,
    _CAMERA,
    f"{_CAM}.intrinsic.f_u",
    f"{_CAM}.intrinsic.f_v",
    f"{_CAM}.intrinsic.c_u",
    f"{_CAM}.intrinsic.c_v",
    f"{_CAM}.intrinsic.k1",
    f"{_CAM}.intrinsic.k2",
    f"{_CAM}.intrinsic.p1",
    f"{_CAM}.intrinsic.p2",
    f"{_CAM}.intrinsic.k3",
    f"{_CAM}.extrinsic.transform",
    f"{_CAM}.width",
    f"{_CAM}.height",
    f"{_CAM}.rolling_shutter_direction",
)
_LIDAR_CALIBRATION_COLUMNS = (
    _SEGMENT,
    _LIDAR,
    f"{_LIDAR_CAL}.extrinsic.transform",
    f"{_LIDAR_CAL}.beam_inclination.min",
    f"{_LIDAR_CAL}.beam_inclination.max",
    f"{_LIDAR_CAL}.beam_inclination.values",
)

VEHICLE_POSE_COLUMNS = (
    _SEGMENT,
    _TIMESTAMP,
    f"{_VEHICLE}.world_from_vehicle.transform",
)
CAMERA_IMAGE_COLUMNS = (
    _SEGMENT,
    _TIMESTAMP,
    _CAMERA,
    f"{_IMAGE}.image",
    f"{_IMAGE}.pose.transform",
    f"{_IMAGE}.velocity.linear_velocity.x",
    f"{_IMAGE}.velocity.linear_velocity.y",
    f"{_IMAGE}.velocity.linear_velocity.z",
    f"{_IMAGE}.velocity.angular_velocity.x",
    f"{_IMAGE}.velocity.angular_velocity.y",
    f"{_IMAGE}.velocity.angular_velocity.z",
    f"{_IMAGE}.pose_timestamp",
    f"{_IMAGE}.rolling_shutter_params.shutter",
    f"{_IMAGE}.rolling_shutter_params.camera_trigger_time",
    f"{_IMAGE}.rolling_shutter_params.camera_readout_done_time",
)


def read_v2_segment_context(files: WaymoV2Files) -> WaymoSegmentContext:
    """Decode both small calibration files once into one shared context."""
    parquet = _import_parquet()
    cameras: dict[int, WaymoCameraCalibration] = {}
    camera_file = parquet.ParquetFile(files.component("camera_calibration"))
    for batch in camera_file.iter_batches(
        batch_size=5,
        columns=list(_CAMERA_CALIBRATION_COLUMNS),
        use_threads=False,
    ):
        for row in range(batch.num_rows):
            segment, sensor_id = _sensor_key(batch, row, _CAMERA)
            _require_requested_segment(segment, files.segment_id)
            intrinsic = _readonly(
                [
                    [
                        _number(batch, f"{_CAM}.intrinsic.f_u", row),
                        0.0,
                        _number(batch, f"{_CAM}.intrinsic.c_u", row),
                    ],
                    [
                        0.0,
                        _number(batch, f"{_CAM}.intrinsic.f_v", row),
                        _number(batch, f"{_CAM}.intrinsic.c_v", row),
                    ],
                    [0.0, 0.0, 1.0],
                ],
                np.float64,
            )
            distortion = _readonly(
                [
                    _number(batch, f"{_CAM}.intrinsic.{name}", row)
                    for name in ("k1", "k2", "p1", "p2", "k3")
                ],
                np.float64,
            )
            cameras[sensor_id] = WaymoCameraCalibration(
                name=WAYMO_CAMERA_ORDER[sensor_id - 1],
                intrinsics=intrinsic,
                distortion_k1_k2_p1_p2_k3=distortion,
                vehicle_from_waymo_camera=_fixed_array(
                    batch, f"{_CAM}.extrinsic.transform", row, np.float64, (4, 4)
                ),
                width=_integer(batch, f"{_CAM}.width", row),
                height=_integer(batch, f"{_CAM}.height", row),
                rolling_shutter_direction=_integer(
                    batch, f"{_CAM}.rolling_shutter_direction", row
                ),
            )

    lidars: dict[int, WaymoLidarCalibration] = {}
    lidar_file = parquet.ParquetFile(files.component("lidar_calibration"))
    for batch in lidar_file.iter_batches(
        batch_size=5,
        columns=list(_LIDAR_CALIBRATION_COLUMNS),
        use_threads=False,
    ):
        for row in range(batch.num_rows):
            segment, sensor_id = _sensor_key(batch, row, _LIDAR)
            _require_requested_segment(segment, files.segment_id)
            values = _optional_list_array(
                batch,
                f"{_LIDAR_CAL}.beam_inclination.values",
                row,
                np.float64,
            )
            lidars[sensor_id] = WaymoLidarCalibration(
                name=WAYMO_LIDAR_ORDER[sensor_id - 1],
                vehicle_from_lidar=_fixed_array(
                    batch, f"{_LIDAR_CAL}.extrinsic.transform", row, np.float64, (4, 4)
                ),
                beam_inclination_min_rad=_number(
                    batch, f"{_LIDAR_CAL}.beam_inclination.min", row
                ),
                beam_inclination_max_rad=_number(
                    batch, f"{_LIDAR_CAL}.beam_inclination.max", row
                ),
                beam_inclination_values_rad=values,
            )
    try:
        return WaymoSegmentContext(
            files.segment_id,
            tuple(cameras[index] for index in range(1, 6)),
            tuple(lidars[index] for index in range(1, 6)),
        )
    except KeyError as exc:
        raise WaymoContractError("calibration payload lacks an audited sensor") from exc


def decode_v2_vehicle_pose(batch: Any, row: int) -> tuple[WaymoFrameKey, np.ndarray]:
    key = _frame_key(batch, row)
    return key, _fixed_array(
        batch,
        f"{_VEHICLE}.world_from_vehicle.transform",
        row,
        np.float64,
        (4, 4),
    )


def decode_v2_camera(
    batch: Any,
    row: int,
    context: WaymoSegmentContext,
) -> tuple[WaymoFrameKey, int, WaymoCameraFrame]:
    key = _frame_key(batch, row)
    sensor_id = _sensor_id(batch, _CAMERA, row)
    _require_requested_segment(key.segment_id, context.segment_id)
    encoded = _binary(batch, f"{_IMAGE}.image", row)
    try:
        import cv2
    except ImportError as exc:
        raise WaymoContractError("camera decoding requires OpenCV") from exc
    bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise WaymoContractError("camera image is not a valid color JPEG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.setflags(write=False)
    calibration = context.camera_calibrations[sensor_id - 1]
    return (
        key,
        sensor_id,
        WaymoCameraFrame(
            calibration=calibration,
            jpeg_bytes=encoded,
            rgb=rgb,
            world_from_vehicle_at_camera_pose_timestamp=_fixed_array(
                batch, f"{_IMAGE}.pose.transform", row, np.float64, (4, 4)
            ),
            global_linear_velocity_mps=_readonly(
                [
                    _number(batch, f"{_IMAGE}.velocity.linear_velocity.{axis}", row)
                    for axis in "xyz"
                ],
                np.float32,
            ),
            vehicle_angular_velocity_radps=_readonly(
                [
                    _number(batch, f"{_IMAGE}.velocity.angular_velocity.{axis}", row)
                    for axis in "xyz"
                ],
                np.float64,
            ),
            pose_timestamp_seconds=_number(batch, f"{_IMAGE}.pose_timestamp", row),
            camera_trigger_time_seconds=_number(
                batch, f"{_IMAGE}.rolling_shutter_params.camera_trigger_time", row
            ),
            camera_readout_done_time_seconds=_number(
                batch, f"{_IMAGE}.rolling_shutter_params.camera_readout_done_time", row
            ),
            shutter_seconds=_number(
                batch, f"{_IMAGE}.rolling_shutter_params.shutter", row
            ),
        ),
    )


def _frame_key(batch: Any, row: int) -> WaymoFrameKey:
    return WaymoFrameKey(
        _text(batch, _SEGMENT, row),
        _integer(batch, _TIMESTAMP, row),
    )


def _sensor_key(batch: Any, row: int, sensor_column: str) -> tuple[str, int]:
    return _text(batch, _SEGMENT, row), _sensor_id(batch, sensor_column, row)


def _sensor_id(batch: Any, sensor_column: str, row: int) -> int:
    value = _integer(batch, sensor_column, row)
    if value not in range(1, 6):
        raise WaymoContractError(f"{sensor_column} must contain IDs 1..5")
    return value


def _fixed_array(
    batch: Any,
    column: str,
    row: int,
    dtype: type[np.generic],
    shape: tuple[int, ...],
) -> np.ndarray:
    list_size = getattr(_column(batch, column).type, "list_size", None)
    if list_size != int(np.prod(shape)):
        raise WaymoContractError(f"{column} must be a fixed-size list")
    return _list_array(batch, column, row, dtype, shape=shape)


def _optional_list_array(
    batch: Any,
    column: str,
    row: int,
    dtype: type[np.generic],
) -> np.ndarray | None:
    scalar = _column(batch, column)[row]
    if not scalar.is_valid:
        return None
    return _copy_primitive_values(scalar.values, column, dtype)


def _list_array(
    batch: Any,
    column: str,
    row: int,
    dtype: type[np.generic],
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    scalar = _column(batch, column)[row]
    if not scalar.is_valid:
        raise WaymoContractError(f"{column} contains null")
    return _copy_primitive_values(scalar.values, column, dtype, shape=shape)


def _copy_primitive_values(
    values: Any,
    column: str,
    dtype: type[np.generic],
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if values.null_count:
        raise WaymoContractError(f"{column} contains null values")
    source = values.to_numpy(zero_copy_only=False)
    expected = np.dtype(dtype)
    if source.dtype != expected:
        raise WaymoContractError(
            f"{column} must contain {expected}, got {source.dtype}"
        )
    if shape is not None and source.size != int(np.prod(shape)):
        raise WaymoContractError(f"{column} value count differs from shape {shape}")
    shaped = source if shape is None else source.reshape(shape)
    result = np.array(shaped, dtype=expected, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly(values: object, dtype: type[np.generic]) -> np.ndarray:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _column(batch: Any, name: str):
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise WaymoContractError(f"missing payload column {name!r}")
    return batch.column(index)


def _text(batch: Any, name: str, row: int) -> str:
    column = _column(batch, name)
    if str(column.type) != "string":
        raise WaymoContractError(f"{name} must contain Arrow strings")
    value = _valid_scalar(column, name, row).as_py()
    if type(value) is not str or not value:
        raise WaymoContractError(f"{name} must contain non-empty strings")
    return value


def _integer(batch: Any, name: str, row: int) -> int:
    column = _column(batch, name)
    _require_primitive_dtype(column, name)
    value = _valid_scalar(column, name, row).as_py()
    if isinstance(value, bool) or not isinstance(value, int):
        raise WaymoContractError(f"{name} must contain integers")
    return int(value)


def _number(batch: Any, name: str, row: int) -> float:
    column = _column(batch, name)
    _require_primitive_dtype(column, name)
    value = _valid_scalar(column, name, row).as_py()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WaymoContractError(f"{name} must contain real numbers")
    return float(value)


def _binary(batch: Any, name: str, row: int) -> bytes:
    column = _column(batch, name)
    if str(column.type) != "binary":
        raise WaymoContractError(f"{name} must contain Arrow binary values")
    value = _valid_scalar(column, name, row).as_buffer().to_pybytes()
    if not value:
        raise WaymoContractError(f"{name} must contain non-empty bytes")
    return value


def _require_requested_segment(actual: str, expected: str) -> None:
    if actual != expected:
        raise WaymoContractError("payload segment does not match requested segment")


def _valid_scalar(column: Any, name: str, row: int):
    scalar = column[row]
    if not scalar.is_valid:
        raise WaymoContractError(f"{name} contains null")
    return scalar


def _require_primitive_dtype(column: Any, name: str) -> None:
    expected = _PRIMITIVE_DTYPES.get(name, np.dtype(np.float64))
    actual = str(column.type)
    if actual != _ARROW_PRIMITIVE_NAMES[expected]:
        raise WaymoContractError(f"{name} must contain {expected}, got {actual}")

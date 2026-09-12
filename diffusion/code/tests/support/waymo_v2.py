"""Generated modular Waymo v2 dataset used by reader integration tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SEGMENT = "segment"
TIMESTAMPS = (100, 200, 300, 400)
CAMERA_FILE_ORDER = (1, 2, 4, 3, 5)
LIDAR_FILE_ORDER = (5, 2, 4, 1, 3)


def write_reader_dataset(
    root: Path,
    *,
    segment: str = SEGMENT,
    partition: str = "validation",
    timestamps: tuple[int, ...] = TIMESTAMPS,
    camera_timestamp_order: tuple[int, ...] | None = None,
) -> None:
    """Write all seven exact components with small but distinct payloads."""
    camera_order = timestamps if camera_timestamp_order is None else camera_timestamp_order
    _write_calibrations(root, partition, segment)
    _write_vehicle_poses(root, partition, segment, timestamps)
    _write_cameras(root, partition, segment, camera_order)
    _write_lidar_pairs(root, partition, segment, timestamps, "lidar", channels=4)
    _write_lidar_pairs(
        root,
        partition,
        segment,
        timestamps,
        "lidar_camera_projection",
        channels=6,
    )
    _write_lidar_poses(root, partition, segment, timestamps)


def _write(
    root: Path,
    partition: str,
    segment: str,
    component: str,
    table: pa.Table,
    row_group_size: int,
) -> None:
    path = root / partition / component / f"{segment}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size)


def _rigid(offset: float) -> list[float]:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (offset, offset + 1.0, offset + 2.0)
    return transform.reshape(-1).tolist()


def _write_calibrations(root: Path, partition: str, segment: str) -> None:
    sensor_ids = [3, 5, 1, 4, 2]
    prefix = "[CameraCalibrationComponent]"
    intrinsics = {
        "f_u": [100.0 + value for value in sensor_ids],
        "f_v": [110.0 + value for value in sensor_ids],
        "c_u": [3.0] * 5,
        "c_v": [4.0] * 5,
        "k1": [0.1] * 5,
        "k2": [-0.02] * 5,
        "p1": [0.001] * 5,
        "p2": [-0.002] * 5,
        "k3": [0.003] * 5,
    }
    camera = pa.table(
        {
            "key.segment_context_name": pa.array([segment] * 5, pa.string()),
            "key.camera_name": pa.array(sensor_ids, pa.int8()),
            **{
                f"{prefix}.intrinsic.{name}": pa.array(values, pa.float64())
                for name, values in intrinsics.items()
            },
            f"{prefix}.extrinsic.transform": pa.array(
                [_rigid(float(value)) for value in sensor_ids],
                pa.list_(pa.float64(), 16),
            ),
            f"{prefix}.width": pa.array([8] * 5, pa.int32()),
            f"{prefix}.height": pa.array([8] * 5, pa.int32()),
            f"{prefix}.rolling_shutter_direction": pa.array([4] * 5, pa.int8()),
        }
    )
    prefix = "[LiDARCalibrationComponent]"
    lidar = pa.table(
        {
            "key.segment_context_name": pa.array([segment] * 5, pa.string()),
            "key.laser_name": pa.array(sensor_ids, pa.int8()),
            f"{prefix}.extrinsic.transform": pa.array(
                [_rigid(10.0 + value) for value in sensor_ids],
                pa.list_(pa.float64(), 16),
            ),
            f"{prefix}.beam_inclination.min": pa.array([-0.2] * 5, pa.float64()),
            f"{prefix}.beam_inclination.max": pa.array([0.1] * 5, pa.float64()),
            f"{prefix}.beam_inclination.values": pa.array(
                [[-0.2, 0.1] if value == 1 else None for value in sensor_ids],
                pa.list_(pa.float64()),
            ),
        }
    )
    _write(root, partition, segment, "camera_calibration", camera, 5)
    _write(root, partition, segment, "lidar_calibration", lidar, 5)


def _write_vehicle_poses(
    root: Path,
    partition: str,
    segment: str,
    timestamps: tuple[int, ...],
) -> None:
    table = pa.table(
        {
            "key.segment_context_name": pa.array([segment] * len(timestamps), pa.string()),
            "key.frame_timestamp_micros": pa.array(timestamps, pa.int64()),
            "[VehiclePoseComponent].world_from_vehicle.transform": pa.array(
                [_rigid(timestamp / 100.0) for timestamp in timestamps],
                pa.list_(pa.float64(), 16),
            ),
        }
    )
    _write(root, partition, segment, "vehicle_pose", table, 20)


def _jpeg(sensor_id: int) -> bytes:
    bgr = np.empty((8, 8, 3), dtype=np.uint8)
    bgr[:] = (10 * sensor_id, 40, 220 - 10 * sensor_id)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    return encoded.tobytes()


def _write_cameras(
    root: Path,
    partition: str,
    segment: str,
    timestamp_order: tuple[int, ...],
) -> None:
    rows = [
        (timestamp, sensor_id) for timestamp in timestamp_order for sensor_id in CAMERA_FILE_ORDER
    ]
    seconds = [timestamp / 1_000_000.0 for timestamp, _ in rows]
    table = pa.table(
        {
            "key.segment_context_name": pa.array([segment] * len(rows), pa.string()),
            "key.frame_timestamp_micros": pa.array(
                [timestamp for timestamp, _ in rows], pa.int64()
            ),
            "key.camera_name": pa.array([sensor_id for _, sensor_id in rows], pa.int8()),
            "[CameraImageComponent].image": pa.array(
                [_jpeg(sensor_id) for _, sensor_id in rows], pa.binary()
            ),
            "[CameraImageComponent].pose.transform": pa.array(
                [_rigid(timestamp / 100.0 + sensor_id / 10.0) for timestamp, sensor_id in rows],
                pa.list_(pa.float64(), 16),
            ),
            "[CameraImageComponent].velocity.linear_velocity.x": pa.array(
                [float(sensor_id) for _, sensor_id in rows], pa.float32()
            ),
            "[CameraImageComponent].velocity.linear_velocity.y": pa.array(
                [-float(sensor_id) for _, sensor_id in rows], pa.float32()
            ),
            "[CameraImageComponent].velocity.linear_velocity.z": pa.array(
                [0.5 * sensor_id for _, sensor_id in rows], pa.float32()
            ),
            "[CameraImageComponent].velocity.angular_velocity.x": pa.array(
                [0.01 * sensor_id for _, sensor_id in rows], pa.float64()
            ),
            "[CameraImageComponent].velocity.angular_velocity.y": pa.array(
                [-0.02 * sensor_id for _, sensor_id in rows], pa.float64()
            ),
            "[CameraImageComponent].velocity.angular_velocity.z": pa.array(
                [0.03 * sensor_id for _, sensor_id in rows], pa.float64()
            ),
            "[CameraImageComponent].pose_timestamp": pa.array(
                [value + 0.024 for value in seconds], pa.float64()
            ),
            "[CameraImageComponent].rolling_shutter_params.shutter": pa.array(
                [0.01] * len(rows), pa.float64()
            ),
            ("[CameraImageComponent].rolling_shutter_params." "camera_trigger_time"): pa.array(
                seconds, pa.float64()
            ),
            ("[CameraImageComponent].rolling_shutter_params." "camera_readout_done_time"): pa.array(
                [value + 0.044 for value in seconds], pa.float64()
            ),
        }
    )
    _write(root, partition, segment, "camera_image", table, 50)


def _write_lidar_pairs(
    root: Path,
    partition: str,
    segment: str,
    timestamps: tuple[int, ...],
    component: str,
    *,
    channels: int,
) -> None:
    rows = [(timestamp, sensor_id) for timestamp in timestamps for sensor_id in LIDAR_FILE_ORDER]
    prefix = "[LiDARComponent]" if component == "lidar" else "[LiDARCameraProjectionComponent]"
    shape = [2, 3, channels]
    first: list[list[float]] = []
    second: list[list[float]] = []
    for timestamp, sensor_id in rows:
        return1, return2 = lidar_pair_values(timestamp, sensor_id, channels)
        first.append(return1.reshape(-1).tolist())
        second.append(return2.reshape(-1).tolist())
    table = pa.table(
        {
            "key.segment_context_name": pa.array([segment] * len(rows), pa.string()),
            "key.frame_timestamp_micros": pa.array(
                [timestamp for timestamp, _ in rows], pa.int64()
            ),
            "key.laser_name": pa.array([sensor_id for _, sensor_id in rows], pa.int8()),
            f"{prefix}.range_image_return1.values": pa.array(first, pa.list_(pa.float32())),
            f"{prefix}.range_image_return1.shape": pa.array(
                [shape] * len(rows), pa.list_(pa.int32(), 3)
            ),
            f"{prefix}.range_image_return2.values": pa.array(second, pa.list_(pa.float32())),
            f"{prefix}.range_image_return2.shape": pa.array(
                [shape] * len(rows), pa.list_(pa.int32(), 3)
            ),
        }
    )
    _write(root, partition, segment, component, table, 50)


def lidar_pair_values(
    timestamp: int,
    sensor_id: int,
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two distinct arrays written for one generated sensor row."""
    shape = (2, 3, channels)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    values = values + timestamp + 10 * sensor_id
    if channels == 6:
        values[..., (0, 3)] = (1.0, 5.0)
        values[..., (1, 2, 4, 5)] += 0.25
    return values, values + 1000.5


def _write_lidar_poses(
    root: Path,
    partition: str,
    segment: str,
    timestamps: tuple[int, ...],
) -> None:
    shape = [2, 3, 6]
    table = pa.table(
        {
            "key.segment_context_name": pa.array([segment] * len(timestamps), pa.string()),
            "key.frame_timestamp_micros": pa.array(timestamps, pa.int64()),
            "key.laser_name": pa.array([1] * len(timestamps), pa.int8()),
            "[LiDARPoseComponent].range_image_return1.values": pa.array(
                [
                    (np.arange(36, dtype=np.float32) + timestamp).tolist()
                    for timestamp in timestamps
                ],
                pa.list_(pa.float32()),
            ),
            "[LiDARPoseComponent].range_image_return1.shape": pa.array(
                [shape] * len(timestamps), pa.list_(pa.int32(), 3)
            ),
        }
    )
    _write(root, partition, segment, "lidar_pose", table, 20)

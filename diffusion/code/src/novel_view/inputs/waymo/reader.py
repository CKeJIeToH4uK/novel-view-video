"""Bounded single-pass reader for one modular Waymo Perception v2 window."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from .camera import WaymoCameraFrame
from .decode import (
    CAMERA_IMAGE_COLUMNS,
    VEHICLE_POSE_COLUMNS,
    _frame_key,
    decode_v2_camera,
    decode_v2_vehicle_pose,
    read_v2_segment_context,
)
from .frame import WaymoFrameBundle, WaymoSegmentContext
from .index import (
    WaymoV2WindowIndex,
    _import_parquet,
    build_waymo_v2_window_index,
)
from .lidar import WaymoLidarFrame, WaymoLidarReturn
from .lidar_decode import (
    LIDAR_POSE_COLUMNS,
    LIDAR_PROJECTION_COLUMNS,
    LIDAR_RANGE_COLUMNS,
    DecodedLidarPair,
    decode_v2_lidar_pose,
    decode_v2_lidar_projections,
    decode_v2_lidar_ranges,
)
from .types import (
    WaymoContractError,
    WaymoFrameKey,
    WaymoOfficialPartition,
)


_T = TypeVar("_T")
_SENSOR_IDS = (1, 2, 3, 4, 5)
_END = object()


class WaymoV2FrameReader(Iterator[WaymoFrameBundle]):
    """One-shot frame iterator with an inspectable key-only index."""

    def __init__(
        self,
        waymo_root: str | Path,
        official_partition: WaymoOfficialPartition,
        segment_id: str,
        start_frame_index: int,
        count: int,
    ) -> None:
        self.index = build_waymo_v2_window_index(
            waymo_root,
            official_partition,
            segment_id,
            start_frame_index,
            count,
        )
        self._frames: Iterator[WaymoFrameBundle] | None = None

    def __iter__(self) -> WaymoV2FrameReader:
        return self

    def __next__(self) -> WaymoFrameBundle:
        if self._frames is None:
            self._frames = _iter_frames(self.index)
        return next(self._frames)


def _iter_frames(index: WaymoV2WindowIndex) -> Iterator[WaymoFrameBundle]:
    parquet = _import_parquet()
    context = read_v2_segment_context(index.files)
    streams = (
        _iter_vehicle_poses(parquet, index),
        _iter_camera_groups(parquet, index, context),
        _iter_lidar_pair_groups(
            parquet,
            index,
            "lidar",
            LIDAR_RANGE_COLUMNS,
            decode_v2_lidar_ranges,
        ),
        _iter_lidar_pair_groups(
            parquet,
            index,
            "lidar_camera_projection",
            LIDAR_PROJECTION_COLUMNS,
            decode_v2_lidar_projections,
        ),
        _iter_pose_groups(parquet, index),
    )
    for expected_key in index.window:
        values = tuple(next(stream, _END) for stream in streams)
        if any(value is _END for value in values):
            raise WaymoContractError(
                "payload component ended before the audited window"
            )
        vehicle, cameras, ranges, projections, poses = values
        component_keys = (
            vehicle[0],
            cameras[0],
            ranges[0],
            projections[0],
            poses[0],
        )
        if any(key != expected_key for key in component_keys):
            raise WaymoContractError(
                "payload component keys differ from audited window"
            )
        yield WaymoFrameBundle(
            key=expected_key,
            context=context,
            world_from_vehicle_for_frame=vehicle[1],
            cameras=cameras[1],
            lidars=_assemble_lidars(
                context,
                ranges[1],
                projections[1],
                poses[1][0],
            ),
        )
        del values, vehicle, cameras, ranges, projections, poses, component_keys
    if any(next(stream, _END) is not _END for stream in streams):
        raise WaymoContractError(
            "payload component continues after the audited window"
        )


def _iter_vehicle_poses(
    parquet: Any,
    index: WaymoV2WindowIndex,
) -> Iterator[tuple[WaymoFrameKey, np.ndarray]]:
    for batch, row, expected_key in _iter_payload_positions(
        parquet,
        index,
        "vehicle_pose",
        VEHICLE_POSE_COLUMNS,
        batch_size=1,
    ):
        key, pose = decode_v2_vehicle_pose(batch, row)
        if key != expected_key:
            raise WaymoContractError("vehicle_pose decoder changed its key")
        yield key, pose


def _iter_camera_groups(
    parquet: Any,
    index: WaymoV2WindowIndex,
    context: WaymoSegmentContext,
) -> Iterator[tuple[WaymoFrameKey, tuple[WaymoCameraFrame, ...]]]:
    rows = (
        decode_v2_camera(batch, row, context)
        for batch, row, _ in _iter_payload_positions(
            parquet,
            index,
            "camera_image",
            CAMERA_IMAGE_COLUMNS,
            batch_size=5,
        )
    )
    yield from _group_sensor_rows(rows, _SENSOR_IDS, "camera_image")


def _iter_lidar_pair_groups(
    parquet: Any,
    index: WaymoV2WindowIndex,
    component: str,
    columns: tuple[str, ...],
    decoder: Callable[[Any, int], DecodedLidarPair],
) -> Iterator[tuple[WaymoFrameKey, tuple[DecodedLidarPair, ...]]]:
    rows = (
        (decoded.key, decoded.sensor_id, decoded)
        for batch, row, _ in _iter_payload_positions(
            parquet,
            index,
            component,
            columns,
            batch_size=5,
        )
        for decoded in (decoder(batch, row),)
    )
    yield from _group_sensor_rows(rows, _SENSOR_IDS, component)


def _iter_pose_groups(
    parquet: Any,
    index: WaymoV2WindowIndex,
) -> Iterator[tuple[WaymoFrameKey, tuple[np.ndarray, ...]]]:
    rows = (
        decode_v2_lidar_pose(batch, row)
        for batch, row, _ in _iter_payload_positions(
            parquet,
            index,
            "lidar_pose",
            LIDAR_POSE_COLUMNS,
            batch_size=1,
        )
    )
    yield from _group_sensor_rows(rows, (1,), "lidar_pose")


def _iter_payload_positions(
    parquet: Any,
    index: WaymoV2WindowIndex,
    component: str,
    columns: tuple[str, ...],
    *,
    batch_size: int,
):
    file = parquet.ParquetFile(index.files.component(component))
    first = index.window[0].frame_timestamp_micros
    last = index.window[-1].frame_timestamp_micros
    previous_timestamp: int | None = None
    for batch in file.iter_batches(
        batch_size=batch_size,
        row_groups=list(index.row_groups(component)),
        columns=list(columns),
        use_threads=False,
    ):
        for row in range(batch.num_rows):
            key = _frame_key(batch, row)
            if key.segment_id != index.files.segment_id:
                raise WaymoContractError(f"{component} contains another segment")
            timestamp = key.frame_timestamp_micros
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise WaymoContractError(f"{component} payload timestamps descend")
            previous_timestamp = timestamp
            if first <= timestamp <= last:
                yield batch, row, key


def _group_sensor_rows(
    rows: Iterator[tuple[WaymoFrameKey, int, _T]],
    expected_sensor_ids: tuple[int, ...],
    component: str,
) -> Iterator[tuple[WaymoFrameKey, tuple[_T, ...]]]:
    current_key: WaymoFrameKey | None = None
    completed_key: WaymoFrameKey | None = None
    current: dict[int, _T] = {}
    for key, sensor_id, value in rows:
        if completed_key is not None and key <= completed_key:
            raise WaymoContractError(
                f"{component} repeats or descends after a complete frame"
            )
        if current_key is None:
            current_key = key
        elif key != current_key:
            raise WaymoContractError(
                f"{component} frame ended before all sensors arrived"
            )
        if sensor_id in current:
            raise WaymoContractError(f"{component} repeats sensor {sensor_id}")
        current[sensor_id] = value
        if len(current) == len(expected_sensor_ids):
            ordered = _ordered_group(current, expected_sensor_ids, component)
            completed_key = current_key
            current_key = None
            current = {}
            yield completed_key, ordered
            del ordered, key, sensor_id, value
    if current_key is not None:
        raise WaymoContractError(f"{component} ends with an incomplete sensor frame")


def _ordered_group(
    values: dict[int, _T],
    expected_sensor_ids: tuple[int, ...],
    component: str,
) -> tuple[_T, ...]:
    if set(values) != set(expected_sensor_ids):
        raise WaymoContractError(f"{component} frame has incomplete sensor payload")
    return tuple(values[sensor_id] for sensor_id in expected_sensor_ids)


def _assemble_lidars(
    context: WaymoSegmentContext,
    ranges: tuple[DecodedLidarPair, ...],
    projections: tuple[DecodedLidarPair, ...],
    top_pose: np.ndarray,
) -> tuple[WaymoLidarFrame, ...]:
    frames: list[WaymoLidarFrame] = []
    for calibration, range_pair, projection_pair in zip(
        context.lidar_calibrations,
        ranges,
        projections,
        strict=True,
    ):
        if range_pair.sensor_id != projection_pair.sensor_id:
            raise WaymoContractError("lidar and projection sensor IDs differ")
        frames.append(
            WaymoLidarFrame(
                calibration=calibration,
                return1=WaymoLidarReturn(
                    range_pair.return1,
                    projection_pair.return1,
                ),
                return2=WaymoLidarReturn(
                    range_pair.return2,
                    projection_pair.return2,
                ),
                pixel_pose_vehicle_to_global_rpy_xyz=(
                    top_pose if calibration.name == "TOP" else None
                ),
            )
        )
    return tuple(frames)


__all__ = ["WaymoV2FrameReader"]

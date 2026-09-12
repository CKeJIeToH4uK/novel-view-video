"""Small key-only index for one modular Waymo Perception v2 segment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frame import select_waymo_frame_window
from .types import (
    WaymoContractError,
    WaymoFrameKey,
    WaymoOfficialPartition,
)


_SEGMENT = "key.segment_context_name"
_TIMESTAMP = "key.frame_timestamp_micros"
_CAMERA = "key.camera_name"
_LIDAR = "key.laser_name"

_TIMED_COMPONENTS = (
    "vehicle_pose",
    "camera_image",
    "lidar",
    "lidar_camera_projection",
    "lidar_pose",
)
_ALL_COMPONENTS = _TIMED_COMPONENTS + (
    "camera_calibration",
    "lidar_calibration",
)
_EXPECTED_SENSOR_IDS = frozenset(range(1, 6))


@dataclass(frozen=True, slots=True)
class WaymoV2Files:
    """Exact seven Parquet files for one segment; no recursive discovery."""

    root: Path
    official_partition: WaymoOfficialPartition
    segment_id: str

    def component(self, name: str) -> Path:
        if name not in _ALL_COMPONENTS:
            raise WaymoContractError(f"unknown Waymo v2 component {name!r}")
        return self.root / self.official_partition / name / f"{self.segment_id}.parquet"


@dataclass(frozen=True, slots=True)
class WaymoV2WindowIndex:
    """Key-only proof and row groups required for one neighbouring window."""

    files: WaymoV2Files
    timeline: tuple[WaymoFrameKey, ...]
    window: tuple[WaymoFrameKey, ...]
    selected_row_groups: tuple[tuple[str, tuple[int, ...]], ...]

    def row_groups(self, component: str) -> tuple[int, ...]:
        for name, groups in self.selected_row_groups:
            if name == component:
                return groups
        raise WaymoContractError(f"component {component!r} is not time-indexed")


def build_waymo_v2_window_index(
    waymo_root: str | Path,
    official_partition: WaymoOfficialPartition,
    segment_id: str,
    start_frame_index: int,
    count: int,
) -> WaymoV2WindowIndex:
    """Audit a segment's key grid and select row groups for one window.

    PyArrow is imported only when this explicit data-ingress leaf is called.
    No camera, LiDAR, pose, or image payload column is read here.
    """
    parquet = _import_parquet()
    files = WaymoV2Files(Path(waymo_root).expanduser(), official_partition, segment_id)
    timeline = _read_timeline(parquet.ParquetFile(files.component("vehicle_pose")))
    if timeline[0].segment_id != files.segment_id:
        raise WaymoContractError(
            "vehicle_pose segment does not match requested segment"
        )
    _audit_segment_keys(parquet, files, timeline)
    window = select_waymo_frame_window(timeline, start_frame_index, count)
    first = window[0].frame_timestamp_micros
    last = window[-1].frame_timestamp_micros
    selected = tuple(
        (
            component,
            _select_overlapping_row_groups(
                parquet.ParquetFile(files.component(component)),
                first,
                last,
                component,
            ),
        )
        for component in _TIMED_COMPONENTS
    )
    return WaymoV2WindowIndex(files, timeline, window, selected)


def _import_parquet():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise WaymoContractError(
            "Waymo v2 reading requires the 'waymo-data' optional dependency"
        ) from exc
    return parquet


def _read_timeline(file: Any) -> tuple[WaymoFrameKey, ...]:
    keys: list[WaymoFrameKey] = []
    for segment, timestamp in _iter_key_rows(file, (_SEGMENT, _TIMESTAMP)):
        keys.append(
            WaymoFrameKey(_require_segment(segment), _require_timestamp(timestamp))
        )
    if not keys:
        raise WaymoContractError("vehicle_pose timeline must not be empty")
    # This also rejects mixed segments, duplicates, and descending input.
    return select_waymo_frame_window(keys, 0, len(keys))


def _audit_segment_keys(
    parquet: Any,
    files: WaymoV2Files,
    timeline: tuple[WaymoFrameKey, ...],
) -> None:
    expected_timestamps = {key.frame_timestamp_micros for key in timeline}
    for component, sensor_column, sensor_ids in (
        ("camera_image", _CAMERA, _EXPECTED_SENSOR_IDS),
        ("lidar", _LIDAR, _EXPECTED_SENSOR_IDS),
        ("lidar_camera_projection", _LIDAR, _EXPECTED_SENSOR_IDS),
        ("lidar_pose", _LIDAR, frozenset({1})),
    ):
        _audit_frame_component(
            parquet.ParquetFile(files.component(component)),
            files.segment_id,
            expected_timestamps,
            sensor_column=sensor_column,
            expected_sensor_ids=sensor_ids,
            component=component,
        )
    _audit_calibrations(
        parquet.ParquetFile(files.component("camera_calibration")),
        files.segment_id,
        _CAMERA,
        "camera_calibration",
    )
    _audit_calibrations(
        parquet.ParquetFile(files.component("lidar_calibration")),
        files.segment_id,
        _LIDAR,
        "lidar_calibration",
    )


def _audit_frame_component(
    file: Any,
    segment_id: str,
    expected_timestamps: set[int],
    *,
    sensor_column: str | None,
    expected_sensor_ids: frozenset[int],
    component: str,
) -> None:
    columns = (_SEGMENT, _TIMESTAMP)
    if sensor_column is not None:
        columns += (sensor_column,)
    counts: Counter[tuple[int, int | None]] = Counter()
    for row in _iter_key_rows(file, columns):
        segment = _require_segment(row[0])
        timestamp = _require_timestamp(row[1])
        sensor_id = None if sensor_column is None else _require_sensor_id(row[2])
        if segment != segment_id:
            raise WaymoContractError(f"{component} contains another segment")
        counts[(timestamp, sensor_id)] += 1
    actual_timestamps = {timestamp for timestamp, _ in counts}
    if actual_timestamps != expected_timestamps:
        raise WaymoContractError(f"{component} timestamp set differs from vehicle_pose")
    for timestamp in expected_timestamps:
        actual_ids = {
            sensor_id
            for (row_timestamp, sensor_id), multiplicity in counts.items()
            if row_timestamp == timestamp and multiplicity > 0 and sensor_id is not None
        }
        if sensor_column is None:
            if counts[(timestamp, None)] != 1:
                raise WaymoContractError(f"{component} must contain one row per frame")
        elif actual_ids != expected_sensor_ids or any(
            counts[(timestamp, sensor_id)] != 1 for sensor_id in expected_sensor_ids
        ):
            raise WaymoContractError(
                f"{component} has an incomplete or duplicate sensor grid"
            )


def _audit_calibrations(
    file: Any,
    segment_id: str,
    sensor_column: str,
    component: str,
) -> None:
    counts: Counter[int] = Counter()
    for segment, sensor in _iter_key_rows(file, (_SEGMENT, sensor_column)):
        if _require_segment(segment) != segment_id:
            raise WaymoContractError(f"{component} contains another segment")
        counts[_require_sensor_id(sensor)] += 1
    if set(counts) != _EXPECTED_SENSOR_IDS or any(
        counts[sensor_id] != 1 for sensor_id in _EXPECTED_SENSOR_IDS
    ):
        raise WaymoContractError(f"{component} must contain five unique sensors")


def _iter_key_rows(file: Any, columns: tuple[str, ...]):
    for row_group in range(file.metadata.num_row_groups):
        for batch in file.iter_batches(
            batch_size=1024,
            row_groups=[row_group],
            columns=list(columns),
            use_threads=False,
        ):
            values = []
            for column in columns:
                index = batch.schema.get_field_index(column)
                if index < 0:
                    raise WaymoContractError(f"missing key column {column!r}")
                array = batch.column(index)
                if array.null_count:
                    raise WaymoContractError(f"key column {column!r} contains nulls")
                values.append(array.to_pylist())
            yield from zip(*values, strict=True)


def _select_overlapping_row_groups(
    file: Any,
    first_timestamp: int,
    last_timestamp: int,
    component: str,
) -> tuple[int, ...]:
    selected: list[int] = []
    for row_group_index in range(file.metadata.num_row_groups):
        row_group = file.metadata.row_group(row_group_index)
        timestamp_columns = [
            row_group.column(index)
            for index in range(row_group.num_columns)
            if row_group.column(index).path_in_schema == _TIMESTAMP
        ]
        if len(timestamp_columns) != 1:
            raise WaymoContractError(f"{component} has no unique timestamp column")
        statistics = timestamp_columns[0].statistics
        if statistics is None or not statistics.has_min_max:
            raise WaymoContractError(
                f"{component} row group {row_group_index} lacks timestamp min/max"
            )
        minimum = _require_timestamp(statistics.min)
        maximum = _require_timestamp(statistics.max)
        if maximum < minimum:
            raise WaymoContractError(f"{component} row-group statistics are reversed")
        if maximum >= first_timestamp and minimum <= last_timestamp:
            selected.append(row_group_index)
    if not selected:
        raise WaymoContractError(f"{component} has no row groups for requested window")
    return tuple(selected)


def _require_segment(value: object) -> str:
    if type(value) is not str or not value:
        raise WaymoContractError("segment key must be a non-empty string")
    return value


def _require_timestamp(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WaymoContractError("frame timestamp must be a non-negative integer")
    return int(value)


def _require_sensor_id(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in range(1, 6)
    ):
        raise WaymoContractError("sensor key must be an official integer ID 1..5")
    return int(value)

"""Упорядоченные центральные Waymo ключи и прежний внешний JSON record."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from novel_view.inputs.waymo.frame import select_waymo_frame_window
from novel_view.inputs.waymo.types import (
    WaymoContractError,
    WaymoFrameKey,
    WaymoOfficialPartition,
)
from novel_view.models.gen3c.spec import GEN3C_WINDOW_SIZE


_SCHEMA_VERSION = "gen3c-waymo-keyset/v1"


@dataclass(frozen=True, slots=True)
class WaymoClipKey:
    """Один центральный клип, выбранный до чтения RGB и геометрии."""

    split_id: str
    official_partition: WaymoOfficialPartition
    segment_id: str
    start_frame_index: int
    frame_timestamps_micros: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PreparedKeyset:
    """Упорядоченные ключи из одного явно названного JSON."""

    keyset_id: str
    keys: tuple[WaymoClipKey, ...]

    @classmethod
    def load(cls, path: str | Path) -> PreparedKeyset:
        """Прочитать прежнюю внешнюю схему без поиска файлов."""
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {
            "schema_version", "keyset_id", "keys",
        }:
            raise WaymoContractError("keyset file has unexpected fields")
        if document["schema_version"] != _SCHEMA_VERSION:
            raise WaymoContractError("unsupported keyset schema_version")
        keyset_id = _text(document["keyset_id"], "keyset_id")
        raw_keys = document["keys"]
        if not isinstance(raw_keys, list) or not raw_keys:
            raise WaymoContractError("keyset keys must be a non-empty list")
        keys = tuple(clip_key_from_mapping(value) for value in raw_keys)
        if len({key.split_id for key in keys}) != 1:
            raise WaymoContractError("one keyset must use one split_id")
        if len({(key.official_partition, key.segment_id) for key in keys}) != len(keys):
            raise WaymoContractError("one keyset must not repeat clip identities")
        return cls(keyset_id, keys)

    def write(self, path: str | Path) -> None:
        """Записать прежний record прямо в переданный путь."""
        document = {
            "schema_version": _SCHEMA_VERSION,
            "keyset_id": self.keyset_id,
            "keys": [
                clip_key_to_mapping(key)
                for key in self.keys
            ],
        }
        Path(path).write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


TimelineLoader = Callable[[WaymoOfficialPartition, str], Sequence[WaymoFrameKey]]


def build_central_keyset(
    *,
    keyset_id: str,
    split_id: str,
    official_partition: WaymoOfficialPartition,
    segment_ids: Sequence[str],
    load_timeline: TimelineLoader,
) -> PreparedKeyset:
    """Выбрать central121 по timeline в точном порядке сегментов."""
    keys = []
    for segment_id in segment_ids:
        timeline = tuple(load_timeline(official_partition, segment_id))
        start = (len(timeline) - GEN3C_WINDOW_SIZE) // 2
        window = select_waymo_frame_window(timeline, start, GEN3C_WINDOW_SIZE)
        keys.append(
            WaymoClipKey(
                split_id, official_partition, segment_id, start,
                tuple(key.frame_timestamp_micros for key in window),
            )
        )
    return PreparedKeyset(keyset_id, tuple(keys))


def clip_key_from_mapping(value: object) -> WaymoClipKey:
    """Read one strict external clip key shared by historical records."""
    expected = {
        "split_id", "official_partition", "segment_id",
        "start_frame_index", "frame_timestamps_micros",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise WaymoContractError("clip key has unexpected fields")
    split_id = _text(value["split_id"], "split_id")
    segment_id = _text(value["segment_id"], "segment_id")
    partition = value["official_partition"]
    if partition not in ("training", "validation"):
        raise WaymoContractError("official_partition must be training or validation")
    start = value["start_frame_index"]
    if type(start) is not int or start < 0:
        raise WaymoContractError("start_frame_index must be a non-negative int")
    timestamps = value["frame_timestamps_micros"]
    if not isinstance(timestamps, list) or len(timestamps) != GEN3C_WINDOW_SIZE:
        raise WaymoContractError("clip timestamps must contain exactly 121 values")
    if any(type(value) is not int or value < 0 for value in timestamps):
        raise WaymoContractError("clip timestamps must be non-negative ints")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise WaymoContractError("clip timestamps must be strictly increasing")
    return WaymoClipKey(split_id, partition, segment_id, start, tuple(timestamps))


def clip_key_to_mapping(key: WaymoClipKey) -> dict[str, object]:
    """Serialize one already selected clip key without another validation layer."""
    return {
        "split_id": key.split_id,
        "official_partition": key.official_partition,
        "segment_id": key.segment_id,
        "start_frame_index": key.start_frame_index,
        "frame_timestamps_micros": list(key.frame_timestamps_micros),
    }


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WaymoContractError(f"{name} must be non-empty text")
    return value


__all__ = [
    "PreparedKeyset",
    "WaymoClipKey",
    "build_central_keyset",
    "clip_key_from_mapping",
    "clip_key_to_mapping",
]

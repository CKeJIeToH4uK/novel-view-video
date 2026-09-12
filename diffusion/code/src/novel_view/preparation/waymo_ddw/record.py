"""One portable ordered record for prepared Waymo DDW items."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from novel_view.inputs.waymo.types import (
    WaymoFrameKey,
    WaymoOfficialPartition,
)
from novel_view.preparation.waymo_ddw.artifacts import LIDAR_DEPTH_FORMAT
from novel_view.preparation.waymo_ddw.selection import SelectedDdwSample
from novel_view.preparation.waymo_ddw.spec import R4C_MODEL_CONTRACT, WaymoDdwSpec


PREPARED_FORMAT = "novel-view/waymo-ddw-prepared/v1"
_FRAME_COUNT = 121
_RECORD_FIELDS = {
    "format",
    "workflow",
    "input",
    "model_contract",
    "condition_depth",
    "depth_supervision",
    "gen3c",
    "items",
}
_ITEM_FIELDS = {
    "sample_id",
    "partition",
    "segment_id",
    "start_frame_index",
    "frame_timestamps_micros",
    "ddw",
    "base_latent",
    "pose_latent",
    "lidar_depth",
}


@dataclass(frozen=True, slots=True)
class PreparedItem:
    sample_id: str
    partition: WaymoOfficialPartition
    segment_id: str
    start_frame_index: int
    frame_timestamps_micros: tuple[int, ...]
    magnitude_m: float
    sign: int
    base_latent: str
    pose_latent: str
    lidar_depth: str


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    reader: str
    dataset: str
    model_contract: str
    depth_backend: str
    depth_checkpoint: str
    tokenizer: str
    text_encoder: str
    empty_prompt: str
    items: tuple[PreparedItem, ...]


def build_prepared_item(
    selected: SelectedDdwSample,
    frame_keys: tuple[WaymoFrameKey, ...],
) -> PreparedItem:
    """Bind one selected sample to the exact frames that were materialized."""
    prefix = f"items/{selected.sample_id}"
    return PreparedItem(
        sample_id=selected.sample_id,
        partition=selected.partition,
        segment_id=selected.segment_id,
        start_frame_index=selected.start_frame_index,
        frame_timestamps_micros=tuple(key.frame_timestamp_micros for key in frame_keys),
        magnitude_m=selected.magnitude_m,
        sign=selected.sign,
        base_latent=f"{prefix}/base.pt",
        pose_latent=f"{prefix}/pose.pt",
        lidar_depth=f"{prefix}/lidar-depth.pt",
    )


def build_prepared_record(
    spec: WaymoDdwSpec,
    items: tuple[PreparedItem, ...],
) -> PreparedRecord:
    """Build the one v1 record after every ordered item bake has succeeded."""
    return PreparedRecord(
        reader=spec.input.reader,
        dataset=spec.input.dataset,
        model_contract=spec.recipe.model_contract,
        depth_backend=spec.recipe.depth_backend,
        depth_checkpoint=spec.recipe.depth_checkpoint,
        tokenizer=spec.recipe.tokenizer,
        text_encoder=spec.recipe.text_encoder,
        empty_prompt="empty-prompt.pt",
        items=items,
    )


def write_prepared_record(path: Path, record: PreparedRecord) -> None:
    """Write the final record directly after successful item preparation."""
    path.write_text(
        json.dumps(_document(record), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_prepared_record(path: Path) -> PreparedRecord:
    """Read and validate the portable producer boundary once."""
    document = _fields(
        json.loads(path.read_text(encoding="utf-8")),
        _RECORD_FIELDS,
        "prepared record",
    )
    if document["format"] != PREPARED_FORMAT:
        raise ValueError("unsupported prepared record format")

    workflow = _mapping(document, "workflow", {"name", "version"})
    if workflow != {"name": "ddw_preparation", "version": 1}:
        raise ValueError("prepared record has the wrong workflow")
    input_values = _mapping(document, "input", {"reader", "dataset"})
    if input_values != {"reader": "waymo_v2", "dataset": "waymo"}:
        raise ValueError("prepared record has the wrong input")
    if document["model_contract"] != R4C_MODEL_CONTRACT:
        raise ValueError("prepared record has the wrong model contract")
    depth = _mapping(document, "condition_depth", {"backend", "checkpoint"})
    if depth["backend"] != "moge":
        raise ValueError("prepared record has the wrong depth backend")
    supervision = _mapping(document, "depth_supervision", {"format"})
    if supervision["format"] != LIDAR_DEPTH_FORMAT:
        raise ValueError("prepared record has the wrong depth supervision")
    gen3c = _mapping(
        document,
        "gen3c",
        {"tokenizer", "text_encoder", "empty_prompt"},
    )
    if gen3c["empty_prompt"] != "empty-prompt.pt":
        raise ValueError("prepared record has the wrong empty prompt locator")

    rows = document["items"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("prepared record items must be a nonempty list")
    items = tuple(_item(row) for row in rows)
    if len({item.sample_id for item in items}) != len(items):
        raise ValueError("prepared record contains duplicate sample_id")
    return PreparedRecord(
        reader="waymo_v2",
        dataset="waymo",
        model_contract=R4C_MODEL_CONTRACT,
        depth_backend="moge",
        depth_checkpoint=_text(depth["checkpoint"], "condition depth checkpoint"),
        tokenizer=_text(gen3c["tokenizer"], "Gen3C tokenizer"),
        text_encoder=_text(gen3c["text_encoder"], "Gen3C text encoder"),
        empty_prompt="empty-prompt.pt",
        items=items,
    )


def _item(value: object) -> PreparedItem:
    row = _fields(value, _ITEM_FIELDS, "prepared item")
    sample_id = _text(row["sample_id"], "sample_id")
    partition = _partition(row["partition"])
    timestamps = row["frame_timestamps_micros"]
    if (
        not isinstance(timestamps, list)
        or len(timestamps) != _FRAME_COUNT
        or any(not _is_int(timestamp) for timestamp in timestamps)
    ):
        raise ValueError("prepared item must contain 121 integer timestamps")
    start = row["start_frame_index"]
    if not _is_int(start) or start < 0:
        raise ValueError("prepared item has the wrong start_frame_index")
    ddw = _mapping(row, "ddw", {"magnitude_m", "sign"})
    magnitude = ddw["magnitude_m"]
    sign = ddw["sign"]
    if (
        not isinstance(magnitude, (int, float))
        or isinstance(magnitude, bool)
        or not math.isfinite(magnitude)
        or magnitude < 0
        or not _is_int(sign)
        or sign not in (-1, 1)
    ):
        raise ValueError("prepared item has the wrong DDW variant")
    prefix = f"items/{sample_id}"
    locators = {
        "base_latent": f"{prefix}/base.pt",
        "pose_latent": f"{prefix}/pose.pt",
        "lidar_depth": f"{prefix}/lidar-depth.pt",
    }
    if any(row[field] != expected for field, expected in locators.items()):
        raise ValueError("prepared item has a non-relative artifact locator")
    return PreparedItem(
        sample_id=sample_id,
        partition=partition,
        segment_id=_text(row["segment_id"], "segment_id"),
        start_frame_index=start,
        frame_timestamps_micros=tuple(int(timestamp) for timestamp in timestamps),
        magnitude_m=float(magnitude),
        sign=sign,
        base_latent=locators["base_latent"],
        pose_latent=locators["pose_latent"],
        lidar_depth=locators["lidar_depth"],
    )


def _document(record: PreparedRecord) -> dict[str, object]:
    return {
        "format": PREPARED_FORMAT,
        "workflow": {"name": "ddw_preparation", "version": 1},
        "input": {"reader": record.reader, "dataset": record.dataset},
        "model_contract": record.model_contract,
        "condition_depth": {
            "backend": record.depth_backend,
            "checkpoint": record.depth_checkpoint,
        },
        "depth_supervision": {"format": LIDAR_DEPTH_FORMAT},
        "gen3c": {
            "tokenizer": record.tokenizer,
            "text_encoder": record.text_encoder,
            "empty_prompt": record.empty_prompt,
        },
        "items": [
            {
                "sample_id": item.sample_id,
                "partition": item.partition,
                "segment_id": item.segment_id,
                "start_frame_index": item.start_frame_index,
                "frame_timestamps_micros": list(item.frame_timestamps_micros),
                "ddw": {"magnitude_m": item.magnitude_m, "sign": item.sign},
                "base_latent": item.base_latent,
                "pose_latent": item.pose_latent,
                "lidar_depth": item.lidar_depth,
            }
            for item in record.items
        ],
    }


def _mapping(
    values: Mapping[str, object],
    field: str,
    expected_fields: set[str],
) -> dict[str, object]:
    value = values[field]
    return _fields(value, expected_fields, field)


def _fields(value: object, expected: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has unexpected fields")
    return value


def _partition(value: object) -> WaymoOfficialPartition:
    if value == "training":
        return "training"
    if value == "validation":
        return "validation"
    raise ValueError("prepared item has the wrong partition")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "PREPARED_FORMAT",
    "PreparedItem",
    "PreparedRecord",
    "build_prepared_item",
    "build_prepared_record",
    "read_prepared_record",
    "write_prepared_record",
]

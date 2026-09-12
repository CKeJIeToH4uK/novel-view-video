"""Strict ordered selection for Waymo DDW preparation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

from novel_view.config.load import load_yaml_mapping, reject_unknown_fields
from novel_view.inputs.waymo.types import WaymoOfficialPartition


_SELECTION_FIELDS = frozenset({"schema_version", "samples"})
_SAMPLE_FIELDS = frozenset(
    {"sample_id", "partition", "segment_id", "start_frame_index", "ddw"}
)
_DDW_FIELDS = frozenset({"magnitude_m", "sign"})


@dataclass(frozen=True, slots=True)
class SelectedDdwSample:
    """One stable Waymo clip identity and its requested DDW displacement."""

    sample_id: str
    partition: WaymoOfficialPartition
    segment_id: str
    start_frame_index: int
    magnitude_m: float
    sign: int


@dataclass(frozen=True, slots=True)
class WaymoDdwSelection:
    """Samples in the exact order declared by the researcher."""

    samples: tuple[SelectedDdwSample, ...]


def load_selection(source: Path) -> WaymoDdwSelection:
    """Read selection YAML without opening Waymo or model files."""
    values = load_yaml_mapping(source)
    reject_unknown_fields(values, _SELECTION_FIELDS)
    _require_fields(values, _SELECTION_FIELDS, "selection")
    if _integer(values, "schema_version") != 1:
        raise ValueError("unsupported Waymo DDW selection schema_version")

    sample_values = _mapping_list(values, "samples")
    if not sample_values:
        raise ValueError("Waymo DDW selection must contain at least one sample")
    samples = tuple(_sample(value) for value in sample_values)
    sample_ids = tuple(sample.sample_id for sample in samples)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Waymo DDW selection contains duplicate sample_id")
    return WaymoDdwSelection(samples=samples)


def _sample(values: Mapping[str, object]) -> SelectedDdwSample:
    reject_unknown_fields(values, _SAMPLE_FIELDS)
    _require_fields(values, _SAMPLE_FIELDS, "sample")
    start_frame_index = _integer(values, "start_frame_index")
    if start_frame_index < 0:
        raise ValueError("start_frame_index must be non-negative")

    ddw = _mapping(values, "ddw")
    reject_unknown_fields(ddw, _DDW_FIELDS)
    _require_fields(ddw, _DDW_FIELDS, "ddw")
    magnitude_m = _number(ddw, "magnitude_m")
    if not math.isfinite(magnitude_m) or magnitude_m < 0:
        raise ValueError("ddw.magnitude_m must be finite and non-negative")
    sign = _integer(ddw, "sign")
    if sign not in (-1, 1):
        raise ValueError("ddw.sign must be -1 or 1")

    return SelectedDdwSample(
        sample_id=_text(values, "sample_id"),
        partition=_partition(values),
        segment_id=_text(values, "segment_id"),
        start_frame_index=start_frame_index,
        magnitude_m=magnitude_m,
        sign=sign,
    )


def _partition(values: Mapping[str, object]) -> WaymoOfficialPartition:
    value = _text(values, "partition")
    if value == "training":
        return "training"
    if value == "validation":
        return "validation"
    raise ValueError("partition must be training or validation")


def _require_fields(
    values: Mapping[str, object],
    fields: frozenset[str],
    name: str,
) -> None:
    missing = fields - values.keys()
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)}")


def _mapping(values: Mapping[str, object], field: str) -> dict[str, object]:
    value = values[field]
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _mapping_list(
    values: Mapping[str, object],
    field: str,
) -> tuple[dict[str, object], ...]:
    value = values[field]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field} must be a list of mappings")
    return tuple(dict(item) for item in value)


def _text(values: Mapping[str, object], field: str) -> str:
    value = values[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _integer(values: Mapping[str, object], field: str) -> int:
    value = values[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(values: Mapping[str, object], field: str) -> float:
    value = values[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    return float(value)


__all__ = ["SelectedDdwSample", "WaymoDdwSelection", "load_selection"]

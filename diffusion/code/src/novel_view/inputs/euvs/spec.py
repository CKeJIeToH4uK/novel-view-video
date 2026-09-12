"""Strict external schema for one ordered EUVS selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from novel_view.config.load import load_yaml_mapping, reject_unknown_fields


_SELECTION_FIELDS = frozenset({"schema_version", "pairs"})
_PAIR_FIELDS = frozenset(
    {"name", "tags", "location", "direction", "channel", "source", "target"}
)
_FRAME_FIELDS = frozenset({"traversal", "image_tokens"})


@dataclass(frozen=True, slots=True)
class EuvsFrameSelection:
    """One traversal and its image tokens in requested order."""

    traversal: str
    image_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EuvsPairSelection:
    """One concrete source-to-target EUVS pair."""

    name: str
    tags: tuple[str, ...]
    location: str
    channel: str
    source: EuvsFrameSelection
    target: EuvsFrameSelection


@dataclass(frozen=True, slots=True)
class EuvsSelection:
    """Pairs in the exact order declared by the researcher."""

    pairs: tuple[EuvsPairSelection, ...]


def load_euvs_selection(source: Path) -> EuvsSelection:
    """Parse selection YAML without opening the dataset or frame files."""
    values = load_yaml_mapping(source)
    reject_unknown_fields(values, _SELECTION_FIELDS)
    _require_fields(values, _SELECTION_FIELDS)
    if _integer(values, "schema_version") != 1:
        raise ValueError("unsupported EUVS selection schema_version")

    pair_values = _mapping_list(values, "pairs")
    if not pair_values:
        raise ValueError("EUVS selection must contain at least one pair")
    pairs = tuple(_pair(value) for value in pair_values)
    names = tuple(pair.name for pair in pairs)
    if len(set(names)) != len(names):
        raise ValueError("EUVS selection contains duplicate pair names")
    return EuvsSelection(pairs=pairs)


def _pair(values: Mapping[str, object]) -> EuvsPairSelection:
    reject_unknown_fields(values, _PAIR_FIELDS)
    _require_fields(values, _PAIR_FIELDS)
    if _string(values, "direction") != "source_to_target":
        raise ValueError("unsupported EUVS selection direction")
    return EuvsPairSelection(
        name=_string(values, "name"),
        tags=_strings(values, "tags", allow_empty=True),
        location=_string(values, "location"),
        channel=_string(values, "channel"),
        source=_frames(_mapping(values, "source")),
        target=_frames(_mapping(values, "target")),
    )


def _frames(values: Mapping[str, object]) -> EuvsFrameSelection:
    reject_unknown_fields(values, _FRAME_FIELDS)
    _require_fields(values, _FRAME_FIELDS)
    tokens = _strings(values, "image_tokens")
    if len(set(tokens)) != len(tokens):
        raise ValueError("EUVS frame selection contains duplicate image tokens")
    return EuvsFrameSelection(
        traversal=_string(values, "traversal"),
        image_tokens=tokens,
    )


def _require_fields(
    values: Mapping[str, object],
    required: frozenset[str],
) -> None:
    missing = required - values.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")


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
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"{field} must be a list of mappings")
    return tuple(dict(item) for item in value)


def _string(values: Mapping[str, object], field: str) -> str:
    value = values[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _strings(
    values: Mapping[str, object],
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = values[field]
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    strings = tuple(value)
    if (not allow_empty and not strings) or any(
        not isinstance(item, str) or not item.strip() for item in strings
    ):
        raise ValueError(f"{field} must contain non-empty text")
    return strings


def _integer(values: Mapping[str, object], field: str) -> int:
    value = values[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


__all__ = [
    "EuvsFrameSelection",
    "EuvsPairSelection",
    "EuvsSelection",
    "load_euvs_selection",
]

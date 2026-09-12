"""Синтаксическое чтение внешнего YAML как одного mapping."""

from pathlib import Path
from typing import Mapping

import yaml


def load_yaml_mapping(source: Path) -> dict[str, object]:
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"YAML root must be a mapping: {source}")
    return dict(values)


def reject_unknown_fields(
    values: Mapping[str, object],
    allowed: frozenset[str],
) -> None:
    """Reject misspelled fields at one external configuration boundary."""
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown, key=str)}")

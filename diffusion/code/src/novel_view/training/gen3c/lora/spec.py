"""Строгая внешняя конфигурация baseline R4c LoRA EDM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from novel_view.config.load import reject_unknown_fields
from novel_view.models.gen3c.lora_weights import Gen3cLoraSpec


R4C_LORA_SPEC = Gen3cLoraSpec()
R4C_LEARNING_RATE = 1e-4
R4C_WEIGHT_DECAY = 0.1
R4C_BETAS = (0.9, 0.99)
R4C_EPSILON = 1e-10
R4C_GRADIENT_CLIP_NORM = 1.0


@dataclass(frozen=True, slots=True)
class R4cTrainingInput:
    """Точные locators подготовленных данных и ordered split."""

    prepared_record: Path
    split: Path
    legacy_index_map: Path | None


@dataclass(frozen=True, slots=True)
class R4cLoraEdmSpec:
    """Изменяемые параметры одного baseline training recipe."""

    base_checkpoint: Path
    epochs: int
    training_seed: int
    validation_seed: int


def parse_r4c_training_input(
    values: Mapping[str, object],
    *,
    allow_legacy_index_map: bool,
) -> R4cTrainingInput:
    """Разобрать input без открытия подготовленных файлов."""
    allowed = {"prepared_record", "split"}
    if allow_legacy_index_map:
        allowed.add("legacy_index_map")
    reject_unknown_fields(values, frozenset(allowed))
    _require_fields(values, frozenset({"prepared_record", "split"}), "input")
    legacy = values.get("legacy_index_map")
    return R4cTrainingInput(
        prepared_record=Path(_text(values, "prepared_record")),
        split=Path(_text(values, "split")),
        legacy_index_map=(
            Path(_text(values, "legacy_index_map")) if legacy is not None else None
        ),
    )


def parse_r4c_lora_edm_spec(
    values: Mapping[str, object],
) -> R4cLoraEdmSpec:
    """Разобрать единственный поддержанный baseline recipe."""
    reject_unknown_fields(values, frozenset({"model", "training"}))
    _require_fields(values, frozenset({"model", "training"}), "parameters")

    model = _mapping(values, "model")
    reject_unknown_fields(model, frozenset({"checkpoint"}))
    _require_fields(model, frozenset({"checkpoint"}), "model")

    training = _mapping(values, "training")
    training_fields = frozenset(
        {"method", "objective", "epochs", "training_seed", "validation_seed"}
    )
    reject_unknown_fields(training, training_fields)
    _require_fields(training, training_fields, "training")
    if _text(training, "method") != "lora":
        raise ValueError("R4c baseline requires method lora")

    objective = _mapping(training, "objective")
    reject_unknown_fields(objective, frozenset({"name", "version"}))
    _require_fields(objective, frozenset({"name", "version"}), "objective")
    if (_text(objective, "name"), _integer(objective, "version")) != (
        "gen3c-kendall-edm",
        1,
    ):
        raise ValueError("R4c baseline requires gen3c-kendall-edm/v1")

    epochs = _positive_integer(training, "epochs")
    return R4cLoraEdmSpec(
        base_checkpoint=Path(_text(model, "checkpoint")),
        epochs=epochs,
        training_seed=_nonnegative_integer(training, "training_seed"),
        validation_seed=_nonnegative_integer(training, "validation_seed"),
    )


def _require_fields(
    values: Mapping[str, object],
    required: frozenset[str],
    name: str,
) -> None:
    missing = required - values.keys()
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)}")


def _mapping(values: Mapping[str, object], field: str) -> dict[str, object]:
    value = values[field]
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


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


def _positive_integer(values: Mapping[str, object], field: str) -> int:
    value = _integer(values, field)
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _nonnegative_integer(values: Mapping[str, object], field: str) -> int:
    value = _integer(values, field)
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


__all__ = [
    "R4C_BETAS",
    "R4C_EPSILON",
    "R4C_GRADIENT_CLIP_NORM",
    "R4C_LEARNING_RATE",
    "R4C_LORA_SPEC",
    "R4C_WEIGHT_DECAY",
    "R4cLoraEdmSpec",
    "R4cTrainingInput",
    "parse_r4c_lora_edm_spec",
    "parse_r4c_training_input",
]

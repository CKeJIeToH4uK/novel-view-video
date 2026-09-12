"""Строгие внешние значения одного Waymo DDW preparation job."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from novel_view.config.load import reject_unknown_fields
from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT


R4C_MODEL_CONTRACT = R4C_GEN3C_MODEL_CONTRACT.contract_id
MOGE_CHECKPOINT = "moge-vitl/model.pt"
GEN3C_TOKENIZER = "gen3c/official/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p"
GEN3C_TEXT_ENCODER = "gen3c/official/checkpoints/google-t5/t5-11b"

_INPUT_FIELDS = frozenset({"reader", "dataset", "selection"})
_PARAMETER_FIELDS = frozenset({"model_contract", "depth", "gen3c"})
_DEPTH_FIELDS = frozenset({"backend", "checkpoint"})
_GEN3C_FIELDS = frozenset({"tokenizer", "text_encoder"})


@dataclass(frozen=True, slots=True)
class WaymoDdwInput:
    """Один concrete Waymo reader и ordered selection."""

    reader: str
    dataset: str
    selection: str


@dataclass(frozen=True, slots=True)
class WaymoDdwRecipe:
    """Закреплённые model identities preparation v1."""

    model_contract: str
    depth_backend: str
    depth_checkpoint: str
    tokenizer: str
    text_encoder: str


@dataclass(frozen=True, slots=True)
class WaymoDdwSpec:
    """Полностью разобранные переносимые настройки preparation v1."""

    input: WaymoDdwInput
    recipe: WaymoDdwRecipe


def parse_waymo_ddw_spec(
    input_values: Mapping[str, object],
    parameter_values: Mapping[str, object],
    *,
    workflow_name: str,
    workflow_version: int,
    execution_preset: str,
) -> WaymoDdwSpec:
    """Разобрать job/recipe без открытия selection, dataset или assets."""
    if (workflow_name, workflow_version) != ("ddw_preparation", 1):
        raise ValueError("Waymo DDW preparation requires ddw_preparation/v1")
    if execution_preset != "inference_cp1":
        raise ValueError("ddw_preparation/v1 requires inference_cp1")

    reject_unknown_fields(input_values, _INPUT_FIELDS)
    _require_fields(input_values, _INPUT_FIELDS, "input")
    reader = _text(input_values, "reader")
    dataset = _text(input_values, "dataset")
    if reader != "waymo_v2":
        raise ValueError("ddw_preparation/v1 requires reader waymo_v2")
    if dataset != "waymo":
        raise ValueError("ddw_preparation/v1 requires dataset waymo")

    reject_unknown_fields(parameter_values, _PARAMETER_FIELDS)
    _require_fields(parameter_values, _PARAMETER_FIELDS, "parameters")
    model_contract = _text(parameter_values, "model_contract")
    if model_contract != R4C_MODEL_CONTRACT:
        raise ValueError("unsupported Waymo DDW model_contract")

    depth = _mapping(parameter_values, "depth")
    reject_unknown_fields(depth, _DEPTH_FIELDS)
    _require_fields(depth, _DEPTH_FIELDS, "depth")
    depth_backend = _text(depth, "backend")
    depth_checkpoint = _text(depth, "checkpoint")
    if depth_backend != "moge":
        raise ValueError("ddw_preparation/v1 requires depth backend moge")

    gen3c = _mapping(parameter_values, "gen3c")
    reject_unknown_fields(gen3c, _GEN3C_FIELDS)
    _require_fields(gen3c, _GEN3C_FIELDS, "gen3c")
    tokenizer = _text(gen3c, "tokenizer")
    text_encoder = _text(gen3c, "text_encoder")

    return WaymoDdwSpec(
        input=WaymoDdwInput(
            reader=reader,
            dataset=dataset,
            selection=_text(input_values, "selection"),
        ),
        recipe=WaymoDdwRecipe(
            model_contract=model_contract,
            depth_backend=depth_backend,
            depth_checkpoint=depth_checkpoint,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        ),
    )


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


def _text(values: Mapping[str, object], field: str) -> str:
    value = values[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


__all__ = [
    "GEN3C_TEXT_ENCODER",
    "GEN3C_TOKENIZER",
    "MOGE_CHECKPOINT",
    "R4C_MODEL_CONTRACT",
    "WaymoDdwInput",
    "WaymoDdwRecipe",
    "WaymoDdwSpec",
    "parse_waymo_ddw_spec",
]

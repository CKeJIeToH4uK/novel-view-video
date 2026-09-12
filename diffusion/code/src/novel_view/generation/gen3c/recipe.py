"""Parse the concrete Gen3C generation choices shared by EUVS and Gaussian."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


BASE_CHECKPOINTS = frozenset({"base", "gen3c/base"})
BASE_MODEL_ID = "gen3c-cosmos-7b-official"
_FIELDS = frozenset({"backend", "checkpoint", "model_id", "seed", "num_steps", "lora"})
_REQUIRED = frozenset({"backend", "checkpoint", "seed"})
_LORA_FIELDS = frozenset({"working_manifest", "evidence_path", "epochs", "strength"})


@dataclass(frozen=True, slots=True)
class Gen3cLoraRecipe:
    """Portable LoRA identity and the output-affecting inference strength."""

    working_manifest: str
    evidence_path: str
    epochs: int
    strength: float


@dataclass(frozen=True, slots=True)
class Gen3cGenerationRecipe:
    """Validated external model and sampling choices, without machine paths."""

    checkpoint: str
    model_id: str | None
    seed: int
    num_steps: int = 35
    lora: Gen3cLoraRecipe | None = None


def parse_generation_recipe(values: Mapping[str, object]) -> Gen3cGenerationRecipe:
    """Read the shared generation mapping once, leaving workflow choices outside."""
    _fields(values, _FIELDS, _REQUIRED, "generation")
    if _text(values, "backend") != "gen3c":
        raise ValueError("generation requires backend gen3c")
    checkpoint = _text(values, "checkpoint")
    model_id = _text(values, "model_id") if "model_id" in values else None
    lora = _lora_recipe(values) if "lora" in values else None
    if checkpoint in BASE_CHECKPOINTS:
        if model_id is not None or lora is not None:
            raise ValueError("base checkpoint cannot carry model or LoRA identity")
    elif model_id is None:
        raise ValueError("model_id is required for a non-base checkpoint")
    elif model_id == BASE_MODEL_ID:
        raise ValueError("a non-base model_id must differ from the base model")
    seed = _integer(values, "seed")
    num_steps = _integer(values, "num_steps") if "num_steps" in values else 35
    if seed < 0 or seed >= 2**63:
        raise ValueError("generation.seed must be within 0..2**63-1")
    if num_steps <= 0:
        raise ValueError("generation.num_steps must be positive")
    return Gen3cGenerationRecipe(checkpoint, model_id, seed, num_steps, lora)


def _lora_recipe(values: Mapping[str, object]) -> Gen3cLoraRecipe:
    lora = values["lora"]
    if not isinstance(lora, dict):
        raise ValueError("generation.lora must be a mapping")
    _fields(lora, _LORA_FIELDS, _LORA_FIELDS, "generation.lora")
    epochs = _integer(lora, "epochs")
    strength = lora["strength"]
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError("generation.lora.strength must be a number")
    strength = float(strength)
    if epochs <= 0 or not math.isfinite(strength) or strength < 0:
        raise ValueError("LoRA epochs must be positive and strength finite/non-negative")
    return Gen3cLoraRecipe(
        _text(lora, "working_manifest"),
        _text(lora, "evidence_path"),
        epochs,
        strength,
    )


def _fields(
    values: Mapping[str, object],
    allowed: frozenset[str],
    required: frozenset[str],
    name: str,
) -> None:
    unknown = values.keys() - allowed
    missing = required - values.keys()
    if unknown or missing:
        raise ValueError(f"{name} fields: unknown={sorted(unknown, key=str)}, missing={sorted(missing)}")


def _text(values: Mapping[str, object], field: str) -> str:
    value = values[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _integer(values: Mapping[str, object], field: str) -> int:
    value = values[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


__all__ = ["Gen3cGenerationRecipe", "Gen3cLoraRecipe", "parse_generation_recipe"]

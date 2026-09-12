"""Читаемый соседний JSON record одного baseline R4c checkpoint."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT
from novel_view.training.gen3c.data import R4C_ORDER_VERSION
from novel_view.training.gen3c.lora.spec import (
    R4C_BETAS,
    R4C_EPSILON,
    R4C_GRADIENT_CLIP_NORM,
    R4C_LEARNING_RATE,
    R4C_LORA_SPEC,
    R4C_WEIGHT_DECAY,
)


R4C_CHECKPOINT_RECORD_FORMAT_V1 = "novel-view/gen3c-lora-checkpoint-record/v1"
_TOP_FIELDS = frozenset(
    {
        "format",
        "objective",
        "ab_training_identity",
        "completed_step",
        "validation_loss",
        "source_attempt",
        "checkpoint",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "base_checkpoint",
        "model_contract",
        "method",
        "optimizer",
        "noise",
        "conditioning",
        "execution",
        "prepared_record",
        "training_sample_ids",
        "training_segment_ids",
        "validation_sample_ids",
        "validation_segment_ids",
        "training_seed",
        "validation_seed",
        "order_version",
        "epochs",
        "checkpoint_every_epochs",
    }
)


@dataclass(frozen=True, slots=True)
class AbTrainingIdentityV1:
    """Изменяемая часть exact base/v1/v2 comparison identity."""

    base_checkpoint: str
    prepared_record: str
    training_sample_ids: tuple[str, ...]
    training_segment_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    validation_segment_ids: tuple[str, ...]
    training_seed: int
    validation_seed: int
    epochs: int


@dataclass(frozen=True, slots=True)
class CheckpointRecordV1:
    """Selection-readable facts written beside one binary checkpoint."""

    identity: AbTrainingIdentityV1
    completed_step: int
    validation_loss: float
    source_attempt: str
    checkpoint: str


def write_checkpoint_record_v1(path: Path, record: CheckpointRecordV1) -> None:
    """Напрямую записать JSON после соответствующего PT checkpoint."""
    path.write_text(
        json.dumps(_record_document(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_checkpoint_record_v1(path: Path) -> CheckpointRecordV1:
    """Строго прочитать scientific facts, не открывая binary checkpoint."""
    document = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(document, "checkpoint record")
    _exact_fields(root, _TOP_FIELDS, "checkpoint record")
    if root["format"] != R4C_CHECKPOINT_RECORD_FORMAT_V1:
        raise ValueError("unsupported R4c checkpoint record format")
    if root["objective"] != _objective_document():
        raise ValueError("R4c checkpoint record objective differs")

    identity = parse_ab_training_identity(root["ab_training_identity"])
    validation_loss = _finite_number(root["validation_loss"], "validation_loss")
    source_attempt = parse_attempt_reference(root["source_attempt"])
    return CheckpointRecordV1(
        identity=identity,
        completed_step=_positive_integer(root["completed_step"], "completed_step"),
        validation_loss=validation_loss,
        source_attempt=source_attempt,
        checkpoint=_text(root["checkpoint"], "checkpoint"),
    )


def parse_ab_training_identity(value: Any) -> AbTrainingIdentityV1:
    """Strictly decode the exact identity shared by v1 and v2 records."""
    identity = _mapping(value, "ab_training_identity")
    _exact_fields(identity, _IDENTITY_FIELDS, "ab_training_identity")
    training_seed = _nonnegative_integer(identity["training_seed"], "training_seed")
    for field, expected in _constant_identity_documents(training_seed).items():
        if identity[field] != expected:
            raise ValueError(f"R4c checkpoint record {field} differs")
    if identity["order_version"] != R4C_ORDER_VERSION:
        raise ValueError("R4c checkpoint record order differs")
    if identity["checkpoint_every_epochs"] != 1:
        raise ValueError("R4c checkpoint cadence differs")

    training_samples = _text_sequence(
        identity["training_sample_ids"], "training_sample_ids"
    )
    training_segments = _text_sequence(
        identity["training_segment_ids"], "training_segment_ids"
    )
    validation_samples = _text_sequence(
        identity["validation_sample_ids"], "validation_sample_ids"
    )
    validation_segments = _text_sequence(
        identity["validation_segment_ids"], "validation_segment_ids"
    )
    if len(training_samples) != len(training_segments) or len(
        validation_samples
    ) != len(validation_segments):
        raise ValueError("R4c checkpoint record sample/segment counts differ")
    if set(training_samples) & set(validation_samples) or set(training_segments) & set(
        validation_segments
    ):
        raise ValueError("R4c checkpoint record split overlaps")

    return AbTrainingIdentityV1(
        base_checkpoint=_text(identity["base_checkpoint"], "base_checkpoint"),
        prepared_record=_text(identity["prepared_record"], "prepared_record"),
        training_sample_ids=training_samples,
        training_segment_ids=training_segments,
        validation_sample_ids=validation_samples,
        validation_segment_ids=validation_segments,
        training_seed=training_seed,
        validation_seed=_nonnegative_integer(
            identity["validation_seed"], "validation_seed"
        ),
        epochs=_positive_integer(identity["epochs"], "epochs"),
    )


def parse_attempt_reference(value: Any) -> str:
    """Decode the recorded exact job/run/attempt reference without discovery."""
    reference = _text(value, "source_attempt")
    if len(reference.split("/")) != 3 or any(not part for part in reference.split("/")):
        raise ValueError("source_attempt must be job/run/attempt")
    return reference


def _record_document(record: CheckpointRecordV1) -> dict[str, Any]:
    identity = record.identity
    return {
        "format": R4C_CHECKPOINT_RECORD_FORMAT_V1,
        "objective": _objective_document(),
        "ab_training_identity": ab_training_identity_document(identity),
        "completed_step": record.completed_step,
        "validation_loss": record.validation_loss,
        "source_attempt": record.source_attempt,
        "checkpoint": record.checkpoint,
    }


def ab_training_identity_document(identity: AbTrainingIdentityV1) -> dict[str, Any]:
    """Encode the exact identity shared by readable v1 and v2 records."""
    return {
        "base_checkpoint": identity.base_checkpoint,
        **_constant_identity_documents(identity.training_seed),
        "prepared_record": identity.prepared_record,
        "training_sample_ids": list(identity.training_sample_ids),
        "training_segment_ids": list(identity.training_segment_ids),
        "validation_sample_ids": list(identity.validation_sample_ids),
        "validation_segment_ids": list(identity.validation_segment_ids),
        "training_seed": identity.training_seed,
        "validation_seed": identity.validation_seed,
        "order_version": R4C_ORDER_VERSION,
        "epochs": identity.epochs,
        "checkpoint_every_epochs": 1,
    }


def _objective_document() -> dict[str, Any]:
    return {"name": "gen3c-kendall-edm", "version": 1}


def _constant_identity_documents(initialization_seed: int) -> dict[str, Any]:
    contract = R4C_GEN3C_MODEL_CONTRACT
    return {
        "model_contract": {
            "id": contract.contract_id,
            "frames": contract.video_frame_count,
            "height": contract.raster_height,
            "width": contract.raster_width,
            "fps": contract.fps,
        },
        "method": {
            "name": "r4c-lora",
            "version": 1,
            "initialization_seed": initialization_seed,
            "rank": R4C_LORA_SPEC.rank,
            "scale": R4C_LORA_SPEC.scale,
            "block_count": R4C_LORA_SPEC.block_count,
            "self_attention_targets": list(R4C_LORA_SPEC.self_attention_targets),
            "cross_attention_targets": list(R4C_LORA_SPEC.cross_attention_targets),
            "trainable": "lora-only",
            "logvar_trainable": False,
        },
        "optimizer": {
            "name": "fused-adam",
            "learning_rate": R4C_LEARNING_RATE,
            "weight_decay": R4C_WEIGHT_DECAY,
            "betas": list(R4C_BETAS),
            "epsilon": R4C_EPSILON,
            "master_weights": "float32",
            "capturable": True,
            "scheduler": "constant",
            "gradient_clip_norm": R4C_GRADIENT_CLIP_NORM,
        },
        "noise": {
            "generation_log_normal": [0.0, 1.0, 4.0],
            "condition_log_normal": [-3.0, 2.0, 1.0],
            "sigma_compute_dtype": "float32",
        },
        "conditioning": {
            "drop_condition": False,
            "prompt_mask": "bfloat16-ones",
            "padding_mask": "bfloat16-zeros",
            "image_size": list(contract.image_size_values),
        },
        "execution": {
            "batch_size": 1,
            "latent_dtype": "bfloat16",
            "activation_recompute": True,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "context_parallel_size": 4,
            "gradient_average_size": 4,
        },
    }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _exact_fields(value: Mapping[str, Any], fields: frozenset[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} fields differ")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    values = tuple(_text(item, name) for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return values


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_number(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


__all__ = [
    "AbTrainingIdentityV1",
    "CheckpointRecordV1",
    "R4C_CHECKPOINT_RECORD_FORMAT_V1",
    "ab_training_identity_document",
    "parse_ab_training_identity",
    "parse_attempt_reference",
    "read_checkpoint_record_v1",
    "write_checkpoint_record_v1",
]

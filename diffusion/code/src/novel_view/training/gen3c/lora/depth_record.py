"""Readable companion JSON for one experimental v2 checkpoint."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novel_view.training.gen3c.lora.depth_spec import (
    DepthObjectiveV2,
    DepthObserverFitIdentity,
    DepthObserverFitResult,
    depth_objective_document,
    observer_fit_identity_document,
    observer_fit_result_document,
    parse_depth_objective,
    parse_observer_fit_identity,
    parse_observer_fit_result,
)
from novel_view.training.gen3c.lora.record import (
    AbTrainingIdentityV1,
    ab_training_identity_document,
    parse_ab_training_identity,
    parse_attempt_reference,
)


R4C_CHECKPOINT_RECORD_FORMAT_V2 = "novel-view/gen3c-lora-checkpoint-record/v2"
_FIELDS = frozenset(
    {
        "format",
        "objective",
        "ab_training_identity",
        "observer_fit_identity",
        "observer_fit_result",
        "observer_lineage",
        "completed_step",
        "validation_edm_loss",
        "validation_depth_loss",
        "validation_total_loss",
        "source_attempt",
        "checkpoint",
    }
)


@dataclass(frozen=True, slots=True)
class CheckpointRecordV2:
    """Selection-readable facts written beside one v2 binary checkpoint."""

    objective: DepthObjectiveV2
    identity: AbTrainingIdentityV1
    observer_fit_identity: DepthObserverFitIdentity
    observer_fit_result: DepthObserverFitResult
    observer_lineage: str
    completed_step: int
    validation_edm_loss: float
    validation_depth_loss: float
    validation_total_loss: float
    source_attempt: str
    checkpoint: str


def write_checkpoint_record_v2(path: Path, record: CheckpointRecordV2) -> None:
    """Write readable v2 scientific facts after the corresponding PT."""
    document = {
        "format": R4C_CHECKPOINT_RECORD_FORMAT_V2,
        "objective": depth_objective_document(record.objective),
        "ab_training_identity": ab_training_identity_document(record.identity),
        "observer_fit_identity": observer_fit_identity_document(
            record.observer_fit_identity
        ),
        "observer_fit_result": observer_fit_result_document(record.observer_fit_result),
        "observer_lineage": record.observer_lineage,
        "completed_step": record.completed_step,
        "validation_edm_loss": record.validation_edm_loss,
        "validation_depth_loss": record.validation_depth_loss,
        "validation_total_loss": record.validation_total_loss,
        "source_attempt": record.source_attempt,
        "checkpoint": record.checkpoint,
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_checkpoint_record_v2(path: Path) -> CheckpointRecordV2:
    """Strictly read v2 facts without opening the binary checkpoint."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != _FIELDS:
        raise ValueError("R4c v2 checkpoint record fields differ")
    document = dict(raw)
    if document["format"] != R4C_CHECKPOINT_RECORD_FORMAT_V2:
        raise ValueError("unsupported R4c v2 checkpoint record format")
    depth_loss = _finite(document["validation_depth_loss"], "validation_depth_loss")
    if depth_loss < 0:
        raise ValueError("validation_depth_loss must be non-negative")
    identity = parse_ab_training_identity(document["ab_training_identity"])
    fit_identity = parse_observer_fit_identity(document["observer_fit_identity"])
    if (
        fit_identity.training_sample_ids,
        fit_identity.training_segment_ids,
        fit_identity.validation_sample_ids,
        fit_identity.validation_segment_ids,
    ) != (
        identity.training_sample_ids,
        identity.training_segment_ids,
        identity.validation_sample_ids,
        identity.validation_segment_ids,
    ):
        raise ValueError("R4c v2 observer fit differs from training identity")
    return CheckpointRecordV2(
        objective=parse_depth_objective(document["objective"]),
        identity=identity,
        observer_fit_identity=fit_identity,
        observer_fit_result=parse_observer_fit_result(document["observer_fit_result"]),
        observer_lineage=_attempt_reference(
            document["observer_lineage"], "observer_lineage"
        ),
        completed_step=_positive_integer(document["completed_step"]),
        validation_edm_loss=_finite(
            document["validation_edm_loss"], "validation_edm_loss"
        ),
        validation_depth_loss=depth_loss,
        validation_total_loss=_finite(
            document["validation_total_loss"], "validation_total_loss"
        ),
        source_attempt=parse_attempt_reference(document["source_attempt"]),
        checkpoint=_text(document["checkpoint"], "checkpoint"),
    )


def _attempt_reference(value: object, name: str) -> str:
    reference = _text(value, name)
    if len(reference.split("/")) != 3 or any(not part for part in reference.split("/")):
        raise ValueError(f"{name} must be job/run/attempt")
    return reference


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _positive_integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("completed_step must be positive")
    return value


def _finite(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


__all__ = [
    "CheckpointRecordV2",
    "R4C_CHECKPOINT_RECORD_FORMAT_V2",
    "read_checkpoint_record_v2",
    "write_checkpoint_record_v2",
]

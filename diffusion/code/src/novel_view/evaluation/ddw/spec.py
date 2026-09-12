"""Strict external contract for matched base/v1/v2 DDW evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from novel_view.config.load import reject_unknown_fields
from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT
from novel_view.training.gen3c.data import R4cPreparedItem
from novel_view.training.gen3c.lora.depth_record import CheckpointRecordV2
from novel_view.training.gen3c.lora.checkpoint import LoadedCheckpointV1
from novel_view.training.gen3c.lora.depth_checkpoint import LoadedCheckpointV2
from novel_view.training.gen3c.lora.record import CheckpointRecordV1


_INPUT_FIELDS = frozenset(
    {
        "prepared_record",
        "split",
        "v1_checkpoint_record",
        "v2_checkpoint_record",
    }
)
_PARAMETER_FIELDS = frozenset(
    {"model_contract", "comparison_step", "sampling", "gen3c", "depth"}
)
_SAMPLING_FIELDS = frozenset({"seed", "steps", "guidance", "condition_augment_sigma"})
_GEN3C_FIELDS = frozenset({"base_checkpoint", "tokenizer"})
_DEPTH_FIELDS = frozenset({"backend", "checkpoint"})


@dataclass(frozen=True, slots=True)
class DdwEvaluationInput:
    """Four explicit record references; no directory discovery."""

    prepared_record: str
    split: str
    v1_checkpoint_record: str
    v2_checkpoint_record: str


@dataclass(frozen=True, slots=True)
class DdwSamplingSpec:
    """The one matched R4c inference schedule."""

    seed: int
    steps: int
    guidance: float
    condition_augment_sigma: float


@dataclass(frozen=True, slots=True)
class DdwEvaluationRecipe:
    """Pinned model identities and comparison step for evaluation v1."""

    model_contract: str
    comparison_step: int
    sampling: DdwSamplingSpec
    base_checkpoint: str
    tokenizer: str
    depth_backend: str
    depth_checkpoint: str


@dataclass(frozen=True, slots=True)
class DdwEvaluationSpec:
    """One fully parsed ddw_evaluation/v1 job."""

    input: DdwEvaluationInput
    recipe: DdwEvaluationRecipe


def parse_ddw_evaluation_spec(
    input_values: Mapping[str, object],
    parameter_values: Mapping[str, object],
    *,
    workflow_name: str,
    workflow_version: int,
    execution_preset: str,
) -> DdwEvaluationSpec:
    """Parse job and recipe without opening records, models, or data."""
    if (workflow_name, workflow_version) != ("ddw_evaluation", 1):
        raise ValueError("DDW evaluation requires ddw_evaluation/v1")
    if execution_preset != "training_cp4":
        raise ValueError("ddw_evaluation/v1 requires training_cp4")

    reject_unknown_fields(input_values, _INPUT_FIELDS)
    _require_fields(input_values, _INPUT_FIELDS, "input")
    evaluation_input = DdwEvaluationInput(
        prepared_record=_text(input_values, "prepared_record"),
        split=_text(input_values, "split"),
        v1_checkpoint_record=_text(input_values, "v1_checkpoint_record"),
        v2_checkpoint_record=_text(input_values, "v2_checkpoint_record"),
    )

    reject_unknown_fields(parameter_values, _PARAMETER_FIELDS)
    _require_fields(parameter_values, _PARAMETER_FIELDS, "parameters")
    model_contract = _text(parameter_values, "model_contract")
    if model_contract != R4C_GEN3C_MODEL_CONTRACT.contract_id:
        raise ValueError("unsupported DDW evaluation model_contract")

    sampling_values = _mapping(parameter_values, "sampling")
    reject_unknown_fields(sampling_values, _SAMPLING_FIELDS)
    _require_fields(sampling_values, _SAMPLING_FIELDS, "sampling")
    sampling = DdwSamplingSpec(
        seed=_nonnegative_integer(sampling_values, "seed"),
        steps=_positive_integer(sampling_values, "steps"),
        guidance=_finite_number(sampling_values, "guidance"),
        condition_augment_sigma=_nonnegative_number(
            sampling_values,
            "condition_augment_sigma",
        ),
    )
    if (
        sampling.steps != 35
        or sampling.guidance != 1.0
        or sampling.condition_augment_sigma != 0.001
    ):
        raise ValueError("ddw_evaluation/v1 requires the pinned R4c schedule")

    gen3c = _mapping(parameter_values, "gen3c")
    reject_unknown_fields(gen3c, _GEN3C_FIELDS)
    _require_fields(gen3c, _GEN3C_FIELDS, "gen3c")
    depth = _mapping(parameter_values, "depth")
    reject_unknown_fields(depth, _DEPTH_FIELDS)
    _require_fields(depth, _DEPTH_FIELDS, "depth")
    depth_backend = _text(depth, "backend")
    if depth_backend != "moge":
        raise ValueError("ddw_evaluation/v1 requires depth backend moge")

    return DdwEvaluationSpec(
        input=evaluation_input,
        recipe=DdwEvaluationRecipe(
            model_contract=model_contract,
            comparison_step=_positive_integer(parameter_values, "comparison_step"),
            sampling=sampling,
            base_checkpoint=_text(gen3c, "base_checkpoint"),
            tokenizer=_text(gen3c, "tokenizer"),
            depth_backend=depth_backend,
            depth_checkpoint=_text(depth, "checkpoint"),
        ),
    )


def require_matched_checkpoint_records(
    training_items: tuple[R4cPreparedItem, ...],
    validation_items: tuple[R4cPreparedItem, ...],
    v1: CheckpointRecordV1,
    v2: CheckpointRecordV2,
    *,
    prepared_record_reference: str,
    base_checkpoint_reference: str,
    comparison_step: int,
) -> None:
    """Require only the scientific identity needed for one honest A/B."""
    if v1.identity != v2.identity:
        raise ValueError("v1 and v2 A/B training identities differ")
    if (v1.completed_step, v2.completed_step) != (
        comparison_step,
        comparison_step,
    ):
        raise ValueError("v1 and v2 must use the requested comparison_step")

    identity = v1.identity
    expected = (
        base_checkpoint_reference,
        prepared_record_reference,
        tuple(item.sample_id for item in training_items),
        tuple(item.segment_id for item in training_items),
        tuple(item.sample_id for item in validation_items),
        tuple(item.segment_id for item in validation_items),
    )
    actual = (
        identity.base_checkpoint,
        identity.prepared_record,
        identity.training_sample_ids,
        identity.training_segment_ids,
        identity.validation_sample_ids,
        identity.validation_segment_ids,
    )
    if actual != expected:
        raise ValueError("checkpoint records differ from the evaluation data identity")


def require_binary_record_facts(
    v1_checkpoint: LoadedCheckpointV1,
    v2_checkpoint: LoadedCheckpointV2,
    v1_record: CheckpointRecordV1,
    v2_record: CheckpointRecordV2,
) -> None:
    """Match only scientific binary facts that the readable records claim."""
    if (
        v1_checkpoint.resume.next_step != v1_record.completed_step
        or v2_checkpoint.baseline.resume.next_step != v2_record.completed_step
    ):
        raise ValueError("binary checkpoint progress differs from its JSON record")
    if (
        v2_checkpoint.objective != v2_record.objective
        or v2_checkpoint.observer_fit_identity != v2_record.observer_fit_identity
        or v2_checkpoint.observer_fit_result != v2_record.observer_fit_result
        or v2_checkpoint.observer_lineage != v2_record.observer_lineage
    ):
        raise ValueError("v2 binary checkpoint facts differ from its JSON record")


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


def _positive_integer(values: Mapping[str, object], field: str) -> int:
    value = values[field]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(values: Mapping[str, object], field: str) -> int:
    value = values[field]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_number(values: Mapping[str, object], field: str) -> float:
    value = values[field]
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _nonnegative_number(values: Mapping[str, object], field: str) -> float:
    value = _finite_number(values, field)
    if value < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return value


__all__ = [
    "DdwEvaluationInput",
    "DdwEvaluationRecipe",
    "DdwEvaluationSpec",
    "DdwSamplingSpec",
    "parse_ddw_evaluation_spec",
    "require_binary_record_facts",
    "require_matched_checkpoint_records",
]

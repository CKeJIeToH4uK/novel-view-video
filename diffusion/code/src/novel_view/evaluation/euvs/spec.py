"""Strict external choices for EUVS evaluation and comparison jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from novel_view.config.job import ResolvedJob
from novel_view.config.load import reject_unknown_fields

_EVALUATION_INPUT_FIELDS = frozenset({"reader", "dataset", "selection", "generation"})
_GENERATION_ATTEMPT_FIELDS = frozenset({"source", "path"})
_GENERATION_RECORD_FIELDS = frozenset({"source", "records"})
_EVALUATION_PARAMETER_FIELDS = frozenset({"evaluation"})
_EVALUATION_FIELDS = frozenset({"mask_recipe"})
_COMPARISON_INPUT_FIELDS = frozenset(
    {
        "reader",
        "dataset",
        "selection",
        "base_evaluation",
        "tuned_evaluation",
    }
)


@dataclass(frozen=True, slots=True)
class GenerationAttemptReference:
    """One exact generation attempt below `/runs`."""

    path: str


@dataclass(frozen=True, slots=True)
class GenerationRecordReferences:
    """Exact legacy generation records in selection order."""

    records: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EuvsEvaluationSpec:
    """Portable external choices of `euvs_evaluation/v1`."""

    dataset: str
    selection: str
    generation: GenerationAttemptReference | GenerationRecordReferences
    mask_recipe_id: str


@dataclass(frozen=True, slots=True)
class EuvsComparisonSpec:
    """Two ready evaluation attempts over one ordered selection."""

    dataset: str
    selection: str
    base_evaluation: str
    tuned_evaluation: str


def parse_euvs_evaluation_job(job: ResolvedJob) -> EuvsEvaluationSpec:
    """Parse one strict job without opening runs, data or model assets."""
    reject_unknown_fields(job.input, _EVALUATION_INPUT_FIELDS)
    _require(job.input, _EVALUATION_INPUT_FIELDS, "input")
    reject_unknown_fields(job.parameters, _EVALUATION_PARAMETER_FIELDS)
    _require(job.parameters, _EVALUATION_PARAMETER_FIELDS, "parameters")
    if _text(job.input, "reader") != "euvs_nuplan":
        raise ValueError("euvs_evaluation/v1 requires reader euvs_nuplan")
    if _text(job.input, "dataset") != "euvs":
        raise ValueError("euvs_evaluation/v1 requires dataset euvs")
    if job.execution.preset != "inference_cp1":
        raise ValueError("euvs_evaluation/v1 requires inference_cp1")

    generation = _mapping(job.input, "generation")
    source = _text(generation, "source")
    if source == "attempt":
        reject_unknown_fields(generation, _GENERATION_ATTEMPT_FIELDS)
        _require(generation, _GENERATION_ATTEMPT_FIELDS, "generation")
        reference: GenerationAttemptReference | GenerationRecordReferences = (
            GenerationAttemptReference(_text(generation, "path"))
        )
    elif source == "records":
        reject_unknown_fields(generation, _GENERATION_RECORD_FIELDS)
        _require(generation, _GENERATION_RECORD_FIELDS, "generation")
        reference = GenerationRecordReferences(_texts(generation, "records"))
    else:
        raise ValueError("generation.source must be attempt or records")

    evaluation = _mapping(job.parameters, "evaluation")
    reject_unknown_fields(evaluation, _EVALUATION_FIELDS)
    _require(evaluation, _EVALUATION_FIELDS, "evaluation")
    return EuvsEvaluationSpec(
        dataset="euvs",
        selection=_text(job.input, "selection"),
        generation=reference,
        mask_recipe_id=_text(evaluation, "mask_recipe"),
    )


def parse_euvs_comparison_job(job: ResolvedJob) -> EuvsComparisonSpec:
    """Parse the pure-CPU comparison boundary without opening its attempts."""
    reject_unknown_fields(job.input, _COMPARISON_INPUT_FIELDS)
    _require(job.input, _COMPARISON_INPUT_FIELDS, "input")
    reject_unknown_fields(job.parameters, frozenset())
    if _text(job.input, "reader") != "euvs_nuplan":
        raise ValueError("euvs_comparison/v1 requires reader euvs_nuplan")
    if _text(job.input, "dataset") != "euvs":
        raise ValueError("euvs_comparison/v1 requires dataset euvs")
    if job.execution.preset != "cpu_test":
        raise ValueError("euvs_comparison/v1 requires cpu_test")
    return EuvsComparisonSpec(
        dataset="euvs",
        selection=_text(job.input, "selection"),
        base_evaluation=_text(job.input, "base_evaluation"),
        tuned_evaluation=_text(job.input, "tuned_evaluation"),
    )


def _require(
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


def _texts(values: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = values[field]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{field} must contain non-empty text")
    return tuple(value)


__all__ = [
    "EuvsComparisonSpec",
    "EuvsEvaluationSpec",
    "GenerationAttemptReference",
    "GenerationRecordReferences",
    "parse_euvs_comparison_job",
    "parse_euvs_evaluation_job",
]

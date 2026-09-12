"""Разбор общей внешней оболочки job и recipe без научной семантики."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

from novel_view.config.load import load_yaml_mapping, reject_unknown_fields


_JOB_FIELDS = frozenset(
    {"schema_version", "name", "workflow", "input", "recipe", "parameters", "execution"}
)
_WORKFLOW_FIELDS = frozenset({"name", "version"})
_EXECUTION_FIELDS = frozenset({"preset"})
_RECIPE_FIELDS = frozenset({"schema_version", "parameters"})


@dataclass(frozen=True, slots=True)
class WorkflowReference:
    name: str
    version: int


@dataclass(frozen=True, slots=True)
class ExecutionReference:
    preset: str


@dataclass(frozen=True, slots=True)
class ResolvedJob:
    schema_version: int
    name: str
    workflow: WorkflowReference
    input: Mapping[str, object]
    recipe: str | None
    parameters: Mapping[str, object]
    execution: ExecutionReference


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


def _string(values: Mapping[str, object], field: str) -> str:
    value = values[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(values: Mapping[str, object], field: str) -> int:
    value = values[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _load_recipe(
    recipe_reference: str,
    config_root: Path,
) -> dict[str, object]:
    recipe_path = config_root / recipe_reference
    values = load_yaml_mapping(recipe_path)
    reject_unknown_fields(values, _RECIPE_FIELDS)
    _require_fields(values, _RECIPE_FIELDS)
    if _integer(values, "schema_version") != 1:
        raise ValueError("unsupported recipe schema_version")
    return _mapping(values, "parameters")


def load_resolved_job(
    job_file: Path,
    config_root: Path,
) -> ResolvedJob:
    values = load_yaml_mapping(job_file)
    reject_unknown_fields(values, _JOB_FIELDS)
    _require_fields(
        values,
        frozenset({"schema_version", "name", "workflow", "input", "execution"}),
    )

    if _integer(values, "schema_version") != 1:
        raise ValueError("unsupported job schema_version")

    has_recipe = "recipe" in values
    has_parameters = "parameters" in values
    if has_recipe == has_parameters:
        raise ValueError("job requires exactly one of recipe or parameters")

    workflow_values = _mapping(values, "workflow")
    reject_unknown_fields(workflow_values, _WORKFLOW_FIELDS)
    _require_fields(workflow_values, _WORKFLOW_FIELDS)

    execution_values = _mapping(values, "execution")
    reject_unknown_fields(execution_values, _EXECUTION_FIELDS)
    _require_fields(execution_values, _EXECUTION_FIELDS)

    recipe = _string(values, "recipe") if has_recipe else None
    parameters = (
        _load_recipe(recipe, config_root)
        if recipe is not None
        else _mapping(values, "parameters")
    )

    return ResolvedJob(
        schema_version=1,
        name=_string(values, "name"),
        workflow=WorkflowReference(
            name=_string(workflow_values, "name"),
            version=_integer(workflow_values, "version"),
        ),
        input=_mapping(values, "input"),
        recipe=recipe,
        parameters=parameters,
        execution=ExecutionReference(
            preset=_string(execution_values, "preset"),
        ),
    )


def resolved_job_to_mapping(job: ResolvedJob) -> dict[str, object]:
    return {
        "schema_version": job.schema_version,
        "name": job.name,
        "workflow": {
            "name": job.workflow.name,
            "version": job.workflow.version,
        },
        "input": dict(job.input),
        "recipe": job.recipe,
        "parameters": dict(job.parameters),
        "execution": {"preset": job.execution.preset},
    }


def resolved_job_from_mapping(values: Mapping[str, object]) -> ResolvedJob:
    """Восстановить уже разрешённую job из собственного attempt record."""
    workflow = cast(Mapping[str, object], values["workflow"])
    execution = cast(Mapping[str, object], values["execution"])
    return ResolvedJob(
        schema_version=cast(int, values["schema_version"]),
        name=cast(str, values["name"]),
        workflow=WorkflowReference(
            name=cast(str, workflow["name"]),
            version=cast(int, workflow["version"]),
        ),
        input=dict(cast(Mapping[str, object], values["input"])),
        recipe=cast(str | None, values["recipe"]),
        parameters=dict(cast(Mapping[str, object], values["parameters"])),
        execution=ExecutionReference(
            preset=cast(str, execution["preset"]),
        ),
    )

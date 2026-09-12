from __future__ import annotations

from pathlib import Path
from typing import Any

from novel_view.config.job import ExecutionReference, ResolvedJob, WorkflowReference
from novel_view.runtime.context import ContainerRoots, ImageMetadata, RuntimeContext


def make_runtime(
    root: Path,
    *,
    attempt: str = "attempt",
    image: str = "core",
) -> RuntimeContext:
    config = root / "config"
    roots = ContainerRoots(
        root / "data",
        root / "models",
        root / "prepared",
        root / "runs",
        root / "cache",
        config / "jobs",
        config / "recipes",
        config / "selections",
    )
    for directory in (
        roots.data, roots.models, roots.prepared, roots.runs,
        roots.cache, roots.jobs, roots.recipes, roots.selections,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    attempt_root = roots.runs / attempt
    attempt_root.mkdir(parents=True)
    return RuntimeContext(
        roots, "run-test", "attempt-test", "fresh",
        attempt_root, "stage5-test",
        ImageMetadata(image, "test", "test", False, "test"),
        None,
    )


def make_job(
    workflow: str,
    version: int,
    input_spec: dict[str, Any],
    parameters: dict[str, Any],
    *,
    preset: str = "inference_cp2",
    name: str | None = None,
    recipe: str | None = None,
) -> ResolvedJob:
    return ResolvedJob(
        1, name or workflow, WorkflowReference(workflow, version), input_spec,
        recipe, parameters, ExecutionReference(preset),
    )

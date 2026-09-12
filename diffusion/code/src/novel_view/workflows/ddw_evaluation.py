"""Lightweight route for one matched DDW heldout-point evaluation."""

from __future__ import annotations

from novel_view.config.job import ResolvedJob
from novel_view.evaluation.ddw.spec import (
    DdwEvaluationSpec,
    parse_ddw_evaluation_spec,
)
from novel_view.runtime.context import RuntimeContext


def parse_job(job: ResolvedJob) -> DdwEvaluationSpec:
    """Parse external evaluation configuration without model imports."""
    return parse_ddw_evaluation_spec(
        job.input,
        job.parameters,
        workflow_name=job.workflow.name,
        workflow_version=job.workflow.version,
        execution_preset=job.execution.preset,
    )


def select_image_variant(job: ResolvedJob) -> str:
    """Use the image variant that contains both Gen3C and MoGe."""
    parse_job(job)
    return "moge"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Run the one explicit generation → decode/MoGe → record order."""
    from novel_view.evaluation.ddw.execute import run_evaluation_v1

    run_evaluation_v1(parse_job(job), runtime)
    return 0


__all__ = ["parse_job", "run_v1", "select_image_variant"]

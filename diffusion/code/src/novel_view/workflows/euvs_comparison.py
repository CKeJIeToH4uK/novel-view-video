"""Compare two ready EUVS evaluation attempts in selection order."""

from __future__ import annotations

from novel_view.config.job import ResolvedJob
from novel_view.evaluation.euvs.spec import parse_euvs_comparison_job
from novel_view.runtime.context import RuntimeContext


def select_image_variant(job: ResolvedJob) -> str:
    """Select the CPU-capable core image without importing saved records."""
    parse_euvs_comparison_job(job)
    return "core"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Read two exact metric matrices and write three direct CSV tables."""
    from novel_view.evaluation.euvs.benchmark import (
        compare_evaluations,
        write_euvs_comparison,
    )
    from novel_view.inputs.euvs.spec import load_euvs_selection

    spec = parse_euvs_comparison_job(job)
    selection = load_euvs_selection(runtime.roots.selections.parent / spec.selection)
    tables = compare_evaluations(
        job.name,
        selection,
        spec.base_evaluation,
        spec.tuned_evaluation,
        runtime.roots.runs,
    )
    write_euvs_comparison(runtime.attempt_root, tables)
    return 0


__all__ = ["run_v1", "select_image_variant"]

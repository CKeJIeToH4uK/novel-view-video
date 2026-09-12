"""Choose one historical Waymo depth backend from eight explicit reports."""

from __future__ import annotations

from typing import cast

from novel_view.config.job import ResolvedJob
from novel_view.preparation.waymo_depth.selection_record import (
    DepthSelectionRecord,
    SELECTION_RECORD_VERSION,
    write_selection_record,
)
from novel_view.preparation.waymo_depth.candidate_record import read_candidate_record
from novel_view.preparation.waymo_depth.selection import (
    DepthCandidateGateSpec,
    DepthGateSpec,
    combine_candidate_outcomes,
    normalize_depth_selection_evidence,
    select_depth_backend_v1,
    select_depth_backend_v2,
)
from novel_view.preparation.waymo_depth.spec import (
    WaymoDepthSelectionSpec,
    load_depth_report_selection,
    parse_depth_selection_spec,
)
from novel_view.runtime.context import RuntimeContext


def parse_job(job: ResolvedJob) -> WaymoDepthSelectionSpec:
    """Parse one strict external v1/v2 selection job."""
    return parse_depth_selection_spec(
        job.input,
        job.parameters,
        workflow_name=job.workflow.name,
        workflow_version=job.workflow.version,
        execution_preset=job.execution.preset,
    )


def select_image_variant(job: ResolvedJob) -> str:
    parse_job(job)
    return "core"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Write the v4 choice or scientific stop after the original count gate."""
    spec = parse_job(job)
    locators, keys, candidates = _read_candidates(spec, runtime)
    reports = combine_candidate_outcomes(keys, candidates)
    gate_spec = cast(DepthGateSpec, spec.gate_spec)
    outcome = select_depth_backend_v1(gate_spec, keys, reports)
    path = runtime.attempt_root / "depth-selection.json"
    write_selection_record(
        path,
        DepthSelectionRecord(
            SELECTION_RECORD_VERSION,
            1,
            spec.gate_id,
            gate_spec,
            keys,
            locators,
            outcome,
            None,
        ),
    )
    print(path)
    return 0


def run_v2(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Normalize shared evidence, then apply the corrected candidate gate."""
    spec = parse_job(job)
    locators, keys, candidates = _read_candidates(spec, runtime)
    evidence = normalize_depth_selection_evidence(keys, candidates)
    reports = combine_candidate_outcomes(keys, candidates)
    gate_spec = cast(DepthCandidateGateSpec, spec.gate_spec)
    outcome = select_depth_backend_v2(gate_spec, keys, reports)
    path = runtime.attempt_root / "depth-selection.json"
    write_selection_record(
        path,
        DepthSelectionRecord(
            SELECTION_RECORD_VERSION,
            2,
            spec.gate_id,
            gate_spec,
            keys,
            locators,
            outcome,
            evidence,
        ),
    )
    print(path)
    return 0


def _read_candidates(
    spec: WaymoDepthSelectionSpec,
    runtime: RuntimeContext,
):
    locators = load_depth_report_selection(
        runtime.roots.selections.parent / spec.selection
    )
    records = tuple(
        read_candidate_record(runtime.roots.runs / locator)
        for locator in locators
    )
    return (
        locators,
        tuple(record.clip_key for record in records),
        tuple(record.outcomes for record in records),
    )


__all__ = ["parse_job", "run_v1", "run_v2", "select_image_variant"]

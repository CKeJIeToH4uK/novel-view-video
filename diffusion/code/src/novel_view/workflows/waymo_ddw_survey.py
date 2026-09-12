"""Measure all eight DDW variants for one explicitly selected survey clip."""

from novel_view.config.job import ResolvedJob
from novel_view.preparation.waymo_ddw.legacy.spec import LegacyDdwSpec, parse_legacy_ddw_spec
from novel_view.runtime.context import RuntimeContext


def parse_job(job: ResolvedJob) -> LegacyDdwSpec:
    return parse_legacy_ddw_spec(
        job.input, job.parameters, workflow_name=job.workflow.name,
        workflow_version=job.workflow.version, execution_preset=job.execution.preset,
    )


def select_image_variant(job: ResolvedJob) -> str:
    parse_job(job)
    return "moge"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    from novel_view.preparation.waymo_ddw.legacy.execution import open_legacy_ddw_measurement
    from novel_view.preparation.waymo_ddw.legacy.record import write_survey_record
    from novel_view.preparation.waymo_ddw.legacy.v2_canary import measure_waymo_ddw_survey

    spec = parse_job(job)
    result = runtime.attempt_root / "result.json"
    with open_legacy_ddw_measurement(spec, runtime) as measurement:
        evidence = measure_waymo_ddw_survey(
            measurement.source, spec.recipe.axis, measurement.warp,
            measurement.reference, measurement.logs,
        )
        write_survey_record(result, spec.input.depth_selection, measurement.depth, evidence)
    print(result)
    return 0 if evidence.passed else 2


__all__ = ["parse_job", "select_image_variant", "run_v1"]

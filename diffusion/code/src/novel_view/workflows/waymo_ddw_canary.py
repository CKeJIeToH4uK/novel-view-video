"""Keep the original and v2 DDW canary orders explicit."""

from novel_view.config.job import ResolvedJob
from novel_view.preparation.waymo_ddw.legacy.spec import (
    LegacyDdwSpec,
    parse_legacy_ddw_spec,
)
from novel_view.runtime.context import RuntimeContext


def parse_job(job: ResolvedJob) -> LegacyDdwSpec:
    return parse_legacy_ddw_spec(
        job.input,
        job.parameters,
        workflow_name=job.workflow.name,
        workflow_version=job.workflow.version,
        execution_preset=job.execution.preset,
    )


def select_image_variant(job: ResolvedJob) -> str:
    parse_job(job)
    return "moge"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Measure all eight variants; a v1 headroom failure remains an error."""
    from novel_view.preparation.waymo_ddw.legacy.canary import (
        measure_waymo_ddw_engineering_canary,
    )
    from novel_view.preparation.waymo_ddw.legacy.execution import open_legacy_ddw_measurement
    from novel_view.preparation.waymo_ddw.legacy.record import write_engineering_canary_record

    spec = parse_job(job)
    result = runtime.attempt_root / "result.json"
    with open_legacy_ddw_measurement(spec, runtime) as measurement:
        evidence = measure_waymo_ddw_engineering_canary(
            measurement.source, spec.recipe.axis, measurement.warp,
            measurement.reference, measurement.logs,
        )
        write_engineering_canary_record(
            result, spec.input.depth_selection, measurement.depth, evidence
        )
    print(result)
    return 0


def run_v2(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Persist the first scientific STOP and return 2 without continuing the axis."""
    from novel_view.preparation.waymo_ddw.legacy.execution import open_legacy_ddw_measurement
    from novel_view.preparation.waymo_ddw.legacy.record import (
        write_v2_canary_record,
        write_v2_canary_stop_record,
    )
    from novel_view.preparation.waymo_ddw.legacy.v2 import DdwV2ScientificStop
    from novel_view.preparation.waymo_ddw.legacy.v2_canary import measure_waymo_ddw_v2_canary

    spec = parse_job(job)
    result = runtime.attempt_root / "result.json"
    with open_legacy_ddw_measurement(spec, runtime) as measurement:
        try:
            evidence = measure_waymo_ddw_v2_canary(
                measurement.source, spec.recipe.axis, measurement.warp,
                measurement.reference, measurement.logs,
            )
        except DdwV2ScientificStop as stop:
            write_v2_canary_stop_record(
                result, spec.input.depth_selection, measurement.source.clip_key,
                measurement.depth, stop,
            )
            print(result)
            return 2
        write_v2_canary_record(result, spec.input.depth_selection, measurement.depth, evidence)
    print(result)
    return 0

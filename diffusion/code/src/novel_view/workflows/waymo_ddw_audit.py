"""Audit one explicitly selected rejection without changing the accepted set."""

from novel_view.config.job import ResolvedJob
from novel_view.preparation.waymo_ddw.legacy.fit_spec import parse_audit_job
from novel_view.runtime.context import RuntimeContext


def select_image_variant(job: ResolvedJob) -> str:
    parse_audit_job(job)
    return "moge"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    import numpy as np

    from novel_view.preparation.waymo_ddw.legacy.audit import (
        materialize_waymo_ddw_fit_audit, select_waymo_ddw_fit_audit,
        write_audit_preview,
    )
    from novel_view.preparation.waymo_ddw.legacy.execution import open_legacy_ddw_clip
    from novel_view.preparation.waymo_ddw.legacy.fit_record import (
        read_fit_audit_source, write_fit_audit_record,
    )

    spec = parse_audit_job(job)
    item = read_fit_audit_source(runtime.roots.runs / spec.item_record)
    selection = select_waymo_ddw_fit_audit(item.decision)
    with open_legacy_ddw_clip(
        selection.decision.assignment.clip_key, spec.moge_checkpoint, runtime,
        runtime.attempt_root / "logs",
    ) as measurement:
        result = materialize_waymo_ddw_fit_audit(
            measurement.source, selection, measurement.warp,
            measurement.logs / "audit-outward.log",
            measurement.logs / "audit-return.log",
        )
        np.save(runtime.attempt_root / "condition_rgb.npy", result.condition_rgb)
        np.save(runtime.attempt_root / "condition_known.npy", result.condition_known)
        write_audit_preview(runtime.attempt_root / "preview.mp4", measurement.source, result)
        write_fit_audit_record(
            runtime.attempt_root / "audit.json", spec.item_record, result, measurement.depth
        )
    print(runtime.attempt_root / "audit.json")
    return 0

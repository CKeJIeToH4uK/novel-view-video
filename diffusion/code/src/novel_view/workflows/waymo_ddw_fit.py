"""Run an explicit subset of the full historical DDW assignment."""

from novel_view.config.job import ResolvedJob
from novel_view.preparation.waymo_ddw.legacy.fit_spec import parse_fit_job
from novel_view.runtime.context import RuntimeContext


def select_image_variant(job: ResolvedJob) -> str:
    parse_fit_job(job)
    return "moge"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    import numpy as np

    from novel_view.preparation.waymo_ddw.legacy.assignment import build_ddw_fit_assignment
    from novel_view.preparation.waymo_ddw.legacy.execution import open_legacy_ddw_clip
    from novel_view.preparation.waymo_ddw.legacy.fit import prepare_waymo_ddw_fit_clip
    from novel_view.preparation.waymo_ddw.legacy.fit_record import (
        build_fit_item_record, write_fit_item_record,
    )
    from novel_view.preparation.waymo_ddw.legacy.fit_spec import load_fit_keyset, load_fit_selection

    spec = parse_fit_job(job)
    selection = load_fit_selection(runtime.roots.selections.parent / spec.selection)
    keyset = load_fit_keyset(runtime.roots.runs / selection.keyset)
    table = build_ddw_fit_assignment(keyset.keys)
    assignments = tuple(table.assignments[index] for index in selection.indices)
    for assignment in assignments:
        item_root = runtime.attempt_root / "items" / f"clip-{assignment.global_fit_index:04d}"
        with open_legacy_ddw_clip(
            assignment.clip_key, spec.moge_checkpoint, runtime, item_root / "logs"
        ) as measurement:
            result = prepare_waymo_ddw_fit_clip(
                measurement.source, assignment, measurement.warp,
                measurement.reference, measurement.logs,
            )
            if result.accepted_condition is not None:
                np.save(item_root / "condition_rgb.npy", result.accepted_condition.condition_rgb)
                np.save(item_root / "condition_known.npy", result.accepted_condition.condition_known)
            write_fit_item_record(
                item_root / "clip.json", build_fit_item_record(result, measurement.depth)
            )
            del result
        del measurement
        print(item_root / "clip.json")
    return 0

"""Collect explicitly named item records in the full assignment order."""

from novel_view.config.job import ResolvedJob
from novel_view.preparation.waymo_ddw.legacy.fit_spec import parse_collection_job
from novel_view.runtime.context import RuntimeContext


def select_image_variant(job: ResolvedJob) -> str:
    parse_collection_job(job)
    return "core"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    from novel_view.preparation.waymo_ddw.legacy.assignment import build_ddw_fit_assignment
    from novel_view.preparation.waymo_ddw.legacy.fit_record import (
        collect_fit_records, read_fit_item_record, write_fit_collection_record,
    )
    from novel_view.preparation.waymo_ddw.legacy.fit_spec import (
        load_fit_keyset, load_fit_collection_selection,
    )

    selection_path = parse_collection_job(job)
    selection = load_fit_collection_selection(runtime.roots.selections.parent / selection_path)
    keyset = load_fit_keyset(runtime.roots.runs / selection.keyset)
    table = build_ddw_fit_assignment(keyset.keys)
    items = tuple(
        (locator, read_fit_item_record(runtime.roots.runs / locator))
        for locator in selection.item_records
    )
    result = collect_fit_records(table.assignments, items)
    output = runtime.attempt_root / "collection.json"
    write_fit_collection_record(output, result)
    print(output)
    return 0

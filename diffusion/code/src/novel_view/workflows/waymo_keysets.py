"""Три исторических Waymo keysets: центральные ключи до RGB и геометрии."""

from __future__ import annotations

from typing import TYPE_CHECKING

from novel_view.config.job import ResolvedJob
from novel_view.config.load import reject_unknown_fields
from novel_view.preparation.waymo_depth.spec import load_split_roles
from novel_view.runtime.context import RuntimeContext

if TYPE_CHECKING:
    from novel_view.preparation.waymo_depth.keyset import PreparedKeyset, TimelineLoader


def parse_job(job: ResolvedJob) -> str:
    """Разобрать только внешние параметры CPU-задания."""
    if (job.workflow.name, job.workflow.version) != ("waymo_keysets", 1):
        raise ValueError("waymo_keysets requires version 1")
    if job.execution.preset != "cpu_test":
        raise ValueError("waymo_keysets requires cpu_test")
    reject_unknown_fields(job.input, frozenset({"reader", "dataset", "selection"}))
    reject_unknown_fields(job.parameters, frozenset())
    if job.input.get("reader") != "waymo_v2" or job.input.get("dataset") != "waymo":
        raise ValueError("waymo_keysets requires waymo_v2 and dataset waymo")
    selection = job.input.get("selection")
    if not isinstance(selection, str) or not selection.strip():
        raise ValueError("waymo_keysets requires an explicit selection")
    return selection


def select_image_variant(job: ResolvedJob) -> str:
    parse_job(job)
    return "core"


def build_named_keysets(
    split_id: str,
    fit_ids: tuple[str, ...],
    dev_ids: tuple[str, ...],
    debug_ids: tuple[str, ...],
    load_timeline: TimelineLoader,
) -> tuple[PreparedKeyset, PreparedKeyset, PreparedKeyset]:
    """Сохранить fit/dev/debug order, переиспользуя уже прочитанные timelines."""
    from novel_view.preparation.waymo_depth.keyset import build_central_keyset

    timelines = {}

    def cached(partition, segment_id):
        identity = (partition, segment_id)
        if identity not in timelines:
            timelines[identity] = tuple(load_timeline(partition, segment_id))
        return timelines[identity]

    fit = build_central_keyset(
        keyset_id="fit-central-v1", split_id=split_id, official_partition="training",
        segment_ids=fit_ids, load_timeline=cached,
    )
    dev = build_central_keyset(
        keyset_id="dev-central-v1", split_id=split_id, official_partition="training",
        segment_ids=dev_ids, load_timeline=cached,
    )
    vertical = build_central_keyset(
        keyset_id="vertical-debug2-v1", split_id=split_id, official_partition="training",
        segment_ids=debug_ids[:2], load_timeline=cached,
    )
    return fit, dev, vertical


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Прочитать явный список, выбрать central121 и записать прежние JSON."""
    from novel_view.inputs.waymo.index import build_waymo_v2_window_index

    selection = parse_job(job)
    roles = load_split_roles(runtime.roots.selections.parent / selection)

    def load_timeline(partition, segment_id):
        return build_waymo_v2_window_index(
            runtime.roots.data / "waymo", partition, segment_id, 0, 1,
        ).timeline

    keysets = build_named_keysets(*roles, load_timeline)
    output = runtime.attempt_root / "keysets"
    output.mkdir(parents=True, exist_ok=True)
    for keyset in keysets:
        path = output / f"{keyset.keyset_id}.json"
        keyset.write(path)
        print(path)
    return 0

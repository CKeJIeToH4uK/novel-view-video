"""Select one v1 or v2 checkpoint from explicit readable records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from novel_view.config.job import ResolvedJob
from novel_view.config.load import reject_unknown_fields
from novel_view.runtime.context import RuntimeContext


@dataclass(frozen=True, slots=True)
class Gen3cCheckpointSelectionInput:
    """Ordered runs-relative checkpoint record locators."""

    checkpoint_records: tuple[Path, ...]


def parse_job(job: ResolvedJob) -> Gen3cCheckpointSelectionInput:
    """Parse one strict CPU-only selection job."""
    if (
        job.workflow.name != "gen3c_checkpoint_selection"
        or job.workflow.version
        not in (
            1,
            2,
        )
    ):
        raise ValueError("checkpoint selection requires a supported version")
    if job.execution.preset != "cpu_test":
        raise ValueError("checkpoint selection requires cpu_test")
    reject_unknown_fields(job.input, frozenset({"checkpoint_records"}))
    reject_unknown_fields(job.parameters, frozenset())
    values = job.input.get("checkpoint_records")
    if not isinstance(values, list) or not values:
        raise ValueError("checkpoint_records must be a non-empty list")
    paths = tuple(Path(value) for value in values if isinstance(value, str) and value)
    if len(paths) != len(values):
        raise ValueError("checkpoint_records must contain non-empty paths")
    return Gen3cCheckpointSelectionInput(paths)


def select_image_variant(job: ResolvedJob) -> str:
    """Use the CPU-capable core image."""
    parse_job(job)
    return "core"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Read only named JSON records and publish one explicit result."""
    from novel_view.training.gen3c.checkpoint_selection import (
        build_checkpoint_selection_v1,
        select_checkpoint_v1,
        write_checkpoint_selection_v1,
    )
    from novel_view.training.gen3c.lora.record import read_checkpoint_record_v1

    spec = parse_job(job)
    candidates = tuple(
        read_checkpoint_record_v1(runtime.roots.runs / path)
        for path in spec.checkpoint_records
    )
    selected = select_checkpoint_v1(candidates)
    result = build_checkpoint_selection_v1(
        spec.checkpoint_records,
        candidates,
        selected,
    )
    output = runtime.attempt_root / "selection.json"
    write_checkpoint_selection_v1(output, result)
    print(output)
    return 0


def run_v2(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Select only among named v2 records of one objective and lineage."""
    from novel_view.training.gen3c.checkpoint_selection import (
        build_checkpoint_selection_v2,
        select_checkpoint_v2,
        write_checkpoint_selection_v2,
    )
    from novel_view.training.gen3c.lora.depth_record import read_checkpoint_record_v2

    spec = parse_job(job)
    candidates = tuple(
        read_checkpoint_record_v2(runtime.roots.runs / path)
        for path in spec.checkpoint_records
    )
    selected = select_checkpoint_v2(candidates)
    result = build_checkpoint_selection_v2(
        spec.checkpoint_records,
        candidates,
        selected,
    )
    output = runtime.attempt_root / "selection.json"
    write_checkpoint_selection_v2(output, result)
    print(output)
    return 0


__all__ = [
    "Gen3cCheckpointSelectionInput",
    "parse_job",
    "run_v1",
    "run_v2",
    "select_image_variant",
]

"""Неизменяемая schema v1 одной operational attempt."""

from dataclasses import dataclass, replace
from typing import Mapping


@dataclass(frozen=True, slots=True)
class AttemptImage:
    variant: str
    image_id: str
    source_revision: str
    source_dirty: bool
    lock_revision: str


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    schema_version: int
    job_name: str
    run_id: str
    attempt_id: str
    attempt_kind: str
    resolved_job: Mapping[str, object]
    selected_checkpoint: str | None
    image: AttemptImage
    container_name: str
    recorded_state: str
    worker_exit_code: int | None
    started_at: str
    finished_at: str | None


def finish_attempt_record(
    record: AttemptRecord,
    *,
    state: str,
    worker_exit_code: int | None,
    finished_at: str,
) -> AttemptRecord:
    return replace(
        record,
        recorded_state=state,
        worker_exit_code=worker_exit_code,
        finished_at=finished_at,
    )


def attempt_record_to_mapping(record: AttemptRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "job_name": record.job_name,
        "run_id": record.run_id,
        "attempt_id": record.attempt_id,
        "attempt_kind": record.attempt_kind,
        "resolved_job": dict(record.resolved_job),
        "selected_checkpoint": record.selected_checkpoint,
        "image": {
            "variant": record.image.variant,
            "id": record.image.image_id,
            "source_revision": record.image.source_revision,
            "source_dirty": record.image.source_dirty,
            "lock_revision": record.image.lock_revision,
        },
        "container_name": record.container_name,
        "recorded_state": record.recorded_state,
        "worker_exit_code": record.worker_exit_code,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }

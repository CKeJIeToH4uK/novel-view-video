"""Прямая запись одного ``attempt.json`` без staging и общего run state."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

from novel_view.runs.record import (
    AttemptImage,
    AttemptRecord,
    attempt_record_to_mapping,
    finish_attempt_record,
)


class RunStore:
    """Владеет переносимой записью одной уже созданной attempt-root."""

    def __init__(self, attempt_root: Path) -> None:
        self._record_file = attempt_root / "attempt.json"

    def start_attempt(
        self,
        *,
        job_name: str,
        run_id: str,
        attempt_id: str,
        attempt_kind: str,
        resolved_job: Mapping[str, object],
        selected_checkpoint: str | None,
        image: AttemptImage,
        container_name: str,
    ) -> AttemptRecord:
        record = AttemptRecord(
            schema_version=1,
            job_name=job_name,
            run_id=run_id,
            attempt_id=attempt_id,
            attempt_kind=attempt_kind,
            resolved_job=dict(resolved_job),
            selected_checkpoint=selected_checkpoint,
            image=image,
            container_name=container_name,
            recorded_state="running",
            worker_exit_code=None,
            started_at=_now(),
            finished_at=None,
        )
        self._write(record)
        return record

    def finish_attempt(
        self,
        record: AttemptRecord,
        *,
        state: str,
        worker_exit_code: int | None,
    ) -> AttemptRecord:
        finished = finish_attempt_record(
            record,
            state=state,
            worker_exit_code=worker_exit_code,
            finished_at=_now(),
        )
        self._write(finished)
        return finished

    def _write(self, record: AttemptRecord) -> None:
        self._record_file.write_text(
            json.dumps(
                attempt_record_to_mapping(record),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

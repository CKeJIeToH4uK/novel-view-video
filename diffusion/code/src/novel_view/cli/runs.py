"""Read-only снимки переносимых ``attempt.json`` для host launcher."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

from novel_view.config.job import ResolvedJob, resolved_job_from_mapping


@dataclass(frozen=True, slots=True)
class AttemptSnapshot:
    attempt_ref: str
    recorded_state: str
    record_present: bool
    started_at: str


def read_attempt_record(record_file: Path) -> dict[str, object]:
    return json.loads(record_file.read_text(encoding="utf-8"))


def print_attempt_record(record_file: Path) -> int:
    record = read_attempt_record(record_file)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def read_attempt_snapshot(
    runs_root: Path,
    job_name: str,
    run_id: str,
    attempt_id: str | None,
) -> AttemptSnapshot:
    if attempt_id is not None:
        record_file = attempt_record_file(
            runs_root,
            job_name,
            run_id,
            attempt_id,
        )
        try:
            record = read_attempt_record(record_file)
        except FileNotFoundError:
            return AttemptSnapshot(
                attempt_ref=f"{job_name}/{run_id}/{attempt_id}",
                recorded_state="absent",
                record_present=False,
                started_at="",
            )
        return _attempt_snapshot(record_file, record)

    attempts_root = runs_root / job_name / run_id / "attempts"
    snapshots = tuple(
        _attempt_snapshot(record_file, read_attempt_record(record_file))
        for record_file in attempts_root.glob("*/attempt.json")
    )
    return max(
        snapshots,
        key=lambda snapshot: (snapshot.started_at, snapshot.attempt_ref),
    )


def print_attempt_snapshot(
    runs_root: Path,
    job_name: str,
    run_id: str,
    attempt_id: str | None,
) -> int:
    snapshot = read_attempt_snapshot(
        runs_root,
        job_name,
        run_id,
        attempt_id,
    )
    print(f"attempt_ref={snapshot.attempt_ref}")
    print(f"recorded_state={snapshot.recorded_state}")
    print(f"record_present={str(snapshot.record_present).lower()}")
    return 0


def read_resolved_job_from_attempt(record_file: Path) -> ResolvedJob:
    record = read_attempt_record(record_file)
    return resolved_job_from_mapping(
        cast(dict[str, object], record["resolved_job"])
    )


def attempt_record_file(
    runs_root: Path,
    job_name: str,
    run_id: str,
    attempt_id: str,
) -> Path:
    return (
        runs_root
        / job_name
        / run_id
        / "attempts"
        / attempt_id
        / "attempt.json"
    )


def _attempt_snapshot(
    record_file: Path,
    record: dict[str, object],
) -> AttemptSnapshot:
    return AttemptSnapshot(
        attempt_ref=(
            f"{record['job_name']}/{record['run_id']}/{record_file.parent.name}"
        ),
        recorded_state=cast(str, record["recorded_state"]),
        record_present=True,
        started_at=cast(str, record["started_at"]),
    )

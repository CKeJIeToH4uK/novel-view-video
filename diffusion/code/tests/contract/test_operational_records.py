"""Один Python writer attempt v1 и read-only восстановление без job YAML."""

import json
from pathlib import Path

import pytest

from novel_view.cli.runs import (
    read_attempt_record,
    read_attempt_snapshot,
    read_resolved_job_from_attempt,
)
from novel_view.config.job import load_resolved_job, resolved_job_to_mapping
from novel_view.runs.record import AttemptImage
from novel_view.runs.store import RunStore

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("state,code", [("succeeded", 0), ("failed", 23)])
def test_attempt_v1_roundtrip_and_saved_job(tmp_path, state, code):
    job_file = tmp_path / "run.yaml"
    job_file.write_bytes((ROOT / "jobs/examples/lifecycle-smoke/run.yaml").read_bytes())
    job = load_resolved_job(job_file, ROOT)
    store = RunStore(tmp_path)
    running = store.start_attempt(
        job_name=job.name,
        run_id="run-1",
        attempt_id="attempt-1",
        attempt_kind="run",
        resolved_job=resolved_job_to_mapping(job),
        selected_checkpoint=None,
        image=AttemptImage("cpu-test", "sha256:image", "revision", False, "sha256:lock"),
        container_name="distil3d-attempt-1",
    )
    finished = store.finish_attempt(running, state=state, worker_exit_code=code)
    assert (running.recorded_state, running.finished_at) == ("running", None)
    assert finished.finished_at and finished.started_at == running.started_at
    job_file.unlink()
    record_file = tmp_path / "attempt.json"
    before = record_file.read_bytes()
    record = read_attempt_record(record_file)
    assert record == {
        "schema_version": 1,
        "job_name": "lifecycle-smoke",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "attempt_kind": "run",
        "resolved_job": resolved_job_to_mapping(job),
        "selected_checkpoint": None,
        "image": {
            "variant": "cpu-test",
            "id": "sha256:image",
            "source_revision": "revision",
            "source_dirty": False,
            "lock_revision": "sha256:lock",
        },
        "container_name": "distil3d-attempt-1",
        "recorded_state": state,
        "worker_exit_code": code,
        "started_at": running.started_at,
        "finished_at": finished.finished_at,
    }
    assert list(record) == sorted(record)
    assert read_resolved_job_from_attempt(record_file) == job
    assert record_file.read_bytes() == before
    assert not any(
        text in json.dumps(record).lower() for text in ("traceback", "/users/", "/home/")
    )


def test_snapshot_is_read_only_and_selects_latest_recorded_attempt(tmp_path):
    missing = read_attempt_snapshot(tmp_path, "job", "run", "missing")
    assert (missing.attempt_ref, missing.recorded_state, missing.record_present) == (
        "job/run/missing",
        "absent",
        False,
    )
    assert list(tmp_path.iterdir()) == []
    for attempt, hour, state in [("a", 10, "succeeded"), ("b", 11, "failed")]:
        record_root = tmp_path / "job/run/attempts" / attempt
        record_root.mkdir(parents=True)
        (record_root / "attempt.json").write_text(
            json.dumps(
                {
                    "job_name": "job",
                    "run_id": "run",
                    "recorded_state": state,
                    "started_at": f"2026-08-27T{hour}:00:00+00:00",
                }
            )
        )
    latest = read_attempt_snapshot(tmp_path, "job", "run", None)
    assert (latest.attempt_ref, latest.recorded_state, latest.record_present) == (
        "job/run/b",
        "failed",
        True,
    )

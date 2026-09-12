"""Настоящие runner exit/records и resume из сохранённой задачи."""

import json
import subprocess
import sys

import pytest

from novel_view.config.job import load_resolved_job
from novel_view.runtime.context import CONTAINER_ROOTS, ImageMetadata, RuntimeContext
from novel_view.workflows.runner import execute_job


def _job(root, mode):
    path = root / "run.yaml"
    path.write_text(
        "schema_version: 1\nname: lifecycle-smoke\n"
        "workflow: {name: _lifecycle_smoke, version: 1}\ninput: {}\n"
        f"parameters: {{mode: {mode}, exit_code: 23}}\n"
        "execution: {preset: cpu_test}\n"
    )
    return path, load_resolved_job(path, root)


def _runtime(root, attempt):
    directory = root / attempt
    directory.mkdir()
    return RuntimeContext(
        roots=CONTAINER_ROOTS,
        run_id="run-1",
        attempt_id=attempt,
        attempt_kind="run",
        attempt_root=directory,
        container_name=f"distil3d-{attempt}",
        image=ImageMetadata(
            "cpu-test", "sha256:integration", "integration", False, "sha256:integration"
        ),
        selected_checkpoint=None,
    )


@pytest.mark.parametrize(
    "mode,code,state", (("success", 0, "succeeded"), ("failure", 23, "failed"))
)
def test_runner_keeps_worker_code_and_two_fresh_records(tmp_path, capfd, mode, code, state):
    _, job = _job(tmp_path, mode)
    records = []
    for attempt in ("first", "second"):
        runtime = _runtime(tmp_path, attempt)
        assert execute_job(job, runtime) == code
        captured = capfd.readouterr()
        record = json.loads((runtime.attempt_root / "attempt.json").read_text())
        assert (record["recorded_state"], record["worker_exit_code"]) == (state, code)
        assert ("Traceback (most recent call last)" in captured.err) == (mode == "failure")
        assert "traceback" not in json.dumps(record).lower()
        records.append(record)
    assert [record["attempt_id"] for record in records] == ["first", "second"]
    assert all(
        record["attempt_kind"] == "run" and record["run_id"] == "run-1"
        for record in records
    )


def test_resume_uses_saved_job_and_keeps_checkpoint_opaque(tmp_path):
    job_file, job = _job(tmp_path, "success")
    source = _runtime(tmp_path, "source")
    assert execute_job(job, source) == 0
    saved = source.attempt_root / "attempt.json"
    expected_job = json.loads(saved.read_text())["resolved_job"]
    job_file.unlink()
    checkpoint = "/models/operator choice/../selected.pt"
    resumed = _runtime(tmp_path, "resume")
    command = (
        sys.executable, "-m", "novel_view.cli.main", "execute",
        "--resume-record", str(saved),
        "--config-root", str(tmp_path),
        "--run-id", "run-1",
        "--attempt-id", "resume",
        "--attempt-kind", "resume",
        "--attempt-root", str(resumed.attempt_root),
        "--container-name", "distil3d-resume",
        "--image-variant", "cpu-test",
        "--image-id", "sha256:integration",
        "--image-source-revision", "integration",
        "--image-source-dirty", "false",
        "--image-lock-revision", "sha256:integration",
        "--selected-checkpoint", checkpoint,
    )
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    record = json.loads((resumed.attempt_root / "attempt.json").read_text())
    assert (record["attempt_kind"], record["selected_checkpoint"]) == ("resume", checkpoint)
    assert record["resolved_job"] == expected_job

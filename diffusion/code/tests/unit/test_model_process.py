"""Публичные результаты изолированного процесса и явного завершения группы."""

import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from unittest.mock import Mock

import pytest

import novel_view.runtime.process as process_module
from novel_view.runtime.distributed import start_torchrun
from novel_view.runtime.process import (
    ProcessError,
    RunningProcess,
    clean_process_environment,
    run_process,
)


def test_environment_and_caller_log_reach_real_worker(tmp_path, monkeypatch):
    removed = (
        "PYTHONHOME",
        "PYTHONPATH",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    )
    for name in removed:
        monkeypatch.setenv(name, "unwanted")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    inherited = clean_process_environment({"CUDA_HOME": "/opt/cuda", "PYTHONUNBUFFERED": "1"})
    assert not set(removed) & inherited.keys()
    assert (inherited["CUDA_VISIBLE_DEVICES"], inherited["PYTHONUNBUFFERED"]) == ("3", "1")
    with pytest.raises(ProcessError):
        clean_process_environment({"PYTHONPATH": "/untrusted"})

    log = tmp_path / "worker.log"
    log.write_text("old output\n")
    names = (*removed, "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PYTHONNOUSERSITE")
    command = "import json, os\nprint(json.dumps([os.environ.get(k) for k in " + repr(names) + "]))"
    run_process(
        [sys.executable, "-c", command],
        log,
        "environment",
        {"CUDA_VISIBLE_DEVICES": "1", "CUDA_HOME": "/opt/cuda"},
    )
    assert json.loads(log.read_text()) == [*[None] * len(removed), "1", "/opt/cuda", "1"]


def test_failure_keeps_full_log_after_scratch_and_original_code(tmp_path):
    log = tmp_path / "worker.log"
    with pytest.raises(ProcessError, match="exited with code 23") as caught:
        with tempfile.TemporaryDirectory(dir=tmp_path) as scratch:
            worker = Path(scratch) / "worker.py"
            worker.write_text(
                "import sys, traceback\n"
                "print('BEGIN:' + 'x' * 12000, flush=True)\n"
                "try:\n"
                "    raise ValueError('original science failure')\n"
                "except ValueError:\n"
                "    traceback.print_exc()\n"
                "    sys.exit(23)\n"
            )
            run_process([sys.executable, str(worker)], log, "worker")
    assert not Path(scratch).exists()
    output = log.read_text()
    assert output.startswith("BEGIN:" + "x" * 12000)
    assert "Traceback (most recent call last)" in output
    assert "ValueError: original science failure" in str(caught.value)


def test_natural_wait_has_no_deadline(tmp_path):
    child = Mock(pid=1234)
    child.wait.return_value = child.poll.return_value = 0
    log = tmp_path / "worker.log"
    with log.open("wb") as stream:
        RunningProcess(child, stream, log, "worker").wait()
        assert stream.closed
    child.wait.assert_called_once_with()


def test_torchrun_returns_live_handle_with_literal_argv(tmp_path, monkeypatch):
    child = Mock(pid=12345)
    child.poll.return_value = None
    popen = Mock(return_value=child)
    monkeypatch.setattr(process_module.os, "getpid", lambda: 4321)
    monkeypatch.setattr(process_module.threading, "get_native_id", lambda: 8765)
    monkeypatch.setattr(process_module.subprocess, "Popen", popen)
    worker = tmp_path / "worker.py"
    log = tmp_path / "worker.log"
    running = start_torchrun(
        Path(sys.executable),
        worker,
        ["--request", "value with spaces"],
        process_count=2,
        log_path=log,
        description="two ranks",
        environment_overrides={"CUDA_VISIBLE_DEVICES": "3,1"},
    )
    assert running.pid == child.pid and running.poll() is None
    child.wait.assert_not_called()
    assert popen.call_args.args[0] == [
        sys.executable, str(Path(process_module.__file__).with_name("_parent_guard.py")),
        "4321", "8765", "--", sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1", "--nproc-per-node=2", "--max-restarts=0",
        str(worker), "--request", "value with spaces",
    ]
    assert "shell" not in popen.call_args.kwargs
    assert popen.call_args.kwargs["start_new_session"]
    assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3,1"
    monkeypatch.setattr(process_module.os, "killpg", Mock(side_effect=ProcessLookupError))
    child.poll.return_value = 0
    running.terminate()


@pytest.mark.parametrize("state", ("exits_on_term", "ignores_term", "leader_exited"))
def test_terminate_handles_grace_and_surviving_descendants(tmp_path, monkeypatch, state):
    child = Mock(pid=2468)
    child.poll.return_value = 1 if state == "leader_exited" else None
    child.wait.side_effect = (
        (subprocess.TimeoutExpired("worker", 0.01), 0) if state == "ignores_term" else (0,)
    )
    kill_group = Mock()
    if state == "exits_on_term":
        kill_group.side_effect = (None, ProcessLookupError)
    monkeypatch.setattr(process_module.os, "killpg", kill_group)
    log = tmp_path / "worker.log"
    with log.open("wb") as stream:
        RunningProcess(child, stream, log, "worker").terminate(grace_seconds=0.01)
        assert stream.closed
    signals = {call.args for call in kill_group.call_args_list}
    assert (child.pid, signal.SIGTERM) in signals
    assert ((child.pid, signal.SIGKILL) in signals) == (state != "exits_on_term")
    if state == "leader_exited":
        child.wait.assert_not_called()
    else:
        assert child.wait.call_args_list[0].kwargs == {"timeout": 0.01}

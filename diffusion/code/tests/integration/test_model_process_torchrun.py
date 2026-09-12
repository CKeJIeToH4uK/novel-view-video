"""Два настоящих CPU-rank: явная остановка и смерть их controller."""

import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress

import pytest

from novel_view.runtime.distributed import start_torchrun

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux parent guard and /proc")


def _live_processes(pids):
    live = set()
    for pid in pids:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if state not in {"X", "Z"}:
            live.add(pid)
    return live


@pytest.mark.parametrize("mode", ("explicit_stop", "controller_death"))
def test_two_ranks_are_reaped(tmp_path, mode):
    worker = tmp_path / "sleeping_rank.py"
    worker.write_text(
        "import os, time\n"
        "print(f'rank={os.environ[\"RANK\"]} pid={os.getpid()}', flush=True)\n"
        "time.sleep(120)\n"
    )
    log = tmp_path / "worker.log"
    running = controller = None
    launcher_pid = None
    tracked = set()
    try:
        if mode == "explicit_stop":
            running = start_torchrun(
                Path(sys.executable),
                worker,
                [],
                process_count=2,
                log_path=log,
                description="CPU explicit stop",
                environment_overrides={"CUDA_VISIBLE_DEVICES": ""},
            )
            launcher_pid = running.pid
            tracked.add(launcher_pid)
        else:
            script = tmp_path / "controller.py"
            script.write_text(
                "import sys, time\n"
                "from pathlib import Path\n"
                "from novel_view.runtime.distributed import start_torchrun\n"
                "root = Path(sys.argv[1])\n"
                "running = start_torchrun(\n"
                "    Path(sys.executable), root / 'sleeping_rank.py', [],\n"
                "    process_count=2, log_path=root / 'worker.log',\n"
                "    description='controller death',\n"
                "    environment_overrides={'CUDA_VISIBLE_DEVICES': ''},\n"
                ")\n"
                "(root / 'launcher.pid').write_text(str(running.pid))\n"
                "time.sleep(120)\n"
            )
            controller = subprocess.Popen(
                [sys.executable, str(script), str(tmp_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            tracked.add(controller.pid)

        rank_pids = {}
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if log.exists():
                rank_pids = {
                    int(rank): int(pid)
                    for rank, pid in re.findall(r"rank=(\d+) pid=(\d+)", log.read_text())
                }
                tracked.update(rank_pids.values())
            if launcher_pid is None and (tmp_path / "launcher.pid").exists():
                launcher_pid = int((tmp_path / "launcher.pid").read_text())
                tracked.add(launcher_pid)
            if set(rank_pids) == {0, 1} and launcher_pid is not None:
                break
            time.sleep(0.05)
        assert set(rank_pids) == {0, 1} and len(set(rank_pids.values())) == 2
        assert launcher_pid is not None
        assert (running or controller).poll() is None

        if running is not None:
            running.terminate(grace_seconds=5)
        else:
            controller.kill()
            controller.wait(timeout=5)
        deadline = time.monotonic() + 10
        live = tracked
        while live and time.monotonic() < deadline:
            live = (
                {pid for pid in rank_pids.values() if Path(f"/proc/{pid}").exists()}
                if mode == "explicit_stop"
                else _live_processes(tracked)
            )
            if live:
                time.sleep(0.05)
        assert not live, f"surviving processes: {live}"
    finally:
        if running is not None:
            running.terminate(grace_seconds=1)
        if controller is not None:
            controller.kill()
            controller.wait(timeout=5)
        for pid in _live_processes(tracked):
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)

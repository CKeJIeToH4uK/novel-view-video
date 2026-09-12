"""Внутренняя synthetic-схема для проверки общего lifecycle v1."""

from dataclasses import dataclass
import os
import signal
import subprocess
import sys
import time
import traceback

from novel_view.config.job import ResolvedJob, reject_unknown_fields
from novel_view.runtime.context import RuntimeContext
from novel_view.runtime.process import run_worker_process


_PARAMETER_FIELDS = frozenset({"mode", "exit_code"})
_MODES = frozenset({"success", "failure", "long", "stubborn", "process_tree"})


@dataclass(frozen=True, slots=True)
class LifecycleSmokeSpec:
    mode: str
    exit_code: int


def parse_lifecycle_smoke_job(job: ResolvedJob) -> LifecycleSmokeSpec:
    reject_unknown_fields(job.input, frozenset())
    reject_unknown_fields(job.parameters, _PARAMETER_FIELDS)

    missing = _PARAMETER_FIELDS - job.parameters.keys()
    if missing:
        raise ValueError(f"missing lifecycle-smoke fields: {sorted(missing)}")

    mode = job.parameters["mode"]
    exit_code = job.parameters["exit_code"]
    if not isinstance(mode, str):
        raise ValueError("lifecycle-smoke mode must be a string")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ValueError("lifecycle-smoke exit_code must be an integer")
    if mode not in _MODES:
        raise ValueError(f"unsupported lifecycle-smoke mode: {mode}")
    return LifecycleSmokeSpec(mode=mode, exit_code=exit_code)


def select_image_variant(job: ResolvedJob) -> str:
    parse_lifecycle_smoke_job(job)
    return "cpu-test"


def run_v1(job: ResolvedJob, _runtime: RuntimeContext) -> int:
    spec = parse_lifecycle_smoke_job(job)
    return run_worker_process(_worker_command(spec, role="leader"))


def run_worker(spec: LifecycleSmokeSpec, *, role: str) -> int:
    if spec.mode == "success":
        print("lifecycle-smoke: success", flush=True)
        return 0
    if spec.mode == "failure":
        try:
            raise RuntimeError("synthetic lifecycle-smoke failure")
        except RuntimeError:
            traceback.print_exc()
        return spec.exit_code
    if spec.mode == "long":
        _install_graceful_sigterm(role)
        _print_role(role)
        return _wait_forever()
    if spec.mode in {"stubborn", "process_tree"}:
        return _run_process_tree_role(spec, role)
    raise ValueError(f"unsupported lifecycle-smoke mode: {spec.mode}")


def _run_process_tree_role(spec: LifecycleSmokeSpec, role: str) -> int:
    if spec.mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        _install_graceful_sigterm(role)
    if role == "grandchild":
        _print_role(role)
        print("ready role=grandchild", flush=True)
        return _wait_forever()

    next_role = "child" if role == "leader" else "grandchild"
    child = subprocess.Popen(
        _worker_command(spec, role=next_role),
        stdout=subprocess.PIPE,
        text=True,
    )
    _forward_until(child, f"ready role={next_role}")
    _print_role(role, child_pid=child.pid)
    if role == "leader":
        print(f"ready process_group={os.getpgrp()}", flush=True)
    else:
        print("ready role=child", flush=True)
    _forward_remaining(child)
    return child.wait()


def _forward_until(child: subprocess.Popen[str], marker: str) -> None:
    assert child.stdout is not None
    for line in child.stdout:
        print(line, end="", flush=True)
        if marker in line:
            return
    return_code = child.wait()
    raise RuntimeError(
        f"synthetic process exited with code {return_code} before {marker}"
    )


def _forward_remaining(child: subprocess.Popen[str]) -> None:
    assert child.stdout is not None
    for line in child.stdout:
        print(line, end="", flush=True)


def _worker_command(spec: LifecycleSmokeSpec, *, role: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "novel_view.cli.main",
        "worker",
        "--mode",
        spec.mode,
        "--exit-code",
        str(spec.exit_code),
        "--role",
        role,
    )


def _install_graceful_sigterm(role: str) -> None:
    def stop_role(_signum: int, _frame: object) -> None:
        print(
            f"signal role={role} signal=SIGTERM",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_role)


def _print_role(role: str, *, child_pid: int | None = None) -> None:
    fields = [
        f"role={role}",
        f"pid={os.getpid()}",
        f"ppid={os.getppid()}",
        f"pgid={os.getpgrp()}",
        f"sid={os.getsid(0)}",
    ]
    if child_pid is not None:
        fields.append(f"child_pid={child_pid}")
    print(" ".join(fields), flush=True)


def _wait_forever() -> int:
    while True:
        time.sleep(60)

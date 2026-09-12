"""Run one child process group and forward an explicit container stop."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import BinaryIO


_ALLOWED_ENVIRONMENT_OVERRIDES = frozenset(
    {
        "COSMOS_CACHE_DIR",
        "CUDA_HOME",
        "CUDA_CACHE_PATH",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "WARP_CACHE_PATH",
        "HF_HUB_OFFLINE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TORCH_EXTENSIONS_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_HOME",
        "TRANSFORMERS_OFFLINE",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
    }
)
_REMOVED_ENVIRONMENT_VARIABLES = (
    "PYTHONHOME",
    "PYTHONPATH",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)
_PARENT_GUARD = Path(__file__).with_name("_parent_guard.py")
_PARENT_GUARD_SEPARATOR = "--"


STOP_EXIT_CODE = 128 + signal.SIGTERM


class StopRequested(Exception):
    """Container runner получил явный SIGTERM от launcher."""


class ProcessError(RuntimeError):
    """One child process cannot be started, stopped, or completed."""


class RunningProcess:
    """Own one child process and its operating-system process group."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        log_file: BinaryIO,
        log_path: Path,
        description: str,
    ) -> None:
        self._process = process
        self._log_file = log_file
        self.log_path = log_path
        self.description = description

    @property
    def pid(self) -> int:
        """Return the process-group leader PID."""
        return self._process.pid

    def poll(self) -> int | None:
        """Return the current exit code without blocking."""
        return self._process.poll()

    def wait(self) -> None:
        """Wait for natural completion without a computation deadline."""
        try:
            returncode = self._process.wait()
        finally:
            if self._process.poll() is not None:
                self._close_log()
        if returncode != 0:
            detail = read_log_tail(self.log_path)
            raise ProcessError(
                f"{self.description} exited with code {returncode}: "
                f"{detail or 'no diagnostic output'}"
            )

    def terminate(self, grace_seconds: float = 10.0) -> None:
        """Stop the process group, escalating after explicit stop grace."""
        self._signal_group(signal.SIGTERM)
        forced = False
        if self._process.poll() is None:
            try:
                self._process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                self._process.wait()
                forced = True
        if not forced and self._process_group_exists():
            self._signal_group(signal.SIGKILL)
        self._close_log()

    def _signal_group(self, signal_number: int) -> None:
        try:
            os.killpg(self._process.pid, signal_number)
        except ProcessLookupError:
            pass

    def _process_group_exists(self) -> bool:
        try:
            os.killpg(self._process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _close_log(self) -> None:
        if not self._log_file.closed:
            self._log_file.close()

    def __enter__(self) -> RunningProcess:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        if exception_type is None:
            self.terminate()
        else:
            try:
                self.terminate()
            except Exception:
                pass


def run_process(
    command: Sequence[str],
    log_path: Path,
    description: str,
    environment_overrides: Mapping[str, str] | None = None,
) -> None:
    """Run one command to natural completion without a shell or timeout."""
    with start_process(
        command,
        log_path,
        description,
        environment_overrides,
    ) as process:
        process.wait()


def start_process(
    command: Sequence[str],
    log_path: Path,
    description: str,
    environment_overrides: Mapping[str, str] | None = None,
) -> RunningProcess:
    """Start one command in a dedicated group with one caller-chosen log."""
    log_file: BinaryIO | None = None
    try:
        log_file = log_path.open("wb")
        process = subprocess.Popen(
            _guarded_command(command),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=clean_process_environment(environment_overrides),
            start_new_session=True,
        )
    except ProcessError:
        if log_file is not None:
            log_file.close()
        raise
    except OSError as error:
        if log_file is not None:
            log_file.close()
        raise ProcessError(f"cannot start {description}: {error}") from error
    return RunningProcess(process, log_file, log_path, description)


def clean_process_environment(
    environment_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Remove user Python paths and tokens, then apply safe overrides."""
    overrides = _validated_environment_overrides(
        {} if environment_overrides is None else environment_overrides
    )
    environment = os.environ.copy()
    for name in _REMOVED_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment.update(overrides)
    return environment


def read_log_tail(path: Path, limit: int = 4_000) -> str:
    """Decode a bounded diagnostic tail from a child-process log."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - limit), os.SEEK_SET)
            return stream.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _guarded_command(command: Sequence[str]) -> list[str]:
    return [
        command[0],
        str(_PARENT_GUARD),
        str(os.getpid()),
        str(threading.get_native_id()),
        _PARENT_GUARD_SEPARATOR,
        *command,
    ]


def _validated_environment_overrides(
    value: Mapping[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, content in value.items():
        if name not in _ALLOWED_ENVIRONMENT_OVERRIDES:
            raise ProcessError(
                f"environment override is not allowed: {name!r}"
            )
        if not isinstance(content, str) or "\x00" in content:
            raise ProcessError(
                f"environment override {name!r} must be a string without NUL"
            )
        result[name] = content
    return result


def run_worker_process(command: Sequence[str]) -> int:
    worker: subprocess.Popen[bytes] | None = None
    stop_requested = False

    def forward_sigterm(_signum: int, _frame: object) -> None:
        nonlocal stop_requested

        stop_requested = True
        if worker is not None and worker.poll() is None:
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    previous_handler = signal.signal(signal.SIGTERM, forward_sigterm)
    try:
        worker = subprocess.Popen(tuple(command), start_new_session=True)
        if stop_requested and worker.poll() is None:
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return_code = worker.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    if stop_requested:
        raise StopRequested
    return return_code

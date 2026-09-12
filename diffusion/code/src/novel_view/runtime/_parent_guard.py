"""Bind one Linux model launcher to the lifetime of its controller."""

from __future__ import annotations

import ctypes
import os
import signal
import sys
from collections.abc import Sequence

_PR_SET_PDEATHSIG = 1


def main(arguments: Sequence[str] | None = None) -> int:
    """Install the parent-death signal, then replace this guard in place."""
    values = tuple(sys.argv[1:] if arguments is None else arguments)
    if len(values) < 4 or values[2] != "--":
        raise RuntimeError(
            "parent guard expects PARENT_PID PARENT_TID -- "
            "COMMAND [ARGUMENT ...]"
        )
    expected_parent_pid = int(values[0])
    expected_parent_tid = int(values[1])
    command = values[3:]
    if sys.platform.startswith("linux"):
        _bind_to_parent(expected_parent_pid, expected_parent_tid)
    os.execvpe(command[0], command, os.environ)
    return 0


def _bind_to_parent(expected_parent_pid: int, expected_parent_tid: int) -> None:
    """Ask Linux to send SIGTERM if the creating controller disappears."""
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    result = prctl(
        _PR_SET_PDEATHSIG,
        int(signal.SIGTERM),
        0,
        0,
        0,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    parent_thread = (
        f"/proc/{expected_parent_pid}/task/{expected_parent_tid}"
    )
    if (
        os.getppid() != expected_parent_pid
        or not os.path.exists(parent_thread)
    ):
        raise RuntimeError("model controller exited before guard setup")


if __name__ == "__main__":
    raise SystemExit(main())

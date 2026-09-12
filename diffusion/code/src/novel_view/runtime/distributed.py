"""Launch one local torchrun group through the no-timeout process seam."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from novel_view.runtime.process import RunningProcess, start_process


def start_torchrun(
    python_executable: Path,
    worker: Path,
    arguments: Sequence[str],
    *,
    process_count: int,
    log_path: Path,
    description: str,
    environment_overrides: Mapping[str, str] | None = None,
) -> RunningProcess:
    """Start the pinned single-node torchrun command without a deadline."""
    command = [
        str(python_executable),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={process_count}",
        "--max-restarts=0",
        str(worker),
        *arguments,
    ]
    return start_process(
        command,
        log_path,
        description,
        environment_overrides,
    )

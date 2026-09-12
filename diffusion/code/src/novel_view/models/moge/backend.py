"""Run one standalone raw MoGe-v1 request in the Gen3C environment."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from novel_view.models.moge.protocol import read_moge_execution, write_moge_request
from novel_view.models.moge.request import MogeExecution, MogeRequest
from novel_view.models.moge.spec import MogeError
from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import ProcessError, run_process


@dataclass(frozen=True, slots=True)
class MogeResources:
    """Machine-local paths and process environment for standalone MoGe-v1."""

    scratch_root: Path
    checkpoint_path: Path
    environment_overrides: Mapping[str, str] = field(
        default_factory=dict,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment_overrides",
            MappingProxyType(dict(self.environment_overrides)),
        )


def run_moge(
    request: MogeRequest,
    resources: MogeResources,
    log_path: Path,
) -> MogeExecution:
    """Execute one raw sequence without a computation deadline."""
    try:
        with tempfile.TemporaryDirectory(
            prefix="moge-v1-",
            dir=resources.scratch_root,
        ) as temporary:
            exchange = Path(temporary)
            write_moge_request(exchange, request)
            run_process(
                [
                    str(GEN3C_PYTHON),
                    str(Path(__file__).with_name("_worker.py")),
                    "--exchange",
                    str(exchange),
                    "--checkpoint",
                    str(resources.checkpoint_path),
                ],
                log_path,
                "MoGe-v1 raw inference",
                resources.environment_overrides,
            )
            return read_moge_execution(exchange)
    except ProcessError as error:
        raise MogeError(str(error)) from error


__all__ = ["MogeResources", "run_moge"]

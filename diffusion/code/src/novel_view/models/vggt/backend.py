"""Run one raw VGGT-Omega request in its fixed container environment."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from novel_view.models.vggt.protocol import (
    read_vggt_omega_execution,
    write_vggt_omega_request,
)
from novel_view.models.vggt.request import VggtOmegaExecution, VggtOmegaRequest
from novel_view.models.vggt.spec import VggtOmegaError
from novel_view.runtime.executables import VGGT_PYTHON
from novel_view.runtime.process import ProcessError, run_process


@dataclass(frozen=True, slots=True)
class VggtOmegaResources:
    """Machine-local paths and process environment for VGGT-Omega."""

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


def run_vggt_omega(
    request: VggtOmegaRequest,
    resources: VggtOmegaResources,
    log_path: Path | None = None,
) -> VggtOmegaExecution:
    """Execute one joint sequence without a computation deadline."""
    try:
        with tempfile.TemporaryDirectory(
            prefix="vggt-omega-",
            dir=resources.scratch_root,
        ) as temporary:
            exchange = Path(temporary)
            write_vggt_omega_request(exchange, request)
            run_process(
                [
                    str(VGGT_PYTHON),
                    str(Path(__file__).with_name("_worker.py")),
                    "--exchange",
                    str(exchange),
                    "--checkpoint",
                    str(resources.checkpoint_path),
                    "--mode",
                    request.input_plan.mode.value,
                ],
                exchange / "worker.log" if log_path is None else log_path,
                f"VGGT-Omega {request.input_plan.mode.value}",
                resources.environment_overrides,
            )
            return read_vggt_omega_execution(exchange, request)
    except ProcessError as error:
        raise VggtOmegaError(str(error)) from error


__all__ = ["VggtOmegaResources", "run_vggt_omega"]

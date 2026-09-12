"""Run one composite Grounding DINO and SAM2 request."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from novel_view.models.grounded_sam2.protocol import (
    read_grounded_sam2_prediction,
    write_grounded_sam2_request,
)
from novel_view.models.grounded_sam2.request import (
    GroundedSam2Prediction,
    GroundedSam2Request,
)
from novel_view.models.grounded_sam2.spec import GroundedSam2ModelError
from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import ProcessError, run_process


@dataclass(frozen=True, slots=True)
class GroundedSam2Resources:
    """Machine-local assets and environment for Grounded-SAM2."""

    scratch_root: Path
    grounding_model_directory: Path
    sam2_checkpoint_path: Path
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


def run_grounded_sam2(
    request: GroundedSam2Request,
    resources: GroundedSam2Resources,
    log_path: Path,
) -> GroundedSam2Prediction:
    """Execute the composite worker once without a computation deadline."""
    try:
        with tempfile.TemporaryDirectory(
            prefix="grounded-sam2-",
            dir=resources.scratch_root,
        ) as temporary:
            exchange = Path(temporary)
            write_grounded_sam2_request(exchange, request)
            run_process(
                [
                    str(GEN3C_PYTHON),
                    str(Path(__file__).with_name("_worker.py")),
                    "--exchange",
                    str(exchange),
                    "--grounding-model-directory",
                    str(resources.grounding_model_directory),
                    "--sam2-checkpoint",
                    str(resources.sam2_checkpoint_path),
                    "--sam2-model-config",
                    request.sam2_model_config,
                    "--prompt",
                    request.prompt,
                    "--box-threshold",
                    str(request.box_threshold),
                    "--text-threshold",
                    str(request.text_threshold),
                ],
                log_path,
                "Grounding DINO and SAM2 raw mask",
                resources.environment_overrides,
            )
            return read_grounded_sam2_prediction(exchange)
    except ProcessError as error:
        raise GroundedSam2ModelError(str(error)) from error


__all__ = ["GroundedSam2Resources", "run_grounded_sam2"]

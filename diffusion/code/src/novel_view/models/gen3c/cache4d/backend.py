"""Run one standalone Cache4D diagnostic in the Gen3C environment."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from novel_view.models.gen3c.cache4d.protocol import (
    read_cache4d_diagnostic_result,
    write_cache4d_conditioning,
)
from novel_view.models.gen3c.cache4d.request import Gen3cConditioning
from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import ProcessError, run_process


class Gen3cCache4dError(RuntimeError):
    """A standalone Cache4D request could not complete."""


@dataclass(frozen=True, slots=True)
class Gen3cCache4dResources:
    """Machine-local locations for the standalone Cache4D worker."""

    scratch_root: Path
    upstream_root: Path
    environment_overrides: Mapping[str, str] = field(
        default_factory=dict,
        hash=False,
    )
    unprojection_chunk_size: int = 13

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment_overrides",
            MappingProxyType(dict(self.environment_overrides)),
        )


@dataclass(frozen=True, slots=True, eq=False)
class Gen3cCache4dResult:
    """Owned numeric result of one standalone Cache4D diagnostic."""

    conditioning: Gen3cConditioning
    coverage_fraction: npt.NDArray[np.float64]
    far_depth_pixel_fraction: npt.NDArray[np.float64]
    selected_global_slots: npt.NDArray[np.int64]
    selected_rgb: npt.NDArray[np.uint8]
    selected_coverage_mask: npt.NDArray[np.bool_]
    overlap_rgb_max_abs: float
    overlap_mask_mismatch_fraction: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    elapsed_seconds: float


def render_gen3c_cache4d(
    conditioning: Gen3cConditioning,
    resources: Gen3cCache4dResources,
    log_path: Path,
) -> Gen3cCache4dResult:
    """Run the pinned Cache4D over one caller-built camera schedule."""
    with tempfile.TemporaryDirectory(
        prefix="gen3c-cache4d-",
        dir=resources.scratch_root,
    ) as temporary:
        exchange = Path(temporary)
        selected_slots = write_cache4d_conditioning(exchange, conditioning)
        try:
            _run_worker(exchange, resources, log_path)
        except ProcessError as error:
            raise Gen3cCache4dError(str(error)) from error
        outputs = read_cache4d_diagnostic_result(exchange)
        try:
            overlap = outputs["overlap"]
            peaks = outputs["peaks"]
            elapsed = outputs["elapsed"]
            return Gen3cCache4dResult(
                conditioning=conditioning,
                coverage_fraction=_owned(outputs["coverage"]),
                far_depth_pixel_fraction=_owned(outputs["far_depth"]),
                selected_global_slots=_owned(selected_slots),
                selected_rgb=_owned(outputs["rgb"]),
                selected_coverage_mask=_owned(outputs["mask"]),
                overlap_rgb_max_abs=float(overlap[0]),
                overlap_mask_mismatch_fraction=float(overlap[1]),
                peak_cuda_allocated_bytes=int(peaks[0]),
                peak_cuda_reserved_bytes=int(peaks[1]),
                elapsed_seconds=float(elapsed[0]),
            )
        finally:
            for value in outputs.values():
                if isinstance(value, np.memmap):
                    value._mmap.close()


def _run_worker(
    exchange: Path,
    resources: Gen3cCache4dResources,
    log_path: Path,
) -> None:
    run_process(
        [
            str(GEN3C_PYTHON),
            str(Path(__file__).with_name("_worker.py")),
            "--exchange",
            str(exchange),
            "--upstream-root",
            str(resources.upstream_root),
            "--unprojection-chunk-size",
            str(resources.unprojection_chunk_size),
        ],
        log_path,
        "pinned Gen3C Cache4D worker",
        resources.environment_overrides,
    )


def _owned(value: npt.NDArray[np.generic]) -> npt.NDArray:
    array = np.array(value, copy=True, order="C")
    array.setflags(write=False)
    return array


__all__ = [
    "Gen3cCache4dError",
    "Gen3cCache4dResources",
    "Gen3cCache4dResult",
    "render_gen3c_cache4d",
]

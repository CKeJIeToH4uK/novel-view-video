"""Generation-only additions to the shared Cache4D NPY protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from novel_view.generation.gen3c.request import Gen3cGenerationRequest
from novel_view.models.gen3c.cache4d.protocol import (
    read_cache4d_conditioning,
    write_cache4d_conditioning,
)

WORKER_READY_FILENAME = "worker.ready"
STOP_FILENAME = "stop"
REQUEST_READY_FILENAME = "request.ready"
REQUEST_DONE_FILENAME = "request.done"
OUTPUT_RGB_PATH_FILENAME = "output_rgb_path.txt"
CONTEXT_DEPTH_OUTPUT_SLOTS_FILENAME = "context_depth_output_slots.npy"
CONTEXT_DEPTH_FILENAME = "context_depth_z_m.npy"
CONTEXT_VALID_FILENAME = "context_valid.npy"
CONTEXT_DIAGNOSTICS_FILENAME = "context_depth_diagnostics.npy"


def write_generation_request(root: Path, request: Gen3cGenerationRequest) -> None:
    """Write conditioning plus the direct caller-owned result path."""
    write_cache4d_conditioning(root, request.conditioning)
    (root / OUTPUT_RGB_PATH_FILENAME).write_text(
        str(request.output_rgb_path),
        encoding="utf-8",
    )
    if request.context_depth_output_slots is not None:
        np.save(
            root / CONTEXT_DEPTH_OUTPUT_SLOTS_FILENAME,
            request.context_depth_output_slots,
            allow_pickle=False,
        )


def read_generation_request(
    root: Path,
) -> tuple[dict[str, np.ndarray], Path]:
    """Open one numeric request and its direct RGB destination."""
    values = read_cache4d_conditioning(root)
    output = Path(
        (root / OUTPUT_RGB_PATH_FILENAME).read_text(encoding="utf-8")
    )
    return values, output


def request_root(session_root: Path, request_index: int) -> Path:
    """Return the ordered exchange directory for one resident request."""
    return session_root / f"request-{request_index:06d}"


def context_output_paths(output_rgb_path: Path) -> tuple[Path, Path]:
    """Return the two caller-owned context result paths."""
    return (
        output_rgb_path.parent / CONTEXT_DEPTH_FILENAME,
        output_rgb_path.parent / CONTEXT_VALID_FILENAME,
    )


def open_generated_rgb(
    path: Path,
    shape: tuple[int, int, int, int],
) -> np.memmap:
    """Open the caller-owned lossless RGB result for direct writing."""
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.uint8,
        shape=shape,
    )


def write_context_result(
    request: Path,
    output_rgb_path: Path,
    depth_z_m: np.ndarray,
    valid: np.ndarray,
    diagnostics: np.ndarray,
) -> None:
    """Write the generation-owned context outputs and request telemetry."""
    depth_path, valid_path = context_output_paths(output_rgb_path)
    np.save(depth_path, depth_z_m, allow_pickle=False)
    np.save(valid_path, valid, allow_pickle=False)
    np.save(
        request / CONTEXT_DIAGNOSTICS_FILENAME,
        diagnostics,
        allow_pickle=False,
    )


def write_rank_telemetry(
    request: Path,
    rank: int,
    cuda_bytes: np.ndarray,
    elapsed_seconds: float,
) -> None:
    """Write the two small telemetry rows for one completed rank."""
    np.save(
        request / f"cuda_bytes_{rank}.npy",
        cuda_bytes,
        allow_pickle=False,
    )
    np.save(
        request / f"elapsed_seconds_{rank}.npy",
        np.asarray((elapsed_seconds,), dtype=np.float64),
        allow_pickle=False,
    )


def write_startup_peak(
    session_root: Path,
    rank: int,
    cuda_bytes: np.ndarray,
) -> None:
    """Write one model-construction CUDA peak row."""
    np.save(
        session_root / f"startup_peak_cuda_bytes_{rank}.npy",
        cuda_bytes,
        allow_pickle=False,
    )


__all__ = [
    "CONTEXT_DEPTH_FILENAME",
    "CONTEXT_DEPTH_OUTPUT_SLOTS_FILENAME",
    "CONTEXT_DIAGNOSTICS_FILENAME",
    "CONTEXT_VALID_FILENAME",
    "OUTPUT_RGB_PATH_FILENAME",
    "REQUEST_DONE_FILENAME",
    "REQUEST_READY_FILENAME",
    "STOP_FILENAME",
    "WORKER_READY_FILENAME",
    "context_output_paths",
    "open_generated_rgb",
    "read_generation_request",
    "request_root",
    "write_context_result",
    "write_generation_request",
    "write_rank_telemetry",
    "write_startup_peak",
]

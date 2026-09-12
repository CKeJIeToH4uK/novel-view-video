"""One isolated native Gen3C forward warp for Waymo DDW preparation."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import run_process


RgbLayout = Literal["uint8_thwc", "normalized_tchw"]


@dataclass(frozen=True, slots=True, eq=False)
class WarpRequest:
    """One source sequence and one target camera pack for a native warp."""

    source_rgb: npt.NDArray[np.uint8] | npt.NDArray[np.float32]
    source_rgb_layout: RgbLayout
    source_depth_z_m: npt.NDArray[np.float32]
    source_depth_valid: npt.NDArray[np.bool_]
    source_rectification_known: npt.NDArray[np.bool_]
    source_K_canvas: npt.NDArray[np.float64]
    source_world_to_camera_cv: npt.NDArray[np.float64]
    target_K_canvas: npt.NDArray[np.float64]
    target_world_to_camera_cv: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class WarpResult:
    """Native frame-major output and process telemetry for one source."""

    rgb_minus_one_to_one: npt.NDArray[np.float32]
    depth_z_m: npt.NDArray[np.float32]
    known: npt.NDArray[np.bool_]
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class WarpResources:
    """Machine-local scratch and the installed Gen3C process environment."""

    scratch_root: Path
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


def run_forward_warp(
    request: WarpRequest,
    resources: WarpResources,
    log_path: Path,
) -> WarpResult:
    """Run one native forward warp without retry or filesystem preflight."""
    with tempfile.TemporaryDirectory(
        prefix="ddw-forward-warp-",
        dir=resources.scratch_root,
    ) as temporary:
        exchange = Path(temporary)
        _write_request(exchange, request)
        run_process(
            [
                str(GEN3C_PYTHON),
                str(Path(__file__).with_name("_warp_worker.py")),
                "--exchange",
                str(exchange),
            ],
            log_path,
            "Gen3C Waymo DDW forward warp",
            resources.environment_overrides,
        )
        return _read_result(exchange)


def _write_request(root: Path, request: WarpRequest) -> None:
    """Write one source, normalizing uint8 RGB one frame at a time."""
    frame_count, height, width = request.source_depth_z_m.shape
    rgb = np.lib.format.open_memmap(
        root / "source_rgb.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, 1, 3, height, width),
    )
    try:
        for index, frame in enumerate(request.source_rgb):
            if request.source_rgb_layout == "uint8_thwc":
                normalized = np.asarray(frame, dtype=np.float32)
                normalized = normalized * (2.0 / 255.0) - 1.0
                rgb[index, 0] = np.moveaxis(normalized, -1, 0)
            else:
                rgb[index, 0] = frame
        rgb.flush()
    finally:
        rgb._mmap.close()

    for filename, value in (
        ("source_depth_z_m.npy", request.source_depth_z_m[:, None]),
        ("source_depth_valid.npy", request.source_depth_valid[:, None]),
        (
            "source_rectification_known.npy",
            request.source_rectification_known[None],
        ),
        ("source_K_canvas.npy", request.source_K_canvas[:, None]),
        (
            "source_world_to_camera_cv.npy",
            request.source_world_to_camera_cv[:, None],
        ),
        ("target_K_canvas.npy", request.target_K_canvas),
        ("target_world_to_camera_cv.npy", request.target_world_to_camera_cv),
    ):
        np.save(root / filename, value, allow_pickle=False)


def _read_result(root: Path) -> WarpResult:
    """Read the concrete worker payload; its arrays are the consumed output."""
    values = {
        name: np.load(root / filename, mmap_mode="r", allow_pickle=False)
        for name, filename in (
            ("rgb", "warped_rgb.npy"),
            ("depth", "warped_depth_z_m.npy"),
            ("known", "warped_known.npy"),
            ("peaks", "peak_cuda_bytes.npy"),
            ("elapsed", "elapsed_seconds.npy"),
        )
    }
    try:
        peaks = values["peaks"]
        elapsed = values["elapsed"]
        return WarpResult(
            rgb_minus_one_to_one=cast(npt.NDArray[np.float32], _owned(values["rgb"])),
            depth_z_m=cast(npt.NDArray[np.float32], _owned(values["depth"])),
            known=cast(npt.NDArray[np.bool_], _owned(values["known"])),
            peak_cuda_allocated_bytes=int(peaks[0]),
            peak_cuda_reserved_bytes=int(peaks[1]),
            elapsed_seconds=float(elapsed[0]),
        )
    finally:
        for value in values.values():
            if isinstance(value, np.memmap):
                cast(Any, value)._mmap.close()


def _owned(value: npt.NDArray[np.generic]) -> npt.NDArray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


__all__ = [
    "RgbLayout",
    "WarpRequest",
    "WarpResources",
    "WarpResult",
    "run_forward_warp",
]

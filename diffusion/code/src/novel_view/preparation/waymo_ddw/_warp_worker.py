"""Private CUDA worker for the pinned Gen3C Waymo forward warp."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any, cast

import numpy as np


_CHUNK_SIZE = 2
_MODULE = "cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", type=Path, required=True)
    return parser.parse_args()


def _official_module() -> Any:
    """Import the image-pinned utility from the installed Gen3C package."""
    return importlib.import_module(_MODULE)


def _load_inputs(root: Path) -> dict[str, np.ndarray]:
    return {
        name: np.load(root / filename, mmap_mode="r", allow_pickle=False)
        for name, filename in (
            ("rgb", "source_rgb.npy"),
            ("depth", "source_depth_z_m.npy"),
            ("valid", "source_depth_valid.npy"),
            ("rect", "source_rectification_known.npy"),
            ("source_k", "source_K_canvas.npy"),
            ("source_w2c", "source_world_to_camera_cv.npy"),
            ("target_k", "target_K_canvas.npy"),
            ("target_w2c", "target_world_to_camera_cv.npy"),
        )
    }


def _tensor(torch: Any, value: np.ndarray, device: Any) -> Any:
    return torch.from_numpy(np.array(value, copy=True, order="C")).to(device)


def _render_chunk(
    module: Any,
    torch: Any,
    rgb: Any,
    depth: Any,
    original_valid: Any,
    rectification_known: Any,
    source_k: Any,
    source_w2c: Any,
    target_k: Any,
    target_w2c: Any,
) -> tuple[Any, Any, Any]:
    """Apply the literal Gen3C sanitization, filtering and splat."""
    safe_depth = torch.nan_to_num(depth, nan=100.0).clamp_(0.0, 100.0)
    reliable = module.reliable_depth_mask_range_batch(
        safe_depth,
        window_size=5,
        ratio_thresh=0.05,
    )
    valid = original_valid & rectification_known & reliable
    points = module.unproject_points(
        safe_depth,
        source_w2c,
        source_k,
        is_depth=True,
        mask=valid,
    )
    warped_rgb, mask, warped_depth, _ = module.forward_warp(
        frame1=rgb,
        mask1=valid,
        depth1=None,
        transformation1=None,
        transformation2=target_w2c,
        intrinsic1=target_k,
        intrinsic2=target_k,
        world_points1=points,
        render_depth=True,
        foreground_masking=False,
        boundary_mask=None,
    )
    known = mask[:, 0] > 0
    warped_rgb = warped_rgb.masked_fill(~known[:, None], -1.0)
    warped_depth = warped_depth.masked_fill(~known, 0.0)
    return warped_rgb, warped_depth, known


def _execute_forward_warp(
    module: Any,
    torch: Any,
    values: dict[str, np.ndarray],
    root: Path,
    device: Any,
) -> None:
    """Render the single source in bounded chronological chunks."""
    frame_count, _, _, height, width = values["rgb"].shape
    use_cuda = str(device).startswith("cuda")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rgb_output = np.lib.format.open_memmap(
        root / "warped_rgb.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, 1, 3, height, width),
    )
    depth_output = np.lib.format.open_memmap(
        root / "warped_depth_z_m.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, 1, height, width),
    )
    known_output = np.lib.format.open_memmap(
        root / "warped_known.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(frame_count, 1, height, width),
    )
    outputs = (rgb_output, depth_output, known_output)
    try:
        with torch.inference_mode():
            for start in range(0, frame_count, _CHUNK_SIZE):
                stop = min(start + _CHUNK_SIZE, frame_count)
                index = slice(start, stop)
                rows = np.arange(start, stop)
                rgb, depth, known = _render_chunk(
                    module,
                    torch,
                    _tensor(torch, values["rgb"][index, 0], device),
                    _tensor(torch, values["depth"][index, 0, None], device),
                    _tensor(torch, values["valid"][index, 0, None], device),
                    _tensor(
                        torch,
                        values["rect"][np.zeros_like(rows), None],
                        device,
                    ),
                    _tensor(torch, values["source_k"][index, 0], device).float(),
                    _tensor(torch, values["source_w2c"][index, 0], device).float(),
                    _tensor(torch, values["target_k"][index], device).float(),
                    _tensor(torch, values["target_w2c"][index], device).float(),
                )
                rgb_output[index, 0] = rgb.cpu().numpy()
                depth_output[index, 0] = depth.cpu().numpy()
                known_output[index, 0] = known.cpu().numpy()
        if use_cuda:
            torch.cuda.synchronize()
        peaks = (
            (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
            if use_cuda
            else (0, 0)
        )
        np.save(
            root / "peak_cuda_bytes.npy",
            np.asarray(peaks, dtype=np.int64),
            allow_pickle=False,
        )
        np.save(
            root / "elapsed_seconds.npy",
            np.asarray((time.monotonic() - started,), dtype=np.float64),
            allow_pickle=False,
        )
        for output in outputs:
            output.flush()
    finally:
        for output in outputs:
            output._mmap.close()


def _run(arguments: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Waymo forward-warp worker requires CUDA")
    torch.cuda.set_device(0)
    values = _load_inputs(arguments.exchange)
    try:
        _execute_forward_warp(
            _official_module(),
            torch,
            values,
            arguments.exchange,
            torch.device("cuda:0"),
        )
    finally:
        for value in values.values():
            if isinstance(value, np.memmap):
                cast(Any, value)._mmap.close()


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()

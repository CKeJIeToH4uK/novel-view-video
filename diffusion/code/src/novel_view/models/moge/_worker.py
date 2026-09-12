"""Private standalone MoGe-v1 CUDA worker."""

from __future__ import annotations

import argparse
import resource
from pathlib import Path

import numpy as np

from novel_view.models.moge._inference import infer_moge_frame, load_moge_model


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def _run(arguments: argparse.Namespace) -> None:
    """Load once, process ordered frames sequentially, and write raw arrays."""
    import torch

    rgb = np.load(arguments.exchange / "rgb.npy", mmap_mode="r", allow_pickle=False)
    fov = np.load(
        arguments.exchange / "fov_x_degrees.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    frame_count, height, width, _ = rgb.shape
    model = load_moge_model(arguments.checkpoint, "cuda")
    torch.cuda.reset_peak_memory_stats()
    depth = np.lib.format.open_memmap(
        arguments.exchange / "depth_model_units.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, height, width),
    )
    valid = np.lib.format.open_memmap(
        arguments.exchange / "model_valid.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(frame_count, height, width),
    )
    intrinsics = np.lib.format.open_memmap(
        arguments.exchange / "model_intrinsics.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, 3, 3),
    )
    for index in range(frame_count):
        image = (
            torch.from_numpy(np.array(rgb[index], copy=True))
            .permute(2, 0, 1)
            .to(device="cuda", dtype=torch.float32)
            .div_(255.0)
        )
        output = infer_moge_frame(model, image, float(fov[index]))
        depth[index] = output["depth"].detach().cpu().numpy()
        valid[index] = output["mask"].detach().cpu().numpy()
        intrinsics[index] = output["intrinsics"].detach().cpu().numpy()
    torch.cuda.synchronize()
    for output in (depth, valid, intrinsics):
        output.flush()
    peak_vram_mib = np.asarray(
        torch.cuda.max_memory_reserved() / (1024.0**2),
        dtype=np.float64,
    )
    peak_ram_mib = np.asarray(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        dtype=np.float64,
    )
    np.save(arguments.exchange / "peak_vram_mib.npy", peak_vram_mib, allow_pickle=False)
    np.save(arguments.exchange / "peak_ram_mib.npy", peak_ram_mib, allow_pickle=False)
    print(
        f"processed={frame_count} peak_ram_mib={float(peak_ram_mib):.3f} "
        f"peak_vram_mib={float(peak_vram_mib):.3f}",
        flush=True,
    )


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()

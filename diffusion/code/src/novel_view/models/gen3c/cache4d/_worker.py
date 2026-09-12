"""One-shot executable for the standalone Cache4D diagnostic."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from novel_view.models.gen3c.cache4d.protocol import (
    read_cache4d_conditioning,
    write_cache4d_diagnostic_result,
)
from novel_view.models.gen3c.cache4d.runtime import (
    create_cache4d,
    dense_cache_inputs,
    import_cache4d,
    render_cache4d,
)


_WINDOW_SIZE = 121
_WINDOW_STEP = 120


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--unprojection-chunk-size", type=int, required=True)
    return parser.parse_args()


def _render_windows(
    cache: Any,
    torch: Any,
    values: dict[str, np.ndarray],
    height: int,
    width: int,
    device: Any,
) -> tuple[np.ndarray, ...]:
    """Render caller-selected windows and retain bounded diagnostics."""
    frame_count = values["source_index"].size
    slots = values["selected_slots"]
    coverage = np.empty(frame_count, dtype=np.float64)
    previews = np.empty((slots.size, height, width, 3), dtype=np.uint8)
    preview_masks = np.empty((slots.size, height, width), dtype=np.bool_)
    filled = np.zeros(slots.size, dtype=np.bool_)
    overlap = np.zeros(2, dtype=np.float64)
    previous_rgb = previous_mask = None

    for start in range(0, frame_count - 1, _WINDOW_STEP):
        stop = start + _WINDOW_SIZE
        rgb, mask = render_cache4d(
            cache,
            torch,
            values["query_w2c"][start:stop],
            values["query_intrinsics"][start:stop],
            start,
            device,
        )
        local_coverage = (
            mask[0, :, :, 0].amax(1).mean((1, 2)).double().cpu().numpy()
        )
        offset = 0
        if previous_rgb is not None and previous_mask is not None:
            overlap[0] = max(
                overlap[0],
                float(torch.max(torch.abs(previous_rgb - rgb[0, 0])).item()),
            )
            overlap[1] = max(
                overlap[1],
                float((previous_mask != mask[0, 0]).float().mean().item()),
            )
            offset = 1
        coverage[start + offset : stop] = local_coverage[offset:]
        for preview_index, global_slot in enumerate(slots):
            if not filled[preview_index] and start <= global_slot < stop:
                local = int(global_slot - start)
                pixels = rgb[0, local, 0].permute(1, 2, 0)
                previews[preview_index] = (
                    torch.clamp(
                        torch.round((pixels + 1.0) * 127.5),
                        0,
                        255,
                    )
                    .byte()
                    .cpu()
                    .numpy()
                )
                preview_masks[preview_index] = (
                    mask[0, local, 0, 0] > 0.5
                ).cpu().numpy()
                filled[preview_index] = True
        previous_rgb = rgb[0, -1].detach().clone()
        previous_mask = mask[0, -1].detach().clone()
        del rgb, mask
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if not np.all(filled):
        raise RuntimeError(f"selected slots were not rendered: {slots[~filled]}")
    return coverage, previews, preview_masks, overlap


def _execute_cache4d(
    module: Any,
    torch: Any,
    values: dict[str, np.ndarray],
    chunk_size: int,
    device: Any,
) -> dict[str, np.ndarray]:
    """Create and render one complete standalone Cache4D request."""
    frame_count = values["source_index"].size
    _, height, width, _ = values["source_rgb"].shape
    use_cuda = str(device).startswith("cuda")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    dense = dense_cache_inputs(values, chunk_size)
    cache = create_cache4d(module, torch, dense, chunk_size, device)
    far_depth = dense.far_depth_pixel_fraction
    del dense
    if cache.input_frame_count() != frame_count:
        raise RuntimeError("Cache4D input frame count changed")
    coverage, rgb, mask, overlap = _render_windows(
        cache,
        torch,
        values,
        height,
        width,
        device,
    )
    if use_cuda:
        torch.cuda.synchronize()
        peaks = np.asarray(
            [
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
            ],
            dtype=np.int64,
        )
    else:
        peaks = np.zeros(2, dtype=np.int64)
    return {
        "coverage": coverage,
        "far_depth": far_depth,
        "rgb": rgb,
        "mask": mask,
        "overlap": overlap,
        "peaks": peaks,
        "elapsed": np.asarray(
            [time.monotonic() - started],
            dtype=np.float64,
        ),
    }


def _run(arguments: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Cache4D worker requires CUDA")
    torch.cuda.set_device(0)
    values = read_cache4d_conditioning(arguments.exchange)
    try:
        outputs = _execute_cache4d(
            import_cache4d(arguments.upstream_root),
            torch,
            values,
            arguments.unprojection_chunk_size,
            torch.device("cuda:0"),
        )
        write_cache4d_diagnostic_result(arguments.exchange, outputs)
    finally:
        for value in values.values():
            if isinstance(value, np.memmap):
                value._mmap.close()


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()

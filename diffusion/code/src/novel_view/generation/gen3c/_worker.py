"""Composition worker for resident Gen3C, Cache4D, and context depth."""

from __future__ import annotations

import argparse
import gc
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from novel_view.generation.gen3c.protocol import (
    CONTEXT_DEPTH_OUTPUT_SLOTS_FILENAME as _CONTEXT_DEPTH_OUTPUT_SLOTS,
    REQUEST_DONE_FILENAME as _REQUEST_DONE,
    REQUEST_READY_FILENAME as _REQUEST_READY,
    STOP_FILENAME as _STOP,
    WORKER_READY_FILENAME as _WORKER_READY,
    open_generated_rgb,
    read_generation_request,
    request_root,
    write_context_result,
    write_rank_telemetry,
    write_startup_peak,
)
from novel_view.models.gen3c.cache4d.runtime import (
    create_cache4d,
    dense_cache_inputs,
    import_cache4d,
    render_cache4d,
)
from novel_view.models.gen3c.lora_weights import Gen3cLoraWeights
from novel_view.models.gen3c.request import (
    Gen3cModelSpec,
    Gen3cSampling,
    Gen3cTopology,
)
from novel_view.models.gen3c.session import Gen3cModelSession
from novel_view.models.gen3c.spec import GEN3C_IMAGE_SIZE_HW

if __package__:
    from .windows import (
        GEN3C_WINDOW_SIZE as _WINDOW_SIZE,
        WINDOW_SEED_AUTOREGRESSIVE as _AUTOREGRESSIVE,
        WINDOW_SEED_SOURCE_RESEED as _SOURCE_RESEED,
        Gen3cWindowCall,
        iter_window_calls,
        materialize_window_cameras,
    )
else:
    from windows import (  # type: ignore[no-redef]
        GEN3C_WINDOW_SIZE as _WINDOW_SIZE,
        WINDOW_SEED_AUTOREGRESSIVE as _AUTOREGRESSIVE,
        WINDOW_SEED_SOURCE_RESEED as _SOURCE_RESEED,
        Gen3cWindowCall,
        iter_window_calls,
        materialize_window_cameras,
    )

_HEIGHT, _WIDTH = GEN3C_IMAGE_SIZE_HW
_COMMAND_REQUEST = 1
_COMMAND_STOP = 2
_POLL_INTERVAL_SECONDS = 0.1


def _arguments() -> argparse.Namespace:
    """Parse the private numeric process protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument(
        "--shared-checkpoint-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--network-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--lora-strength", type=float)
    parser.add_argument("--context-depth-model", type=Path)
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--unprojection-chunk-size",
        type=int,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument(
        "--window-seed-policy",
        choices=(_AUTOREGRESSIVE, _SOURCE_RESEED),
        required=True,
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    return parser.parse_args()


def _context_depth_modules() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Import pinned MoGe plus official alignment/filter/warp functions."""
    from cosmos_predict1.diffusion.inference.camera_utils import align_depth
    from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
        forward_warp,
        reliable_depth_mask_range_batch,
    )
    from novel_view.models.moge.gen3c import (
        load_gen3c_moge_predictor,
        load_moge_model,
        predict_gen3c_context_depth,
    )
    from novel_view.generation.gen3c.context_depth import prepare_context_depth

    return (
        load_moge_model,
        load_gen3c_moge_predictor,
        predict_gen3c_context_depth,
        align_depth,
        reliable_depth_mask_range_batch,
        forward_warp,
        prepare_context_depth,
    )


def _render_window(
    cache: Any,
    torch: Any,
    values: Mapping[str, np.ndarray],
    call: Gen3cWindowCall,
    device: Any,
) -> tuple[Any, Any]:
    """Render one canonical global camera window from the shared cache."""
    query_w2c, query_intrinsics = materialize_window_cameras(
        values["query_w2c"],
        values["query_intrinsics"],
        values["source_w2c"],
        values["source_intrinsics"],
        call,
    )
    return render_cache4d(
        cache,
        torch,
        query_w2c,
        query_intrinsics,
        call.start,
        device,
    )


def _generate_windows(
    cache: Any,
    model: Gen3cModelSession,
    values: Mapping[str, np.ndarray],
    output: np.ndarray | None,
    window_seed_policy: str = _AUTOREGRESSIVE,
) -> None:
    """Generate one sequence and drop every repeated local seed row."""
    previous_last_frame: np.ndarray | None = None
    for call in iter_window_calls(
        values["source_index"],
        window_seed_policy,
    ):
        rendered_rgb, rendered_mask = _render_window(
            cache,
            model.torch,
            values,
            call,
            model.device,
        )
        if call.seed_kind == "source":
            current_seed = model.encode_seed(
                values["source_rgb"][call.source_index]
            )
        else:
            if previous_last_frame is None:
                raise RuntimeError("autoregressive window seed is unavailable")
            current_seed = model.encode_seed(previous_last_frame)
        video = model.generate_window(
            rendered_rgb,
            rendered_mask,
            current_seed,
        )
        if output is not None:
            offset = call.generated_slice_start
            output[call.start + offset : call.stop] = video[offset:]
        if window_seed_policy == _AUTOREGRESSIVE:
            previous_last_frame = np.array(
                video[-1], copy=True, order="C"
            )
        del rendered_rgb, rendered_mask, current_seed, video


def _prepare_context_depth_output(
    request: Path,
    cache: Any,
    moge_model: Any,
    context_modules: tuple[Any, Any, Any, Any, Any, Any],
    values: Mapping[str, np.ndarray],
    generated_path: Path,
    torch: Any,
    device: Any,
) -> None:
    """Estimate, align, and filter up to twenty trailing context views."""
    slots = np.load(
        request / _CONTEXT_DEPTH_OUTPUT_SLOTS,
        allow_pickle=False,
    )
    generated = np.load(generated_path, mmap_mode="r", allow_pickle=False)
    try:
        query_w2c = torch.from_numpy(
            np.asarray(values["query_w2c"][slots], dtype=np.float32)
        ).unsqueeze(0).to(device)
        query_k = torch.from_numpy(
            np.asarray(values["query_intrinsics"][slots], dtype=np.float32)
        ).unsqueeze(0).to(device)
        with torch.inference_mode():
            reference_depth, reference_mask = cache.render_cache(
                query_w2c,
                query_k,
                render_depth=True,
                start_frame_idx=int(slots[0]),
            )
        reference_depth = reference_depth[0, :, 0].cpu().numpy()
        reference_mask = (reference_mask[0, :, 0, 0] > 0.5).cpu().numpy()
        context_rgb = np.asarray(generated[slots])
        moge_depth = np.empty(reference_depth.shape, dtype=np.float32)
        moge_valid = np.empty(reference_mask.shape, dtype=np.bool_)
        for index, rgb in enumerate(context_rgb):
            with torch.inference_mode():
                prediction = context_modules[1](
                    context_modules[0],
                    rgb,
                    reference_depth.shape[1],
                    reference_depth.shape[2],
                    device,
                    moge_model,
                )
            moge_depth[index] = prediction.depth_model_units
            moge_valid[index] = prediction.valid
        result = context_modules[5](
            moge_depth,
            moge_valid,
            reference_depth.astype(np.float32, copy=False),
            reference_mask,
            np.asarray(values["query_w2c"][slots], dtype=np.float64),
            np.asarray(values["query_intrinsics"][slots], dtype=np.float64),
            torch=torch,
            align_depth=context_modules[2],
            reliable_depth_mask_range_batch=context_modules[3],
            forward_warp=context_modules[4],
        )
        write_context_result(
            request,
            generated_path,
            result.depth_z_m,
            result.valid,
            np.stack(
                (
                    result.alignment_scale,
                    result.alignment_bias,
                    result.alignment_mae_m,
                    result.valid.mean(axis=(1, 2)),
                ),
                axis=1,
            ),
        )
    finally:
        if isinstance(generated, np.memmap):
            generated._mmap.close()


def _next_command(
    session_root: Path,
    request_index: int,
    rank: int,
    torch: Any,
    device: Any,
) -> int:
    """Let rank zero observe files and broadcast one numeric command."""
    command = torch.zeros(2, dtype=torch.int64, device=device)
    if rank == 0:
        request = request_root(session_root, request_index)
        while True:
            if (session_root / _STOP).is_file():
                command[0] = _COMMAND_STOP
                command[1] = request_index
                break
            if (request / _REQUEST_READY).is_file():
                command[0] = _COMMAND_REQUEST
                command[1] = request_index
                break
            time.sleep(_POLL_INTERVAL_SECONDS)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(command, src=0)
    opcode = int(command[0].item())
    received_index = int(command[1].item())
    if received_index != request_index:
        raise RuntimeError("Gen3C request index diverged between ranks")
    if opcode not in (_COMMAND_REQUEST, _COMMAND_STOP):
        raise RuntimeError(f"unknown Gen3C session command: {opcode}")
    return opcode


def _close_memmap(value: np.ndarray | None) -> None:
    """Close one NumPy memory map when it owns a mapped file."""
    if isinstance(value, np.memmap):
        value._mmap.close()


def _run_request(
    request: Path,
    arguments: argparse.Namespace,
    cache_module: Any,
    model: Gen3cModelSession,
    moge_model: Any = None,
    context_modules: tuple[Any, Any, Any, Any, Any] | None = None,
) -> None:
    """Generate one request and release every request-local allocation."""
    torch = model.torch
    device = model.device
    rank = model.rank
    values: dict[str, np.ndarray] = {}
    output_path: Path | None = None
    output: np.memmap | None = None
    dense: Any = None
    cache: Any = None
    started = time.monotonic()
    before = model.cuda_bytes()
    model.reset_peak_cuda_bytes()
    try:
        values, output_path = read_generation_request(request)
        frame_count = values["source_index"].size
        _, height, width, _ = values["source_rgb"].shape
        if (height, width) != (_HEIGHT, _WIDTH):
            raise RuntimeError(
                f"Gen3C source raster must be {(_HEIGHT, _WIDTH)}"
            )
        dense = dense_cache_inputs(
            values,
            arguments.unprojection_chunk_size,
        )
        cache = create_cache4d(
            cache_module,
            torch,
            dense,
            arguments.unprojection_chunk_size,
            device,
        )
        dense = None
        if cache.input_frame_count() != frame_count:
            raise RuntimeError("Cache4D input frame count changed")
        if rank == 0:
            output = open_generated_rgb(
                output_path,
                (frame_count, height, width, 3),
            )
        _generate_windows(
            cache,
            model,
            values,
            output,
            arguments.window_seed_policy,
        )
        if output is not None:
            output.flush()
        if rank == 0 and (request / _CONTEXT_DEPTH_OUTPUT_SLOTS).is_file():
            if moge_model is None or context_modules is None:
                raise RuntimeError("context depth requested without resident MoGe")
            _prepare_context_depth_output(
                request,
                cache,
                moge_model,
                context_modules,
                values,
                output_path,
                torch,
                device,
            )
        torch.cuda.synchronize(device)
        peak = model.peak_cuda_bytes()
    finally:
        _close_memmap(output)
        output = None
        cache = None
        dense = None
        for value in values.values():
            _close_memmap(value)
        values.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    after = model.cuda_bytes()
    write_rank_telemetry(
        request,
        rank,
        np.concatenate((before, peak, after)),
        time.monotonic() - started,
    )
    model.barrier()
    if rank == 0:
        (request / _REQUEST_DONE).touch(exist_ok=False)


def _run(arguments: argparse.Namespace) -> None:
    """Build one Gen3C pipeline and serve sequential numeric requests."""
    cache_module = import_cache4d(arguments.upstream_root)
    lora = (
        None
        if arguments.lora_checkpoint is None
        else Gen3cLoraWeights(
            arguments.lora_checkpoint,
            arguments.lora_strength,
        )
    )
    spec = Gen3cModelSpec(
        arguments.shared_checkpoint_root,
        arguments.network_checkpoint,
        lora,
    )
    sampling = Gen3cSampling(
        prompt=arguments.prompt,
        negative_prompt=arguments.negative_prompt,
        guidance=arguments.guidance,
        steps=arguments.num_steps,
        seed=arguments.seed,
    )

    model: Gen3cModelSession | None = None
    moge_model: Any = None
    try:
        model = Gen3cModelSession.open(
            spec,
            Gen3cTopology(arguments.context_parallel_size),
            sampling,
        )
        torch = model.torch
        rank = model.rank
        device = model.device
        context_modules = None
        if arguments.context_depth_model is not None and rank == 0:
            load_moge_model, load_predictor, *context_functions = (
                _context_depth_modules()
            )
            moge_model = load_moge_model(arguments.context_depth_model, device)
            context_modules = (load_predictor(), *context_functions)
        torch.cuda.synchronize(device)
        write_startup_peak(
            arguments.session_root,
            model.rank,
            model.peak_cuda_bytes(),
        )
        model.barrier()
        if rank == 0:
            (arguments.session_root / _WORKER_READY).touch(exist_ok=False)

        request_index = 0
        while True:
            opcode = _next_command(
                arguments.session_root,
                request_index,
                rank,
                torch,
                device,
            )
            if opcode == _COMMAND_STOP:
                break
            model.restore_request_rng()
            _run_request(
                request_root(arguments.session_root, request_index),
                arguments,
                cache_module,
                model,
                moge_model,
                context_modules,
            )
            request_index += 1
        model.barrier()
    finally:
        moge_model = None
        if model is not None:
            model.close()


def main() -> None:
    """Serve sequential Gen3C requests in one isolated worker group."""
    _run(_arguments())


if __name__ == "__main__":
    main()

"""Явная CP4/A100/Apex/NCCL-диагностика перед Gen3C training."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
import subprocess
from typing import Any, cast

from novel_view.runtime.presets import ExecutionPreset, TRAINING_CP4


_PROFILE_NAME = "a100-cp4"
_GPU_NAME_FRAGMENT = "A100"
_COMPUTE_CAPABILITY = "8.0"
_MINIMUM_TOTAL_MEMORY_BYTES = 75 * 1024**3
_MINIMUM_FREE_MEMORY_FRACTION = 0.90


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    return parser.parse_args()


def _resolve_profile(name: str) -> ExecutionPreset:
    if name != _PROFILE_NAME:
        raise RuntimeError(f"unsupported Gen3C diagnostic profile: {name}")
    return TRAINING_CP4


def _visible_devices(process_count: int) -> tuple[str, ...]:
    devices = tuple(
        part.strip()
        for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if part.strip()
    )
    if len(devices) != process_count or len(set(devices)) != process_count:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must contain {process_count} distinct devices"
        )
    return devices


def _require_mig_disabled(device: str) -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=mig.mode.current",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"timed out while reading MIG mode for GPU {device}"
        ) from error
    except OSError as error:
        raise RuntimeError(f"cannot execute nvidia-smi: {error}") from error
    modes = tuple(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )
    if completed.returncode or len(modes) != 1:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"cannot read MIG mode for GPU {device}: {detail}")
    if modes[0].casefold() != "disabled":
        raise RuntimeError(f"GPU {device} must have MIG disabled, got {modes[0]}")
    return modes[0]


def _distributed_environment(profile: ExecutionPreset) -> tuple[int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("preflight must be launched by torchrun") from error
    expected = profile.gpu_count
    if (world_size, local_world_size) != (expected, expected):
        raise RuntimeError(
            f"{_PROFILE_NAME} requires WORLD_SIZE=LOCAL_WORLD_SIZE={expected}"
        )
    if rank != local_rank or not 0 <= rank < expected:
        raise RuntimeError(f"{_PROFILE_NAME} requires one local rank per GPU")
    return rank, local_rank


def _training_extensions() -> tuple[str, str]:
    try:
        import amp_C
        from apex.optimizers import FusedAdam
    except ImportError as error:
        raise RuntimeError("preflight requires Apex FusedAdam and amp_C") from error
    return (
        f"{FusedAdam.__module__}.{FusedAdam.__name__}",
        str(amp_C.__name__),
    )


def _training_runtime() -> tuple[Any, str, str]:
    import torch

    fused_adam, amp_extension = _training_extensions()
    return torch, fused_adam, amp_extension


def _run_collectives(
    torch: Any,
    process_count: int,
    rank: int,
) -> None:
    reduced = torch.tensor(float(rank + 1), device="cuda", dtype=torch.bfloat16)
    torch.distributed.all_reduce(reduced)
    expected_sum = process_count * (process_count + 1) / 2
    if float(reduced) != expected_sum:
        raise RuntimeError("NCCL all_reduce returned the wrong value")

    gathered = [torch.empty_like(reduced) for _ in range(process_count)]
    local = torch.tensor(float(rank), device="cuda", dtype=torch.bfloat16)
    torch.distributed.all_gather(gathered, local)
    if [float(value) for value in gathered] != list(range(process_count)):
        raise RuntimeError("NCCL all_gather returned the wrong rank order")

    matrix = torch.eye(32, device="cuda", dtype=torch.bfloat16)
    if not torch.isfinite(matrix @ matrix).all():
        raise RuntimeError("BF16 CUDA operation produced NaN or Inf")
    torch.distributed.barrier()


def _validate_hardware(rows: list[dict[str, object]], process_count: int) -> None:
    if len(rows) != process_count:
        raise RuntimeError("incomplete hardware all_gather")
    if any(_GPU_NAME_FRAGMENT not in str(row["name"]) for row in rows):
        raise RuntimeError(f"{_PROFILE_NAME} requires {_GPU_NAME_FRAGMENT}")
    if any(str(row["compute_capability"]) != _COMPUTE_CAPABILITY for row in rows):
        raise RuntimeError(
            f"{_PROFILE_NAME} requires compute capability {_COMPUTE_CAPABILITY}"
        )
    signatures = {
        (row["name"], row["compute_capability"], row["total_memory_bytes"])
        for row in rows
    }
    if len(signatures) != 1:
        raise RuntimeError("all visible GPUs must be homogeneous")
    for row in rows:
        total = int(cast(int, row["total_memory_bytes"]))
        free = int(cast(int, row["free_memory_bytes"]))
        if total < _MINIMUM_TOTAL_MEMORY_BYTES:
            raise RuntimeError(f"{_PROFILE_NAME} requires at least 75 GiB per GPU")
        if not 0 <= free <= total or free / total < _MINIMUM_FREE_MEMORY_FRACTION:
            raise RuntimeError(f"{_PROFILE_NAME} requires at least 90% free GPU memory")


def _run_distributed_preflight(
    torch: Any,
    profile: ExecutionPreset,
    *,
    rank: int,
    local_rank: int,
    fused_adam: str,
    amp_extension: str,
) -> dict[str, object] | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != profile.gpu_count:
        raise RuntimeError(
            f"preflight requires exactly {profile.gpu_count} visible CUDA devices"
        )
    torch.cuda.set_device(local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    mig_mode = _require_mig_disabled(f"GPU-{properties.uuid}")
    free_memory, _ = torch.cuda.mem_get_info(local_rank)
    local_hardware = {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "free_memory_bytes": int(free_memory),
        "mig_mode": mig_mode,
    }

    try:
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=120),
        )
        gathered_hardware: list[dict[str, object] | None] = [None] * profile.gpu_count
        torch.distributed.all_gather_object(gathered_hardware, local_hardware)
        if any(item is None for item in gathered_hardware):
            raise RuntimeError("incomplete hardware all_gather")
        rows = [dict(item) for item in gathered_hardware if item is not None]
        _validate_hardware(rows, profile.gpu_count)

        _run_collectives(torch, profile.gpu_count, rank)
        torch.cuda.synchronize()
        if rank != 0:
            return None
        nccl = torch.cuda.nccl.version()
        return {
            "status": "passed",
            "profile": _PROFILE_NAME,
            "world_size": profile.gpu_count,
            "gpus": rows,
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "nccl_version": ".".join(str(part) for part in nccl),
            "fused_adam": fused_adam,
            "amp_extension": amp_extension,
        }
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> int:
    arguments = _arguments()
    profile = _resolve_profile(arguments.profile)
    _visible_devices(profile.gpu_count)
    rank, local_rank = _distributed_environment(profile)
    torch, fused_adam, amp_extension = _training_runtime()
    result = _run_distributed_preflight(
        torch,
        profile,
        rank=rank,
        local_rank=local_rank,
        fused_adam=fused_adam,
        amp_extension=amp_extension,
    )
    if result is not None:
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

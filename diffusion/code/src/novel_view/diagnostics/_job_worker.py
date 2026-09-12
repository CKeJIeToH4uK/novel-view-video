"""Explicit backend imports and small CUDA probes, without loading models."""

from __future__ import annotations

import argparse
from datetime import timedelta
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


def _imports(backend: str) -> tuple[str, ...]:
    names: tuple[str, ...]
    if backend == "vggt":
        names = ("vggt_omega.models", "vggt_omega.utils.pose_enc")
    elif backend == "moge":
        names = ("moge.model.v1",)
    elif backend == "legacy_ddw":
        names = (
            "moge.model.v1",
            "cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch",
            "cosmos_predict1.diffusion.inference.cache_3d",
        )
    elif backend in ("generation", "generation_moge"):
        names = (
            "cosmos_predict1.diffusion.inference.gen3c_pipeline",
            "cosmos_predict1.diffusion.inference.inference_utils",
            "cosmos_predict1.diffusion.inference.cache_3d",
        )
        if backend == "generation_moge":
            names += ("moge.model.v1",)
    elif backend == "ddw_preparation":
        names = (
            "cosmos_predict1.auxiliary.t5_text_encoder",
            "cosmos_predict1.diffusion.module.pretrained_vae",
            "cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch",
            "moge.model.v1",
        )
    elif backend == "gen3c_training":
        names = (
            "cosmos_predict1.diffusion.inference.inference_utils",
            "transformer_engine.pytorch",
            "apex.optimizers",
            "amp_C",
            "megatron.core.parallel_state",
        )
    elif backend == "ddw_evaluation":
        names = (
            "cosmos_predict1.diffusion.inference.inference_utils",
            "cosmos_predict1.diffusion.module.pretrained_vae",
            "moge.model.v1",
        )
    else:  # EUVS metrics have no Gen3C diffusion dependency.
        names = ("sam2.build_sam", "sam2.sam2_image_predictor", "lpips")
        sys.path.insert(0, "/opt/upstream/dinov2")
        try:
            importlib.import_module("dinov2.hub.backbones")
        finally:
            sys.path.pop(0)
    for name in names:
        importlib.import_module(name)
    return names


def _read_byte(path: Path) -> None:
    with path.open("rb") as source:
        source.read(1)


def _declared_assets(arguments: argparse.Namespace) -> None:
    if arguments.vae_root is not None:
        for filename in (
            "encoder.jit",
            "decoder.jit",
            "mean_std.pt",
            "image_mean_std.pt",
        ):
            _read_byte(arguments.vae_root / filename)
    if arguments.t5_root is not None:
        from transformers import T5Config, T5TokenizerFast

        T5Config.from_pretrained(arguments.t5_root, local_files_only=True)
        T5TokenizerFast.from_pretrained(arguments.t5_root, local_files_only=True)
        _read_byte(arguments.t5_root / "pytorch_model.bin")
    if arguments.grounding_root is not None:
        from transformers import AutoConfig, AutoProcessor

        AutoConfig.from_pretrained(arguments.grounding_root, local_files_only=True)
        AutoProcessor.from_pretrained(arguments.grounding_root, local_files_only=True)
        _read_byte(arguments.grounding_root / "model.safetensors")
    if arguments.alexnet_root is not None:
        _read_byte(arguments.alexnet_root / "hub/checkpoints/alexnet-owt-7be5be79.pth")


def _nccl_smoke(process_count: int) -> int:
    import torch

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    try:
        torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=60))
        value = torch.ones(1, device="cuda", dtype=torch.bfloat16)
        torch.distributed.all_reduce(value)
        if float(value) != process_count:
            raise RuntimeError("Diagnostic returned an incorrect NCCL sum")
        torch.cuda.synchronize()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        choices=(
            "vggt",
            "moge",
            "legacy_ddw",
            "generation",
            "generation_moge",
            "euvs_evaluation",
            "ddw_preparation",
            "gen3c_training",
            "ddw_evaluation",
        ),
    )
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--t5-root", type=Path)
    parser.add_argument("--vae-root", type=Path)
    parser.add_argument("--grounding-root", type=Path)
    parser.add_argument("--alexnet-root", type=Path)
    parser.add_argument("--nccl", action="store_true")
    arguments = parser.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != arguments.gpu_count:
        raise RuntimeError(f"doctor requires {arguments.gpu_count} visible CUDA devices")
    if arguments.nccl:
        return _nccl_smoke(arguments.gpu_count)
    modules = _imports(arguments.backend)
    _declared_assets(arguments)
    if arguments.backend == "gen3c_training":
        distributed_worker = [
            "-m",
            "novel_view.diagnostics.training",
            "--profile",
            "a100-cp4",
        ]
    else:
        for device in range(arguments.gpu_count):
            matrix = torch.eye(32, device=f"cuda:{device}", dtype=torch.bfloat16)
            if not bool(torch.isfinite(matrix @ matrix).all()):
                raise RuntimeError("CUDA diagnostic produced non-finite values")
            torch.cuda.synchronize(device)
        distributed_worker = [
            "-m",
            "novel_view.diagnostics._job_worker",
            "--backend",
            arguments.backend,
            "--gpu-count",
            str(arguments.gpu_count),
            "--nccl",
        ]
    if arguments.gpu_count > 1:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={arguments.gpu_count}",
                "--max-restarts=0",
                *distributed_worker,
            ],
            check=False,
        )
        if completed.returncode:
            return completed.returncode
    print(
        json.dumps(
            {
                "backend": arguments.backend,
                "imports": modules,
                "gpu_count": arguments.gpu_count,
                "status": "passed",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

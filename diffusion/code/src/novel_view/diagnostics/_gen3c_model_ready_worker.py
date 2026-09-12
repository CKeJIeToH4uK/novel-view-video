"""Приватный prompt/VAE worker Gen3C model-ready диагностики."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_COMPONENT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_COMPONENT_ROOT / "src"))

from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT  # noqa: E402
from novel_view.preparation.waymo_ddw._prompt_worker import (  # noqa: E402
    _encode_empty_prompt,
    _load_encoder,
)
from novel_view.preparation.waymo_ddw._vae_worker import (  # noqa: E402
    _cpu_latent,
    _load_model,
    _normalize_condition,
    _normalize_target,
    _read_target,
)
from novel_view.preparation.waymo_ddw.artifacts import (  # noqa: E402
    BASE_LATENT_SHAPE,
    POSE_LATENT_SHAPE,
    read_base_latents,
    read_empty_prompt,
    read_pose_latent,
    write_base_latents,
    write_empty_prompt,
    write_pose_latent,
)
from novel_view.preparation.waymo_ddw.bake import (  # noqa: E402
    read_condition_pixels,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    prompt = phases.add_parser("prompt")
    prompt.add_argument("--text-encoder", type=Path, required=True)
    prompt.add_argument("--scratch-root", type=Path, required=True)
    vae = phases.add_parser("vae")
    vae.add_argument("--sample-id", required=True)
    vae.add_argument("--magnitude-m", type=float, required=True)
    vae.add_argument("--sign", type=int, choices=(-1, 1), required=True)
    vae.add_argument("--target-rgb", type=Path, required=True)
    vae.add_argument("--condition-rgb", type=Path, required=True)
    vae.add_argument("--condition-known", type=Path, required=True)
    vae.add_argument("--tokenizer", type=Path, required=True)
    vae.add_argument("--scratch-root", type=Path, required=True)
    return parser.parse_args()


def _rng_state(torch: Any) -> tuple[Any, Any]:
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()


def _require_rng(torch: Any, expected: tuple[Any, Any]) -> None:
    actual = _rng_state(torch)
    if not torch.equal(actual[0], expected[0]) or not torch.equal(
        actual[1], expected[1]
    ):
        raise RuntimeError("model-ready encode changed Torch RNG state")


def _report(
    torch: Any,
    phase: str,
    started: float,
    loaded: float,
    status: str = "pass",
    **fields: object,
) -> None:
    torch.cuda.synchronize()
    finished = time.monotonic()
    document = {
        "status": status,
        "phase": phase,
        "load_seconds": loaded - started,
        "check_seconds": finished - loaded,
        "total_seconds": finished - started,
        "cuda_bytes": {
            "allocated": int(torch.cuda.memory_allocated(0)),
            "reserved": int(torch.cuda.memory_reserved(0)),
            "max_allocated": int(torch.cuda.max_memory_allocated(0)),
            "max_reserved": int(torch.cuda.max_memory_reserved(0)),
        },
        **fields,
    }
    print(json.dumps(document, sort_keys=True), flush=True)


def _difference_metrics(
    torch: Any,
    left: Any,
    right: Any,
) -> dict[str, int | float]:
    delta = (left.to(torch.float32) - right.to(torch.float32)).abs()
    mismatch_count = int(torch.count_nonzero(left != right).item())
    return {
        "mismatch_count": mismatch_count,
        "mismatch_fraction": mismatch_count / left.numel(),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "rms_abs": float(delta.square().mean().sqrt().item()),
    }


def _run_prompt(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    torch, encoder = _load_encoder(arguments.text_encoder)
    torch.cuda.synchronize()
    loaded = time.monotonic()
    rng = _rng_state(torch)
    first = _encode_empty_prompt(torch, encoder)
    _require_rng(torch, rng)
    repeated = _encode_empty_prompt(torch, encoder)
    _require_rng(torch, rng)
    if not torch.equal(first, repeated):
        raise RuntimeError("empty prompt encode is not exact-repeatable")

    with tempfile.TemporaryDirectory(
        prefix="gen3c-model-ready-prompt-",
        dir=arguments.scratch_root,
    ) as directory:
        path = Path(directory) / "empty_prompt.pt"
        write_empty_prompt(path, first)
        if not torch.equal(read_empty_prompt(path).t5_text_embeddings, first):
            raise RuntimeError("empty prompt artifact round-trip differs")
    _report(torch, "prompt", started, loaded)


def _target_tensor(torch: Any, rgb: np.ndarray, device: Any) -> Any:
    value = torch.from_numpy(rgb).permute(3, 0, 1, 2).unsqueeze(0)
    return _normalize_target(torch, value, device)


def _condition_tensors(
    torch: Any,
    rgb_path: Path,
    known_path: Path,
    device: Any,
) -> tuple[Any, Any]:
    contract = R4C_GEN3C_MODEL_CONTRACT
    rgb, known = read_condition_pixels(
        rgb_path,
        known_path,
        contract.raster_size_hw,
    )
    if (
        rgb.dtype != np.uint8
        or rgb.shape != contract.rgb_thwc_shape
        or known.dtype != np.bool_
        or known.shape != contract.known_thw_shape
    ):
        raise ValueError("model-ready condition has the wrong array contract")
    return _normalize_condition(
        torch,
        torch.from_numpy(rgb).permute(3, 0, 1, 2).unsqueeze(0),
        torch.from_numpy(known).unsqueeze(0).unsqueeze(0),
        device,
    )


def _run_vae(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    torch, model, helper = _load_model(arguments.tokenizer)
    torch.cuda.synchronize()
    loaded = time.monotonic()
    rng = _rng_state(torch)
    device = torch.device("cuda", 0)

    target = _read_target(arguments.target_rgb)
    target_device = _target_tensor(torch, target, device)
    clean_a = _cpu_latent(
        torch,
        model.encode(target_device),
        BASE_LATENT_SHAPE,
        "clean_latent",
    )
    source_raw, padded = helper(
        model,
        target_device[:, :, :1].contiguous(),
        num_frames_condition=1,
    )
    del padded
    source_a = _cpu_latent(
        torch,
        source_raw,
        BASE_LATENT_SHAPE,
        "source_latent",
    )
    _require_rng(torch, rng)
    clean_repeat = _cpu_latent(
        torch,
        model.encode(target_device),
        BASE_LATENT_SHAPE,
        "clean_latent",
    )
    source_raw, padded = helper(
        model,
        target_device[:, :, :1].contiguous(),
        num_frames_condition=1,
    )
    del padded, target_device
    source_repeat = _cpu_latent(
        torch,
        source_raw,
        BASE_LATENT_SHAPE,
        "source_latent",
    )
    _require_rng(torch, rng)
    np.subtract(255, target[1:], out=target[1:])
    target_device = _target_tensor(torch, target, device)
    clean_b = _cpu_latent(
        torch,
        model.encode(target_device),
        BASE_LATENT_SHAPE,
        "clean_latent",
    )
    source_raw, padded = helper(
        model,
        target_device[:, :, :1].contiguous(),
        num_frames_condition=1,
    )
    del padded, target_device, target
    source_b = _cpu_latent(
        torch,
        source_raw,
        BASE_LATENT_SHAPE,
        "source_latent",
    )
    _require_rng(torch, rng)
    metrics = {
        "clean_repeat": _difference_metrics(torch, clean_a, clean_repeat),
        "source_repeat": _difference_metrics(torch, source_a, source_repeat),
        "clean_target_effect": _difference_metrics(torch, clean_a, clean_b),
        "source_target_effect": _difference_metrics(torch, source_a, source_b),
    }
    if metrics["clean_target_effect"]["mismatch_count"] == 0:
        raise RuntimeError("base latent target barrier failed")

    condition, known = _condition_tensors(
        torch,
        arguments.condition_rgb,
        arguments.condition_known,
        device,
    )
    pose = _cpu_latent(
        torch,
        model.encode_warped_frames(condition, known, torch.bfloat16),
        POSE_LATENT_SHAPE,
        "pose_latent",
    )
    _require_rng(torch, rng)
    pose_repeat = _cpu_latent(
        torch,
        model.encode_warped_frames(condition, known, torch.bfloat16),
        POSE_LATENT_SHAPE,
        "pose_latent",
    )
    _require_rng(torch, rng)
    del condition, known
    metrics["pose_repeat"] = _difference_metrics(torch, pose, pose_repeat)
    if not bool(torch.all(pose[:, 32:] == 0)):
        raise RuntimeError("DDW pose second buffer is not zero")

    with tempfile.TemporaryDirectory(
        prefix="gen3c-model-ready-vae-",
        dir=arguments.scratch_root,
    ) as directory:
        base_path = Path(directory) / "base.pt"
        pose_path = Path(directory) / "pose.pt"
        write_base_latents(base_path, arguments.sample_id, clean_a, source_a)
        loaded_base = read_base_latents(
            base_path,
            arguments.sample_id,
        )
        write_pose_latent(
            pose_path,
            arguments.sample_id,
            arguments.magnitude_m,
            arguments.sign,
            pose,
        )
        loaded_pose = read_pose_latent(
            pose_path,
            arguments.sample_id,
            arguments.magnitude_m,
            arguments.sign,
        )
        if (
            not torch.equal(loaded_base.clean_latent, clean_a)
            or not torch.equal(loaded_base.source_latent, source_a)
            or not torch.equal(loaded_pose.pose_latent, pose)
        ):
            raise RuntimeError("VAE artifact round-trip differs")
    stop = any(
        metrics[name]["mismatch_count"] > 0
        for name in (
            "clean_repeat",
            "source_repeat",
            "source_target_effect",
            "pose_repeat",
        )
    )
    fields: dict[str, object] = {
        "sample_id": arguments.sample_id,
        "magnitude_m": arguments.magnitude_m,
        "sign": arguments.sign,
        "metrics": metrics,
    }
    if stop:
        fields["reason"] = "non_bit_exact_latents"
    _report(
        torch,
        "vae",
        started,
        loaded,
        "stop" if stop else "pass",
        **fields,
    )
    if stop:
        raise RuntimeError("VAE latents are not bit-exact-repeatable")


def _run(arguments: argparse.Namespace) -> None:
    if arguments.phase == "prompt":
        _run_prompt(arguments)
    else:
        _run_vae(arguments)


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()

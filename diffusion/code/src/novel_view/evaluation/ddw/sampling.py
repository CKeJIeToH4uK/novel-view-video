"""Exact matched base/v1/v2 sampling for one prepared R4c item."""

from __future__ import annotations

import importlib
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from novel_view.models.gen3c.lora_weights import (
    gen3c_lora_enabled,
    replace_gen3c_lora_adapter,
)
from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT
from novel_view.evaluation.ddw.spec import DdwSamplingSpec


TEMP_LATENT_FORMAT = "novel-view/ddw-evaluation-latent/v1"
DdwVariant = Literal["base", "v1", "v2"]


@dataclass(frozen=True, slots=True)
class SampledDdwVariants:
    """Three CPU BF16 latents produced from one immutable sampling state."""

    base: Any
    v1: Any
    v2: Any


@dataclass(frozen=True, slots=True)
class R4cSamplingState:
    """Conditions, explicit noises, RNG boundary, and original LoRA layout."""

    condition: Any
    uncondition: Any
    initial_noise: Any
    condition_noise: Any
    rng: tuple[Any, ...]
    initial_adapter: Mapping[str, Any]


def prepare_r4c_sampling(
    model: Any,
    source_latent: Any,
    pose_latent: Any,
    prompt_embedding: Any,
    sampling: DdwSamplingSpec,
    initial_adapter: Mapping[str, Any],
) -> R4cSamplingState:
    """Build the one condition and two explicit noises shared by all variants."""
    torch = importlib.import_module("torch")
    data_batch = _conditioner_inputs(torch, prompt_embedding)
    condition, uncondition = model.conditioner.get_condition_uncondition(data_batch)
    condition.video_cond_bool = True
    condition = model.add_condition_video_indicator_and_video_input_mask(
        source_latent,
        condition,
        1,
    )
    condition = model.add_condition_pose(pose_latent, condition)
    uncondition.video_cond_bool = True
    uncondition = model.add_condition_video_indicator_and_video_input_mask(
        source_latent,
        uncondition,
        1,
    )
    uncondition = model.add_condition_pose(
        pose_latent,
        uncondition,
        drop_out_latent=True,
    )
    parallel_state = importlib.import_module("megatron.core.parallel_state")
    if parallel_state.is_initialized():
        model_v2w = importlib.import_module("cosmos_predict1.diffusion.model.model_v2w")
        condition = model_v2w.broadcast_condition(
            condition,
            to_tp=False,
            to_cp=model.net.is_context_parallel_enabled,
        )
        uncondition = model_v2w.broadcast_condition(
            uncondition,
            to_tp=False,
            to_cp=model.net.is_context_parallel_enabled,
        )

    generator = torch.Generator(device=source_latent.device)
    generator.manual_seed(sampling.seed)
    initial_noise = torch.randn(
        R4C_GEN3C_MODEL_CONTRACT.base_latent_shape,
        generator=generator,
        device=source_latent.device,
        dtype=model.tensor_kwargs["dtype"],
    )
    initial_noise = initial_noise * model.scheduler.init_noise_sigma
    condition_noise = torch.randn(
        source_latent.shape,
        generator=generator,
        device=source_latent.device,
        dtype=torch.float32,
    )
    return R4cSamplingState(
        condition=condition,
        uncondition=uncondition,
        initial_noise=initial_noise,
        condition_noise=condition_noise,
        rng=_capture_rng(torch, source_latent.device),
        initial_adapter=initial_adapter,
    )


def sample_r4c_variant(
    model: Any,
    state: R4cSamplingState,
    sampling: DdwSamplingSpec,
    adapter: Mapping[str, Any] | None,
) -> Any:
    """Sample base or one full replacement adapter from the same boundary."""
    torch = importlib.import_module("torch")
    device = state.initial_noise.device
    _restore_rng(torch, device, state.rng)
    replace_gen3c_lora_adapter(model.model, state.initial_adapter)
    if adapter is not None:
        replace_gen3c_lora_adapter(model.model, adapter)
    with torch.inference_mode(), gen3c_lora_enabled(
        model.model,
        adapter is not None,
    ):
        latent = _sample_once(
            torch,
            model,
            state.condition,
            state.uncondition,
            state.initial_noise.clone(),
            state.condition_noise.clone(),
            sampling,
        )
    result = latent.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(result.shape) != R4C_GEN3C_MODEL_CONTRACT.base_latent_shape or not bool(
        torch.isfinite(result).all()
    ):
        raise RuntimeError("sampled R4c latent violates its scientific contract")
    return result


def sample_r4c_variants(
    model: Any,
    state: R4cSamplingState,
    sampling: DdwSamplingSpec,
    v1_adapter: Mapping[str, Any],
    v2_adapter: Mapping[str, Any],
) -> SampledDdwVariants:
    """Run base, v1, and v2 in their fixed matched order."""
    return SampledDdwVariants(
        base=sample_r4c_variant(model, state, sampling, None),
        v1=sample_r4c_variant(model, state, sampling, v1_adapter),
        v2=sample_r4c_variant(model, state, sampling, v2_adapter),
    )


def write_temporary_latent(
    path: Path,
    sample_id: str,
    variant: DdwVariant,
    latent: Any,
) -> None:
    """Write one strict process-boundary PT inside the attempt cache."""
    importlib.import_module("torch").save(
        {
            "format": TEMP_LATENT_FORMAT,
            "sample_id": sample_id,
            "variant": variant,
            "latent": latent,
        },
        path,
    )


def read_temporary_latent(
    path: Path,
    sample_id: str,
    variant: DdwVariant,
) -> Any:
    """Read one generated latent at the decode process boundary."""
    torch = importlib.import_module("torch")
    raw = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(raw, Mapping) or set(raw) != {
        "format",
        "sample_id",
        "variant",
        "latent",
    }:
        raise ValueError("DDW evaluation latent fields differ")
    if (
        raw["format"] != TEMP_LATENT_FORMAT
        or raw["sample_id"] != sample_id
        or raw["variant"] != variant
    ):
        raise ValueError("DDW evaluation latent identity differs")
    latent = raw["latent"]
    if (
        not torch.is_tensor(latent)
        or latent.device.type != "cpu"
        or latent.dtype != torch.bfloat16
        or tuple(latent.shape) != R4C_GEN3C_MODEL_CONTRACT.base_latent_shape
        or not latent.is_contiguous()
        or not bool(torch.isfinite(latent).all())
    ):
        raise ValueError("DDW evaluation latent tensor differs")
    return latent


def _conditioner_inputs(torch: Any, prompt_embedding: Any) -> dict[str, Any]:
    contract = R4C_GEN3C_MODEL_CONTRACT
    device = prompt_embedding.device
    return {
        "t5_text_embeddings": prompt_embedding,
        "t5_text_mask": torch.ones(
            contract.prompt_mask_shape,
            dtype=torch.bfloat16,
            device=device,
        ),
        "fps": torch.tensor([contract.fps], dtype=torch.bfloat16, device=device),
        "num_frames": torch.tensor(
            [contract.video_frame_count],
            dtype=torch.bfloat16,
            device=device,
        ),
        "image_size": torch.tensor(
            [contract.image_size_values],
            dtype=torch.bfloat16,
            device=device,
        ),
        "padding_mask": torch.zeros(
            contract.padding_mask_shape,
            dtype=torch.bfloat16,
            device=device,
        ),
    }


def _sample_once(
    torch: Any,
    model: Any,
    condition: Any,
    uncondition: Any,
    initial_noise: Any,
    condition_noise: Any,
    sampling: DdwSamplingSpec,
) -> Any:
    model.scheduler.set_timesteps(sampling.steps)
    state = initial_noise
    parallel = None
    if model.net.is_context_parallel_enabled:
        parallel = importlib.import_module("cosmos_predict1.diffusion.module.parallel")
        state = parallel.split_inputs_cp(
            state,
            seq_dim=2,
            cp_group=model.net.cp_group,
        )

    for timestep in model.scheduler.timesteps:
        model.scheduler._init_step_index(timestep)
        sigma = model.scheduler.sigmas[model.scheduler.step_index].to(
            **model.tensor_kwargs
        )
        augmented, clean_condition, indicator = _augment_condition(
            torch,
            model,
            state,
            sigma,
            condition,
            condition_noise,
            parallel,
            sampling,
        )
        network_input = model.scheduler.scale_model_input(
            augmented.to(**model.tensor_kwargs),
            timestep=timestep,
        )
        timestep = timestep.to(**model.tensor_kwargs)
        conditional = model.net(
            x=network_input,
            timesteps=timestep,
            **condition.to_dict(),
        )
        unconditional = model.net(
            x=network_input,
            timesteps=timestep,
            **uncondition.to_dict(),
        )
        guided = conditional + sampling.guidance * (conditional - unconditional)
        condition_output = model._reverse_precondition_output(
            clean_condition,
            xt=augmented,
            sigma=sigma,
        )
        output = indicator * condition_output + (1 - indicator) * guided
        state = model.scheduler.step(output, timestep, augmented).prev_sample

    if parallel is not None:
        state = parallel.cat_outputs_cp(
            state,
            seq_dim=2,
            cp_group=model.net.cp_group,
        )
    return state


def _augment_condition(
    torch: Any,
    model: Any,
    state: Any,
    sigma: Any,
    condition: Any,
    condition_noise: Any,
    parallel: Any | None,
    sampling: DdwSamplingSpec,
) -> tuple[Any, Any, Any]:
    clean = condition.gt_latent
    indicator = condition.condition_video_indicator
    if sampling.condition_augment_sigma >= sigma:
        indicator = torch.zeros_like(indicator)
    augmented = clean + condition_noise * sampling.condition_augment_sigma
    augmented = model.scheduler.precondition_inputs(
        augmented,
        sampling.condition_augment_sigma,
    )
    augmented = model._reverse_precondition_input(augmented, sigma)
    if parallel is not None:
        clean = parallel.split_inputs_cp(
            clean,
            seq_dim=2,
            cp_group=model.net.cp_group,
        )
        indicator = parallel.split_inputs_cp(
            indicator,
            seq_dim=2,
            cp_group=model.net.cp_group,
        )
        augmented = parallel.split_inputs_cp(
            augmented,
            seq_dim=2,
            cp_group=model.net.cp_group,
        )
    return indicator * augmented + (1 - indicator) * state, clean, indicator


def _capture_rng(torch: Any, device: Any) -> tuple[Any, ...]:
    cuda_state = (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        cuda_state,
    )


def _restore_rng(torch: Any, device: Any, state: tuple[Any, ...]) -> None:
    python_state, numpy_state, cpu_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(cast(tuple[Any, ...], numpy_state))
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


__all__ = [
    "DdwVariant",
    "R4cSamplingState",
    "SampledDdwVariants",
    "prepare_r4c_sampling",
    "read_temporary_latent",
    "sample_r4c_variant",
    "sample_r4c_variants",
    "write_temporary_latent",
]

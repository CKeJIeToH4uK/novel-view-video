"""Буквальный Kendall EDM objective baseline R4c LoRA."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT

if TYPE_CHECKING:
    from novel_view.training.gen3c.topology import Gen3cTrainingTopology


@dataclass(frozen=True, slots=True)
class Gen3cTrainingNoise:
    """Four CP-consistent draws used by one baseline forward."""

    sigma: Any
    epsilon: Any
    condition_sigma: Any
    condition_epsilon: Any


@dataclass(frozen=True, slots=True)
class EdmForwardResult:
    """One differentiable EDM value and its CP-local v2 seam."""

    loss: Any
    prediction: Any
    clean_target: Any
    generation_mask: Any


def build_r4c_conditioner_inputs(prompt_embedding: Any) -> dict[str, Any]:
    """Build the six fixed native conditioner inputs beside one prompt."""
    torch = importlib.import_module("torch")
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


def prepare_r4c_condition(
    model: Any,
    prompt_embedding: Any,
    source_latent: Any,
    pose_latent: Any,
) -> Any:
    """Build the native Gen3C condition from the four ready batch inputs."""
    data_batch = build_r4c_conditioner_inputs(prompt_embedding)
    condition, _ = model.conditioner.get_condition_uncondition(data_batch)
    condition.video_cond_bool = True
    condition = model.add_condition_video_indicator_and_video_input_mask(
        source_latent,
        condition,
        1,
    )
    return model.add_condition_pose(pose_latent, condition)


def sample_r4c_noise(
    clean_latent: Any,
    topology: Gen3cTrainingTopology,
) -> Gen3cTrainingNoise:
    """Draw all four baseline tensors locally, then synchronize their values."""
    torch = importlib.import_module("torch")
    batch_size = clean_latent.shape[0]
    device = clean_latent.device
    dtype = clean_latent.dtype

    generation_log_sigma = torch.randn(
        batch_size,
        device=device,
        dtype=torch.float32,
    )
    condition_log_sigma = torch.randn(
        batch_size,
        device=device,
        dtype=torch.float32,
    )
    epsilon = torch.randn(clean_latent.shape, device=device, dtype=dtype)
    condition_epsilon = torch.randn(clean_latent.shape, device=device, dtype=dtype)

    sigma = generation_log_sigma.exp().mul(4.0).to(dtype=dtype)
    condition_sigma = condition_log_sigma.mul(2.0).add(-3.0).exp().to(dtype=dtype)
    group = topology.context_parallel_group
    values = (sigma, epsilon, condition_sigma, condition_epsilon)
    return Gen3cTrainingNoise(
        *(_broadcast_from_cp_source(value, group) for value in values)
    )


def forward_r4c_edm(
    model: Any,
    network: Any,
    clean_latent: Any,
    noise: Gen3cTrainingNoise,
    condition: Any,
) -> EdmForwardResult:
    """Run one native forward with condition augmentation and Kendall loss."""
    sigma = noise.sigma
    noisy_latent = clean_latent + _batch_view(sigma, clean_latent) * noise.epsilon
    clean_target = clean_latent
    if model.net.is_context_parallel_enabled:
        parallel = importlib.import_module("cosmos_predict1.diffusion.module.parallel")
        noisy_latent = parallel.split_inputs_cp(
            noisy_latent,
            seq_dim=2,
            cp_group=model.net.cp_group,
        )
        clean_target = parallel.split_inputs_cp(
            clean_target,
            seq_dim=2,
            cp_group=model.net.cp_group,
        )

    network_input, generation_mask = _augment_training_condition(
        model,
        noisy_latent,
        sigma,
        condition,
        noise,
    )
    network_input = network_input.to(**model.tensor_kwargs)
    model_input = model.scheduler.precondition_inputs(network_input, sigma)
    model_input.requires_grad_(True)
    timestep = model.scheduler.precondition_noise(sigma)
    model_output = network(
        x=model_input,
        timesteps=timestep,
        **condition.to_dict(),
    )
    prediction = model.scheduler.precondition_outputs(
        network_input,
        model_output,
        sigma,
    )
    logvar = model.model.logvar(timestep)
    loss = kendall_edm_loss(
        prediction,
        clean_target,
        generation_mask,
        sigma,
        model.sigma_data,
        logvar,
    )
    return EdmForwardResult(loss, prediction, clean_target, generation_mask)


def kendall_edm_loss(
    prediction: Any,
    clean_target: Any,
    generation_mask: Any,
    sigma: Any,
    sigma_data: float,
    logvar: Any,
) -> Any:
    """Apply the frozen full-tensor Kendall EDM formula."""
    difference = prediction.float() - clean_target.float()
    sigma_float = sigma.float()
    weight = (sigma_float.square() + sigma_data**2) / (
        sigma_float * sigma_data
    ).square()
    base = (
        difference.square() * generation_mask.float() * _batch_view(weight, difference)
    )
    logvar_view = _batch_view(logvar.float(), difference)
    return (base * (-logvar_view).exp() + logvar_view).mean()


def _augment_training_condition(
    model: Any,
    noisy_latent: Any,
    sigma: Any,
    condition: Any,
    noise: Gen3cTrainingNoise,
) -> tuple[Any, Any]:
    source = condition.gt_latent
    indicator = condition.condition_video_indicator
    augmented = (
        source + _batch_view(noise.condition_sigma, source) * noise.condition_epsilon
    )
    augmented = model.scheduler.precondition_inputs(augmented, noise.condition_sigma)
    augmented = model._reverse_precondition_input(augmented, sigma)
    if model.net.is_context_parallel_enabled:
        parallel = importlib.import_module("cosmos_predict1.diffusion.module.parallel")
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
    network_input = indicator * augmented + (1 - indicator) * noisy_latent
    return network_input, 1 - indicator


def _broadcast_from_cp_source(value: Any, group: Any) -> Any:
    if group is None:
        return value
    distributed = importlib.import_module("torch.distributed")
    source_rank = min(distributed.get_process_group_ranks(group))
    distributed.broadcast(value, source_rank, group=group)
    return value


def _batch_view(value: Any, reference: Any) -> Any:
    if value.ndim == 0:
        value = value.unsqueeze(0)
    return value.view(value.shape[0], *((1,) * (reference.ndim - 1)))


__all__ = [
    "EdmForwardResult",
    "Gen3cTrainingNoise",
    "build_r4c_conditioner_inputs",
    "forward_r4c_edm",
    "kendall_edm_loss",
    "prepare_r4c_condition",
    "sample_r4c_noise",
]

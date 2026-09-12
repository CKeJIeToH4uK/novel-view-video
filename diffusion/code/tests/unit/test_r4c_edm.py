"""Independent Kendall EDM values, vector gradients and the four RNG draws."""

import math
from types import SimpleNamespace as NS

import pytest
import torch

from novel_view.training.gen3c.lora import objective
from tests.support.gen3c_training import ObjectiveCondition, ObjectiveModel


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_kendall_value_and_full_gradient_across_cp_shards(cp_size):
    prediction = torch.linspace(-1.0, 1.0, 16, requires_grad=True).view(1, 1, 16, 1, 1)
    target = torch.linspace(0.5, -0.5, 16).view_as(prediction)
    mask = torch.ones_like(target)
    mask[:, :, 0] = 0
    reference = ((prediction - target).square() * mask * 0.625 + math.log(2)).mean()
    losses = [
        objective.kendall_edm_loss(p, t, m, torch.tensor([2.0]), 1.0, torch.tensor([math.log(2)]))
        for p, t, m in zip(
            prediction.chunk(cp_size, 2), target.chunk(cp_size, 2), mask.chunk(cp_size, 2)
        )
    ]
    actual = torch.stack(losses).mean()
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(
        torch.autograd.grad(actual, prediction)[0], torch.autograd.grad(reference, prediction)[0]
    )


def test_four_draws_keep_fp32_log_sigma_and_one_cp_source(monkeypatch):
    latent = torch.zeros(2, 1, 2, 1, 1, dtype=torch.bfloat16)
    draws = iter(
        [
            torch.tensor([1.0, -1.0]),
            torch.tensor([0.05, -0.05]),
            torch.zeros_like(latent),
            torch.ones_like(latent),
        ]
    )
    events = []

    def randn(shape, *, device, dtype):
        events.append(dtype)
        return next(draws).to(device=device, dtype=dtype)

    def broadcast(value, source, *, group):
        events.append((source, group))

    monkeypatch.setattr(torch, "randn", randn)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [5, 3])
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    noise = objective.sample_r4c_noise(latent, NS(context_parallel_group="cp"))
    assert (
        events == [torch.float32, torch.float32, torch.bfloat16, torch.bfloat16] + [(3, "cp")] * 4
    )
    torch.testing.assert_close(
        noise.sigma, (torch.tensor([1.0, -1.0]).exp() * 4).bfloat16(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        noise.condition_sigma,
        (torch.tensor([0.05, -0.05]) * 2 - 3).exp().bfloat16(),
        rtol=0,
        atol=0,
    )
    assert not noise.epsilon.any() and noise.condition_epsilon.eq(1).all()


@pytest.mark.parametrize("cp", [False, True])
def test_forward_preserves_condition_and_sequence_split(cp, monkeypatch):
    model = ObjectiveModel(context_parallel=cp)
    clean = torch.tensor([10.0, 2.0, 30.0, 4.0]).view(1, 1, 4, 1, 1)
    condition = ObjectiveCondition(
        torch.tensor([3.0, 0.0, 5.0, 0.0]).view_as(clean),
        torch.tensor([1.0, 0.0, 1.0, 0.0]).view_as(clean),
    )
    noise = objective.Gen3cTrainingNoise(
        torch.ones(1), torch.zeros_like(clean), torch.tensor([2.0]), torch.ones_like(clean)
    )
    seen, split = [], []
    original = objective.importlib.import_module

    def split_inputs(value, *, seq_dim, cp_group):
        split.append((seq_dim, cp_group))
        return value.chunk(2, dim=seq_dim)[0]

    monkeypatch.setattr(
        objective.importlib,
        "import_module",
        lambda name: (
            NS(split_inputs_cp=split_inputs) if name.endswith(".parallel") else original(name)
        ),
    )

    def network(*, x, timesteps, **kwargs):
        seen.append(x)
        return torch.zeros_like(x)

    result = objective.forward_r4c_edm(model, network, clean, noise, condition)
    assert seen[0].requires_grad
    assert seen[0][0, 0, 0, 0, 0].item() == pytest.approx(5 / math.sqrt(5))
    torch.testing.assert_close(result.clean_target, clean[:, :, :2] if cp else clean)
    expected_mask = (
        (1 - condition.condition_video_indicator)[:, :, :2]
        if cp
        else 1 - condition.condition_video_indicator
    )
    torch.testing.assert_close(result.generation_mask, expected_mask)
    assert result.loss.item() == (4.0 if cp else 10.0)
    assert split == ([(2, "cp")] * 4 if cp else [])


def test_conditioner_fixed_fields_do_not_consume_rng():
    prompt = torch.zeros((1, 512, 1024), dtype=torch.bfloat16)
    before = torch.get_rng_state().clone()
    values = objective.build_r4c_conditioner_inputs(prompt)
    assert set(values) == {
        "t5_text_embeddings",
        "t5_text_mask",
        "fps",
        "num_frames",
        "image_size",
        "padding_mask",
    }
    assert values["t5_text_embeddings"] is prompt
    assert values["t5_text_mask"].eq(1).all() and not values["padding_mask"].any()
    assert values["fps"].item() == 10 and values["num_frames"].item() == 121
    assert values["image_size"].tolist() == [[704, 1280, 704, 1280]]
    assert torch.equal(before, torch.get_rng_state())

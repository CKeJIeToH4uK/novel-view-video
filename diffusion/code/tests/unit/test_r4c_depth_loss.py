"""Frozen teacher, masked Smooth-L1 and complete CP gradient equivalence."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch.nn import functional as F

from novel_view.training.gen3c.lora import depth, depth_method, depth_objective
from novel_view.training.gen3c.lora.objective import EdmForwardResult
from novel_view.training.gen3c.topology import Gen3cTrainingTopology
from tests.support.gen3c_training import local_topology, loop_batch


def test_teacher_detached_prediction_gradient_and_zero_equal_readouts():
    prediction = torch.ones((1, 16, 2, 1, 1), requires_grad=True)
    clean = torch.zeros_like(prediction, requires_grad=True)
    valid = torch.tensor([True, False]).view(1, 1, 2, 1, 1)
    grid = depth.DepthGrid(torch.zeros_like(valid, dtype=torch.float32), valid)
    observer = depth.LatentDepthObserver().requires_grad_(False)
    with torch.no_grad():
        for parameter in observer.parameters():
            parameter.zero_()
        observer.input.weight[0, 0] = observer.output.weight[0, 0] = 1
    edm = EdmForwardResult(torch.tensor(0.0), prediction, clean, valid.float())
    result = depth_objective.depth_loss_from_edm(edm, grid, observer, local_topology())
    result.backward_loss.backward()
    assert result.global_loss > 0 and result.global_valid_count == 1
    assert prediction.grad[:, :, 0].abs().sum() > 0 and not prediction.grad[:, :, 1].any()
    assert clean.grad is None and all(parameter.grad is None for parameter in observer.parameters())
    same = prediction.detach()
    equal = EdmForwardResult(torch.tensor(0.0), same, same, valid.float())
    assert (
        depth_objective.depth_loss_from_edm(
            equal, grid, observer, local_topology()
        ).global_loss.item()
        == 0.0
    )
    with pytest.raises(RuntimeError):
        depth_objective.depth_loss_from_edm(
            equal, depth.DepthGrid(grid.values, torch.zeros_like(valid)), observer, local_topology()
        )


@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_sparse_full_gradient_matches_global_mean_even_with_empty_ranks(cp_size, monkeypatch):
    prediction = torch.linspace(-1.0, 1.0, 8, requires_grad=True).view(1, 1, 8, 1, 1)
    clean = torch.zeros_like(prediction)
    valid = torch.tensor([False] * 4 + [True] * 4).view_as(clean)
    grid = depth.DepthGrid(clean, valid)
    errors = F.smooth_l1_loss(prediction, clean, beta=0.1, reduction="none")
    reference = errors.masked_select(valid).mean()
    expected_gradient = torch.autograd.grad(reference, prediction, retain_graph=True)[0]
    total_error = errors.detach().masked_select(valid).sum()
    rank = 0

    def split(value, _group):
        return depth.DepthGrid(
            value.values.chunk(cp_size, 2)[rank], value.valid.chunk(cp_size, 2)[rank]
        )

    def reduce(value, *, group):
        assert group == "cp"
        value.fill_(int(valid.sum()) if value.dtype == torch.int64 else total_error)

    monkeypatch.setattr(depth_objective, "split_depth_grid_cp", split)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    local_losses = []
    for rank in range(cp_size):
        local = EdmForwardResult(
            torch.tensor(0.0),
            prediction.chunk(cp_size, 2)[rank],
            clean.chunk(cp_size, 2)[rank],
            torch.ones_like(clean.chunk(cp_size, 2)[rank]),
        )
        topology = Gen3cTrainingTopology(cp_size, cp_size, rank, rank, "cp", "cp", cp_size)
        result = depth_objective.depth_loss_from_edm(local, grid, torch.nn.Identity(), topology)
        local_losses.append(result.backward_loss)
        torch.testing.assert_close(result.global_loss, reference.detach())
    simulated_ddp = torch.stack(local_losses).mean()
    torch.testing.assert_close(simulated_ddp, reference)
    torch.testing.assert_close(torch.autograd.grad(simulated_ddp, prediction)[0], expected_gradient)


def test_depth_weighted_step_clipping_and_validation_mean(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    losses = iter([(parameter - 1).square(), torch.tensor(1.0), torch.tensor(3.0)])
    depths = iter(
        [((parameter - 2).square(), 0.5), (torch.tensor(2.0), 2.0), (torch.tensor(4.0), 4.0)]
    )
    monkeypatch.setattr(
        depth_method,
        "sample_r4c_noise",
        lambda *_: NS(sigma=torch.ones(1), condition_sigma=torch.zeros(1)),
    )
    monkeypatch.setattr(depth_method, "prepare_r4c_condition", lambda *_: None)
    monkeypatch.setattr(depth_method, "forward_r4c_edm", lambda *_: NS(loss=next(losses)))

    def depth_loss(*_):
        backward, reported = next(depths)
        return NS(backward_loss=backward, global_loss=torch.tensor(reported), global_valid_count=1)

    monkeypatch.setattr(depth_method, "depth_loss_from_edm", depth_loss)
    common = dict(
        model=None, network=None, observer=None, depth_weight=0.1, topology=local_topology()
    )
    batch = loop_batch("sample")
    step = depth_method.run_training_step_v2(
        **common,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_parameters=(parameter,),
        batch=batch,
        depth=None
    )
    validation = depth_method.run_validation_v2(**common, batches=[(batch, None)] * 2)
    assert (parameter.item(), step.gradient_norm) == pytest.approx((0.1, 2.4))
    assert (step.edm_loss, step.depth_loss, step.total_loss) == (1.0, 0.5, 1.05)
    assert validation == depth_method.ValidationRecordV2(2.0, 3.0, 2.3, 2)

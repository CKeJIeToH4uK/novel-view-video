"""Экспериментальный latent-depth objective с точной CP-нормировкой."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from novel_view.training.gen3c.lora.depth import (
    DepthGrid,
    LatentDepthObserver,
    split_depth_grid_cp,
)
from novel_view.training.gen3c.lora.objective import EdmForwardResult
from novel_view.training.gen3c.topology import Gen3cTrainingTopology


@dataclass(frozen=True, slots=True, eq=False)
class DepthLossResult:
    """Differentiable local scalar and detached CP-global measurement."""

    backward_loss: Tensor
    global_loss: Tensor
    global_valid_count: int


def depth_loss_from_edm(
    edm: EdmForwardResult,
    depth: DepthGrid,
    observer: LatentDepthObserver,
    topology: Gen3cTrainingTopology,
) -> DepthLossResult:
    """Compare predicted/clean readouts and preserve global-mean gradients."""
    local_depth = (
        depth
        if topology.context_parallel_group is None
        else split_depth_grid_cp(depth, topology.context_parallel_group)
    )
    predicted = observer(edm.prediction.float())
    clean = observer(edm.clean_target.float()).detach()
    valid = local_depth.valid & edm.generation_mask.bool()
    errors = F.smooth_l1_loss(predicted, clean, beta=0.1, reduction="none")
    local_error_sum = errors.masked_select(valid).sum()
    global_count = valid.sum(dtype=torch.int64)
    global_error_sum = local_error_sum.detach().clone()

    group = topology.gradient_average_group
    if group is not None:
        distributed = torch.distributed
        distributed.all_reduce(global_count, group=group)
        distributed.all_reduce(global_error_sum, group=group)
    count = int(global_count.item())
    if count == 0:
        raise RuntimeError("R4c depth objective has no globally valid LiDAR cells")

    return DepthLossResult(
        backward_loss=topology.gradient_average_size * local_error_sum / global_count,
        global_loss=global_error_sum / global_count,
        global_valid_count=count,
    )


__all__ = ["DepthLossResult", "depth_loss_from_edm"]

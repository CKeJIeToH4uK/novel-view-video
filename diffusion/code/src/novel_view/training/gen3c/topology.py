"""Фактические CP4-группы одного R4c training worker."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Gen3cTrainingTopology:
    """Один снимок групп, которые реально использует Gen3C training."""

    world_size: int
    context_parallel_size: int
    rank: int
    context_parallel_rank: int
    context_parallel_group: Any
    gradient_average_group: Any
    gradient_average_size: int


def initialize_r4c_cp4() -> Gen3cTrainingTopology:
    """Поднять прежнюю TP1/PP1/DP1/CP4 схему и один раз захватить группы."""
    distributed_runtime = importlib.import_module("cosmos_predict1.utils.distributed")
    parallel_state = importlib.import_module("megatron.core.parallel_state")
    torch = importlib.import_module("torch")

    distributed_runtime.init()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        context_parallel_size=4,
    )
    context_group = parallel_state.get_context_parallel_group()
    gradient_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    return Gen3cTrainingTopology(
        world_size=torch.distributed.get_world_size(),
        context_parallel_size=torch.distributed.get_world_size(context_group),
        rank=torch.distributed.get_rank(),
        context_parallel_rank=torch.distributed.get_rank(context_group),
        context_parallel_group=context_group,
        gradient_average_group=gradient_group,
        gradient_average_size=torch.distributed.get_world_size(gradient_group),
    )


__all__ = ["Gen3cTrainingTopology", "initialize_r4c_cp4"]

"""Один baseline R4c LoRA update и отдельное измерение validation."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from novel_view.training.gen3c.data import R4cDataCursor, R4cTrainingBatch
from novel_view.training.gen3c.lora.objective import (
    forward_r4c_edm,
    prepare_r4c_condition,
    sample_r4c_noise,
)
from novel_view.training.gen3c.lora.spec import R4C_GRADIENT_CLIP_NORM
from novel_view.training.gen3c.topology import Gen3cTrainingTopology


@dataclass(frozen=True, slots=True)
class TrainingStepRecord:
    """Отделённые от графа измерения одного завершённого update."""

    sample_id: str
    next_cursor: R4cDataCursor
    loss: float
    gradient_norm: float
    learning_rate: float
    sigma: tuple[float, ...]
    condition_sigma: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ValidationRecordV1:
    """Средний CP-global EDM loss одного validation-прохода."""

    loss: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class TrainingUpdateMeasurements:
    """Два числа одного завершённого LoRA optimizer update."""

    gradient_norm: float
    learning_rate: float


def run_training_step_v1(
    *,
    model: Any,
    network: Any,
    optimizer: Any,
    scheduler: Any,
    trainable_parameters: tuple[Any, ...],
    batch: R4cTrainingBatch,
    topology: Gen3cTrainingTopology,
) -> TrainingStepRecord:
    """Выполнить один update в неизменном baseline-порядке."""
    optimizer.zero_grad(set_to_none=True)
    noise = sample_r4c_noise(batch.clean_latent, topology)
    condition = prepare_r4c_condition(
        model,
        batch.prompt_embedding,
        batch.source_latent,
        batch.pose_latent,
    )
    result = forward_r4c_edm(
        model,
        network,
        batch.clean_latent,
        noise,
        condition,
    )
    update = apply_r4c_lora_update(
        loss=result.loss,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_parameters=trainable_parameters,
        topology=topology,
    )

    report_loss = r4c_cp_mean(result.loss.detach(), topology)
    return TrainingStepRecord(
        sample_id=batch.sample_id,
        next_cursor=batch.next_cursor,
        loss=float(report_loss),
        gradient_norm=update.gradient_norm,
        learning_rate=update.learning_rate,
        sigma=_as_float_tuple(noise.sigma),
        condition_sigma=_as_float_tuple(noise.condition_sigma),
    )


def apply_r4c_lora_update(
    *,
    loss: Any,
    optimizer: Any,
    scheduler: Any,
    trainable_parameters: tuple[Any, ...],
    topology: Gen3cTrainingTopology,
) -> TrainingUpdateMeasurements:
    """Apply the shared finite/backward/clip/optimizer/scheduler order."""
    torch = importlib.import_module("torch")
    require_finite_r4c_loss(loss, topology)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        R4C_GRADIENT_CLIP_NORM,
        error_if_nonfinite=True,
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return TrainingUpdateMeasurements(
        gradient_norm=float(gradient_norm.detach()),
        learning_rate=float(optimizer.param_groups[0]["lr"]),
    )


def run_validation_v1(
    *,
    model: Any,
    network: Any,
    batches: Iterable[R4cTrainingBatch],
    topology: Gen3cTrainingTopology,
) -> ValidationRecordV1:
    """Измерить objective без update; seed/mode/RNG принадлежат epoch loop."""
    torch = importlib.import_module("torch")
    losses: list[float] = []
    with torch.no_grad():
        for batch in batches:
            noise = sample_r4c_noise(batch.clean_latent, topology)
            condition = prepare_r4c_condition(
                model,
                batch.prompt_embedding,
                batch.source_latent,
                batch.pose_latent,
            )
            result = forward_r4c_edm(
                model,
                network,
                batch.clean_latent,
                noise,
                condition,
            )
            require_finite_r4c_loss(result.loss, topology)
            losses.append(float(r4c_cp_mean(result.loss.detach(), topology)))
    return ValidationRecordV1(loss=sum(losses) / len(losses), sample_count=len(losses))


def require_finite_r4c_loss(loss: Any, topology: Gen3cTrainingTopology) -> None:
    torch = importlib.import_module("torch")
    finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
    if topology.context_parallel_group is not None:
        distributed = importlib.import_module("torch.distributed")
        distributed.all_reduce(
            finite,
            op=distributed.ReduceOp.MIN,
            group=topology.context_parallel_group,
        )
    if not bool(finite.item()):
        raise FloatingPointError("Gen3C loss is non-finite on at least one CP rank")


def r4c_cp_mean(value: Any, topology: Gen3cTrainingTopology) -> Any:
    if topology.context_parallel_group is None:
        return value
    distributed = importlib.import_module("torch.distributed")
    value = value.clone()
    distributed.all_reduce(value, group=topology.context_parallel_group)
    return value / topology.context_parallel_size


def _as_float_tuple(value: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().float().cpu().flatten())


__all__ = [
    "TrainingStepRecord",
    "TrainingUpdateMeasurements",
    "ValidationRecordV1",
    "apply_r4c_lora_update",
    "r4c_cp_mean",
    "require_finite_r4c_loss",
    "run_training_step_v1",
    "run_validation_v1",
]

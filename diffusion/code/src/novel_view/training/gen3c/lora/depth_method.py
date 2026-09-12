"""Concrete experimental R4c LoRA loop with frozen latent-depth readout."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novel_view.training.gen3c.data import (
    R4cDataCursor,
    R4cPreparedItem,
    R4cTrainingBatch,
    iter_r4c_training_batches,
    r4c_data_cursor,
)
from novel_view.training.gen3c.lora.checkpoint import (
    TrainingStateV1,
    capture_rank_rng_state,
    gather_rank_rng_states,
)
from novel_view.training.gen3c.lora.depth import (
    DepthGrid,
    LatentDepthObserver,
    R4cDepthItem,
    fit_depth_observer,
    read_r4c_depth_grid,
)
from novel_view.training.gen3c.lora.depth_checkpoint import (
    LoadedCheckpointV2,
    TrainingStateV2,
    restore_depth_observer_v2,
    restore_lora_v2,
    restore_training_state_v2,
    save_checkpoint_v2,
)
from novel_view.training.gen3c.lora.depth_objective import depth_loss_from_edm
from novel_view.training.gen3c.lora.depth_record import (
    CheckpointRecordV2,
    write_checkpoint_record_v2,
)
from novel_view.training.gen3c.lora.depth_spec import (
    DepthObjectiveV2,
    DepthObserverFitIdentity,
    R4cLoraDepthSpec,
)
from novel_view.training.gen3c.lora.method import (
    R4cLoraMethod,
    build_r4c_lora_method,
    build_r4c_lora_model,
    build_r4c_training_identity,
    checkpoint_barrier,
)
from novel_view.training.gen3c.lora.objective import (
    forward_r4c_edm,
    prepare_r4c_condition,
    sample_r4c_noise,
)
from novel_view.training.gen3c.lora.step import (
    apply_r4c_lora_update,
    r4c_cp_mean,
    require_finite_r4c_loss,
)
from novel_view.training.gen3c.topology import Gen3cTrainingTopology


@dataclass(frozen=True, slots=True)
class TrainingStepRecordV2:
    """Отделённые измерения одного экспериментального v2 update."""

    sample_id: str
    next_cursor: R4cDataCursor
    edm_loss: float
    depth_loss: float
    total_loss: float
    depth_valid_count: int
    gradient_norm: float
    learning_rate: float
    sigma: tuple[float, ...]
    condition_sigma: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ValidationRecordV2:
    """Item-balanced CP-global validation measurements."""

    edm_loss: float
    depth_loss: float
    total_loss: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class TrainingProgressV2:
    """One completed v2 update and optional epoch checkpoint facts."""

    completed_step: int
    step: TrainingStepRecordV2
    validation: ValidationRecordV2 | None
    checkpoint: Path | None
    checkpoint_record: Path | None


def run_training_step_v2(
    *,
    model: Any,
    network: Any,
    optimizer: Any,
    scheduler: Any,
    trainable_parameters: tuple[Any, ...],
    batch: R4cTrainingBatch,
    depth: DepthGrid,
    observer: LatentDepthObserver,
    depth_weight: float,
    topology: Gen3cTrainingTopology,
) -> TrainingStepRecordV2:
    """Run the v1 forward, add depth, then perform one shared LoRA update."""
    optimizer.zero_grad(set_to_none=True)
    noise = sample_r4c_noise(batch.clean_latent, topology)
    condition = prepare_r4c_condition(
        model,
        batch.prompt_embedding,
        batch.source_latent,
        batch.pose_latent,
    )
    edm = forward_r4c_edm(model, network, batch.clean_latent, noise, condition)
    depth_result = depth_loss_from_edm(edm, depth, observer, topology)
    total = edm.loss + depth_weight * depth_result.backward_loss
    update = apply_r4c_lora_update(
        loss=total,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_parameters=trainable_parameters,
        topology=topology,
    )
    edm_loss = float(r4c_cp_mean(edm.loss.detach(), topology))
    depth_loss = float(depth_result.global_loss)
    return TrainingStepRecordV2(
        sample_id=batch.sample_id,
        next_cursor=batch.next_cursor,
        edm_loss=edm_loss,
        depth_loss=depth_loss,
        total_loss=edm_loss + depth_weight * depth_loss,
        depth_valid_count=depth_result.global_valid_count,
        gradient_norm=update.gradient_norm,
        learning_rate=update.learning_rate,
        sigma=_as_float_tuple(noise.sigma),
        condition_sigma=_as_float_tuple(noise.condition_sigma),
    )


def run_validation_v2(
    *,
    model: Any,
    network: Any,
    batches: Iterable[tuple[R4cTrainingBatch, DepthGrid]],
    observer: LatentDepthObserver,
    depth_weight: float,
    topology: Gen3cTrainingTopology,
) -> ValidationRecordV2:
    """Measure each CP-global item first, then average item measurements."""
    torch = importlib.import_module("torch")
    edm_losses: list[float] = []
    depth_losses: list[float] = []
    with torch.no_grad():
        for batch, depth in batches:
            noise = sample_r4c_noise(batch.clean_latent, topology)
            condition = prepare_r4c_condition(
                model,
                batch.prompt_embedding,
                batch.source_latent,
                batch.pose_latent,
            )
            edm = forward_r4c_edm(
                model,
                network,
                batch.clean_latent,
                noise,
                condition,
            )
            depth_result = depth_loss_from_edm(edm, depth, observer, topology)
            require_finite_r4c_loss(
                edm.loss + depth_weight * depth_result.backward_loss,
                topology,
            )
            edm_losses.append(float(r4c_cp_mean(edm.loss.detach(), topology)))
            depth_losses.append(float(depth_result.global_loss))
    edm_loss = math.fsum(edm_losses) / len(edm_losses)
    depth_loss = math.fsum(depth_losses) / len(depth_losses)
    return ValidationRecordV2(
        edm_loss,
        depth_loss,
        edm_loss + depth_weight * depth_loss,
        len(edm_losses),
    )


def run_training_v2(
    *,
    spec: R4cLoraDepthSpec,
    base_checkpoint: Path,
    prepared_record_reference: str,
    artifact_root: Path,
    training_items: tuple[R4cDepthItem, ...],
    validation_items: tuple[R4cDepthItem, ...],
    output: Path,
    source_attempt: str,
    resume: LoadedCheckpointV2 | None,
    topology: Gen3cTrainingTopology,
    device: object,
) -> Iterator[TrainingProgressV2]:
    """Fit or restore one observer, then run the explicit v2 epoch order."""
    if resume is None:
        observer = LatentDepthObserver()
        fit_identity = _observer_fit_identity(training_items, validation_items, spec)
        fit_result = fit_depth_observer(
            observer,
            training_items,
            validation_items,
            spec.observer,
            topology,
            artifact_root=artifact_root,
            device=device,
        )
        objective = DepthObjectiveV2(spec.depth_weight)
        observer_lineage = source_attempt
    else:
        observer = restore_depth_observer_v2(resume, device=device)
        fit_identity = resume.observer_fit_identity
        fit_result = resume.observer_fit_result
        objective = resume.objective
        observer_lineage = resume.observer_lineage

    model = build_r4c_lora_model(upstream_root=None, base_checkpoint=base_checkpoint)
    if resume is not None:
        restore_lora_v2(resume, model.model)
    method = build_r4c_lora_method(model, topology)
    prepared_training = tuple(item.prepared for item in training_items)
    prepared_validation = tuple(item.prepared for item in validation_items)
    if resume is None:
        completed_steps = 0
        cursor = r4c_data_cursor(prepared_training, training_seed=spec.training_seed)
    else:
        restored = restore_training_state_v2(
            resume,
            method,
            global_rank=topology.rank,
        )
        completed_steps = restored.next_step
        cursor = restored.data_cursor

    batches = iter_r4c_training_batches(
        prepared_training,
        cursor,
        artifact_root=artifact_root,
        device=device,
        context_parallel_group=topology.context_parallel_group,
    )
    depths = {item.prepared.sample_id: item for item in training_items}
    identity = build_r4c_training_identity(
        base_checkpoint=spec.base_checkpoint,
        prepared_record_reference=prepared_record_reference,
        training_items=prepared_training,
        validation_items=prepared_validation,
        training_seed=spec.training_seed,
        validation_seed=spec.validation_seed,
        epochs=spec.epochs,
    )
    final_step = spec.epochs * len(training_items)

    for completed_step in range(completed_steps + 1, final_step + 1):
        batch = next(batches)
        step = run_training_step_v2(
            model=method.model,
            network=method.network,
            optimizer=method.optimizer,
            scheduler=method.scheduler,
            trainable_parameters=method.trainable_parameters,
            batch=batch,
            depth=_depth_on_device(
                read_r4c_depth_grid(
                    depths[batch.sample_id], artifact_root=artifact_root
                ),
                device,
            ),
            observer=observer,
            depth_weight=objective.depth_weight,
            topology=topology,
        )
        validation = None
        checkpoint_path = None
        record_path = None
        if step.next_cursor.offset == 0:
            validation = _run_validation_epoch_v2(
                spec=spec,
                method=method,
                validation_items=validation_items,
                artifact_root=artifact_root,
                observer=observer,
                depth_weight=objective.depth_weight,
                topology=topology,
                device=device,
            )
            checkpoint_path = output / f"step-{completed_step:09d}.pt"
            record_path = checkpoint_path.with_suffix(".json")
            rng_states = gather_rank_rng_states(
                capture_rank_rng_state(topology.rank),
                topology,
            )
            if topology.rank == 0:
                baseline_state = TrainingStateV1(
                    method.model.model,
                    method.optimizer,
                    method.scheduler,
                    completed_step,
                    step.next_cursor,
                    topology,
                    rng_states,
                )
                save_checkpoint_v2(
                    checkpoint_path,
                    TrainingStateV2(
                        baseline_state,
                        objective,
                        observer,
                        fit_identity,
                        fit_result,
                        observer_lineage,
                    ),
                )
                write_checkpoint_record_v2(
                    record_path,
                    CheckpointRecordV2(
                        objective,
                        identity,
                        fit_identity,
                        fit_result,
                        observer_lineage,
                        completed_step,
                        validation.edm_loss,
                        validation.depth_loss,
                        validation.total_loss,
                        source_attempt,
                        checkpoint_path.name,
                    ),
                )
            checkpoint_barrier(topology)

        yield TrainingProgressV2(
            completed_step,
            step,
            validation,
            checkpoint_path,
            record_path,
        )


def _run_validation_epoch_v2(
    *,
    spec: R4cLoraDepthSpec,
    method: R4cLoraMethod,
    validation_items: tuple[R4cDepthItem, ...],
    artifact_root: Path,
    observer: LatentDepthObserver,
    depth_weight: float,
    topology: Gen3cTrainingTopology,
    device: object,
) -> ValidationRecordV2:
    torch = importlib.import_module("torch")
    cpu_rng = torch.get_rng_state()
    cuda_device = torch.cuda.current_device()
    cuda_rng = torch.cuda.get_rng_state(cuda_device)
    was_training = method.model.model.training
    prepared = tuple(item.prepared for item in validation_items)
    depths = {item.prepared.sample_id: item for item in validation_items}
    try:
        torch.manual_seed(spec.validation_seed)
        torch.cuda.manual_seed(spec.validation_seed)
        method.model.model.eval()
        cursor = r4c_data_cursor(prepared, training_seed=spec.validation_seed)
        batches = iter_r4c_training_batches(
            prepared,
            cursor,
            artifact_root=artifact_root,
            device=device,
            context_parallel_group=topology.context_parallel_group,
        )
        return run_validation_v2(
            model=method.model,
            network=method.network,
            batches=(
                (
                    batch,
                    _depth_on_device(
                        read_r4c_depth_grid(
                            depths[batch.sample_id], artifact_root=artifact_root
                        ),
                        device,
                    ),
                )
                for batch in (next(batches) for _ in prepared)
            ),
            observer=observer,
            depth_weight=depth_weight,
            topology=topology,
        )
    finally:
        method.model.model.train(was_training)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, cuda_device)


def _observer_fit_identity(
    training_items: tuple[R4cDepthItem, ...],
    validation_items: tuple[R4cDepthItem, ...],
    spec: R4cLoraDepthSpec,
) -> DepthObserverFitIdentity:
    return DepthObserverFitIdentity(
        tuple(item.prepared.sample_id for item in training_items),
        tuple(item.prepared.segment_id for item in training_items),
        tuple(item.prepared.sample_id for item in validation_items),
        tuple(item.prepared.segment_id for item in validation_items),
        spec.observer,
    )


def _depth_on_device(depth: DepthGrid, device: object) -> DepthGrid:
    return DepthGrid(depth.values.to(device=device), depth.valid.to(device=device))


def _as_float_tuple(value: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().float().cpu().flatten())


__all__ = [
    "TrainingProgressV2",
    "TrainingStepRecordV2",
    "ValidationRecordV2",
    "run_training_step_v2",
    "run_training_v2",
    "run_validation_v2",
]

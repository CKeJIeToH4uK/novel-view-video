"""Конкретная сборка baseline R4c LoRA без общего trainer."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from novel_view.models.gen3c import lora_weights as model_lora
from novel_view.training.gen3c.data import (
    R4cPreparedItem,
    iter_r4c_training_batches,
    r4c_data_cursor,
)
from novel_view.training.gen3c.lora.checkpoint import (
    LoadedCheckpointV1,
    TrainingStateV1,
    capture_rank_rng_state,
    gather_rank_rng_states,
    restore_lora_v1,
    restore_training_state_v1,
    save_checkpoint_v1,
)
from novel_view.training.gen3c.lora.record import (
    AbTrainingIdentityV1,
    CheckpointRecordV1,
    write_checkpoint_record_v1,
)
from novel_view.training.gen3c.lora.spec import (
    R4C_BETAS,
    R4C_EPSILON,
    R4C_GRADIENT_CLIP_NORM,
    R4C_LEARNING_RATE,
    R4C_LORA_SPEC,
    R4C_WEIGHT_DECAY,
    R4cLoraEdmSpec,
)
from novel_view.training.gen3c.lora.step import (
    TrainingStepRecord,
    run_training_step_v1,
    run_validation_v1,
)
from novel_view.training.gen3c.topology import Gen3cTrainingTopology


@dataclass(slots=True)
class R4cLoraMethod:
    """Живущие объекты одного baseline R4c LoRA worker."""

    model: Any
    network: Any
    optimizer: Any
    scheduler: Any
    trainable_parameters: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class TrainingProgressV1:
    """One completed update and any epoch-boundary checkpoint facts."""

    completed_step: int
    step: TrainingStepRecord
    validation_loss: float | None
    checkpoint: Path | None
    checkpoint_record: Path | None


def build_r4c_training_identity(
    *,
    base_checkpoint: Path,
    prepared_record_reference: str,
    training_items: tuple[R4cPreparedItem, ...],
    validation_items: tuple[R4cPreparedItem, ...],
    training_seed: int,
    validation_seed: int,
    epochs: int,
) -> AbTrainingIdentityV1:
    """Build the exact A/B facts shared by the two concrete LoRA loops."""
    return AbTrainingIdentityV1(
        base_checkpoint=str(base_checkpoint),
        prepared_record=prepared_record_reference,
        training_sample_ids=tuple(item.sample_id for item in training_items),
        training_segment_ids=tuple(item.segment_id for item in training_items),
        validation_sample_ids=tuple(item.sample_id for item in validation_items),
        validation_segment_ids=tuple(item.segment_id for item in validation_items),
        training_seed=training_seed,
        validation_seed=validation_seed,
        epochs=epochs,
    )


def build_r4c_lora_model(
    *,
    upstream_root: Path | None,
    base_checkpoint: Path,
) -> Any:
    """Создать точный rank-8/scale-1 layout в 28 Gen3C-блоках."""
    return model_lora.build_gen3c_lora(
        upstream_root,
        base_checkpoint,
        R4C_LORA_SPEC,
    )


def build_r4c_lora_method(
    model: Any,
    topology: Gen3cTrainingTopology,
) -> R4cLoraMethod:
    """Включить CP/recompute/DDP и создать точный baseline optimizer."""
    model.net.enable_context_parallel(topology.context_parallel_group)
    _apply_activation_recompute(model.net)
    model.model.train()

    distributed_runtime = importlib.import_module("cosmos_predict1.utils.distributed")
    config_module = importlib.import_module("cosmos_predict1.utils.config")
    network = distributed_runtime.parallel_model_wrapper(
        config_module.DDPConfig(),
        model.net,
    )
    lora_parameters = model_lora.gen3c_lora_state_by_name(
        model.model,
        keep_vars=True,
    )
    trainable = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in model.model.named_parameters()
                if parameter.requires_grad
            ),
            key=lambda item: item[0],
        )
    )
    if not lora_parameters or tuple(lora_parameters) != tuple(
        name for name, _ in trainable
    ):
        raise RuntimeError("R4c training must expose only LoRA parameters")
    trainable_parameters = tuple(parameter for _, parameter in trainable)

    optimizer_module = importlib.import_module(
        "cosmos_predict1.diffusion.training.utils.optim_instantiate"
    )
    optimizer = optimizer_module.get_base_optimizer(
        model.model,
        lr=R4C_LEARNING_RATE,
        weight_decay=R4C_WEIGHT_DECAY,
        betas=list(R4C_BETAS),
        eps=R4C_EPSILON,
        optim_type="fusedadam",
        sharding=False,
        master_weights=True,
        capturable=True,
    )
    scheduler_module = importlib.import_module("torch.optim.lr_scheduler")
    scheduler = scheduler_module.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    return R4cLoraMethod(
        model=model,
        network=network,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_parameters=trainable_parameters,
    )


def run_training_v1(
    *,
    spec: R4cLoraEdmSpec,
    base_checkpoint: Path,
    prepared_record_reference: str,
    artifact_root: Path,
    training_items: tuple[R4cPreparedItem, ...],
    validation_items: tuple[R4cPreparedItem, ...],
    output: Path,
    source_attempt: str,
    resume: LoadedCheckpointV1 | None,
    topology: Gen3cTrainingTopology,
    device: object,
) -> Iterator[TrainingProgressV1]:
    """Run the concrete baseline epoch order and write one checkpoint per epoch."""
    model = build_r4c_lora_model(
        upstream_root=None,
        base_checkpoint=base_checkpoint,
    )
    if resume is not None:
        restore_lora_v1(resume, model.model)
    method = build_r4c_lora_method(model, topology)

    if resume is None:
        completed_steps = 0
        cursor = r4c_data_cursor(
            training_items,
            training_seed=spec.training_seed,
        )
    else:
        restored = restore_training_state_v1(
            resume,
            method,
            global_rank=topology.rank,
        )
        completed_steps = restored.next_step
        cursor = restored.data_cursor

    batches = iter_r4c_training_batches(
        training_items,
        cursor,
        artifact_root=artifact_root,
        device=device,
        context_parallel_group=topology.context_parallel_group,
    )
    identity = build_r4c_training_identity(
        base_checkpoint=spec.base_checkpoint,
        prepared_record_reference=prepared_record_reference,
        training_items=training_items,
        validation_items=validation_items,
        training_seed=spec.training_seed,
        validation_seed=spec.validation_seed,
        epochs=spec.epochs,
    )
    final_step = spec.epochs * len(training_items)

    for completed_step in range(completed_steps + 1, final_step + 1):
        step = run_training_step_v1(
            model=method.model,
            network=method.network,
            optimizer=method.optimizer,
            scheduler=method.scheduler,
            trainable_parameters=method.trainable_parameters,
            batch=next(batches),
            topology=topology,
        )
        validation_loss = None
        checkpoint_path = None
        record_path = None
        if step.next_cursor.offset == 0:
            validation_loss = _run_validation_epoch(
                spec=spec,
                method=method,
                validation_items=validation_items,
                artifact_root=artifact_root,
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
                save_checkpoint_v1(
                    checkpoint_path,
                    TrainingStateV1(
                        model=method.model.model,
                        optimizer=method.optimizer,
                        scheduler=method.scheduler,
                        next_step=completed_step,
                        data_cursor=step.next_cursor,
                        topology=topology,
                        rank_rng_states=rng_states,
                    ),
                )
                write_checkpoint_record_v1(
                    record_path,
                    CheckpointRecordV1(
                        identity=identity,
                        completed_step=completed_step,
                        validation_loss=validation_loss,
                        source_attempt=source_attempt,
                        checkpoint=checkpoint_path.name,
                    ),
                )
            checkpoint_barrier(topology)

        yield TrainingProgressV1(
            completed_step,
            step,
            validation_loss,
            checkpoint_path,
            record_path,
        )


def _run_validation_epoch(
    *,
    spec: R4cLoraEdmSpec,
    method: R4cLoraMethod,
    validation_items: tuple[R4cPreparedItem, ...],
    artifact_root: Path,
    topology: Gen3cTrainingTopology,
    device: object,
) -> float:
    torch = importlib.import_module("torch")
    cpu_rng = torch.get_rng_state()
    cuda_device = torch.cuda.current_device()
    cuda_rng = torch.cuda.get_rng_state(cuda_device)
    was_training = method.model.model.training
    try:
        torch.manual_seed(spec.validation_seed)
        torch.cuda.manual_seed(spec.validation_seed)
        method.model.model.eval()
        cursor = r4c_data_cursor(
            validation_items,
            training_seed=spec.validation_seed,
        )
        batches = iter_r4c_training_batches(
            validation_items,
            cursor,
            artifact_root=artifact_root,
            device=device,
            context_parallel_group=topology.context_parallel_group,
        )
        validation = run_validation_v1(
            model=method.model,
            network=method.network,
            batches=(next(batches) for _ in validation_items),
            topology=topology,
        )
        return validation.loss
    finally:
        method.model.model.train(was_training)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, cuda_device)


def checkpoint_barrier(topology: Gen3cTrainingTopology) -> None:
    if topology.context_parallel_group is not None:
        importlib.import_module("torch.distributed").barrier(
            group=topology.context_parallel_group
        )


def _apply_activation_recompute(network: Any) -> None:
    block_types = _activation_block_types(network)
    if not any(isinstance(module, block_types) for module in network.modules()):
        raise RuntimeError("Gen3C network has no activation-recompute blocks")
    checkpoint = importlib.import_module(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
    )
    transformer_engine = importlib.import_module(
        "transformer_engine.pytorch.distributed"
    )
    wrapper = partial(
        checkpoint.checkpoint_wrapper,
        checkpoint_impl=checkpoint.CheckpointImpl.REENTRANT,
        checkpoint_fn=transformer_engine.checkpoint,
    )
    checkpoint.apply_activation_checkpointing(
        network,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, block_types),
    )


def _activation_block_types(network: Any) -> tuple[type, ...]:
    declared = getattr(network, "fsdp_wrap_block_cls", ())
    block_types = (
        tuple(declared) if isinstance(declared, (list, tuple, set)) else (declared,)
    )
    block_types = tuple(item for item in block_types if isinstance(item, type))
    if block_types:
        return block_types
    return tuple(dict.fromkeys(type(block) for block in network.blocks.values()))


__all__ = [
    "R4C_BETAS",
    "R4C_EPSILON",
    "R4C_GRADIENT_CLIP_NORM",
    "R4C_LEARNING_RATE",
    "R4C_LORA_SPEC",
    "R4C_WEIGHT_DECAY",
    "R4cLoraMethod",
    "TrainingProgressV1",
    "build_r4c_training_identity",
    "build_r4c_lora_method",
    "build_r4c_lora_model",
    "checkpoint_barrier",
    "run_training_v1",
]

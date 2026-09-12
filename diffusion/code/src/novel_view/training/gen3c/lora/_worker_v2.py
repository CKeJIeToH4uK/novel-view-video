"""Private CP4 worker for experimental R4c LoRA LiDAR-depth training."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

from novel_view.preparation.waymo_ddw.record import read_prepared_record
from novel_view.training.gen3c.data import load_r4c_training_split
from novel_view.training.gen3c.lora.depth import resolve_r4c_depth_items
from novel_view.training.gen3c.lora.depth_checkpoint import load_checkpoint_v2
from novel_view.training.gen3c.lora.depth_method import (
    TrainingProgressV2,
    run_training_v2,
)
from novel_view.training.gen3c.lora.depth_spec import (
    DepthObserverFitSpec,
    R4cLoraDepthSpec,
)
from novel_view.training.gen3c.topology import initialize_r4c_cp4


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prepared-record", type=Path, required=True)
    parser.add_argument("--prepared-record-reference", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-attempt", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--validation-seed", type=int, required=True)
    parser.add_argument("--depth-weight", type=float, required=True)
    parser.add_argument("--observer-init-seed", type=int, required=True)
    parser.add_argument("--observer-fit-order-seed", type=int, required=True)
    parser.add_argument("--observer-fit-epochs", type=int, required=True)
    parser.add_argument("--observer-items-per-step", type=int, required=True)
    parser.add_argument("--observer-optimizer", required=True)
    parser.add_argument("--observer-learning-rate", type=float, required=True)
    parser.add_argument("--observer-beta1", type=float, required=True)
    parser.add_argument("--observer-beta2", type=float, required=True)
    parser.add_argument("--observer-epsilon", type=float, required=True)
    parser.add_argument("--observer-weight-decay", type=float, required=True)
    parser.add_argument("--observer-scheduler", required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def _run(arguments: argparse.Namespace) -> None:
    prepared = read_prepared_record(arguments.prepared_record)
    split = load_r4c_training_split(arguments.split)
    training_items, validation_items = resolve_r4c_depth_items(prepared, split)

    torch = importlib.import_module("torch")
    local_rank = int(os.environ["LOCAL_RANK"])
    _set_rank_local_compiler_caches(local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    parallel_state = importlib.import_module("megatron.core.parallel_state")
    started = time.monotonic()
    try:
        topology = initialize_r4c_cp4()
        resume = (
            None
            if arguments.resume_checkpoint is None
            else load_checkpoint_v2(arguments.resume_checkpoint)
        )
        importlib.import_module("cosmos_predict1.utils.misc").set_random_seed(
            arguments.training_seed,
            by_rank=False,
        )
        arguments.output.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier(group=topology.context_parallel_group)
        spec = _training_spec(arguments)
        for progress in run_training_v2(
            spec=spec,
            base_checkpoint=arguments.base_checkpoint,
            prepared_record_reference=arguments.prepared_record_reference,
            artifact_root=arguments.prepared_record.parent,
            training_items=training_items,
            validation_items=validation_items,
            output=arguments.output,
            source_attempt=arguments.source_attempt,
            resume=resume,
            topology=topology,
            device=device,
        ):
            if topology.rank == 0:
                print(json.dumps(_progress_event(progress)), flush=True)
        torch.cuda.synchronize()
        if topology.rank == 0:
            print(
                json.dumps(
                    {
                        "event": "training_complete",
                        "completed_step": spec.epochs * len(training_items),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    finally:
        try:
            parallel_state.destroy_model_parallel()
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


def _training_spec(arguments: argparse.Namespace) -> R4cLoraDepthSpec:
    observer = DepthObserverFitSpec(
        init_seed=arguments.observer_init_seed,
        fit_order_seed=arguments.observer_fit_order_seed,
        fit_epochs=arguments.observer_fit_epochs,
        items_per_step=arguments.observer_items_per_step,
        optimizer_name=arguments.observer_optimizer,
        learning_rate=arguments.observer_learning_rate,
        betas=(arguments.observer_beta1, arguments.observer_beta2),
        epsilon=arguments.observer_epsilon,
        weight_decay=arguments.observer_weight_decay,
        scheduler=arguments.observer_scheduler,
    )
    return R4cLoraDepthSpec(
        base_checkpoint=arguments.base_checkpoint_reference,
        epochs=arguments.epochs,
        training_seed=arguments.training_seed,
        validation_seed=arguments.validation_seed,
        depth_weight=arguments.depth_weight,
        observer=observer,
    )


def _progress_event(progress: TrainingProgressV2) -> dict[str, object]:
    step = progress.step
    event: dict[str, object] = {
        "event": "training_step",
        "completed_step": progress.completed_step,
        "sample_id": step.sample_id,
        "edm_loss": step.edm_loss,
        "depth_loss": step.depth_loss,
        "total_loss": step.total_loss,
        "depth_valid_count": step.depth_valid_count,
        "gradient_norm": step.gradient_norm,
        "learning_rate": step.learning_rate,
    }
    if progress.validation is not None:
        event.update(
            {
                "validation_edm_loss": progress.validation.edm_loss,
                "validation_depth_loss": progress.validation.depth_loss,
                "validation_total_loss": progress.validation.total_loss,
                "checkpoint": str(progress.checkpoint),
                "checkpoint_record": str(progress.checkpoint_record),
            }
        )
    return event


def _set_rank_local_compiler_caches(local_rank: int) -> None:
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        root = os.environ.get(name)
        if root:
            cache = Path(root) / f"rank-{local_rank}"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(cache)


if __name__ == "__main__":
    _run(_parse_arguments())

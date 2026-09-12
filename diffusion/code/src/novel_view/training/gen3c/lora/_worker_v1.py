"""Private CP4 worker for the supported R4c LoRA EDM baseline."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

from novel_view.preparation.waymo_ddw.record import read_prepared_record
from novel_view.training.gen3c.data import (
    load_r4c_training_split,
    resolve_r4c_items,
)
from novel_view.training.gen3c.lora.checkpoint import load_checkpoint_v1
from novel_view.training.gen3c.lora.method import TrainingProgressV1, run_training_v1
from novel_view.training.gen3c.lora.spec import R4cLoraEdmSpec
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
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--legacy-index-map", type=Path)
    return parser.parse_args()


def _run(arguments: argparse.Namespace) -> None:
    prepared = read_prepared_record(arguments.prepared_record)
    split = load_r4c_training_split(arguments.split)
    training_items, validation_items = resolve_r4c_items(prepared, split)

    torch = importlib.import_module("torch")
    local_rank = int(os.environ["LOCAL_RANK"])
    _set_rank_local_compiler_caches(local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    parallel_state = importlib.import_module("megatron.core.parallel_state")
    model_parallel_attempted = True
    started = time.monotonic()
    try:
        topology = initialize_r4c_cp4()
        resume = (
            None
            if arguments.resume_checkpoint is None
            else load_checkpoint_v1(
                arguments.resume_checkpoint,
                legacy_index_map=arguments.legacy_index_map,
            )
        )
        importlib.import_module("cosmos_predict1.utils.misc").set_random_seed(
            arguments.training_seed,
            by_rank=False,
        )
        arguments.output.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier(group=topology.context_parallel_group)
        spec = R4cLoraEdmSpec(
            base_checkpoint=arguments.base_checkpoint_reference,
            epochs=arguments.epochs,
            training_seed=arguments.training_seed,
            validation_seed=arguments.validation_seed,
        )
        for progress in run_training_v1(
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
        if model_parallel_attempted:
            try:
                parallel_state.destroy_model_parallel()
            finally:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()


def _progress_event(progress: TrainingProgressV1) -> dict[str, object]:
    step = progress.step
    event = {
        "event": "training_step",
        "completed_step": progress.completed_step,
        "sample_id": step.sample_id,
        "loss": step.loss,
        "gradient_norm": step.gradient_norm,
        "learning_rate": step.learning_rate,
    }
    if progress.validation_loss is not None:
        event.update(
            {
                "validation_loss": progress.validation_loss,
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

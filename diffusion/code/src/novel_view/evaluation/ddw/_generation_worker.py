"""Private CP4 worker for matched base/v1/v2 DDW latent generation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from novel_view.evaluation.ddw.samples import resolve_ddw_evaluation_samples
from novel_view.evaluation.ddw.sampling import (
    prepare_r4c_sampling,
    sample_r4c_variants,
    write_temporary_latent,
)
from novel_view.evaluation.ddw.spec import (
    DdwSamplingSpec,
    require_binary_record_facts,
    require_matched_checkpoint_records,
)
from novel_view.models.gen3c import lora_weights as model_lora
from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT
from novel_view.preparation.waymo_ddw import artifacts
from novel_view.preparation.waymo_ddw.record import read_prepared_record
from novel_view.training.gen3c.data import (
    load_r4c_training_split,
    raise_r4c_source_errors,
)
from novel_view.training.gen3c.lora.checkpoint import load_checkpoint_v1
from novel_view.training.gen3c.lora.depth_checkpoint import load_checkpoint_v2
from novel_view.training.gen3c.lora.depth_record import read_checkpoint_record_v2
from novel_view.training.gen3c.lora.record import read_checkpoint_record_v1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prepared-record", type=Path, required=True)
    parser.add_argument("--prepared-record-reference", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--v1-record", type=Path, required=True)
    parser.add_argument("--v2-record", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint-reference", required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--comparison-step", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--sampling-steps", type=int, required=True)
    parser.add_argument("--sampling-guidance", type=float, required=True)
    parser.add_argument("--condition-augment-sigma", type=float, required=True)
    return parser.parse_args()


def _run(arguments: argparse.Namespace) -> None:
    prepared = read_prepared_record(arguments.prepared_record)
    split = load_r4c_training_split(arguments.split)
    training_items, evaluation_samples = resolve_ddw_evaluation_samples(
        prepared,
        split,
    )
    validation_items = tuple(sample.model_input for sample in evaluation_samples)
    v1_record = read_checkpoint_record_v1(arguments.v1_record)
    v2_record = read_checkpoint_record_v2(arguments.v2_record)
    require_matched_checkpoint_records(
        training_items,
        validation_items,
        v1_record,
        v2_record,
        prepared_record_reference=arguments.prepared_record_reference,
        base_checkpoint_reference=arguments.base_checkpoint_reference,
        comparison_step=arguments.comparison_step,
    )
    v1_checkpoint = load_checkpoint_v1(
        arguments.v1_record.parent / v1_record.checkpoint
    )
    v2_checkpoint = load_checkpoint_v2(
        arguments.v2_record.parent / v2_record.checkpoint
    )
    require_binary_record_facts(
        v1_checkpoint,
        v2_checkpoint,
        v1_record,
        v2_record,
    )

    sampling = DdwSamplingSpec(
        arguments.sampling_seed,
        arguments.sampling_steps,
        arguments.sampling_guidance,
        arguments.condition_augment_sigma,
    )
    torch = importlib.import_module("torch")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    parallel_state = importlib.import_module("megatron.core.parallel_state")
    initialized = False
    started = time.monotonic()
    try:
        importlib.import_module("cosmos_predict1.utils.distributed").init()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            context_parallel_size=4,
        )
        initialized = True
        context_group = parallel_state.get_context_parallel_group()
        ranks = torch.distributed.get_process_group_ranks(context_group)
        source_rank = min(ranks)
        rank = torch.distributed.get_rank()
        importlib.import_module("cosmos_predict1.utils.misc").set_random_seed(
            v1_record.identity.training_seed,
            by_rank=False,
        )
        lora_spec = model_lora.decode_gen3c_lora_spec(
            cast(Mapping[str, Any], v1_checkpoint.adapter["spec"])
        )
        model = model_lora.build_gen3c_lora(
            None,
            arguments.base_checkpoint,
            lora_spec,
        )
        model.cuda()
        model.net.enable_context_parallel(context_group)
        for parameter in model.model.parameters():
            parameter.requires_grad_(False)
        model.model.eval()
        initial_adapter = {
            name: tensor.detach().clone()
            for name, tensor in model_lora.gen3c_lora_state_by_name(model.model).items()
        }
        v1_adapter = cast(Mapping[str, Any], v1_checkpoint.adapter["state_dict"])
        v2_adapter = cast(
            Mapping[str, Any],
            v2_checkpoint.baseline.adapter["state_dict"],
        )
        del v1_checkpoint, v2_checkpoint

        for item_index, sample in enumerate(evaluation_samples):
            source, pose, prompt = _materialize_inputs(
                torch,
                sample.model_input,
                arguments.prepared_record.parent,
                device,
                context_group,
                ranks,
                source_rank,
            )
            state = prepare_r4c_sampling(
                model,
                source,
                pose,
                prompt,
                sampling,
                initial_adapter,
            )
            variants = sample_r4c_variants(
                model,
                state,
                sampling,
                v1_adapter,
                v2_adapter,
            )
            _write_variants(
                torch,
                arguments.latent_root / f"item-{item_index:04d}",
                sample.prepared.sample_id,
                variants,
                context_group,
                source_rank,
                rank,
            )
            if rank == source_rank:
                print(
                    json.dumps(
                        {
                            "event": "ddw_evaluation_generation_item",
                            "sample_id": sample.prepared.sample_id,
                            "item_index": item_index,
                        }
                    ),
                    flush=True,
                )
            del variants, state, source, pose, prompt

        torch.cuda.synchronize()
        if rank == source_rank:
            print(
                json.dumps(
                    {
                        "event": "ddw_evaluation_generation_complete",
                        "item_count": len(evaluation_samples),
                        "comparison_step": arguments.comparison_step,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    finally:
        if initialized:
            try:
                parallel_state.destroy_model_parallel()
            finally:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()


def _materialize_inputs(
    torch: Any,
    item: Any,
    artifact_root: Path,
    device: Any,
    group: Any,
    ranks: list[int],
    source_rank: int,
) -> tuple[Any, Any, Any]:
    rank = torch.distributed.get_rank()
    local_error = None
    if rank == source_rank:
        try:
            base = artifacts.read_base_latents(
                artifact_root / item.base_latent,
                item.sample_id,
            )
            pose_artifact = artifacts.read_pose_latent(
                artifact_root / item.pose_latent,
                item.sample_id,
                item.magnitude_m,
                item.sign,
            )
            prompt_artifact = artifacts.read_empty_prompt(
                artifact_root / item.prompt_embedding
            )
            source, pose, prompt = (
                value.to(device=device)
                for value in (
                    base.source_latent,
                    pose_artifact.pose_latent,
                    prompt_artifact.t5_text_embeddings,
                )
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            source = pose = prompt = None
    else:
        contract = R4C_GEN3C_MODEL_CONTRACT
        source, pose, prompt = (
            torch.empty(shape, dtype=torch.bfloat16, device=device)
            for shape in (
                contract.base_latent_shape,
                contract.pose_latent_shape,
                contract.prompt_embedding_shape,
            )
        )
    raise_r4c_source_errors(
        torch.distributed,
        group,
        ranks,
        local_error,
        operation="evaluation data materialization",
    )
    for tensor in (source, pose, prompt):
        torch.distributed.broadcast(tensor, source_rank, group=group)
    return source, pose, prompt


def _write_variants(
    torch: Any,
    root: Path,
    sample_id: str,
    variants: Any,
    group: Any,
    source_rank: int,
    rank: int,
) -> None:
    error = None
    if rank == source_rank:
        try:
            root.mkdir(parents=True, exist_ok=True)
            for variant, latent in (
                ("base", variants.base),
                ("v1", variants.v1),
                ("v2", variants.v2),
            ):
                write_temporary_latent(
                    root / f"{variant}.pt",
                    sample_id,
                    variant,
                    latent,
                )
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
    status = [error]
    torch.distributed.broadcast_object_list(status, src=source_rank, group=group)
    if status[0] is not None:
        raise RuntimeError(f"DDW evaluation latent write failed: {status[0]}")
    torch.distributed.barrier(group=group)


if __name__ == "__main__":
    _run(_arguments())

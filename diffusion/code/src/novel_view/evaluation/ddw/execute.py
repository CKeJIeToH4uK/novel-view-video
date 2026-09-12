"""Two-process execution order for one matched DDW evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from novel_view.evaluation.ddw.metrics import aggregate_ddw_items
from novel_view.evaluation.ddw.record import (
    DdwEvaluationRecord,
    DdwItemResult,
    EvaluationCheckpoint,
    read_item_result,
    write_evaluation_record,
)
from novel_view.evaluation.ddw.samples import (
    DdwEvaluationSample,
    read_raw_front_target,
    resolve_ddw_evaluation_samples,
)
from novel_view.evaluation.ddw.spec import (
    DdwEvaluationSpec,
    require_matched_checkpoint_records,
)
from novel_view.models.gen3c.environment import gen3c_environment
from novel_view.runtime.context import RuntimeContext
from novel_view.runtime.distributed import start_torchrun
from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import run_process
from novel_view.preparation.waymo_ddw.record import read_prepared_record
from novel_view.training.gen3c.data import load_r4c_training_split
from novel_view.training.gen3c.lora.depth_record import read_checkpoint_record_v2
from novel_view.training.gen3c.lora.depth_spec import (
    R4C_DEPTH_OBJECTIVE_NAME,
    R4C_DEPTH_OBJECTIVE_VERSION,
)
from novel_view.training.gen3c.lora.record import read_checkpoint_record_v1


def run_evaluation_v1(
    spec: DdwEvaluationSpec,
    runtime: RuntimeContext,
) -> Path:
    """Generate once, decode each item, then write one final JSON record."""
    prepared_path = runtime.roots.prepared / spec.input.prepared_record
    split_path = runtime.roots.selections.parent / spec.input.split
    v1_record_path = runtime.roots.runs / spec.input.v1_checkpoint_record
    v2_record_path = runtime.roots.runs / spec.input.v2_checkpoint_record
    prepared = read_prepared_record(prepared_path)
    split = load_r4c_training_split(split_path)
    training_items, samples = resolve_ddw_evaluation_samples(prepared, split)
    validation_items = tuple(sample.model_input for sample in samples)
    v1_record = read_checkpoint_record_v1(v1_record_path)
    v2_record = read_checkpoint_record_v2(v2_record_path)
    require_matched_checkpoint_records(
        training_items,
        validation_items,
        v1_record,
        v2_record,
        prepared_record_reference=spec.input.prepared_record,
        base_checkpoint_reference=spec.recipe.base_checkpoint,
        comparison_step=spec.recipe.comparison_step,
    )

    log_root = runtime.attempt_root / "logs" / "ddw-evaluation"
    video_root = runtime.attempt_root / "videos"
    log_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="ddw-evaluation-",
        dir=runtime.roots.cache,
    ) as temporary:
        scratch = Path(temporary)
        latent_root = scratch / "latents"
        latent_root.mkdir()
        environment = gen3c_environment(runtime, scratch)
        _run_generation(
            spec,
            prepared_path,
            split_path,
            v1_record_path,
            v2_record_path,
            runtime.roots.models / spec.recipe.base_checkpoint,
            latent_root,
            log_root / "generation.log",
            environment,
        )
        item_results = tuple(
            _evaluate_sample(
                spec,
                runtime,
                prepared_path,
                sample,
                item_index,
                latent_root,
                scratch,
                video_root,
                log_root,
                environment,
            )
            for item_index, sample in enumerate(samples)
        )

    record = DdwEvaluationRecord(
        prepared_record=spec.input.prepared_record,
        split=spec.input.split,
        model_contract=spec.recipe.model_contract,
        base_checkpoint=spec.recipe.base_checkpoint,
        tokenizer=spec.recipe.tokenizer,
        depth_checkpoint=spec.recipe.depth_checkpoint,
        comparison_step=spec.recipe.comparison_step,
        sampling=spec.recipe.sampling,
        v1_checkpoint=EvaluationCheckpoint(
            record=spec.input.v1_checkpoint_record,
            checkpoint=str(
                Path(spec.input.v1_checkpoint_record).parent / v1_record.checkpoint
            ),
            source_attempt=v1_record.source_attempt,
            completed_step=v1_record.completed_step,
            objective_name="gen3c-kendall-edm",
            objective_version=1,
            depth_weight=None,
            observer_lineage=None,
        ),
        v2_checkpoint=EvaluationCheckpoint(
            record=spec.input.v2_checkpoint_record,
            checkpoint=str(
                Path(spec.input.v2_checkpoint_record).parent / v2_record.checkpoint
            ),
            source_attempt=v2_record.source_attempt,
            completed_step=v2_record.completed_step,
            objective_name=R4C_DEPTH_OBJECTIVE_NAME,
            objective_version=R4C_DEPTH_OBJECTIVE_VERSION,
            depth_weight=v2_record.objective.depth_weight,
            observer_lineage=v2_record.observer_lineage,
        ),
        items=item_results,
        aggregates=aggregate_ddw_items(tuple(item.metrics for item in item_results)),
    )
    result_path = runtime.attempt_root / "ddw-heldout-point-evaluation.json"
    write_evaluation_record(result_path, record)
    print(
        json.dumps(
            {
                "event": "ddw_evaluation_complete",
                "comparison_step": spec.recipe.comparison_step,
                "item_count": len(item_results),
                "record": str(result_path),
            }
        ),
        flush=True,
    )
    return result_path


def _run_generation(
    spec: DdwEvaluationSpec,
    prepared_path: Path,
    split_path: Path,
    v1_record_path: Path,
    v2_record_path: Path,
    base_checkpoint: Path,
    latent_root: Path,
    log_path: Path,
    environment: dict[str, str],
) -> None:
    worker = Path(__file__).with_name("_generation_worker.py")
    process = start_torchrun(
        GEN3C_PYTHON,
        worker,
        [
            "--prepared-record",
            str(prepared_path),
            "--prepared-record-reference",
            spec.input.prepared_record,
            "--split",
            str(split_path),
            "--v1-record",
            str(v1_record_path),
            "--v2-record",
            str(v2_record_path),
            "--base-checkpoint",
            str(base_checkpoint),
            "--base-checkpoint-reference",
            spec.recipe.base_checkpoint,
            "--latent-root",
            str(latent_root),
            "--comparison-step",
            str(spec.recipe.comparison_step),
            "--sampling-seed",
            str(spec.recipe.sampling.seed),
            "--sampling-steps",
            str(spec.recipe.sampling.steps),
            "--sampling-guidance",
            str(spec.recipe.sampling.guidance),
            "--condition-augment-sigma",
            str(spec.recipe.sampling.condition_augment_sigma),
        ],
        process_count=4,
        log_path=log_path,
        description="matched DDW base/v1/v2 generation",
        environment_overrides=environment,
    )
    process.wait()


def _evaluate_sample(
    spec: DdwEvaluationSpec,
    runtime: RuntimeContext,
    prepared_path: Path,
    sample: DdwEvaluationSample,
    item_index: int,
    latent_root: Path,
    scratch: Path,
    video_root: Path,
    log_root: Path,
    environment: dict[str, str],
) -> DdwItemResult:
    with TemporaryDirectory(
        prefix=f"item-{item_index:04d}-",
        dir=scratch,
    ) as temporary:
        sample_scratch = Path(temporary)
        target = read_raw_front_target(
            runtime.roots.data / "waymo",
            sample,
        )
        target_path = sample_scratch / "target.npy"
        np.save(target_path, target.rgb_thwc, allow_pickle=False)
        del target
        video_locator = f"videos/item-{item_index:04d}.mp4"
        result_path = sample_scratch / "result.json"
        run_process(
            [
                str(GEN3C_PYTHON),
                str(Path(__file__).with_name("_decode_worker.py")),
                "--sample-id",
                sample.prepared.sample_id,
                "--latent-root",
                str(latent_root / f"item-{item_index:04d}"),
                "--target",
                str(target_path),
                "--lidar-depth",
                str(prepared_path.parent / sample.prepared.lidar_depth),
                "--tokenizer",
                str(runtime.roots.models / spec.recipe.tokenizer),
                "--moge-checkpoint",
                str(runtime.roots.models / spec.recipe.depth_checkpoint),
                "--scratch",
                str(sample_scratch),
                "--video",
                str(video_root / f"item-{item_index:04d}.mp4"),
                "--video-locator",
                video_locator,
                "--result",
                str(result_path),
            ],
            log_root / f"item-{item_index:04d}.log",
            f"DDW evaluation item {item_index}",
            environment,
        )
        return read_item_result(result_path)


__all__ = ["run_evaluation_v1"]

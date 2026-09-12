"""Лёгкий разбор двух явных версий R4c training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from novel_view.config.job import ResolvedJob
from novel_view.models.gen3c.environment import gen3c_environment
from novel_view.runtime.context import RuntimeContext
from novel_view.runtime.distributed import start_torchrun
from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.training.gen3c.lora.spec import (
    R4cTrainingInput,
    parse_r4c_lora_edm_spec,
    parse_r4c_training_input,
)


@dataclass(frozen=True, slots=True)
class ResolvedGen3cTrainingJob:
    """Полностью разобранный v1 либо v2 job без model imports."""

    name: str
    version: int
    input: R4cTrainingInput
    recipe: object


def resolve_training_job(job: ResolvedJob) -> ResolvedGen3cTrainingJob:
    """Разобрать точную workflow version и соответствующий recipe."""
    if job.workflow.name != "gen3c_training" or job.workflow.version not in (1, 2):
        raise ValueError("Gen3C training requires gen3c_training/v1 or v2")
    if job.execution.preset != "training_cp4":
        raise ValueError("Gen3C training requires training_cp4")

    if job.workflow.version == 1:
        input_spec = parse_r4c_training_input(
            job.input,
            allow_legacy_index_map=True,
        )
        recipe = parse_r4c_lora_edm_spec(job.parameters)
    else:
        from novel_view.training.gen3c.lora.depth_spec import (
            parse_r4c_lora_depth_spec,
        )

        input_spec = parse_r4c_training_input(
            job.input,
            allow_legacy_index_map=False,
        )
        recipe = parse_r4c_lora_depth_spec(job.parameters)

    return ResolvedGen3cTrainingJob(
        name=job.name,
        version=job.workflow.version,
        input=input_spec,
        recipe=recipe,
    )


def select_image_variant(job: ResolvedJob) -> str:
    """Обе версии используют основной Gen3C image."""
    resolve_training_job(job)
    return "core"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Launch exactly one installed CP4 worker for the supported baseline."""
    resolved = resolve_training_job(job)
    if resolved.version != 1:
        raise ValueError("run_v1 requires gen3c_training/v1")
    extra_arguments: list[str] = []
    input_spec = resolved.input
    if input_spec.legacy_index_map is not None:
        extra_arguments.extend(
            (
                "--legacy-index-map",
                str(runtime.roots.selections.parent / input_spec.legacy_index_map),
            )
        )
    return _run_exact_training_worker(
        job,
        runtime,
        resolved,
        worker_name="_worker_v1.py",
        extra_arguments=extra_arguments,
    )


def run_v2(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Launch only the experimental worker with its explicit depth recipe."""
    resolved = resolve_training_job(job)
    if resolved.version != 2:
        raise ValueError("run_v2 requires gen3c_training/v2")
    observer = resolved.recipe.observer
    return _run_exact_training_worker(
        job,
        runtime,
        resolved,
        worker_name="_worker_v2.py",
        extra_arguments=[
            "--depth-weight",
            str(resolved.recipe.depth_weight),
            "--observer-init-seed",
            str(observer.init_seed),
            "--observer-fit-order-seed",
            str(observer.fit_order_seed),
            "--observer-fit-epochs",
            str(observer.fit_epochs),
            "--observer-items-per-step",
            str(observer.items_per_step),
            "--observer-optimizer",
            observer.optimizer_name,
            "--observer-learning-rate",
            str(observer.learning_rate),
            "--observer-beta1",
            str(observer.betas[0]),
            "--observer-beta2",
            str(observer.betas[1]),
            "--observer-epsilon",
            str(observer.epsilon),
            "--observer-weight-decay",
            str(observer.weight_decay),
            "--observer-scheduler",
            observer.scheduler,
        ],
    )


def _run_exact_training_worker(
    job: ResolvedJob,
    runtime: RuntimeContext,
    resolved: ResolvedGen3cTrainingJob,
    *,
    worker_name: str,
    extra_arguments: list[str],
) -> int:
    """Assemble the shared process seam after two exact workers exist."""
    spec = resolved.recipe
    input_spec = resolved.input
    prepared_record = runtime.roots.prepared / input_spec.prepared_record
    log_root = runtime.attempt_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    arguments = [
        "--prepared-record",
        str(prepared_record),
        "--prepared-record-reference",
        str(input_spec.prepared_record),
        "--split",
        str(runtime.roots.selections.parent / input_spec.split),
        "--base-checkpoint",
        str(runtime.roots.models / spec.base_checkpoint),
        "--base-checkpoint-reference",
        str(spec.base_checkpoint),
        "--output",
        str(runtime.attempt_root / "checkpoints"),
        "--source-attempt",
        f"{job.name}/{runtime.run_id}/{runtime.attempt_id}",
        "--epochs",
        str(spec.epochs),
        "--training-seed",
        str(spec.training_seed),
        "--validation-seed",
        str(spec.validation_seed),
        *extra_arguments,
    ]
    if runtime.selected_checkpoint is not None:
        arguments.extend(("--resume-checkpoint", runtime.selected_checkpoint))

    with TemporaryDirectory(
        prefix=f"gen3c-training-v{resolved.version}-",
        dir=runtime.roots.cache,
    ) as temporary_root:
        process = start_torchrun(
            GEN3C_PYTHON,
            Path(__file__).resolve().parents[1]
            / "training"
            / "gen3c"
            / "lora"
            / worker_name,
            arguments,
            process_count=4,
            log_path=log_root / f"training-v{resolved.version}.log",
            description=f"Gen3C R4c LoRA training v{resolved.version}",
            environment_overrides=gen3c_environment(
                runtime,
                Path(temporary_root),
            ),
        )
        process.wait()
    return 0


__all__ = [
    "ResolvedGen3cTrainingJob",
    "resolve_training_job",
    "run_v1",
    "run_v2",
    "select_image_variant",
]

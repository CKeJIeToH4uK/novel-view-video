"""Run ordered EUVS generation through one resident Gen3C session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping
from urllib.parse import quote

from novel_view.config.job import ResolvedJob, reject_unknown_fields
from novel_view.generation.gen3c.recipe import (
    Gen3cGenerationRecipe,
    parse_generation_recipe,
)
from novel_view.runtime.context import RuntimeContext

if TYPE_CHECKING:
    from novel_view.generation.euvs.record import Gen3cRunSampling
    from novel_view.generation.gen3c.session import (
        Gen3cGenerationParameters,
    )
    from novel_view.inputs.euvs.spec import EuvsPairSelection, EuvsSelection


_INPUT_FIELDS = frozenset(
    {"reader", "dataset", "selection", "source_views_attempt"}
)
_PARAMETER_FIELDS = frozenset({"geometry", "generation"})
_GEOMETRY_FIELDS = frozenset({"backend"})
_GENERATED_RGB = "generated_rgb.npy"
_RUN_RECORD = "run.json"


@dataclass(frozen=True, slots=True)
class EuvsGenerationSpec:
    """External choices owned by `euvs_generation/v1` and `/v2`."""

    dataset: str
    selection: str
    source_views_attempt: str
    generation: Gen3cGenerationRecipe


@dataclass(frozen=True, slots=True)
class EuvsGenerationOutput:
    """One selected pair and its exact input/output directories."""

    selection: EuvsPairSelection
    source_view_directory: Path
    output_directory: Path


def parse_euvs_generation_job(job: ResolvedJob) -> EuvsGenerationSpec:
    """Parse the strict external fields of one EUVS generation job."""
    reject_unknown_fields(job.input, _INPUT_FIELDS)
    _require_fields(job.input, _INPUT_FIELDS, "input")
    reject_unknown_fields(job.parameters, _PARAMETER_FIELDS)
    _require_fields(job.parameters, _PARAMETER_FIELDS, "parameters")
    if _text(job.input, "reader") != "euvs_nuplan":
        raise ValueError("euvs_generation requires reader euvs_nuplan")

    geometry = _mapping(job.parameters, "geometry")
    reject_unknown_fields(geometry, _GEOMETRY_FIELDS)
    _require_fields(geometry, _GEOMETRY_FIELDS, "geometry")
    if _text(geometry, "backend") != "vggt_omega":
        raise ValueError("euvs_generation requires geometry vggt_omega")

    dataset = _text(job.input, "dataset")
    if dataset != "euvs":
        raise ValueError("euvs_generation requires dataset euvs")
    return EuvsGenerationSpec(
        dataset=dataset,
        selection=_text(job.input, "selection"),
        source_views_attempt=_text(job.input, "source_views_attempt"),
        generation=parse_generation_recipe(_mapping(job.parameters, "generation")),
    )


def select_image_variant(job: ResolvedJob) -> str:
    """Select the literal production image without importing model code."""
    parse_euvs_generation_job(job)
    return "core"


def plan_euvs_generation_outputs(
    selection: EuvsSelection,
    source_views_attempt: Path,
    attempt_root: Path,
) -> tuple[EuvsGenerationOutput, ...]:
    """Preserve selection order in direct source-view and pair paths."""
    from novel_view.generation.euvs.source_views import (
        source_view_directory_name,
    )

    source_root = source_views_attempt / "source-views"
    pair_root = attempt_root / "pairs"
    return tuple(
        EuvsGenerationOutput(
            pair,
            source_root / source_view_directory_name(pair.source.image_tokens),
            pair_root / quote(pair.name, safe=""),
        )
        for pair in selection.pairs
    )


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Generate selection-order pairs with autoregressive window seeds."""
    from novel_view.generation.gen3c.windows import WINDOW_SEED_AUTOREGRESSIVE

    return _run(job, runtime, WINDOW_SEED_AUTOREGRESSIVE)


def run_v2(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Generate selection-order pairs with a source seed for every window."""
    from novel_view.generation.gen3c.windows import WINDOW_SEED_SOURCE_RESEED

    return _run(job, runtime, WINDOW_SEED_SOURCE_RESEED)


def _run(
    job: ResolvedJob,
    runtime: RuntimeContext,
    window_seed_policy: str,
) -> int:
    """Run the one shared EUVS selection and resident-session order."""
    from novel_view.generation.euvs.plan import build_euvs_generation_plan
    from novel_view.generation.euvs.record import (
        build_euvs_gen3c_run_record,
        write_run_record,
    )
    from novel_view.generation.euvs.source_views import (
        bind_source_view,
        read_source_view,
    )
    from novel_view.generation.gen3c.model_request import build_model_request
    from novel_view.generation.gen3c.resources import build_generation_resources
    from novel_view.generation.gen3c.session import Gen3cGenerationSession
    from novel_view.inputs.euvs.index import FramesIndex
    from novel_view.inputs.euvs.spec import load_euvs_selection
    from novel_view.models.gen3c.spec import GEN3C_IMAGE_SIZE_HW
    from novel_view.runtime.presets import load_execution_preset

    spec = parse_euvs_generation_job(job)
    selection = load_euvs_selection(
        runtime.roots.selections.parent / spec.selection
    )
    source_attempt = runtime.roots.runs / spec.source_views_attempt
    outputs = plan_euvs_generation_outputs(
        selection,
        source_attempt,
        runtime.attempt_root,
    )
    frames = FramesIndex.load(runtime.roots.data / spec.dataset)
    model, parameters = build_model_request(
        runtime.roots.models,
        spec.generation,
        window_seed_policy=window_seed_policy,
    )
    context_parallel_size = load_execution_preset(
        job.execution.preset
    ).gpu_count
    (runtime.attempt_root / "pairs").mkdir()

    with Gen3cGenerationSession(
        model,
        parameters,
        build_generation_resources(runtime, context_parallel_size),
    ) as session:
        for output in outputs:
            source_view = read_source_view(
                output.selection,
                frames,
                GEN3C_IMAGE_SIZE_HW,
            )
            geometry = bind_source_view(
                source_view,
                output.source_view_directory,
            )
            plan = build_euvs_generation_plan(
                source_view.physical_pair,
                source_view.geometry_input.source,
                geometry,
            )
            output.output_directory.mkdir()
            generated = session.generate(
                plan.request,
                output.output_directory / _GENERATED_RGB,
            )
            record = build_euvs_gen3c_run_record(
                experiment_name=job.name,
                pair=output.selection,
                target_output_index=tuple(
                    map(int, plan.timeline.target_output_index)
                ),
                target_source_sequence_index=tuple(
                    map(int, plan.timeline.source_sequence_index)
                ),
                geometry_backend="vggt-omega",
                geometry_directory=output.source_view_directory,
                geometry_provenance=geometry.provenance,
                model_id=model.model_id,
                network_checkpoint=model.network_checkpoint,
                lora_working_manifest=(
                    None
                    if model.lora_provenance is None
                    else model.lora_provenance.working_manifest
                ),
                lora_evidence_path=(
                    None
                    if model.lora_provenance is None
                    else model.lora_provenance.evidence_path
                ),
                lora_epochs=(
                    None
                    if model.lora_provenance is None
                    else model.lora_provenance.epochs
                ),
                lora_strength=(
                    None
                    if model.lora_weights is None
                    else model.lora_weights.strength
                ),
                sampling=_record_sampling(parameters),
                context_parallel_size=context_parallel_size,
                output_rgb_path=generated.generated_rgb_path,
                output_frame_count=plan.timeline.model_frame_count,
                runs_root=runtime.roots.runs,
                models_root=runtime.roots.models,
            )
            write_run_record(output.output_directory / _RUN_RECORD, record)
    return 0


def _record_sampling(
    parameters: Gen3cGenerationParameters,
) -> Gen3cRunSampling:
    from novel_view.generation.euvs.record import Gen3cRunSampling

    return Gen3cRunSampling(
        seed=parameters.seed,
        prompt=parameters.prompt,
        negative_prompt=parameters.negative_prompt,
        guidance=parameters.guidance,
        num_steps=parameters.num_steps,
        window_seed_policy=parameters.window_seed_policy,
    )


def _require_fields(
    values: Mapping[str, object],
    fields: frozenset[str],
    name: str,
) -> None:
    missing = fields - values.keys()
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)}")


def _text(values: Mapping[str, object], field: str) -> str:
    value = values[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _mapping(values: Mapping[str, object], field: str) -> dict[str, object]:
    value = values[field]
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


__all__ = [
    "EuvsGenerationOutput",
    "EuvsGenerationSpec",
    "parse_euvs_generation_job",
    "plan_euvs_generation_outputs",
    "run_v1",
    "run_v2",
    "select_image_variant",
]

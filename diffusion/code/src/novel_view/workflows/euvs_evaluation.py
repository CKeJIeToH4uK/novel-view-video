"""Run exact EUVS generation records through one evaluation attempt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from novel_view.config.job import ResolvedJob
from novel_view.evaluation.euvs.spec import (
    EuvsEvaluationSpec,
    GenerationAttemptReference,
    GenerationRecordReferences,
    parse_euvs_evaluation_job,
)
from novel_view.runtime.context import RuntimeContext

if TYPE_CHECKING:
    from novel_view.evaluation.euvs.execute import EuvsMetricResources
    from novel_view.evaluation.euvs.masks import GroundedSam2Resources
    from novel_view.inputs.euvs.spec import EuvsPairSelection, EuvsSelection

_MODEL_DIRECTORY = Path("euvs/official/euvs-eval-v1")
_DINOV2_REPOSITORY = Path("/opt/upstream/dinov2")


@dataclass(frozen=True, slots=True)
class EuvsEvaluationRecord:
    """One selected pair, exact generation record and direct result path."""

    selection: EuvsPairSelection
    record_path: Path
    record_locator: str
    result_path: Path


def select_image_variant(job: ResolvedJob) -> str:
    """Select the literal image without importing readers or model owners."""
    parse_euvs_evaluation_job(job)
    return "core"


def plan_euvs_evaluation_records(
    spec: EuvsEvaluationSpec,
    selection: EuvsSelection,
    runs_root: Path,
    attempt_root: Path,
) -> tuple[EuvsEvaluationRecord, ...]:
    """Build exact selection-order record/result paths without discovery."""
    generation = spec.generation
    if isinstance(generation, GenerationAttemptReference):
        locators = tuple(
            (
                Path(generation.path)
                / "pairs"
                / quote(pair.name, safe="")
                / "run.json"
            ).as_posix()
            for pair in selection.pairs
        )
    else:
        if len(generation.records) != len(selection.pairs):
            raise ValueError("generation.records must match selection order")
        locators = tuple(Path(value).as_posix() for value in generation.records)
    return tuple(
        EuvsEvaluationRecord(
            pair,
            runs_root / locator,
            locator,
            attempt_root / "metrics" / f"{quote(pair.name, safe='')}.json",
        )
        for pair, locator in zip(selection.pairs, locators, strict=True)
    )


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Evaluate every exact record sequentially and stop on the first error."""
    from novel_view.evaluation.euvs.execute import evaluate_pair
    from novel_view.evaluation.euvs.masks import (
        EUVS_GROUNDED_SAM2_RECIPE,
        load_or_predict_masks,
    )
    from novel_view.evaluation.euvs.record import write_euvs_metric_result
    from novel_view.evaluation.euvs.samples import load_euvs_samples
    from novel_view.evaluation.euvs.support_views import build_euvs_support
    from novel_view.inputs.euvs.index import FramesIndex
    from novel_view.inputs.euvs.spec import load_euvs_selection

    spec = parse_euvs_evaluation_job(job)
    if spec.mask_recipe_id != EUVS_GROUNDED_SAM2_RECIPE.recipe_id:
        raise ValueError(f"unsupported mask recipe: {spec.mask_recipe_id}")
    selection = load_euvs_selection(
        runtime.roots.selections.parent / spec.selection
    )
    records = plan_euvs_evaluation_records(
        spec,
        selection,
        runtime.roots.runs,
        runtime.attempt_root,
    )
    frames = FramesIndex.load(runtime.roots.data / spec.dataset)
    mask_cache = (
        runtime.roots.cache
        / "euvs"
        / "masks"
        / spec.mask_recipe_id
    )
    (runtime.attempt_root / "metrics").mkdir()
    legacy = isinstance(spec.generation, GenerationRecordReferences)

    for planned in records:
        log_directory = (
            runtime.attempt_root / "logs" / quote(planned.selection.name, safe="")
        )
        log_directory.mkdir(parents=True, exist_ok=True)
        source_resources, target_resources, metric_resources = _resources(
            runtime, log_directory,
        )
        samples = load_euvs_samples(
            planned.record_path,
            frames,
            allow_legacy_record=legacy,
        )
        _require_selected_pair(planned.selection, samples.record.pair)
        source_masks = load_or_predict_masks(
            samples.pair.source,
            mask_cache,
            source_resources,
        )
        target_masks = load_or_predict_masks(
            samples.pair.target,
            mask_cache,
            target_resources,
        )
        support = build_euvs_support(
            samples,
            source_masks,
            target_masks,
            runtime.roots.runs,
            allow_legacy_geometry=legacy,
        )
        result = evaluate_pair(
            support,
            metric_resources,
            planned.record_locator,
        )
        write_euvs_metric_result(planned.result_path, result)
    return 0


def _resources(
    runtime: RuntimeContext,
    log_directory: Path,
) -> tuple[GroundedSam2Resources, GroundedSam2Resources, EuvsMetricResources]:
    from novel_view.evaluation.euvs.execute import EuvsMetricResources
    from novel_view.evaluation.euvs.masks import GroundedSam2Resources

    model_root = runtime.roots.models / _MODEL_DIRECTORY
    common = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    masks = tuple(
        GroundedSam2Resources(
            scratch_root=runtime.roots.cache,
            log_path=log_directory / f"{role}-masks.log",
            grounding_model_directory=model_root / "grounding-dino-tiny",
            sam2_checkpoint_path=model_root / "sam2/sam2.1_hiera_large.pt",
            environment_overrides=common,
        )
        for role in ("source", "target")
    )
    torch_home = model_root / "torch"
    metrics = EuvsMetricResources(
        scratch_root=runtime.roots.cache,
        log_path=log_directory / "metrics.log",
        dinov2_repository=_DINOV2_REPOSITORY,
        dinov2_checkpoint=model_root / "dinov2/dinov2_vitb14_pretrain.pth",
        torch_home=torch_home,
        environment_overrides={**common, "TORCH_HOME": str(torch_home)},
    )
    return masks[0], masks[1], metrics


def _require_selected_pair(
    selection: EuvsPairSelection,
    pair: EuvsPairSelection,
) -> None:
    """Bind the external selection to the exact scientific record identity."""
    actual = (
        pair.name,
        pair.tags,
        pair.location,
        pair.channel,
        pair.source.traversal,
        pair.source.image_tokens,
        pair.target.traversal,
        pair.target.image_tokens,
    )
    expected = (
        selection.name,
        selection.tags,
        selection.location,
        selection.channel,
        selection.source.traversal,
        selection.source.image_tokens,
        selection.target.traversal,
        selection.target.image_tokens,
    )
    if actual != expected:
        raise ValueError("generation record disagrees with selected EUVS pair")


__all__ = [
    "EuvsEvaluationRecord",
    "plan_euvs_evaluation_records",
    "run_v1",
    "select_image_variant",
]

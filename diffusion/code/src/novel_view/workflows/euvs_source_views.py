"""Run reusable EUVS source views as one explicit versioned workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from novel_view.config.job import ResolvedJob, reject_unknown_fields
from novel_view.runtime.context import RuntimeContext

if TYPE_CHECKING:
    from novel_view.inputs.euvs.spec import (
        EuvsPairSelection,
        EuvsSelection,
    )


_INPUT_FIELDS = frozenset({"reader", "dataset", "selection"})
_PARAMETER_FIELDS = frozenset({"geometry"})
_GEOMETRY_FIELDS = frozenset({"backend"})
_SOURCE_SIZE_HW = (704, 1_280)
_CHECKPOINT = Path("vggt-omega/vggt_omega_1b_512.pt")


@dataclass(frozen=True, slots=True)
class EuvsSourceViewsSpec:
    """The portable external choices of `euvs_source_views/v1`."""

    dataset: str
    selection: str


@dataclass(frozen=True, slots=True)
class EuvsSourceViewOutput:
    """One unique selected source and its direct model-owned directory."""

    selection: EuvsPairSelection
    directory: Path


def parse_euvs_source_views_job(job: ResolvedJob) -> EuvsSourceViewsSpec:
    """Parse only the external fields owned by this concrete workflow."""
    reject_unknown_fields(job.input, _INPUT_FIELDS)
    _require_fields(job.input, _INPUT_FIELDS, "input")
    reject_unknown_fields(job.parameters, _PARAMETER_FIELDS)
    _require_fields(job.parameters, _PARAMETER_FIELDS, "parameters")

    reader = _text(job.input, "reader")
    dataset = _text(job.input, "dataset")
    selection = _text(job.input, "selection")
    if reader != "euvs_nuplan":
        raise ValueError("euvs_source_views/v1 requires reader euvs_nuplan")
    if dataset != "euvs":
        raise ValueError("euvs_source_views/v1 requires dataset euvs")
    if job.execution.preset != "inference_cp1":
        raise ValueError("euvs_source_views/v1 requires inference_cp1")

    geometry = _mapping(job.parameters, "geometry")
    reject_unknown_fields(geometry, _GEOMETRY_FIELDS)
    _require_fields(geometry, _GEOMETRY_FIELDS, "geometry")
    if _text(geometry, "backend") != "vggt_omega":
        raise ValueError("euvs_source_views/v1 requires backend vggt_omega")
    return EuvsSourceViewsSpec(dataset=dataset, selection=selection)


def select_image_variant(job: ResolvedJob) -> str:
    """Select the literal production image without importing readers/models."""
    parse_euvs_source_views_job(job)
    return "core"


def plan_source_view_outputs(
    selection: EuvsSelection,
    attempt_root: Path,
) -> tuple[EuvsSourceViewOutput, ...]:
    """Map first-seen exact source tuples to direct attempt directories."""
    from novel_view.generation.euvs.source_views import (
        source_view_directory_name,
        unique_source_views,
    )

    root = attempt_root / "source-views"
    return tuple(
        EuvsSourceViewOutput(
            pair,
            root / source_view_directory_name(pair.source.image_tokens),
        )
        for pair in unique_source_views(selection)
    )


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Read, predict and directly save every unique source in selection order."""
    from novel_view.generation.euvs.source_views import (
        build_vggt_source_request,
        read_source_view,
    )
    from novel_view.inputs.euvs.index import FramesIndex
    from novel_view.inputs.euvs.spec import load_euvs_selection
    from novel_view.models.vggt.backend import (
        VggtOmegaResources,
        run_vggt_omega,
    )
    from novel_view.models.vggt.record import save_vggt_omega_prediction
    from novel_view.models.vggt.spec import VggtOmegaInputMode

    spec = parse_euvs_source_views_job(job)
    selection = load_euvs_selection(runtime.roots.selections.parent / spec.selection)
    outputs = plan_source_view_outputs(selection, runtime.attempt_root)
    index = FramesIndex.load(runtime.roots.data / spec.dataset)
    (runtime.attempt_root / "source-views").mkdir()
    resources = VggtOmegaResources(
        scratch_root=runtime.roots.cache,
        checkpoint_path=runtime.roots.models / _CHECKPOINT,
    )

    for ordinal, output in enumerate(outputs):
        source_view = read_source_view(
            output.selection,
            index,
            _SOURCE_SIZE_HW,
        )
        execution = run_vggt_omega(
            build_vggt_source_request(
                source_view,
                VggtOmegaInputMode.BALANCED_896,
            ),
            resources,
            runtime.attempt_root / f"source-view-{ordinal:04d}.log",
        )
        source = source_view.geometry_input.source
        save_vggt_omega_prediction(
            execution.prediction,
            source.sequence_id,
            source.image_size_hw,
            output.directory,
        )
    return 0


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


def _mapping(
    values: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    value = values[field]
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


__all__ = [
    "EuvsSourceViewOutput",
    "EuvsSourceViewsSpec",
    "parse_euvs_source_views_job",
    "plan_source_view_outputs",
    "run_v1",
    "select_image_variant",
]

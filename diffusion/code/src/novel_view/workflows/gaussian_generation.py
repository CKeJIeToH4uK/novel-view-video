"""Run the explicit Gaussian generation workflow versions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import cv2
import numpy as np

from novel_view.config.job import ResolvedJob, reject_unknown_fields
from novel_view.generation.gen3c.recipe import (
    Gen3cGenerationRecipe,
    parse_generation_recipe,
)
from novel_view.inputs.gaussian.spec import (
    GaussianHandoffInputSpec,
    parse_gaussian_handoff_job_input,
    parse_gaussian_job_input,
)
from novel_view.runtime.context import RuntimeContext

if TYPE_CHECKING:
    from novel_view.generation.gaussian.clips import GaussianClip
    from novel_view.generation.gaussian.full import GaussianFullSequence
    from novel_view.generation.gen3c.session import (
        Gen3cGenerationModel,
        Gen3cGenerationParameters,
        Gen3cGenerationSession,
    )


_PARAMETER_FIELDS = frozenset({"frame_rate", "generation"})
_DENSE_PARAMETER_FIELDS = frozenset({"frame_rate", "dense", "generation"})
_OVERLAP_PARAMETER_FIELDS = frozenset(
    {"frame_rate", "dense", "generation", "context_depth"}
)
_HANDOFF_PARAMETER_FIELDS = frozenset({"bake_dense_stride"})
_DENSE_FIELDS = frozenset(
    {"densification_factor", "bake_stride", "target_capacity"}
)
_OVERLAP_DENSE_FIELDS = _DENSE_FIELDS | {"production_capacity"}
_CONTEXT_DEPTH_FIELDS = frozenset({"backend", "checkpoint"})
_GENERATED_RGB = "generated_rgb.npy"
_TARGET_VIDEO = "targets.mp4"
_RUN_RECORD = "run.json"


@dataclass(frozen=True, slots=True)
class GaussianGenerationSpec:
    """Model and fixed-rate choices shared by Gaussian generation versions."""

    frame_rate: int
    generation: Gen3cGenerationRecipe


@dataclass(frozen=True, slots=True)
class GaussianDenseGenerationSpec:
    """Dense camera sampling and the model request used by workflow v3."""

    generation: GaussianGenerationSpec
    densification_factor: int
    bake_stride: int
    target_capacity: int


@dataclass(frozen=True, slots=True)
class GaussianDenseOverlapSpec:
    """Overlap21 schedule, model request, and resident MoGe checkpoint."""

    generation: GaussianGenerationSpec
    densification_factor: int
    bake_stride: int
    production_capacity: int
    target_capacity: int
    context_depth_checkpoint: str


@dataclass(frozen=True, slots=True)
class GaussianDenseHandoffSpec:
    """Exact attempt references and publication stride of workflow v5."""

    input: GaussianHandoffInputSpec
    bake_dense_stride: int


def parse_gaussian_generation_job(job: ResolvedJob) -> GaussianGenerationSpec:
    """Parse the strict external full-sequence or native-clip recipe."""
    reject_unknown_fields(job.parameters, _PARAMETER_FIELDS)
    _require_fields(job.parameters, _PARAMETER_FIELDS, "parameters")
    frame_rate = _integer(job.parameters, "frame_rate")
    if frame_rate != 24:
        raise ValueError("gaussian_generation requires frame_rate 24")

    spec = _parse_generation(job.parameters, frame_rate)
    if spec.generation.lora is not None:
        raise ValueError("Gaussian run v1 cannot record LoRA identity")
    return spec


def parse_gaussian_dense_generation_job(
    job: ResolvedJob,
) -> GaussianDenseGenerationSpec:
    """Parse strict trajectory/chunk and model choices for workflow v3."""
    from novel_view.generation.gen3c.timeline import GEN3C_WINDOW_STEP
    reject_unknown_fields(job.parameters, _DENSE_PARAMETER_FIELDS)
    _require_fields(job.parameters, _DENSE_PARAMETER_FIELDS, "parameters")
    frame_rate = _integer(job.parameters, "frame_rate")
    if frame_rate != 24:
        raise ValueError("gaussian_generation requires frame_rate 24")
    dense = _mapping(job.parameters, "dense")
    reject_unknown_fields(dense, _DENSE_FIELDS)
    _require_fields(dense, _DENSE_FIELDS, "dense")
    factor = _integer(dense, "densification_factor")
    stride = _integer(dense, "bake_stride")
    capacity = _integer(dense, "target_capacity")
    if factor <= 0 or stride <= 0:
        raise ValueError("dense factor and bake stride must be positive")
    if capacity != GEN3C_WINDOW_STEP:
        raise ValueError(
            f"dense target_capacity must be {GEN3C_WINDOW_STEP}"
        )
    return GaussianDenseGenerationSpec(
        _parse_generation(job.parameters, frame_rate),
        factor,
        stride,
        capacity,
    )


def parse_gaussian_dense_overlap_job(job: ResolvedJob) -> GaussianDenseOverlapSpec:
    """Parse the strict overlap21 schedule and resident MoGe identity."""
    from novel_view.generation.gen3c.timeline import GEN3C_WINDOW_STEP

    reject_unknown_fields(job.parameters, _OVERLAP_PARAMETER_FIELDS)
    _require_fields(job.parameters, _OVERLAP_PARAMETER_FIELDS, "parameters")
    frame_rate = _integer(job.parameters, "frame_rate")
    if frame_rate != 24:
        raise ValueError("gaussian_generation requires frame_rate 24")
    dense = _mapping(job.parameters, "dense")
    reject_unknown_fields(dense, frozenset(_OVERLAP_DENSE_FIELDS))
    _require_fields(dense, frozenset(_OVERLAP_DENSE_FIELDS), "dense")
    factor = _integer(dense, "densification_factor")
    stride = _integer(dense, "bake_stride")
    production = _integer(dense, "production_capacity")
    capacity = _integer(dense, "target_capacity")
    if factor <= 0 or stride <= 0 or production <= 0:
        raise ValueError("dense factor, bake stride, and production must be positive")
    if capacity != GEN3C_WINDOW_STEP or capacity - production != 20:
        raise ValueError("overlap21 requires 100 owned and 20 trailing slots")
    if production % factor:
        raise ValueError("overlap21 chunk starts must remain on source poses")
    context_depth = _mapping(job.parameters, "context_depth")
    reject_unknown_fields(context_depth, _CONTEXT_DEPTH_FIELDS)
    _require_fields(context_depth, _CONTEXT_DEPTH_FIELDS, "context_depth")
    if _text(context_depth, "backend") != "moge":
        raise ValueError("gaussian_generation/v4 requires context backend moge")
    return GaussianDenseOverlapSpec(
        _parse_generation(job.parameters, frame_rate),
        factor,
        stride,
        production,
        capacity,
        _text(context_depth, "checkpoint"),
    )


def parse_gaussian_dense_handoff_job(job: ResolvedJob) -> GaussianDenseHandoffSpec:
    """Parse the pure-CPU dense PNG publication boundary."""
    input_spec = parse_gaussian_handoff_job_input(job)
    reject_unknown_fields(job.parameters, _HANDOFF_PARAMETER_FIELDS)
    _require_fields(job.parameters, _HANDOFF_PARAMETER_FIELDS, "parameters")
    stride = _integer(job.parameters, "bake_dense_stride")
    if stride <= 0:
        raise ValueError("bake_dense_stride must be positive")
    if job.execution.preset != "cpu_test":
        raise ValueError("gaussian_generation/v5 requires cpu_test")
    return GaussianDenseHandoffSpec(input_spec, stride)


def select_image_variant(job: ResolvedJob) -> str:
    """Select the core image without importing the Gen3C worker."""
    if job.workflow.version == 5:
        parse_gaussian_dense_handoff_job(job)
        return "core"
    if job.workflow.version in (3, 4):
        from novel_view.inputs.gaussian.spec import parse_gaussian_dense_job_input

        parse_gaussian_dense_job_input(job)
        if job.workflow.version == 3:
            parse_gaussian_dense_generation_job(job)
            return "core"
        parse_gaussian_dense_overlap_job(job)
        return "moge"
    else:
        parse_gaussian_job_input(job)
        parse_gaussian_generation_job(job)
    return "core"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Generate one full producer sequence through one resident session."""
    from novel_view.generation.gaussian.full import build_gaussian_full_sequence
    from novel_view.generation.gen3c.model_request import build_model_request
    from novel_view.generation.gen3c.resources import build_generation_resources
    from novel_view.generation.gen3c.session import Gen3cGenerationSession
    from novel_view.inputs.gaussian.reader import read_gaussian_export
    from novel_view.inputs.gaussian.spec import (
        GaussianFullSequenceSelection,
        load_gaussian_selection,
    )
    from novel_view.runtime.presets import load_execution_preset

    input_spec = parse_gaussian_job_input(job)
    spec = parse_gaussian_generation_job(job)
    selection = load_gaussian_selection(
        runtime.roots.selections.parent / input_spec.selection
    )
    if not isinstance(selection, GaussianFullSequenceSelection):
        raise ValueError("gaussian_generation/v1 requires full_sequence selection")
    export = read_gaussian_export(runtime.roots.data / input_spec.info_file)
    sequence = build_gaussian_full_sequence(export, selection, spec.frame_rate)
    model, parameters = build_model_request(
        runtime.roots.models, spec.generation
    )
    context_parallel_size = load_execution_preset(job.execution.preset).gpu_count
    with Gen3cGenerationSession(
        model,
        parameters,
        build_generation_resources(runtime, context_parallel_size),
    ) as session:
        _run_sequence(
            sequence,
            session,
            model,
            parameters,
            runtime.attempt_root,
            context_parallel_size,
        )
    return 0


def run_v2(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Generate selected independent clips through one resident session."""
    from novel_view.generation.gaussian.clips import plan_native_clips
    from novel_view.generation.gaussian.full import build_gaussian_full_sequence
    from novel_view.generation.gen3c.model_request import build_model_request
    from novel_view.generation.gen3c.resources import build_generation_resources
    from novel_view.generation.gen3c.session import Gen3cGenerationSession
    from novel_view.inputs.gaussian.reader import read_gaussian_export
    from novel_view.inputs.gaussian.spec import (
        GaussianIndependentClipsSelection,
        load_gaussian_selection,
    )
    from novel_view.runtime.presets import load_execution_preset

    input_spec = parse_gaussian_job_input(job)
    spec = parse_gaussian_generation_job(job)
    selection = load_gaussian_selection(
        runtime.roots.selections.parent / input_spec.selection
    )
    if not isinstance(selection, GaussianIndependentClipsSelection):
        raise ValueError(
            "gaussian_generation/v2 requires independent_clips selection"
        )
    export = read_gaussian_export(runtime.roots.data / input_spec.info_file)
    sequence = build_gaussian_full_sequence(
        export,
        selection,
        spec.frame_rate,
    )
    clips = plan_native_clips(sequence, selection.clip_indices)
    model, parameters = build_model_request(
        runtime.roots.models, spec.generation
    )
    context_parallel_size = load_execution_preset(job.execution.preset).gpu_count
    with Gen3cGenerationSession(
        model,
        parameters,
        build_generation_resources(runtime, context_parallel_size),
    ) as session:
        for clip in clips:
            output = runtime.attempt_root / "clips" / f"clip-{clip.index - 1:03d}"
            output.mkdir(parents=True)
            _run_sequence(
                clip.sequence,
                session,
                model,
                parameters,
                output,
                context_parallel_size,
                independent_clip=clip,
            )
    return 0


def run_v3(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Generate exact dense independent chunks through one resident session."""
    from novel_view.generation.gaussian.dense import (
        build_dense_camera_table,
        build_dense_independent_sequence,
        plan_independent_chunks,
        select_dense_export,
        write_standard_camera_table,
    )
    from novel_view.generation.gaussian.record import (
        build_gaussian_run_v2,
        write_gaussian_run_record,
    )
    from novel_view.generation.gen3c.model_request import build_model_request
    from novel_view.generation.gen3c.resources import build_generation_resources
    from novel_view.generation.gen3c.session import Gen3cGenerationSession
    from novel_view.inputs.gaussian.reader import read_gaussian_export
    from novel_view.inputs.gaussian.spec import (
        GaussianDenseIndependentSelection,
        load_gaussian_selection,
        parse_gaussian_dense_job_input,
    )
    from novel_view.runtime.presets import load_execution_preset

    input_spec = parse_gaussian_dense_job_input(job)
    dense_spec = parse_gaussian_dense_generation_job(job)
    generation_spec = dense_spec.generation
    selection = load_gaussian_selection(
        runtime.roots.selections.parent / input_spec.selection
    )
    if not isinstance(selection, GaussianDenseIndependentSelection):
        raise ValueError(
            "gaussian_generation/v3 requires dense_independent selection"
        )
    export = read_gaussian_export(runtime.roots.data / input_spec.info_file)
    if export.scene_id != selection.scene_id:
        raise ValueError("Gaussian export disagrees with selected scene")
    export = select_dense_export(export, selection.source_pose_range)
    trajectory, camera_table = build_dense_camera_table(
        export,
        runtime.roots.data / input_spec.source_transforms,
        runtime.roots.data / input_spec.source_intrinsics,
        densification_factor=dense_spec.densification_factor,
    )
    write_standard_camera_table(
        camera_table,
        runtime.attempt_root / "dense-camera-table",
    )
    chunks = plan_independent_chunks(
        trajectory,
        target_capacity=dense_spec.target_capacity,
    )
    model, parameters = build_model_request(
        runtime.roots.models, generation_spec.generation
    )
    context_parallel_size = load_execution_preset(job.execution.preset).gpu_count
    with Gen3cGenerationSession(
        model,
        parameters,
        build_generation_resources(runtime, context_parallel_size),
    ) as session:
        for chunk in chunks:
            sequence = build_dense_independent_sequence(
                export,
                trajectory,
                chunk,
            )
            output = runtime.attempt_root / "chunks" / f"chunk-{chunk.chunk_index:03d}"
            output.mkdir(parents=True)
            result = session.generate(sequence, output / _GENERATED_RGB)
            video = _write_target_video(
                result.generated_rgb_path,
                sequence.target_output_index,
                output / _TARGET_VIDEO,
                generation_spec.frame_rate,
            )
            write_gaussian_run_record(
                output / _RUN_RECORD,
                build_gaussian_run_v2(
                    sequence,
                    camera_table,
                    len(chunks),
                    model,
                    parameters,
                    result,
                    video,
                    context_parallel_size,
                ),
            )
    return 0


def run_v4(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Generate overlap21 chunks through one resident Gen3C and MoGe process."""
    from novel_view.generation.gaussian.dense import (
        build_dense_camera_table,
        build_dense_overlap_sequence,
        load_dense_overlap_context,
        plan_overlap_chunks,
        select_dense_export,
        write_standard_camera_table,
    )
    from novel_view.generation.gaussian.record import (
        build_gaussian_run_v3,
        write_gaussian_run_record,
    )
    from novel_view.generation.gen3c.model_request import build_model_request
    from novel_view.generation.gen3c.resources import build_generation_resources
    from novel_view.generation.gen3c.session import Gen3cGenerationSession
    from novel_view.inputs.gaussian.reader import read_gaussian_export
    from novel_view.inputs.gaussian.spec import (
        GaussianDenseOverlapSelection,
        load_gaussian_selection,
        parse_gaussian_dense_job_input,
    )
    from novel_view.runtime.presets import load_execution_preset

    input_spec = parse_gaussian_dense_job_input(job)
    overlap_spec = parse_gaussian_dense_overlap_job(job)
    generation_spec = overlap_spec.generation
    selection = load_gaussian_selection(
        runtime.roots.selections.parent / input_spec.selection
    )
    if not isinstance(selection, GaussianDenseOverlapSelection):
        raise ValueError(
            "gaussian_generation/v4 requires dense_overlap21 selection"
        )
    export = read_gaussian_export(runtime.roots.data / input_spec.info_file)
    if export.scene_id != selection.scene_id:
        raise ValueError("Gaussian export disagrees with selected scene")
    export = select_dense_export(export, selection.source_pose_range)
    trajectory, camera_table = build_dense_camera_table(
        export,
        runtime.roots.data / input_spec.source_transforms,
        runtime.roots.data / input_spec.source_intrinsics,
        densification_factor=overlap_spec.densification_factor,
    )
    write_standard_camera_table(
        camera_table,
        runtime.attempt_root / "dense-camera-table",
    )
    chunks = plan_overlap_chunks(
        trajectory,
        production_capacity=overlap_spec.production_capacity,
        target_capacity=overlap_spec.target_capacity,
    )
    model, parameters = build_model_request(
        runtime.roots.models, generation_spec.generation
    )
    context_parallel_size = load_execution_preset(job.execution.preset).gpu_count
    context_depth_model = (
        runtime.roots.models / overlap_spec.context_depth_checkpoint
    )
    previous = None
    with Gen3cGenerationSession(
        model,
        parameters,
        build_generation_resources(
            runtime,
            context_parallel_size,
            context_depth_model=context_depth_model,
        ),
    ) as session:
        for chunk in chunks:
            sequence = build_dense_overlap_sequence(
                export,
                trajectory,
                chunk,
                previous,
            )
            output = runtime.attempt_root / "chunks" / f"chunk-{chunk.chunk_index:03d}"
            output.mkdir(parents=True)
            result = session.generate(sequence, output / _GENERATED_RGB)
            video = _write_target_video(
                result.generated_rgb_path,
                sequence.target_output_index,
                output / _TARGET_VIDEO,
                generation_spec.frame_rate,
            )
            write_gaussian_run_record(
                output / _RUN_RECORD,
                build_gaussian_run_v3(
                    sequence,
                    camera_table,
                    len(chunks),
                    model,
                    parameters,
                    result,
                    video,
                    context_parallel_size,
                    context_depth_model,
                    previous,
                ),
            )
            previous = (
                load_dense_overlap_context(sequence, result)
                if chunk.context_frame_count
                else None
            )
    return 0


def run_v5(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Publish exact independent and overlap dense outputs as native PNGs."""
    from shutil import copyfile

    from novel_view.generation.gaussian.handoff import (
        materialize_dense_png,
        select_dense_outputs,
    )
    from novel_view.inputs.gaussian.spec import (
        GaussianDenseHandoffSelection,
        load_gaussian_selection,
    )

    spec = parse_gaussian_dense_handoff_job(job)
    selection = load_gaussian_selection(
        runtime.roots.selections.parent / spec.input.selection
    )
    if not isinstance(selection, GaussianDenseHandoffSelection):
        raise ValueError("gaussian_generation/v5 requires dense_handoff selection")
    independent, overlap = select_dense_outputs(
        runtime.roots.runs / spec.input.independent_attempt,
        runtime.roots.runs / spec.input.overlap_attempt,
        selection.dense_camera_ids,
        scene_id=selection.scene_id,
        bake_dense_stride=spec.bake_dense_stride,
    )
    table_root = runtime.attempt_root / "dense-camera-table"
    table_root.mkdir(parents=True)
    transforms = table_root / "transforms.json"
    intrinsics = table_root / "intrinsics.json"
    copyfile(independent.dense_transforms, transforms)
    copyfile(independent.dense_intrinsics, intrinsics)
    for handoff in (independent, overlap):
        materialize_dense_png(
            handoff,
            runtime.attempt_root / handoff.method / "selected",
            bake_dense_stride=spec.bake_dense_stride,
            dense_transforms=transforms,
            dense_intrinsics=intrinsics,
        )
    return 0


def _parse_generation(
    parameters: Mapping[str, object],
    frame_rate: int,
) -> GaussianGenerationSpec:
    return GaussianGenerationSpec(
        frame_rate,
        parse_generation_recipe(_mapping(parameters, "generation")),
    )


def _run_sequence(
    sequence: GaussianFullSequence,
    session: Gen3cGenerationSession,
    model: Gen3cGenerationModel,
    parameters: Gen3cGenerationParameters,
    output: Path,
    context_parallel_size: int,
    *,
    independent_clip: GaussianClip | None = None,
) -> None:
    """Write one direct NPY, target video and historical v1 record."""
    from novel_view.generation.gaussian.record import (
        build_gaussian_run_v1,
        write_gaussian_run_record,
    )

    result = session.generate(sequence, output / _GENERATED_RGB)
    video = _write_target_video(
        result.generated_rgb_path,
        sequence.target_output_index,
        output / _TARGET_VIDEO,
        sequence.frame_rate,
    )
    record = build_gaussian_run_v1(
        sequence,
        model,
        parameters,
        result,
        video,
        context_parallel_size,
        independent_clip=independent_clip,
    )
    write_gaussian_run_record(output / _RUN_RECORD, record)


def _write_target_video(
    generated_rgb_path: Path,
    target_output_index: np.ndarray,
    output_path: Path,
    frame_rate: int,
) -> Path:
    """Encode the exact target slots without source or padded frames."""
    generated = np.load(generated_rgb_path, mmap_mode="r", allow_pickle=False)
    height, width = generated.shape[1:3]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (width, height),
    )
    if not writer.isOpened():
        generated._mmap.close()
        raise RuntimeError("cannot open Gaussian target MP4 writer")
    try:
        for slot in target_output_index:
            writer.write(cv2.cvtColor(generated[int(slot)], cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        generated._mmap.close()
    return output_path


def _require_fields(
    values: Mapping[str, object],
    required: frozenset[str],
    context: str,
) -> None:
    missing = required - values.keys()
    if missing:
        raise ValueError(f"{context} missing fields: {sorted(missing)}")


def _mapping(values: Mapping[str, object], field: str) -> dict[str, object]:
    value = values[field]
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _text(values: Mapping[str, object], field: str) -> str:
    value = values[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _integer(values: Mapping[str, object], field: str) -> int:
    value = values[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


__all__ = [
    "GaussianDenseGenerationSpec",
    "GaussianDenseHandoffSpec",
    "GaussianDenseOverlapSpec",
    "GaussianGenerationSpec",
    "parse_gaussian_dense_generation_job",
    "parse_gaussian_dense_handoff_job",
    "parse_gaussian_dense_overlap_job",
    "parse_gaussian_generation_job",
    "run_v1",
    "run_v2",
    "run_v3",
    "run_v4",
    "run_v5",
    "select_image_variant",
]

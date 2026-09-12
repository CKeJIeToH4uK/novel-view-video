"""Явная проверка только ресурсов выбранной задачи, без научного запуска."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from novel_view.config.job import ResolvedJob, load_resolved_job
from novel_view.runtime.context import CONTAINER_ROOTS, ContainerRoots
from novel_view.runtime.executables import GEN3C_PYTHON, VGGT_PYTHON
from novel_view.workflows.runner import select_job

if TYPE_CHECKING:
    from novel_view.inputs.euvs.spec import EuvsSelection
    from novel_view.inputs.waymo.types import WaymoOfficialPartition
    from novel_view.workflows.euvs_generation import EuvsGenerationSpec
    from novel_view.workflows.gaussian_generation import GaussianGenerationSpec


def doctor_job(
    job_file: Path,
    config_root: Path,
    *,
    roots: ContainerRoots = CONTAINER_ROOTS,
) -> int:
    """Прочитать названные входы и проверить нужный задаче Python-prefix."""
    job = load_resolved_job(job_file, config_root)
    selected = select_job(job)
    if selected.usage_notes:
        print(json.dumps({"usage_notes": list(selected.usage_notes)}), flush=True)
    gpu_count = selected.preset.gpu_count
    name = job.workflow.name
    code = 0
    if name == "euvs_source_views":
        code = _euvs_source_views(job, roots, gpu_count)
    elif name == "euvs_generation":
        code = _euvs_generation(job, roots, gpu_count)
    elif name == "euvs_evaluation":
        code = _euvs_evaluation(job, roots, gpu_count)
    elif name == "euvs_comparison":
        _euvs_comparison(job, roots)
    elif name == "gaussian_generation":
        code = _gaussian_generation(job, roots, gpu_count)
    elif name == "ddw_preparation":
        code = _ddw_preparation(job, roots, gpu_count)
    elif name == "gen3c_training":
        code = _gen3c_training(job, roots, gpu_count)
    elif name == "ddw_evaluation":
        code = _ddw_evaluation(job, roots, gpu_count)
    elif name == "gen3c_checkpoint_selection":
        _checkpoint_selection(job, roots)
    elif name == "waymo_depth_comparison":
        code = _waymo_depth_comparison(job, roots, gpu_count)
    elif name == "waymo_depth_selection":
        _waymo_depth_selection(job, roots)
    elif name in ("waymo_ddw_canary", "waymo_ddw_probe", "waymo_ddw_survey"):
        code = _legacy_ddw(job, roots, gpu_count)
    print(
        json.dumps(
            {
                "doctor": "job",
                "job": job.name,
                "status": "passed" if code == 0 else "failed",
            }
        )
    )
    return code


def _open_file(path: Path) -> None:
    with path.open("rb") as stream:
        stream.read(1)


def _backend(name: str, gpu_count: int, *arguments: str) -> int:
    python = VGGT_PYTHON if name == "vggt" else GEN3C_PYTHON
    return subprocess.run(
        [
            str(python),
            "-m",
            "novel_view.diagnostics._job_worker",
            "--backend",
            name,
            "--gpu-count",
            str(gpu_count),
            *arguments,
        ],
        check=False,
    ).returncode


def _euvs_frames(
    selection: EuvsSelection,
    dataset: Path,
    *,
    target_rgb: bool = False,
) -> None:
    from novel_view.inputs.euvs.index import FramesIndex

    frames = FramesIndex.load(dataset)
    for pair in selection.pairs:
        for view, read_rgb in ((pair.source, True), (pair.target, target_rgb)):
            for frame in frames.resolve(
                view.image_tokens,
                location=pair.location,
                traversal=view.traversal,
                channel=pair.channel,
            ):
                if read_rgb:
                    _open_file(frame.image_path)
                _open_file(frame.db_path)


def _saved_source_views(directory: Path) -> None:
    from novel_view.models.vggt.record import PREDICTION_FILES

    _open_file(directory / "summary.json")
    for filename in PREDICTION_FILES.values():
        _open_file(directory / filename)


def _euvs_source_views(job: ResolvedJob, roots: ContainerRoots, count: int) -> int:
    from novel_view.generation.euvs.source_views import unique_source_views
    from novel_view.inputs.euvs.spec import EuvsSelection, load_euvs_selection
    from novel_view.workflows.euvs_source_views import (
        _CHECKPOINT,
        parse_euvs_source_views_job,
    )

    spec = parse_euvs_source_views_job(job)
    selection = load_euvs_selection(roots.selections.parent / spec.selection)
    _euvs_frames(
        EuvsSelection(unique_source_views(selection)), roots.data / spec.dataset
    )
    _open_file(roots.models / _CHECKPOINT)
    return _backend("vggt", count)


def _generation_weights(
    spec: EuvsGenerationSpec | GaussianGenerationSpec,
    roots: ContainerRoots,
    count: int,
    backend: str = "generation",
) -> int:
    from novel_view.generation.gen3c.model_request import build_model_request
    from novel_view.generation.gen3c.recipe import Gen3cGenerationRecipe

    model, _ = build_model_request(roots.models, spec.generation)
    _open_file(model.artifact.network_checkpoint)
    shared = model.artifact.shared_checkpoint_root
    if model.artifact.lora is not None:
        base, _ = build_model_request(
            roots.models,
            Gen3cGenerationRecipe("gen3c/base", None, spec.generation.seed),
        )
        _open_file(base.artifact.network_checkpoint)
    return _backend(
        backend,
        count,
        "--t5-root",
        str(shared / "google-t5/t5-11b"),
        "--vae-root",
        str(shared / "Cosmos-Tokenize1-CV8x8x8-720p"),
    )


def _euvs_generation(job: ResolvedJob, roots: ContainerRoots, count: int) -> int:
    from novel_view.generation.euvs.source_views import source_view_directory_name
    from novel_view.inputs.euvs.spec import load_euvs_selection
    from novel_view.workflows.euvs_generation import parse_euvs_generation_job

    spec = parse_euvs_generation_job(job)
    selection = load_euvs_selection(roots.selections.parent / spec.selection)
    _euvs_frames(selection, roots.data / spec.dataset)
    for pair in selection.pairs:
        _saved_source_views(
            roots.runs
            / spec.source_views_attempt
            / "source-views"
            / source_view_directory_name(pair.source.image_tokens)
        )
    return _generation_weights(spec, roots, count)


def _euvs_evaluation(job: ResolvedJob, roots: ContainerRoots, count: int) -> int:
    from novel_view.evaluation.euvs.spec import (
        GenerationAttemptReference,
        parse_euvs_evaluation_job,
    )
    from novel_view.generation.euvs.record import load_run_record
    from novel_view.inputs.euvs.spec import load_euvs_selection
    from novel_view.workflows.euvs_evaluation import (
        _MODEL_DIRECTORY,
        plan_euvs_evaluation_records,
    )

    spec = parse_euvs_evaluation_job(job)
    selection = load_euvs_selection(roots.selections.parent / spec.selection)
    planned = plan_euvs_evaluation_records(spec, selection, roots.runs, roots.runs)
    _euvs_frames(selection, roots.data / spec.dataset, target_rgb=True)
    legacy = not isinstance(spec.generation, GenerationAttemptReference)
    for item in planned:
        # Используется только входной record_path; result_path не открывается.
        path = item.record_path
        record = load_run_record(path, allow_legacy=legacy)
        _open_file(path.with_name(record.output.file))
        _saved_source_views(roots.runs / record.geometry.runs_relative_directory)
    models = roots.models / _MODEL_DIRECTORY
    _open_file(models / "sam2/sam2.1_hiera_large.pt")
    _open_file(models / "dinov2/dinov2_vitb14_pretrain.pth")
    return _backend(
        "euvs_evaluation",
        count,
        "--grounding-root",
        str(models / "grounding-dino-tiny"),
        "--alexnet-root",
        str(models / "torch"),
    )


def _euvs_comparison(job: ResolvedJob, roots: ContainerRoots) -> None:
    from novel_view.evaluation.euvs.benchmark import _load_cell
    from novel_view.evaluation.euvs.spec import parse_euvs_comparison_job
    from novel_view.inputs.euvs.spec import load_euvs_selection

    spec = parse_euvs_comparison_job(job)
    selection = load_euvs_selection(roots.selections.parent / spec.selection)
    for variant, attempt in (
        ("base", spec.base_evaluation),
        ("tuned", spec.tuned_evaluation),
    ):
        for pair in selection.pairs:
            _load_cell(variant, pair, attempt, roots.runs)


def _gaussian_generation(job: ResolvedJob, roots: ContainerRoots, count: int) -> int:
    from novel_view.inputs.gaussian.reader import read_gaussian_export
    from novel_view.inputs.gaussian.spec import (
        GaussianFullSequenceSelection,
        GaussianIndependentClipsSelection,
        GaussianDenseIndependentSelection,
        GaussianDenseOverlapSelection,
        GaussianDenseHandoffSelection,
        load_gaussian_selection,
        parse_gaussian_job_input,
        parse_gaussian_dense_job_input,
    )
    from novel_view.workflows.gaussian_generation import (
        parse_gaussian_generation_job,
        parse_gaussian_dense_generation_job,
        parse_gaussian_dense_overlap_job,
        parse_gaussian_dense_handoff_job,
    )

    version = job.workflow.version
    if version == 5:
        from novel_view.generation.gaussian.handoff import select_dense_outputs

        spec = parse_gaussian_dense_handoff_job(job)
        handoff_selection = load_gaussian_selection(
            roots.selections.parent / spec.input.selection
        )
        if not isinstance(handoff_selection, GaussianDenseHandoffSelection):
            raise ValueError("gaussian_generation/v5 requires dense_handoff selection")
        outputs = select_dense_outputs(
            roots.runs / spec.input.independent_attempt,
            roots.runs / spec.input.overlap_attempt,
            handoff_selection.dense_camera_ids,
            scene_id=handoff_selection.scene_id,
            bake_dense_stride=spec.bake_dense_stride,
        )
        for output in outputs:
            for row in output.rows:
                _open_file(row.generated_rgb)
        return 0
    backend = "generation"
    if version in (1, 2):
        from novel_view.generation.gaussian.full import build_gaussian_full_sequence
        from novel_view.generation.gaussian.clips import plan_native_clips

        input_spec = parse_gaussian_job_input(job)
        selection = load_gaussian_selection(
            roots.selections.parent / input_spec.selection
        )
        if version == 1 and not isinstance(selection, GaussianFullSequenceSelection):
            raise ValueError("gaussian_generation/v1 requires full_sequence selection")
        if version == 2 and not isinstance(
            selection, GaussianIndependentClipsSelection
        ):
            raise ValueError(
                "gaussian_generation/v2 requires independent_clips selection"
            )
        generation = parse_gaussian_generation_job(job)
        export = read_gaussian_export(roots.data / input_spec.info_file)
        sequence = build_gaussian_full_sequence(
            export,
            cast(
                GaussianFullSequenceSelection | GaussianIndependentClipsSelection,
                selection,
            ),
            generation.frame_rate,
        )
        if version == 1:
            frames = sequence.frames
        else:
            clips = plan_native_clips(
                sequence,
                cast(GaussianIndependentClipsSelection, selection).clip_indices,
            )
            frames = tuple(frame for clip in clips for frame in clip.sequence.frames)
    else:
        from novel_view.generation.gaussian.dense import select_dense_export

        dense_input = parse_gaussian_dense_job_input(job)
        selection = load_gaussian_selection(
            roots.selections.parent / dense_input.selection
        )
        if version == 3:
            if not isinstance(selection, GaussianDenseIndependentSelection):
                raise ValueError(
                    "gaussian_generation/v3 requires dense_independent selection"
                )
            generation = parse_gaussian_dense_generation_job(job).generation
        else:
            if not isinstance(selection, GaussianDenseOverlapSelection):
                raise ValueError(
                    "gaussian_generation/v4 requires dense_overlap21 selection"
                )
            overlap = parse_gaussian_dense_overlap_job(job)
            generation = overlap.generation
            _open_file(roots.models / overlap.context_depth_checkpoint)
            backend = "generation_moge"
        export = read_gaussian_export(roots.data / dense_input.info_file)
        if export.scene_id != selection.scene_id:
            raise ValueError("Gaussian export disagrees with selected scene")
        frames = select_dense_export(export, selection.source_pose_range).frames
        _open_file(roots.data / dense_input.source_transforms)
        _open_file(roots.data / dense_input.source_intrinsics)
    for frame in frames:
        for path in (frame.rgb_path, frame.depth_path, frame.mask_path):
            _open_file(path)
    return _generation_weights(generation, roots, count, backend)


def _waymo_segment(
    root: Path, partition: WaymoOfficialPartition, segment_id: str
) -> None:
    from novel_view.inputs.waymo.index import WaymoV2Files, _ALL_COMPONENTS

    files = WaymoV2Files(root, partition, segment_id)
    for component in _ALL_COMPONENTS:
        _open_file(files.component(component))


def _ddw_preparation(job: ResolvedJob, roots: ContainerRoots, count: int) -> int:
    from novel_view.preparation.waymo_ddw.selection import load_selection
    from novel_view.workflows.ddw_preparation import parse_job

    spec = parse_job(job)
    selection = load_selection(roots.selections.parent / spec.input.selection)
    for sample in selection.samples:
        _waymo_segment(
            roots.data / spec.input.dataset, sample.partition, sample.segment_id
        )
    _open_file(roots.models / spec.recipe.depth_checkpoint)
    return _backend(
        "ddw_preparation",
        count,
        "--t5-root",
        str(roots.models / spec.recipe.text_encoder),
        "--vae-root",
        str(roots.models / spec.recipe.tokenizer),
    )


def _waymo_depth_comparison(job: ResolvedJob, roots: ContainerRoots, count: int) -> int:
    from novel_view.preparation.waymo_depth.spec import load_depth_clip_selection
    from novel_view.workflows.waymo_depth_comparison import parse_job

    spec = parse_job(job)
    clip = load_depth_clip_selection(roots.selections.parent / spec.input.selection)
    _waymo_segment(roots.data / spec.input.dataset, clip.official_partition, clip.segment_id)
    _open_file(roots.models / spec.recipe.moge_checkpoint)
    _open_file(roots.models / spec.recipe.vggt_checkpoint)
    code = _backend("moge", count)
    return code if code else _backend("vggt", count)


def _waymo_depth_selection(job: ResolvedJob, roots: ContainerRoots) -> None:
    from novel_view.preparation.waymo_depth.candidate_record import read_candidate_record
    from novel_view.preparation.waymo_depth.selection import (
        combine_candidate_outcomes,
        normalize_depth_selection_evidence,
    )
    from novel_view.preparation.waymo_depth.spec import load_depth_report_selection
    from novel_view.workflows.waymo_depth_selection import parse_job

    spec = parse_job(job)
    locators = load_depth_report_selection(roots.selections.parent / spec.selection)
    records = tuple(read_candidate_record(roots.runs / locator) for locator in locators)
    keys = tuple(record.clip_key for record in records)
    outcomes = tuple(record.outcomes for record in records)
    combine_candidate_outcomes(keys, outcomes)
    if job.workflow.version == 2:
        normalize_depth_selection_evidence(keys, outcomes)


def _legacy_ddw(job: ResolvedJob, roots: ContainerRoots, count: int) -> int:
    from novel_view.preparation.waymo_depth.selection import DepthGateDecision
    from novel_view.preparation.waymo_depth.selection_record import read_selection_record
    from novel_view.preparation.waymo_depth.spec import load_depth_clip_selection
    from novel_view.workflows.waymo_ddw_canary import parse_job

    spec = parse_job(job)
    selection = read_selection_record(roots.runs / spec.input.depth_selection)
    if not isinstance(selection.outcome, DepthGateDecision) or (
        selection.outcome.selected_backend != "moge-v1-lidar-scale"
    ):
        raise ValueError("legacy DDW requires a selected MoGe depth result")
    clip = load_depth_clip_selection(roots.selections.parent / spec.input.selection)
    _waymo_segment(roots.data / spec.input.dataset, clip.official_partition, clip.segment_id)
    _open_file(roots.models / spec.recipe.moge_checkpoint)
    return _backend("legacy_ddw", count)


def _gen3c_training(job: ResolvedJob, roots: ContainerRoots, count: int) -> int:
    from novel_view.preparation.waymo_ddw.record import read_prepared_record
    from novel_view.training.gen3c.data import (
        load_r4c_legacy_index_map,
        load_r4c_training_split,
        resolve_r4c_items,
    )
    from novel_view.workflows.gen3c_training import resolve_training_job
    from novel_view.training.gen3c.lora.spec import R4cLoraEdmSpec
    from novel_view.training.gen3c.lora.depth_spec import R4cLoraDepthSpec

    spec = resolve_training_job(job)
    path = roots.prepared / spec.input.prepared_record
    prepared = read_prepared_record(path)
    split = load_r4c_training_split(roots.selections.parent / spec.input.split)
    training, validation = resolve_r4c_items(prepared, split)
    by_id = {item.sample_id: item for item in prepared.items}
    for item in training + validation:
        for relative in (item.base_latent, item.pose_latent, item.prompt_embedding):
            _open_file(path.parent / relative)
        if job.workflow.version == 2:
            _open_file(path.parent / by_id[item.sample_id].lidar_depth)
    if spec.input.legacy_index_map is not None:
        load_r4c_legacy_index_map(roots.selections.parent / spec.input.legacy_index_map)
    recipe = cast(R4cLoraEdmSpec | R4cLoraDepthSpec, spec.recipe)
    _open_file(roots.models / recipe.base_checkpoint)
    return _backend("gen3c_training", count)


def _checkpoint_selection(job: ResolvedJob, roots: ContainerRoots) -> None:
    from novel_view.training.gen3c.lora.record import read_checkpoint_record_v1
    from novel_view.training.gen3c.lora.depth_record import read_checkpoint_record_v2
    from novel_view.workflows.gen3c_checkpoint_selection import parse_job

    reader = (
        read_checkpoint_record_v1
        if job.workflow.version == 1
        else read_checkpoint_record_v2
    )
    for path in parse_job(job).checkpoint_records:
        reader(roots.runs / path)


def _ddw_evaluation(job: ResolvedJob, roots: ContainerRoots, count: int) -> int:
    from novel_view.evaluation.ddw.samples import resolve_ddw_evaluation_samples
    from novel_view.preparation.waymo_ddw.record import read_prepared_record
    from novel_view.training.gen3c.data import load_r4c_training_split
    from novel_view.training.gen3c.lora.record import (
        CheckpointRecordV1,
        read_checkpoint_record_v1,
    )
    from novel_view.training.gen3c.lora.depth_record import (
        CheckpointRecordV2,
        read_checkpoint_record_v2,
    )
    from novel_view.workflows.ddw_evaluation import parse_job

    spec = parse_job(job)
    path = roots.prepared / spec.input.prepared_record
    prepared = read_prepared_record(path)
    split = load_r4c_training_split(roots.selections.parent / spec.input.split)
    _, samples = resolve_ddw_evaluation_samples(prepared, split)
    for sample in samples:
        item = sample.model_input
        for relative in (
            item.base_latent,
            item.pose_latent,
            item.prompt_embedding,
            Path(sample.prepared.lidar_depth),
        ):
            _open_file(path.parent / relative)
        source = sample.prepared
        _waymo_segment(roots.data / "waymo", source.partition, source.segment_id)
    checkpoints: tuple[
        tuple[str, Callable[[Path], CheckpointRecordV1 | CheckpointRecordV2]], ...
    ] = (
        (spec.input.v1_checkpoint_record, read_checkpoint_record_v1),
        (spec.input.v2_checkpoint_record, read_checkpoint_record_v2),
    )
    for checkpoint_locator, reader in checkpoints:
        record_path = roots.runs / checkpoint_locator
        record = reader(record_path)
        _open_file(record_path.parent / record.checkpoint)
    _open_file(roots.models / spec.recipe.base_checkpoint)
    _open_file(roots.models / spec.recipe.depth_checkpoint)
    return _backend(
        "ddw_evaluation", count, "--vae-root", str(roots.models / spec.recipe.tokenizer)
    )

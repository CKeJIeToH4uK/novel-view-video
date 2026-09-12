"""Run the historical three-camera MoGe then VGGT comparison."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from novel_view.config.job import ResolvedJob
from novel_view.preparation.waymo_depth.spec import (
    WaymoDepthComparisonSpec,
    load_depth_clip_selection,
    parse_depth_comparison_spec,
)
from novel_view.runtime.context import RuntimeContext


def parse_job(job: ResolvedJob) -> WaymoDepthComparisonSpec:
    """Parse the strict external comparison job without opening data or models."""
    return parse_depth_comparison_spec(
        job.input,
        job.parameters,
        workflow_name=job.workflow.name,
        workflow_version=job.workflow.version,
        execution_preset=job.execution.preset,
    )


def select_image_variant(job: ResolvedJob) -> str:
    parse_job(job)
    return "moge"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Build shared evidence once, then persist MoGe before starting VGGT."""
    from novel_view.inputs.waymo.reader import WaymoV2FrameReader
    from novel_view.models.gen3c.environment import gen3c_environment
    from novel_view.models.moge.backend import MogeResources
    from novel_view.models.vggt.backend import VggtOmegaResources
    from novel_view.preparation.waymo_depth.clip import build_depth_comparison_clip
    from novel_view.preparation.waymo_depth.comparison import (
        DepthComparisonResources,
        compare_depth_candidates,
    )
    from novel_view.preparation.waymo_depth.candidate_record import write_candidate_record

    spec = parse_job(job)
    clip_key = load_depth_clip_selection(
        runtime.roots.selections.parent / spec.input.selection
    )
    reader = WaymoV2FrameReader(
        runtime.roots.data / spec.input.dataset,
        clip_key.official_partition,
        clip_key.segment_id,
        clip_key.start_frame_index,
        len(clip_key.frame_timestamps_micros),
    )
    log_root = runtime.attempt_root / "logs"
    log_root.mkdir(exist_ok=True)
    environment = gen3c_environment(runtime, runtime.roots.cache)
    resources = DepthComparisonResources(
        MogeResources(
            runtime.roots.cache,
            runtime.roots.models / spec.recipe.moge_checkpoint,
            environment,
        ),
        VggtOmegaResources(
            runtime.roots.cache,
            runtime.roots.models / spec.recipe.vggt_checkpoint,
        ),
    )
    with TemporaryDirectory(prefix="waymo-depth-", dir=runtime.roots.cache) as temporary:
        clip = build_depth_comparison_clip(clip_key, reader, Path(temporary))
        outcomes = compare_depth_candidates(
            clip,
            resources,
            runtime.attempt_root,
            log_root,
        )
    result = runtime.attempt_root / "result.json"
    write_candidate_record(result, clip_key, outcomes)
    print(result)
    return 0


__all__ = ["parse_job", "run_v1", "select_image_variant"]

"""Prepare ordered Waymo FRONT DDW items for Gen3C R4c training."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from novel_view.config.job import ResolvedJob
from novel_view.preparation.waymo_ddw.spec import (
    WaymoDdwSpec,
    parse_waymo_ddw_spec,
)
from novel_view.runtime.context import RuntimeContext


def parse_job(job: ResolvedJob) -> WaymoDdwSpec:
    """Parse the strict external DDW preparation job once."""
    return parse_waymo_ddw_spec(
        job.input,
        job.parameters,
        workflow_name=job.workflow.name,
        workflow_version=job.workflow.version,
        execution_preset=job.execution.preset,
    )


def select_image_variant(job: ResolvedJob) -> str:
    """Select the literal MoGe image without importing model stacks."""
    parse_job(job)
    return "moge"


def run_v1(job: ResolvedJob, runtime: RuntimeContext) -> int:
    """Prepare selection-order DDW items through the one supported v1 path."""
    from novel_view.generation.gen3c.resources import build_generation_resources
    from novel_view.models.moge.backend import MogeResources
    from novel_view.preparation.waymo_ddw.bake import (
        bake_items,
        bake_prompt,
        iter_vae_batches,
        stage_bake_input,
    )
    from novel_view.preparation.waymo_ddw.condition import build_condition
    from novel_view.preparation.waymo_ddw.depth import predict_moge_metric_depth
    from novel_view.preparation.waymo_ddw.raster import build_front_raster
    from novel_view.preparation.waymo_ddw.record import (
        build_prepared_item,
        build_prepared_record,
        write_prepared_record,
    )
    from novel_view.preparation.waymo_ddw.selection import load_selection
    from novel_view.preparation.waymo_ddw.source import prepare_front_source
    from novel_view.preparation.waymo_ddw.target_path import build_target_path
    from novel_view.preparation.waymo_ddw.warp import WarpResources

    spec = parse_job(job)
    selection = load_selection(runtime.roots.selections.parent / spec.input.selection)
    output_root = runtime.roots.prepared / "waymo-ddw" / job.name
    log_root = runtime.attempt_root / "logs" / "ddw-preparation"
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    process_resources = build_generation_resources(runtime, 1)
    environment = process_resources.environment_overrides
    moge = MogeResources(
        runtime.roots.cache,
        runtime.roots.models / spec.recipe.depth_checkpoint,
        environment,
    )
    warp = WarpResources(runtime.roots.cache, environment)
    bake_prompt(
        runtime.roots.models / spec.recipe.text_encoder,
        output_root / "empty-prompt.pt",
        log_root / "empty-prompt.log",
        environment,
    )

    prepared_items = []
    for batch_index, selected_batch in enumerate(iter_vae_batches(selection.samples)):
        with TemporaryDirectory(
            prefix="ddw-batch-",
            dir=runtime.roots.cache,
        ) as temporary:
            scratch_root = Path(temporary)
            bake_tasks = []
            batch_items = []
            for selected in selected_batch:
                source = prepare_front_source(
                    runtime.roots.data / spec.input.dataset,
                    selected,
                )
                front = build_front_raster(source)
                depth = predict_moge_metric_depth(
                    front,
                    moge,
                    log_root / f"{selected.sample_id}-moge.log",
                )
                path = build_target_path(
                    front.world_to_camera_cv,
                    selected.magnitude_m,
                    selected.sign,
                )
                condition = build_condition(
                    front,
                    depth,
                    path,
                    warp,
                    log_root / f"{selected.sample_id}-warp-out.log",
                    log_root / f"{selected.sample_id}-warp-back.log",
                )
                bake_tasks.append(
                    stage_bake_input(
                        front,
                        condition,
                        scratch_root,
                        output_root,
                    )
                )
                batch_items.append(build_prepared_item(selected, front.frame_keys))
                del source, front, depth, path, condition

            bake_items(
                tuple(bake_tasks),
                runtime.roots.models / spec.recipe.tokenizer,
                scratch_root / "vae-tasks.json",
                log_root / f"vae-batch-{batch_index:04d}.log",
                environment,
            )
            prepared_items.extend(batch_items)

    record_path = output_root / "prepared.json"
    write_prepared_record(
        record_path,
        build_prepared_record(spec, tuple(prepared_items)),
    )
    print(record_path)
    return 0


__all__ = ["parse_job", "run_v1", "select_image_variant"]

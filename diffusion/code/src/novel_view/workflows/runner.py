"""Ленивый выбор и operational lifecycle явного versioned workflow."""

from collections.abc import Callable
from dataclasses import dataclass

from novel_view.config.job import ResolvedJob
from novel_view.config.job import resolved_job_to_mapping
from novel_view.runs.record import AttemptImage
from novel_view.runs.store import RunStore
from novel_view.runtime.context import RuntimeContext
from novel_view.runtime.presets import ExecutionPreset, load_execution_preset
from novel_view.runtime.process import STOP_EXIT_CODE, StopRequested


@dataclass(frozen=True, slots=True)
class JobSelection:
    job_name: str
    image_variant: str
    preset: ExecutionPreset
    usage_notes: tuple[str, ...] = ()


def select_job(job: ResolvedJob) -> JobSelection:
    workflow_key = (job.workflow.name, job.workflow.version)
    usage_notes: tuple[str, ...] = ()
    if workflow_key == ("_lifecycle_smoke", 1):
        from novel_view.workflows._lifecycle_smoke import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("euvs_source_views", 1):
        from novel_view.models.vggt.spec import vggt_usage_note
        from novel_view.workflows.euvs_source_views import _CHECKPOINT, select_image_variant

        image_variant = select_image_variant(job)
        usage_notes = (vggt_usage_note(str(_CHECKPOINT)),)
    elif workflow_key in {
        ("euvs_generation", 1),
        ("euvs_generation", 2),
    }:
        from novel_view.workflows.euvs_generation import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("euvs_evaluation", 1):
        from novel_view.workflows.euvs_evaluation import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("euvs_comparison", 1):
        from novel_view.workflows.euvs_comparison import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key in (
        ("gaussian_generation", 1),
        ("gaussian_generation", 2),
        ("gaussian_generation", 3),
        ("gaussian_generation", 4),
        ("gaussian_generation", 5),
    ):
        from novel_view.workflows.gaussian_generation import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("waymo_keysets", 1):
        from novel_view.workflows.waymo_keysets import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("waymo_depth_comparison", 1):
        from novel_view.models.vggt.spec import vggt_usage_note
        from novel_view.workflows.waymo_depth_comparison import parse_job

        spec = parse_job(job)
        image_variant = "moge"
        usage_notes = (vggt_usage_note(str(spec.recipe.vggt_checkpoint)),)
    elif workflow_key in {
        ("waymo_depth_selection", 1),
        ("waymo_depth_selection", 2),
    }:
        from novel_view.workflows.waymo_depth_selection import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key in {("waymo_ddw_canary", 1), ("waymo_ddw_canary", 2)}:
        from novel_view.workflows.waymo_ddw_canary import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("waymo_ddw_probe", 1):
        from novel_view.workflows.waymo_ddw_probe import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("waymo_ddw_survey", 1):
        from novel_view.workflows.waymo_ddw_survey import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("waymo_ddw_fit", 1):
        from novel_view.workflows.waymo_ddw_fit import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("waymo_ddw_fit_collection", 1):
        from novel_view.workflows.waymo_ddw_fit_collection import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("waymo_ddw_audit", 1):
        from novel_view.workflows.waymo_ddw_audit import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("ddw_preparation", 1):
        from novel_view.workflows.ddw_preparation import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key == ("ddw_evaluation", 1):
        from novel_view.workflows.ddw_evaluation import select_image_variant

        image_variant = select_image_variant(job)
    elif workflow_key in {
        ("gen3c_checkpoint_selection", 1),
        ("gen3c_checkpoint_selection", 2),
    }:
        from novel_view.workflows.gen3c_checkpoint_selection import (
            select_image_variant,
        )

        image_variant = select_image_variant(job)
    elif workflow_key in {
        ("gen3c_training", 1),
        ("gen3c_training", 2),
    }:
        from novel_view.workflows.gen3c_training import select_image_variant

        image_variant = select_image_variant(job)
    else:
        raise ValueError(
            f"unsupported workflow: {job.workflow.name} v{job.workflow.version}"
        )

    return JobSelection(
        job_name=job.name,
        image_variant=image_variant,
        preset=load_execution_preset(job.execution.preset),
        usage_notes=usage_notes,
    )


def execute_job(job: ResolvedJob, runtime: RuntimeContext) -> int:
    run_workflow: Callable[[ResolvedJob, RuntimeContext], int]
    workflow_key = (job.workflow.name, job.workflow.version)
    if workflow_key == ("_lifecycle_smoke", 1):
        from novel_view.workflows._lifecycle_smoke import run_v1 as run_workflow
    elif workflow_key == ("euvs_source_views", 1):
        from novel_view.workflows.euvs_source_views import run_v1 as run_workflow
    elif workflow_key == ("euvs_generation", 1):
        from novel_view.workflows.euvs_generation import run_v1 as run_workflow
    elif workflow_key == ("euvs_generation", 2):
        from novel_view.workflows.euvs_generation import run_v2 as run_workflow
    elif workflow_key == ("euvs_evaluation", 1):
        from novel_view.workflows.euvs_evaluation import run_v1 as run_workflow
    elif workflow_key == ("euvs_comparison", 1):
        from novel_view.workflows.euvs_comparison import run_v1 as run_workflow
    elif workflow_key == ("gaussian_generation", 1):
        from novel_view.workflows.gaussian_generation import run_v1 as run_workflow
    elif workflow_key == ("gaussian_generation", 2):
        from novel_view.workflows.gaussian_generation import run_v2 as run_workflow
    elif workflow_key == ("gaussian_generation", 3):
        from novel_view.workflows.gaussian_generation import run_v3 as run_workflow
    elif workflow_key == ("gaussian_generation", 4):
        from novel_view.workflows.gaussian_generation import run_v4 as run_workflow
    elif workflow_key == ("gaussian_generation", 5):
        from novel_view.workflows.gaussian_generation import run_v5 as run_workflow
    elif workflow_key == ("waymo_keysets", 1):
        from novel_view.workflows.waymo_keysets import run_v1 as run_workflow
    elif workflow_key == ("waymo_depth_comparison", 1):
        from novel_view.workflows.waymo_depth_comparison import run_v1 as run_workflow
    elif workflow_key == ("waymo_depth_selection", 1):
        from novel_view.workflows.waymo_depth_selection import run_v1 as run_workflow
    elif workflow_key == ("waymo_depth_selection", 2):
        from novel_view.workflows.waymo_depth_selection import run_v2 as run_workflow
    elif workflow_key == ("waymo_ddw_canary", 1):
        from novel_view.workflows.waymo_ddw_canary import run_v1 as run_workflow
    elif workflow_key == ("waymo_ddw_canary", 2):
        from novel_view.workflows.waymo_ddw_canary import run_v2 as run_workflow
    elif workflow_key == ("waymo_ddw_probe", 1):
        from novel_view.workflows.waymo_ddw_probe import run_v1 as run_workflow
    elif workflow_key == ("waymo_ddw_survey", 1):
        from novel_view.workflows.waymo_ddw_survey import run_v1 as run_workflow
    elif workflow_key == ("waymo_ddw_fit", 1):
        from novel_view.workflows.waymo_ddw_fit import run_v1 as run_workflow
    elif workflow_key == ("waymo_ddw_fit_collection", 1):
        from novel_view.workflows.waymo_ddw_fit_collection import run_v1 as run_workflow
    elif workflow_key == ("waymo_ddw_audit", 1):
        from novel_view.workflows.waymo_ddw_audit import run_v1 as run_workflow
    elif workflow_key == ("ddw_preparation", 1):
        from novel_view.workflows.ddw_preparation import run_v1 as run_workflow
    elif workflow_key == ("ddw_evaluation", 1):
        from novel_view.workflows.ddw_evaluation import run_v1 as run_workflow
    elif workflow_key == ("gen3c_training", 1):
        from novel_view.workflows.gen3c_training import run_v1 as run_workflow
    elif workflow_key == ("gen3c_training", 2):
        from novel_view.workflows.gen3c_training import run_v2 as run_workflow
    elif workflow_key == ("gen3c_checkpoint_selection", 1):
        from novel_view.workflows.gen3c_checkpoint_selection import (
            run_v1 as run_workflow,
        )
    elif workflow_key == ("gen3c_checkpoint_selection", 2):
        from novel_view.workflows.gen3c_checkpoint_selection import (
            run_v2 as run_workflow,
        )
    else:
        raise ValueError(
            f"unsupported workflow: {job.workflow.name} v{job.workflow.version}"
        )

    load_execution_preset(job.execution.preset)
    store = RunStore(runtime.attempt_root)
    attempt = store.start_attempt(
        job_name=job.name,
        run_id=runtime.run_id,
        attempt_id=runtime.attempt_id,
        attempt_kind=runtime.attempt_kind,
        resolved_job=resolved_job_to_mapping(job),
        selected_checkpoint=runtime.selected_checkpoint,
        image=AttemptImage(
            variant=runtime.image.variant,
            image_id=runtime.image.image_id,
            source_revision=runtime.image.source_revision,
            source_dirty=runtime.image.source_dirty,
            lock_revision=runtime.image.lock_revision,
        ),
        container_name=runtime.container_name,
    )

    try:
        worker_exit_code = run_workflow(job, runtime)
    except StopRequested:
        try:
            store.finish_attempt(
                attempt,
                state="stopped",
                worker_exit_code=STOP_EXIT_CODE,
            )
        except OSError:
            pass
        raise
    except BaseException:
        try:
            store.finish_attempt(
                attempt,
                state="failed",
                worker_exit_code=None,
            )
        except OSError:
            pass
        raise

    state = "succeeded" if worker_exit_code == 0 else "failed"
    store.finish_attempt(
        attempt,
        state=state,
        worker_exit_code=worker_exit_code,
    )
    return worker_exit_code

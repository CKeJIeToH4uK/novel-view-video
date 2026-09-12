"""Лёгкий внутренний CLI нового контейнерного orchestration."""

import argparse
from pathlib import Path
from typing import Sequence

from novel_view.cli.jobs import (
    HostPlanInputs,
    print_job_selection,
    print_job_plan,
    select_job_for_launcher,
)
from novel_view.cli.runs import (
    attempt_record_file,
    print_attempt_record,
    print_attempt_snapshot,
    read_resolved_job_from_attempt,
)
from novel_view.config.job import load_resolved_job
from novel_view.runtime.context import (
    CONTAINER_ROOTS,
    ImageMetadata,
    RuntimeContext,
)
from novel_view.runtime.process import STOP_EXIT_CODE, StopRequested


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="novel_view.cli.main")
    commands = parser.add_subparsers(dest="command", required=True)

    select_parser = commands.add_parser("select")
    select_parser.add_argument("--job", type=Path, required=True)
    select_parser.add_argument("--config-root", type=Path, required=True)

    doctor_parser = commands.add_parser("doctor")
    doctor_parser.add_argument("--job", type=Path, required=True)
    doctor_parser.add_argument("--config-root", type=Path, required=True)

    waymo_doctor = commands.add_parser("doctor-waymo")
    waymo_doctor.add_argument("diagnostic", choices=("ingress", "raster"))
    waymo_doctor.add_argument("arguments", nargs=argparse.REMAINDER)

    plan_parser = commands.add_parser("plan")
    plan_source = plan_parser.add_mutually_exclusive_group(required=True)
    plan_source.add_argument("--job", type=Path)
    plan_source.add_argument("--resume-record", type=Path)
    plan_parser.add_argument("--selected-preset", required=True)
    plan_parser.add_argument("--selected-checkpoint")
    plan_parser.add_argument("--config-root", type=Path, required=True)
    plan_parser.add_argument("--image-variant", required=True)
    plan_parser.add_argument("--image-id", required=True)
    plan_parser.add_argument("--image-source-revision", required=True)
    plan_parser.add_argument(
        "--image-source-dirty",
        choices=("true", "false"),
        required=True,
    )
    plan_parser.add_argument("--image-lock-revision", required=True)
    plan_parser.add_argument("--host-data-root", required=True)
    plan_parser.add_argument("--host-models-root", required=True)
    plan_parser.add_argument("--host-prepared-root", required=True)
    plan_parser.add_argument("--host-runs-root", required=True)
    plan_parser.add_argument("--host-cache-root", required=True)
    plan_parser.add_argument("--host-jobs-root", required=True)
    plan_parser.add_argument("--host-recipes-root", required=True)
    plan_parser.add_argument("--host-selections-root", required=True)
    plan_parser.add_argument("--host-uid", type=int, required=True)
    plan_parser.add_argument("--host-gid", type=int, required=True)
    plan_parser.add_argument("--host-gpu-ids", required=True)
    plan_parser.add_argument("--container-gpu-indices", required=True)
    plan_parser.add_argument("--ipc", required=True)
    plan_parser.add_argument("--memlock", required=True)

    execute_parser = commands.add_parser("execute")
    job_source = execute_parser.add_mutually_exclusive_group(required=True)
    job_source.add_argument("--job", type=Path)
    job_source.add_argument("--resume-record", type=Path)
    execute_parser.add_argument("--config-root", type=Path, required=True)
    execute_parser.add_argument("--run-id", required=True)
    execute_parser.add_argument("--attempt-id", required=True)
    execute_parser.add_argument(
        "--attempt-kind",
        choices=("run", "resume"),
        required=True,
    )
    execute_parser.add_argument("--attempt-root", type=Path, required=True)
    execute_parser.add_argument("--container-name", required=True)
    execute_parser.add_argument("--selected-checkpoint")
    _add_image_arguments(execute_parser)

    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--mode", required=True)
    worker_parser.add_argument("--exit-code", type=int, required=True)
    worker_parser.add_argument(
        "--role",
        choices=("leader", "child", "grandchild"),
        default="leader",
    )

    read_parser = commands.add_parser("read-attempt")
    read_parser.add_argument("--record", type=Path, required=True)

    resume_parser = commands.add_parser("select-resume")
    resume_parser.add_argument("--runs-root", type=Path, required=True)
    resume_parser.add_argument("--job-name", required=True)
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--attempt-id", required=True)

    snapshot_parser = commands.add_parser("describe-attempt")
    snapshot_parser.add_argument("--runs-root", type=Path, required=True)
    snapshot_parser.add_argument("--job-name", required=True)
    snapshot_parser.add_argument("--run-id", required=True)
    snapshot_parser.add_argument("--attempt-id")
    return parser


def _add_image_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-variant", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--image-source-revision", required=True)
    parser.add_argument(
        "--image-source-dirty",
        choices=("true", "false"),
        required=True,
    )
    parser.add_argument("--image-lock-revision", required=True)


def _image_metadata(arguments: argparse.Namespace) -> ImageMetadata:
    return ImageMetadata(
        variant=arguments.image_variant,
        image_id=arguments.image_id,
        source_revision=arguments.image_source_revision,
        source_dirty=arguments.image_source_dirty == "true",
        lock_revision=arguments.image_lock_revision,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if (
        arguments.command == "plan"
        and arguments.resume_record is not None
        and arguments.selected_checkpoint is None
    ):
        parser.error("plan --resume-record requires --selected-checkpoint")
    if arguments.command == "select":
        return select_job_for_launcher(arguments.job, arguments.config_root)

    if arguments.command == "doctor":
        from novel_view.diagnostics.job import doctor_job

        return doctor_job(arguments.job, arguments.config_root)

    if arguments.command == "doctor-waymo":
        from novel_view.diagnostics.waymo import run_raster_main, run_v2_ingress_main

        if arguments.diagnostic == "ingress":
            return run_v2_ingress_main(arguments.arguments)
        return run_raster_main(arguments.arguments)

    if arguments.command == "execute":
        from novel_view.workflows.runner import execute_job

        runtime = RuntimeContext(
            roots=CONTAINER_ROOTS,
            run_id=arguments.run_id,
            attempt_id=arguments.attempt_id,
            attempt_kind=arguments.attempt_kind,
            attempt_root=arguments.attempt_root,
            container_name=arguments.container_name,
            image=_image_metadata(arguments),
            selected_checkpoint=arguments.selected_checkpoint,
        )
        if arguments.resume_record is None:
            job = load_resolved_job(arguments.job, arguments.config_root)
        else:
            job = read_resolved_job_from_attempt(arguments.resume_record)
        try:
            return execute_job(job, runtime)
        except StopRequested:
            return STOP_EXIT_CODE

    if arguments.command == "worker":
        from novel_view.workflows._lifecycle_smoke import (
            LifecycleSmokeSpec,
            run_worker,
        )

        return run_worker(
            LifecycleSmokeSpec(
                mode=arguments.mode,
                exit_code=arguments.exit_code,
            ),
            role=arguments.role,
        )

    if arguments.command == "read-attempt":
        return print_attempt_record(arguments.record)

    if arguments.command == "select-resume":
        resume_record = attempt_record_file(
            arguments.runs_root,
            arguments.job_name,
            arguments.run_id,
            arguments.attempt_id,
        )
        return print_job_selection(
            read_resolved_job_from_attempt(resume_record),
            resume_record=resume_record,
        )

    if arguments.command == "describe-attempt":
        return print_attempt_snapshot(
            arguments.runs_root,
            arguments.job_name,
            arguments.run_id,
            arguments.attempt_id,
        )

    return print_job_plan(
        job_file=arguments.job,
        config_root=arguments.config_root,
        selected_preset=arguments.selected_preset,
        resume_record=arguments.resume_record,
        selected_checkpoint=arguments.selected_checkpoint,
        image=_image_metadata(arguments),
        host=HostPlanInputs(
            data_root=arguments.host_data_root,
            models_root=arguments.host_models_root,
            prepared_root=arguments.host_prepared_root,
            runs_root=arguments.host_runs_root,
            cache_root=arguments.host_cache_root,
            jobs_root=arguments.host_jobs_root,
            recipes_root=arguments.host_recipes_root,
            selections_root=arguments.host_selections_root,
            uid=arguments.host_uid,
            gid=arguments.host_gid,
            host_gpu_ids=(
                tuple(arguments.host_gpu_ids.split(","))
                if arguments.host_gpu_ids
                else ()
            ),
            container_gpu_indices=(
                tuple(map(int, arguments.container_gpu_indices.split(",")))
                if arguments.container_gpu_indices
                else ()
            ),
            ipc=arguments.ipc or None,
            memlock=arguments.memlock or None,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())

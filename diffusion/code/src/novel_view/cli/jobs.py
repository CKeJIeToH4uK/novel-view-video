"""Внутренние команды выбора job и построения читаемого plan."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence, cast

from novel_view.cli.runs import read_attempt_record
from novel_view.config.job import (
    ResolvedJob,
    load_resolved_job,
    resolved_job_from_mapping,
    resolved_job_to_mapping,
)
from novel_view.runtime.context import CONTAINER_ROOTS, ImageMetadata
from novel_view.workflows.runner import select_job


@dataclass(frozen=True, slots=True)
class HostPlanInputs:
    data_root: str
    models_root: str
    prepared_root: str
    runs_root: str
    cache_root: str
    jobs_root: str
    recipes_root: str
    selections_root: str
    uid: int
    gid: int
    host_gpu_ids: tuple[str, ...]
    container_gpu_indices: tuple[int, ...]
    ipc: str | None
    memlock: str | None


def select_job_for_launcher(job_file: Path, config_root: Path) -> int:
    return print_job_selection(load_resolved_job(job_file, config_root))


def print_job_selection(
    job: ResolvedJob,
    *,
    resume_record: Path | None = None,
) -> int:
    selection = select_job(job)
    print(f"job_name={selection.job_name}")
    print(f"image_variant={selection.image_variant}")
    print(f"preset={selection.preset.name}")
    if resume_record is not None:
        print(f"resume_record={resume_record}")
    return 0


def _mounts(host: HostPlanInputs) -> list[dict[str, object]]:
    return [
        {
            "source": host.data_root,
            "target": str(CONTAINER_ROOTS.data),
            "read_only": True,
        },
        {
            "source": host.models_root,
            "target": str(CONTAINER_ROOTS.models),
            "read_only": True,
        },
        {
            "source": host.prepared_root,
            "target": str(CONTAINER_ROOTS.prepared),
            "read_only": False,
        },
        {
            "source": host.runs_root,
            "target": str(CONTAINER_ROOTS.runs),
            "read_only": False,
        },
        {
            "source": host.cache_root,
            "target": str(CONTAINER_ROOTS.cache),
            "read_only": False,
        },
        {
            "source": host.jobs_root,
            "target": str(CONTAINER_ROOTS.jobs),
            "read_only": True,
        },
        {
            "source": host.recipes_root,
            "target": str(CONTAINER_ROOTS.recipes),
            "read_only": True,
        },
        {
            "source": host.selections_root,
            "target": str(CONTAINER_ROOTS.selections),
            "read_only": True,
        },
    ]


def _mount_argument(mount: dict[str, object]) -> str:
    value = f"type=bind,src={mount['source']},dst={mount['target']}"
    if mount["read_only"]:
        return f"{value},readonly"
    return value


def _future_attempt_command(
    *,
    job_file: Path | None,
    resume_record: Path | None,
    selected_checkpoint: str | None,
    run_id: str,
    job_name: str,
    image: ImageMetadata,
    host: HostPlanInputs,
    mounts: Sequence[dict[str, object]],
) -> list[str]:
    command = [
        "docker",
        "create",
        "--name",
        "<container-name>",
        "--label",
        f"io.distil3d.job={job_name}",
        "--label",
        f"io.distil3d.run-id={run_id}",
        "--label",
        "io.distil3d.attempt-id=<attempt-id>",
        "--label",
        f"io.distil3d.attempt-ref={job_name}/{run_id}/<attempt-id>",
        "--user",
        f"{host.uid}:{host.gid}",
        "--network",
        "none",
    ]
    if host.host_gpu_ids:
        command.extend(
            (
                "--gpus",
                f'"device={",".join(host.host_gpu_ids)}"',
                "--env",
                f'CUDA_VISIBLE_DEVICES={",".join(map(str, host.container_gpu_indices))}',
            )
        )
    if host.ipc is not None:
        command.extend(("--ipc", host.ipc))
    if host.memlock is not None:
        command.extend(("--ulimit", f"memlock={host.memlock}"))
    for mount in mounts:
        command.extend(("--mount", _mount_argument(mount)))
    command.extend(
        (
            image.image_id,
            "/opt/envs/core/bin/python",
            "-m",
            "novel_view.cli.main",
            "execute",
            "--job" if resume_record is None else "--resume-record",
            str(job_file if resume_record is None else resume_record),
            "--config-root",
            str(CONTAINER_ROOTS.jobs.parent),
            "--run-id",
            run_id,
            "--attempt-id",
            "<attempt-id>",
            "--attempt-kind",
            "run" if resume_record is None else "resume",
            "--attempt-root",
            f"{CONTAINER_ROOTS.runs}/{job_name}/{run_id}/attempts/<attempt-id>",
            "--container-name",
            "<container-name>",
            "--image-variant",
            image.variant,
            "--image-id",
            image.image_id,
            "--image-source-revision",
            image.source_revision,
            "--image-source-dirty",
            str(image.source_dirty).lower(),
            "--image-lock-revision",
            image.lock_revision,
        )
    )
    if resume_record is not None:
        command.extend(("--selected-checkpoint", cast(str, selected_checkpoint)))
    return command


def build_job_plan(
    *,
    job_file: Path | None,
    config_root: Path,
    selected_preset: str,
    image: ImageMetadata,
    host: HostPlanInputs,
    resume_record: Path | None = None,
    selected_checkpoint: str | None = None,
) -> dict[str, object]:
    if resume_record is None:
        job = load_resolved_job(cast(Path, job_file), config_root)
        run_id = "<run-id>"
    else:
        record = read_attempt_record(resume_record)
        job = resolved_job_from_mapping(cast(dict[str, object], record["resolved_job"]))
        run_id = cast(str, record["run_id"])
    selection = select_job(job)
    if selection.image_variant != image.variant:
        raise ValueError(
            f"selected image variant {selection.image_variant} does not match {image.variant}"
        )
    if selection.preset.name != selected_preset:
        raise ValueError(
            f"selected preset {selection.preset.name} does not match {selected_preset}"
        )

    mounts = _mounts(host)
    return {
        "schema_version": 1,
        "job": resolved_job_to_mapping(job),
        "usage_notes": list(selection.usage_notes),
        "execution": {
            "preset": selection.preset.name,
            "gpu_count": selection.preset.gpu_count,
            "host_gpu_ids": list(host.host_gpu_ids),
            "container_gpu_indices": list(host.container_gpu_indices),
            "ipc": host.ipc,
            "memlock": host.memlock,
            "network": "none",
        },
        "image": {
            "variant": image.variant,
            "id": image.image_id,
            "source_revision": image.source_revision,
            "source_dirty": image.source_dirty,
            "lock_revision": image.lock_revision,
        },
        "mounts": mounts,
        "future_attempt": {
            "command": _future_attempt_command(
                job_file=job_file,
                resume_record=resume_record,
                selected_checkpoint=selected_checkpoint,
                run_id=run_id,
                job_name=job.name,
                image=image,
                host=host,
                mounts=mounts,
            )
        },
    }


def print_job_plan(
    *,
    job_file: Path | None,
    config_root: Path,
    selected_preset: str,
    image: ImageMetadata,
    host: HostPlanInputs,
    resume_record: Path | None = None,
    selected_checkpoint: str | None = None,
) -> int:
    plan = build_job_plan(
        job_file=job_file,
        config_root=config_root,
        selected_preset=selected_preset,
        image=image,
        host=host,
        resume_record=resume_record,
        selected_checkpoint=selected_checkpoint,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0

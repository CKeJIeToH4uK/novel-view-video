"""Внешние job/recipe и config-only plan, без открытия machine roots."""

import json
from pathlib import Path

import pytest

from novel_view.cli.jobs import HostPlanInputs, ImageMetadata, build_job_plan
from novel_view.config.job import load_resolved_job
from novel_view.inputs.euvs.spec import load_euvs_selection
from novel_view.inputs.gaussian.spec import load_gaussian_selection
from novel_view.workflows.runner import select_job

ROOT = Path(__file__).resolve().parents[4]
SMOKE = ROOT / "jobs/examples/lifecycle-smoke/run.yaml"


def _plan(job_file, config_root=ROOT, image="cpu-test", preset="cpu_test", gpu_count=0, **resume):
    return build_job_plan(
        job_file=job_file,
        config_root=config_root,
        selected_preset=preset,
        image=ImageMetadata(image, "sha256:test-image", "test-revision", False, "sha256:test-lock"),
        host=HostPlanInputs(
            data_root="/missing/data",
            models_root="/missing/models",
            prepared_root="/missing/prepared",
            runs_root="/missing/runs",
            cache_root="/missing/cache",
            jobs_root="/checkout/jobs",
            recipes_root="/checkout/recipes",
            selections_root="/checkout/selections",
            uid=1000,
            gid=1000,
            host_gpu_ids=tuple(str(i) for i in range(gpu_count)),
            container_gpu_indices=tuple(range(gpu_count)),
            ipc=None,
            memlock=None,
        ),
        **resume,
    )


def test_plan_resolves_tracked_job_without_machine_inputs(tmp_path):
    job_file = tmp_path / "run.yaml"
    job_file.write_bytes(SMOKE.read_bytes())
    plan = _plan(job_file)
    assert plan["schema_version"] == 1
    assert plan["job"] == {
        "schema_version": 1,
        "name": "lifecycle-smoke",
        "workflow": {"name": "_lifecycle_smoke", "version": 1},
        "input": {},
        "recipe": None,
        "parameters": {"mode": "success", "exit_code": 0},
        "execution": {"preset": "cpu_test"},
    }
    assert plan["execution"] == {
        "preset": "cpu_test",
        "gpu_count": 0,
        "host_gpu_ids": [],
        "container_gpu_indices": [],
        "ipc": None,
        "memlock": None,
        "network": "none",
    }
    assert plan["image"]["id"] == "sha256:test-image"
    assert [mount["target"] for mount in plan["mounts"]] == [
        "/data",
        "/models",
        "/prepared",
        "/runs",
        "/cache",
        "/project-config/jobs",
        "/project-config/recipes",
        "/project-config/selections",
    ]
    command = plan["future_attempt"]["command"]
    assert command[:3] == ["docker", "create", "--name"]
    assert not {"--gpus", "--ipc", "--ulimit", "--env"}.intersection(command)
    assert "<run-id>" in command and "<attempt-id>" in command
    assert "io.distil3d.attempt-ref=lifecycle-smoke/<run-id>/<attempt-id>" in command
    assert not (tmp_path / "attempts").exists()


def test_resume_uses_saved_job_run_and_literal_checkpoint(tmp_path):
    job_file = tmp_path / "run.yaml"
    job_file.write_bytes(SMOKE.read_bytes())
    saved = _plan(job_file)["job"]
    record = tmp_path / "attempt.json"
    record.write_text(
        json.dumps(
            {"run_id": "original-run", "resolved_job": saved, "image": {"id": "sha256:old-image"}}
        )
    )
    job_file.unlink()
    checkpoint = "/unavailable/checkpoint with spaces/../chosen.pt"
    plan = _plan(None, resume_record=record, selected_checkpoint=checkpoint)
    assert plan["job"] == saved and plan["image"]["id"] == "sha256:test-image"
    command = plan["future_attempt"]["command"]
    for flag, value in [
        ("--resume-record", str(record)),
        ("--run-id", "original-run"),
        ("--attempt-id", "<attempt-id>"),
        ("--attempt-kind", "resume"),
        ("--selected-checkpoint", checkpoint),
    ]:
        assert command[command.index(flag) + 1] == value
    assert "--job" not in command and "sha256:old-image" not in command
    assert "io.distil3d.attempt-ref=lifecycle-smoke/original-run/<attempt-id>" in command
    assert "/runs/lifecycle-smoke/original-run/attempts/<attempt-id>" in command
    assert not (tmp_path / "attempts").exists()


@pytest.mark.parametrize(
    "relative,workflow,version,image,preset,kind",
    [
        (f"euvs/{scope}/{suffix}", workflow, version, "core", "inference_cp1", None)
        for scope in ("primary35", "extended16", "raw29", "location-2-tr2-to-tr6")
        for suffix, workflow, version in [
            ("source-views", "euvs_source_views", 1),
            ("", "euvs_generation", 1),
            ("source-reseed", "euvs_generation", 2),
        ]
    ]
    + [
        (f"scene9/{name}", "gaussian_generation", version, image, preset, kind)
        for name, version, image, preset, kind in [
            ("full-sequence", 1, "core", "inference_cp2", "full_sequence"),
            ("one-independent-clip", 2, "core", "inference_cp2", "independent_clips"),
            ("all-independent-clips", 2, "core", "inference_cp2", "independent_clips"),
            ("dense-independent", 3, "core", "inference_cp2", "dense_independent"),
            ("dense-overlap21", 4, "moge", "inference_cp2", "dense_overlap21"),
            ("dense-png-handoff", 5, "core", "cpu_test", "dense_handoff"),
        ]
    ],
)
def test_all_legacy_forms_plan_without_machine_inputs(
    relative, workflow, version, image, preset, kind
):
    path = ROOT / "jobs/legacy" / relative / "run.yaml"
    job = load_resolved_job(path, ROOT)
    count = select_job(job).preset.gpu_count
    plan = _plan(path, image=image, preset=preset, gpu_count=count)
    assert plan["job"]["workflow"] == {"name": workflow, "version": version}
    selection_file = ROOT / job.input["selection"]
    if kind is None:
        assert load_euvs_selection(selection_file).pairs
    else:
        assert load_gaussian_selection(selection_file).kind == kind
    assert ("--gpus" in plan["future_attempt"]["command"]) == bool(count)


def test_recipe_resolves_into_saved_parameters(tmp_path):
    (tmp_path / "recipe.yaml").write_text(
        "schema_version: 1\nparameters: {mode: success, exit_code: 0}\n"
    )
    job_file = tmp_path / "run.yaml"
    job_file.write_text(
        "schema_version: 1\nname: recipe-job\nworkflow: {name: _lifecycle_smoke, version: 1}\n"
        "input: {}\nrecipe: recipe.yaml\nexecution: {preset: cpu_test}\n"
    )
    job = load_resolved_job(job_file, tmp_path)
    assert job.recipe == "recipe.yaml" and dict(job.parameters) == {
        "mode": "success",
        "exit_code": 0,
    }


@pytest.mark.parametrize(
    "change", ["unknown", "version", "ambiguous-recipe", "workflow", "source-cp2"]
)
def test_bad_external_job_is_rejected_once(tmp_path, change):
    if change == "source-cp2":
        body = (ROOT / "jobs/acceptance/euvs-source-views-v1/run.yaml").read_text()
        body = body.replace("inference_cp1", "inference_cp2")
    else:
        body = SMOKE.read_text()
        if change == "unknown":
            body += "unexpected: true\n"
        elif change == "version":
            body = body.replace("schema_version: 1", "schema_version: 2")
        elif change == "ambiguous-recipe":
            body += "recipe: missing.yaml\n"
        else:
            body = body.replace("_lifecycle_smoke", "unknown_workflow")
    job_file = tmp_path / "run.yaml"
    job_file.write_text(body)
    with pytest.raises(ValueError):
        select_job(load_resolved_job(job_file, ROOT))


@pytest.mark.parametrize("selection", [{"image": "core"}, {"preset": "inference_cp1"}])
def test_final_plan_rejects_changed_external_selection(selection):
    with pytest.raises(ValueError):
        _plan(SMOKE, **selection)

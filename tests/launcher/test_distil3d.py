"""Внешние команды launcher; обычный жизненный цикл проверяет реальный Docker."""

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from distil3d_acceptance import FAILURE_CASES, accept_inputs, interrupt, summary

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "distil3d"
JOB = "jobs/examples/lifecycle-smoke"
REFERENCE = "lifecycle-smoke/run-kept/source-attempt"
TE_CASE = (
    "ac:test:tc:model:test-gen3c-activation-checkpointing:gen3c-activation-checkpointing-tests"
)
ACCEPT_CASES = (
    "euvs-geometry-campaign-v1",
    "euvs-pair-autoregressive-v1",
    "euvs-pair-autoregressive-v1:cp2",
    "euvs-pair-source-reseed-v1",
    "euvs-evaluation-run-v1",
    "euvs-base-tuned-comparison-v1",
    "gaussian-full-sequence-v1",
    "gaussian-independent-clip-v1",
    "gaussian-dense-independent-v1",
    "gaussian-dense-overlap21-v1",
    "gaussian-dense-png-handoff-v1",
    "ddw-preparation-v1",
    "gen3c-training-fresh-v1",
    "gen3c-training-resume-v1",
    "gen3c-checkpoint-selection-v1",
    "gen3c-training-fresh-v2",
    "gen3c-training-resume-v2",
    "gen3c-checkpoint-selection-v2",
    "gen3c-ddw-evaluation-v2",
    "waymo-depth-candidates-v2",
    "waymo-depth-selection-gate-v2",
    "waymo-ddw-engineering-canary-v1",
    "waymo-ddw-v2-canary-v1",
    "waymo-ddw-v3-probe-v1",
    "waymo-ddw-survey-v1",
)


@pytest.fixture
def fake_docker(tmp_path):
    docker_root, log = tmp_path / "docker-root", tmp_path / "docker.tsv"
    docker_root.mkdir()
    for name in ("docker", "git", "nvidia-smi"):
        executable = tmp_path / name
        executable.write_bytes(
            (ROOT / "tests/support" / f"fake_{name.replace('-', '_')}").read_bytes()
        )
        executable.chmod(0o755)
    return (
        dict(
            os.environ,
            PATH=f"{tmp_path}:{os.environ['PATH']}",
            FAKE_DOCKER_ROOT=str(docker_root),
            FAKE_DOCKER_LOG=str(log),
        ),
        log,
    )


def run(arguments, environment):
    return subprocess.run(
        (str(LAUNCHER), *map(str, arguments)),
        cwd="/tmp",
        env=environment,
        capture_output=True,
        text=True,
    )


def calls(log, command=None):
    rows = [line.split("\t") for line in log.read_text().splitlines()] if log.exists() else []
    return [row for row in rows if command is None or row[0] == command]


def option(argv, flag):
    return argv[argv.index(flag) + 1]


def image(argv):
    return argv[argv.index("--entrypoint") + 2]


def resources(argv):
    return {
        flag: option(argv, flag)
        for flag in ("--network", "--gpus", "--env", "--ipc", "--ulimit")
        if flag in argv
    }


def profile(root, gpu_ids="", *, materialize=False):
    path = root / "profile"
    path.mkdir()
    roots = {
        name: root / "roots" / name for name in ("data", "models", "prepared", "runs", "cache")
    }
    if materialize:
        for directory in roots.values():
            directory.mkdir(parents=True)
    path.joinpath(".env").write_text(
        "".join(f"DISTIL3D_{name.upper()}_ROOT={directory}\n" for name, directory in roots.items())
        + f"DISTIL3D_GPU_IDS={gpu_ids}\n"
    )
    return path, roots


def test_accept_reports_are_distinct_complete_and_bounded(fake_docker, tmp_path):
    environment, log = fake_docker
    arguments, output, candidate_id = accept_inputs(environment, tmp_path)
    source = tmp_path / "profile/.env"
    source.write_text(source.read_text().replace("GPU_IDS=0,1,2,3", "GPU_IDS=3,1,0,2"))
    assert run(arguments, environment).returncode == run(arguments, environment).returncode == 0
    campaigns = sorted(output.iterdir())
    assert len(campaigns) == 2 and all(path.name.endswith(candidate_id) for path in campaigns)
    campaign = campaigns[0]
    assert campaign.stat().st_mode & 0o777 == 0o700
    assert {path.name for path in campaign.iterdir()} == {
        "bootstrap.log",
        "candidate.json",
        "commands.tsv",
        "summary.tsv",
        "host",
        "images",
        "plans",
        "cases",
        "records",
    }
    rows = list(csv.DictReader((campaign / "summary.tsv").open(), delimiter="\t"))
    assert [row["case_id"] for row in rows] == [
        "bootstrap",
        "foundation",
        *(f"ac:scenario:sc:{name}" for name in ACCEPT_CASES[:12]),
        TE_CASE,
        *(f"ac:scenario:sc:{name}" for name in ACCEPT_CASES[12:]),
        "closure",
    ]
    assert all(row["status"] == "succeeded" for row in rows)
    assert [row["image_variant"] for row in rows[22:28]] == ["moge", "core", *(["moge"] * 4)]
    native = next(row for row in calls(log, "create") if "-m" in row and "pytest" in row)
    assert image(native) == environment["FAKE_DOCKER_CORE_ID"]
    assert option(native, "--user") == f"{os.getuid()}:{os.getgid()}"
    assert resources(native) == {
        "--network": "none",
        "--gpus": '"device=3"',
        "--env": "CUDA_VISIBLE_DEVICES=0",
        "--ipc": "host",
        "--ulimit": "memlock=-1:-1",
    }
    assert native[native.index("/opt/envs/gen3c/bin/python") :] == [
        "/opt/envs/gen3c/bin/python",
        "-m",
        "pytest",
        "-c",
        "/pyproject.toml",
        "-p",
        "no:cacheprovider",
        "-q",
        "/tests/integration/test_model_api.py::test_te_recompute_preserves_result_and_gradients",
        "--junitxml=/evidence/junit.xml",
    ]
    mounts = [native[i + 1] for i, value in enumerate(native) if value == "--mount"]
    assert mounts[:2] == [
        f"type=bind,src={ROOT}/diffusion/code/tests,dst=/tests,readonly",
        f"type=bind,src={ROOT}/diffusion/code/pyproject.toml,dst=/pyproject.toml,readonly",
    ]
    assert len(mounts) == 3 and mounts[2].endswith(",dst=/evidence")
    assert rows[14]["recorded_state"] == "absent" and rows[14]["container_exit_code"] == "0"
    assert "io.distil3d.attempt-ref=" + rows[14]["attempt_ref"] in native
    assert sum(value.startswith("io.distil3d.") for value in native) == 4
    assert len(list((campaign / "records").rglob("junit.xml"))) == 1
    assert not (tmp_path / "roots/runs/native-te-recompute").exists()
    readers = [
        row
        for row in calls(log, "run")
        if "novel_view.cli.acceptance" in row
        and any(argument.startswith("waymo-") for argument in row)
    ][:6]
    assert option(readers[1], "--reports-selection") == (
        "/project-config/selections/local/stage11/legacy/waymo-depth-choice-8.yaml"
    )
    gate_job, gate_run, gate_attempt = rows[23]["attempt_ref"].split("/")
    assert all(
        option(row, "--expected-depth-selection")
        == f"{gate_job}/{gate_run}/attempts/{gate_attempt}/depth-selection.json"
        for row in readers[2:]
    )
    assert (rows[0]["container_exit_code"], rows[0]["attempt_ref"]) == ("0", "none")
    assert all(rows[0][name].endswith("Z") for name in ("started", "finished"))
    facts = [
        json.loads((campaign / "images" / f"{name}-inspect.json").read_text())
        for name in ("core", "moge")
    ]
    assert [item["variant"] for item in facts] == ["core", "moge"]
    assert all(item["candidate_id"] == candidate_id for item in facts)
    assert len(list((campaign / "records").rglob("record.json"))) == 9
    assert {path.name for path in (campaign / "records").rglob("*.csv")} == {"summary.csv"}
    job = ROOT / "jobs/local" / campaign.name / "stage11-waymo-one-item/run.yaml"
    assert "name: stage11-waymo-one-item" in job.read_text()
    gate = job.parents[1] / "waymo-depth-selection-v2/run.yaml"
    assert (
        "selection: selections/local/stage11/legacy/waymo-depth-choice-8.yaml" in gate.read_text()
    )
    returned = "".join(
        path.read_text(errors="replace") for path in campaign.rglob("*") if path.is_file()
    )
    assert "token-value" not in returned and not (campaign / "images.tar").exists()


@pytest.mark.parametrize("setting,value,failed,skipped,independent", FAILURE_CASES)
def test_accept_failure_skips_dependents_not_independent_branches(
    fake_docker, tmp_path, setting, value, failed, skipped, independent
):
    environment, _ = fake_docker
    arguments, output, _ = accept_inputs(environment, tmp_path)
    environment[setting] = value
    assert run(arguments, environment).returncode != 0
    rows = summary(output)
    assert [
        next(row["status"] for row in rows if key in row["case_id"])
        for key in (failed, skipped, independent)
    ] == ["failed", "skipped", "succeeded"]


@pytest.mark.parametrize("exit_code,oom", [(23, "false"), (0, "true")])
def test_accept_te_error_skips_both_trainers_not_legacy(fake_docker, tmp_path, exit_code, oom):
    environment, _ = fake_docker
    arguments, output, _ = accept_inputs(environment, tmp_path)
    environment.update(
        FAKE_DOCKER_FAIL_CASE="native-te-recompute",
        FAKE_DOCKER_FAIL_EXIT=str(exit_code),
        FAKE_DOCKER_FAIL_OOM=oom,
    )
    assert run(arguments, environment).returncode != 0
    rows = {row["case_id"]: row for row in summary(output)}
    assert (rows[TE_CASE]["status"], rows[TE_CASE]["recorded_state"]) == ("failed", "absent")
    assert rows[TE_CASE]["container_exit_code"] == str(exit_code)
    assert rows[TE_CASE]["oom_killed"] == oom
    if oom == "true":
        assert not list(output.rglob("junit.xml"))
    for version in (1, 2):
        assert rows[f"ac:scenario:sc:gen3c-training-fresh-v{version}"]["status"] == "skipped"
    assert rows["ac:scenario:sc:waymo-ddw-survey-v1"]["status"] == "succeeded"


@pytest.mark.parametrize(
    "setting,value", [("FAKE_NVIDIA_COUNT", "3"), ("FAKE_NVIDIA_MEMORY", "40960")]
)
def test_accept_foundation_failure_starts_no_attempt(fake_docker, tmp_path, setting, value):
    environment, log = fake_docker
    arguments, output, _ = accept_inputs(environment, tmp_path)
    environment[setting] = value
    assert run(arguments, environment).returncode != 0
    assert (summary(output)[1]["case_id"], summary(output)[1]["status"]) == ("foundation", "failed")
    assert not calls(log, "create")


@pytest.mark.parametrize(
    "active,created",
    [("euvs-source-views-v1", 1), ("native-te-recompute", 13), ("waymo-ddw-canary-v2", 24)],
)
def test_accept_signal_stops_before_next_case(fake_docker, tmp_path, active, created):
    environment, log = fake_docker
    arguments, output, _ = accept_inputs(environment, tmp_path)
    ready = tmp_path / "logs-ready"
    environment.update(
        FAKE_DOCKER_LOGS_WAIT="1",
        FAKE_DOCKER_LOGS_READY=str(ready),
        FAKE_DOCKER_ACTIVE_CASE=active,
    )
    assert interrupt(LAUNCHER, ROOT, arguments, environment, ready) == 130
    assert any(row["status"] == "interrupted" for row in summary(output))
    assert len(calls(log, "create")) == created
    assert calls(log, "stop") == [
        [
            "stop",
            "--signal",
            "SIGTERM",
            "--time",
            "10",
            Path(environment["FAKE_DOCKER_ROOT"]).joinpath("container-id").read_text().strip(),
        ]
    ]


@pytest.mark.parametrize(
    "active,signum,index,phase",
    [
        ("euvs-source-views-v1", signal.SIGTERM, 2, "CREATE"),
        ("native-te-recompute", signal.SIGINT, 14, "CREATE"),
        ("euvs-source-views-v1", signal.SIGINT, 2, "PLAN"),
    ],
)
def test_accept_signal_during_create_leaves_no_container(
    fake_docker, tmp_path, active, signum, index, phase
):
    environment, log = fake_docker
    arguments, output, _ = accept_inputs(environment, tmp_path)
    ready = tmp_path / "create-ready"
    environment[f"FAKE_DOCKER_{phase}_WAIT_CASE"] = active
    environment["FAKE_DOCKER_CREATE_READY"] = str(ready)
    if phase == "PLAN":
        environment["FAKE_DOCKER_ACTIVE_CASE"] = active
    process = subprocess.Popen(
        (str(LAUNCHER), *arguments),
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for _ in range(2000):
        if ready.exists() or process.poll() is not None:
            break
        time.sleep(0.01)
    assert ready.exists()
    process.send_signal(signum)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 130, stdout + stderr
    rows = summary(output)
    assert rows[index]["status"] == "interrupted"
    assert all(row["status"] == "skipped" for row in rows[index + 1 : -1])
    assert not (Path(environment["FAKE_DOCKER_ROOT"]) / "container-id").exists()


@pytest.mark.parametrize(
    "state,command,active,index,created",
    [
        ("created", "RM", "", 2, 1),
        ("running", "STOP", "", 2, 1),
        ("exited", "STOP", "native-te-recompute", 14, 13),
    ],
)
def test_accept_cleanup_failure_aborts_campaign(
    fake_docker, tmp_path, state, command, active, index, created
):
    environment, log = fake_docker
    arguments, output, _ = accept_inputs(environment, tmp_path)
    environment.update(FAKE_DOCKER_CONTAINER_STATE=state, FAKE_DOCKER_ACTIVE_CASE=active)
    environment[f"FAKE_DOCKER_{command}_EXIT"] = "23"
    assert run(arguments, environment).returncode != 0
    rows = summary(output)
    assert rows[index]["status"] == "interrupted"
    assert "cleanup-unknown" in rows[index]["note"]
    assert all(row["status"] == "skipped" for row in rows[index + 1 : -1])
    assert len(calls(log, "create")) == created and len(calls(log, command.lower())) == 1


@pytest.mark.parametrize("choice", ["vggt", "stop", "invalid-record"])
def test_accept_depth_gate_controls_only_historical_ddw(fake_docker, tmp_path, choice):
    environment, log = fake_docker
    arguments, output, _ = accept_inputs(environment, tmp_path)
    if choice == "invalid-record":
        environment["FAKE_DOCKER_RECORD_FAIL_CASE"] = "waymo-depth-selection-v2"
    else:
        environment["FAKE_DOCKER_DEPTH_CHOICE"] = choice
    assert run(arguments, environment).returncode != 0
    rows = summary(output)
    assert rows[22]["status"] == "succeeded"  # The new comparison is independent of the old eight.
    assert rows[23]["status"] == ("failed" if choice == "invalid-record" else "succeeded")
    assert all(row["status"] == "skipped" for row in rows[24:28])
    assert not any(
        "io.distil3d.attempt-ref=waymo-ddw-" in " ".join(row) for row in calls(log, "create")
    )


@pytest.mark.parametrize("job,index", [("canary-v2", 25), ("probe", 26), ("survey", 27)])
def test_accept_scientific_stop_keeps_actual_failed_attempt(fake_docker, tmp_path, job, index):
    environment, _ = fake_docker
    arguments, output, _ = accept_inputs(environment, tmp_path)
    environment["FAKE_DOCKER_SCIENTIFIC_STOP_CASE"] = f"waymo-ddw-{job}"
    assert run(arguments, environment).returncode == 0
    row = summary(output)[index]
    assert (
        row["status"],
        row["recorded_state"],
        row["container_exit_code"],
        row["oom_killed"],
    ) == ("succeeded", "failed", "2", "false")
    assert "scientific_stop" in row["note"]


@pytest.mark.parametrize("failure", ["exit2", "oom", "invalid-record", "candidate"])
def test_accept_does_not_turn_process_or_record_error_into_scientific_stop(
    fake_docker, tmp_path, failure
):
    environment, _ = fake_docker
    arguments, output, _ = accept_inputs(environment, tmp_path)
    job = "waymo-ddw-canary-v2"
    if failure == "exit2":
        environment.update(FAKE_DOCKER_FAIL_CASE=job, FAKE_DOCKER_FAIL_EXIT="2")
    elif failure == "oom":
        environment.update(FAKE_DOCKER_SCIENTIFIC_STOP_CASE=job, FAKE_DOCKER_STOP_OOM="true")
    elif failure == "invalid-record":
        environment["FAKE_DOCKER_RECORD_FAIL_CASE"] = job
    else:
        environment["FAKE_DOCKER_FAIL_CASE"] = "waymo-depth-candidates"
    assert run(arguments, environment).returncode != 0
    rows = summary(output)
    assert rows[22 if failure == "candidate" else 25]["status"] == "failed"
    assert all(rows[index]["status"] == "succeeded" for index in (23, 26, 27))


@pytest.mark.parametrize(
    "setting,value,early",
    [
        ("missing-archive", "", True),
        ("FAKE_DOCKER_LOAD_EXIT", "19", True),
        ("FAKE_DOCKER_CANDIDATE_EXIT", "2", True),
        ("FAKE_DOCKER_TAG_CORE_ID", f"sha256:{'e' * 64}", False),
        ("FAKE_DOCKER_LABEL_VARIANT", "wrong", False),
        ("FAKE_GIT_DIRTY", "true", False),
    ],
)
def test_accept_bootstrap_preserves_early_error_report(
    fake_docker, tmp_path, setting, value, early
):
    environment, _ = fake_docker
    arguments, output, candidate_id = accept_inputs(environment, tmp_path)
    if setting == "missing-archive":
        (tmp_path / "bundle/images.tar").unlink()
    else:
        environment[setting] = value
    assert run(arguments, environment).returncode != 0
    (campaign,) = output.iterdir()
    assert (campaign / "bootstrap.log").stat().st_size > 0
    assert ("bootstrap" in campaign.name) is early
    if early:
        assert not (campaign / "summary.tsv").exists()
    else:
        assert campaign.name.endswith(candidate_id)
        rows = summary(output)
        assert len(rows) == 1 and rows[0]["status"] == "failed"


def test_help_lists_only_public_commands_without_docker(fake_docker):
    environment, log = fake_docker
    result = run(("--help",), environment)
    assert result.returncode == 0 and not calls(log)
    assert all(
        command in result.stdout
        for command in (
            "doctor host",
            "doctor job --profile",
            "build [--variant core|moge]",
            "test [unit|contract|integration|all]",
            "plan --profile",
            "run --profile",
            "resume --profile",
            "status --profile",
            "logs --profile",
            "stop --profile",
            "remove --profile",
            "accept a100-4 --candidate",
        )
    )


@pytest.mark.parametrize(
    "arguments,variants", [((), ["core"]), (("--variant", "moge"), ["core", "moge"])]
)
def test_build_selects_only_requested_images(fake_docker, arguments, variants):
    environment, log = fake_docker
    dirty = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=normal"), cwd=ROOT
    )
    assert run(("build", *arguments), environment).returncode == 0
    builds = calls(log, "build")
    assert [option(call, "--target") for call in builds] == [f"production-{v}" for v in variants]
    assert [option(call, "--tag") for call in builds] == [
        f"distil3d:cuda124-sm80-{v}" for v in variants
    ]
    assert all(f"DISTIL3D_SOURCE_DIRTY={'true' if dirty else 'false'}" in call for call in builds)
    assert {call[0] for call in calls(log)} == {"build", "image"}
    assert all("{{.Id}}" in call for call in calls(log, "image"))


@pytest.mark.parametrize(
    "arguments",
    [
        ("build", "--variant", "vggt"),
        ("build", "--profile", "profiles/local/server"),
        ("build", "--variant", "core", "--candidate-output", "bundle"),
        ("test", "models"),
        (
            "resume",
            "--profile",
            "absent",
            "lifecycle-smoke/run-kept",
            "--checkpoint",
            "/models/chosen.pt",
        ),
    ],
)
def test_invalid_external_command_stops_before_docker(fake_docker, arguments):
    environment, log = fake_docker
    assert run(arguments, environment).returncode == 2 and not calls(log)


def test_dirty_candidate_does_not_build_or_create_bundle(fake_docker, tmp_path):
    environment, log = fake_docker
    environment.update(FAKE_GIT_REVISION="a" * 40, FAKE_GIT_DIRTY="true")
    output = tmp_path / "bundle"
    assert (
        run(("build", "--variant", "moge", "--candidate-output", output), environment).returncode
        == 2
    )
    assert not calls(log) and not output.exists()


@pytest.mark.parametrize(
    "group,setting,exit_code,commands",
    [
        ("unit", "", 0, ["build", "run"]),
        ("contract", "FAKE_DOCKER_BUILD_EXIT", 47, ["build"]),
        ("integration", "FAKE_DOCKER_RUN_EXIT", 23, ["build", "run"]),
    ],
)
def test_cpu_group_argv_and_natural_exit(fake_docker, group, setting, exit_code, commands):
    environment, log = fake_docker
    if setting:
        environment[setting] = str(exit_code)
    assert run(("test", group), environment).returncode == exit_code
    rows = calls(log)
    assert [row[0] for row in rows] == commands
    build = rows[0]
    assert build[1:3] == ["--platform", "linux/amd64"]
    assert option(build, "--target") == "cpu-test" and option(build, "--tag") == "distil3d:cpu-test"
    assert all(
        any(value.startswith(f"DISTIL3D_{key}=") for value in build)
        for key in ("SOURCE_REVISION", "SOURCE_DIRTY", "LOCK_REVISION")
    )
    assert "DISTIL3D_IMAGE_VARIANT=cpu-test" in build
    if len(rows) == 2:
        command = rows[1]
        assert command[:7] == [
            "run",
            "--rm",
            "--network",
            "none",
            "--hostname",
            "localhost",
            "distil3d:cpu-test",
        ]
        expected_paths = [f"tests/{group}"]
        if group == "integration":
            expected_paths.append("/workspace/tests/launcher/test_distil3d.py")
        assert command[-len(expected_paths) :] == expected_paths
        assert command[command.index("-m", command.index("pytest")) + 1] == "not native_api"


def test_profile_values_are_literal_and_host_doctor_only_probes_explicit_roots(
    fake_docker, tmp_path
):
    environment, log = fake_docker
    path, roots = profile(tmp_path, materialize=True)
    sentinel = tmp_path / "not-created"
    source = path / ".env"
    source.write_text(
        source.read_text().replace(str(roots["data"]), f"$(touch {sentinel})=literal#value")
    )
    result = run(("doctor", "host", "--profile", path), environment)
    assert (
        result.returncode == 0 and "$(touch" in result.stdout and "=literal#value" in result.stdout
    )
    assert not sentinel.exists()
    assert [row[0] for row in calls(log)] == ["version", "info", "info", "info"]


@pytest.mark.parametrize(
    "extra,gpus",
    [("DISTIL3D_UNKNOWN=x\n", ""), ("DISTIL3D_GPU_IDS=\n", ""), ("", "7,,2"), ("", "7,7")],
)
def test_bad_external_profile_is_rejected_once(fake_docker, tmp_path, extra, gpus):
    environment, log = fake_docker
    path, _ = profile(tmp_path, gpus)
    source = path / ".env"
    source.write_text(source.read_text() + extra)
    assert run(("plan", "--profile", path, JOB), environment).returncode == 2
    assert not calls(log)


@pytest.mark.parametrize(
    "variant,preset,gpus,selected,indices",
    [
        ("cpu-test", "cpu_test", "", "", ""),
        ("core", "inference_cp1", "7,2", "7", "0"),
        ("moge", "inference_cp2", "7,2", "7,2", "0,1"),
        ("core", "training_cp4", "GPU-one,2,GPU-three,7,9", "GPU-one,2,GPU-three,7", "0,1,2,3"),
    ],
)
def test_plan_is_config_only_and_uses_selected_image(
    fake_docker, tmp_path, variant, preset, gpus, selected, indices
):
    environment, log = fake_docker
    environment.update(FAKE_DOCKER_SELECTOR_VARIANT=variant, FAKE_DOCKER_SELECTOR_PRESET=preset)
    path, roots = profile(tmp_path, gpus)
    result = run(("plan", "--profile", path, JOB), environment)
    assert result.returncode == 0, result.stderr
    rows = calls(log)
    selector, final = calls(log, "run")
    chosen = f"sha256:fake-{'cpu' if variant == 'cpu-test' else variant}-image"
    tag = "distil3d:cpu-test" if variant == "cpu-test" else f"distil3d:cuda124-sm80-{variant}"
    assert image(selector) == "distil3d:cuda124-sm80-core" and "select" in selector
    assert image(final) == chosen and "plan" in final
    for command in (selector, final):
        assert command[1:4] == ["--rm", "--network", "none"] and resources(command) == {
            "--network": "none"
        }
        mounts = [command[i + 1] for i, flag in enumerate(command) if flag == "--mount"]
        assert len(mounts) == 3 and all(
            "/project-config/" in value and value.endswith(",readonly") for value in mounts
        )
        assert all(str(tmp_path / "roots") not in value for value in mounts)
        assert option(command, "--entrypoint") == ""
    metadata = {
        "--selected-preset": preset,
        "--image-variant": variant,
        "--image-id": chosen,
        "--image-source-revision": "fake-revision",
        "--image-source-dirty": "false",
        "--image-lock-revision": "sha256:fake-lock",
        "--host-gpu-ids": selected,
        "--container-gpu-indices": indices,
    }
    assert all(option(final, flag) == value for flag, value in metadata.items())
    assert (
        f"--ipc={'host' if selected else ''}" in final
        and f"--memlock={'-1:-1' if selected else ''}" in final
    )
    assert all(
        option(final, f"--host-{name}-root") == str(directory) for name, directory in roots.items()
    )
    assert all(
        option(final, f"--host-{name}-root") == str(ROOT / name)
        for name in ("jobs", "recipes", "selections")
    )
    ids = [call for call in calls(log, "image") if "{{.Id}}" in call]
    labels = [call for call in calls(log, "image") if "{{.Id}}" not in call]
    assert len(ids) == 2 and all(call[-1] == tag for call in ids)
    assert labels and all(call[-1] == chosen for call in labels)
    assert not {"build", "create", "start"}.intersection(row[0] for row in rows)
    assert not any(directory.exists() for directory in roots.values())


@pytest.mark.parametrize(
    "command,setting,value,code",
    [
        ("plan", "FAKE_DOCKER_PLAN_EXIT", "37", 37),
        ("doctor", "FAKE_DOCKER_PLAN_EXIT", "37", 37),
        ("doctor", "FAKE_DOCKER_DOCTOR_EXIT", "23", 23),
        ("run", "FAKE_DOCKER_LABEL_VARIANT", "moge", 2),
        ("run", "FAKE_DOCKER_MISMATCH_EXIT", "41", 41),
        ("run", "FAKE_DOCKER_TAG_RACE", "1", 2),
        ("run", "FAKE_DOCKER_SELECTOR_PRESET", "training_cp4", 2),
    ],
)
def test_plan_or_diagnostic_failure_starts_no_attempt(
    fake_docker, tmp_path, command, setting, value, code
):
    environment, log = fake_docker
    environment.update(
        FAKE_DOCKER_SELECTOR_VARIANT="core", FAKE_DOCKER_SELECTOR_PRESET="inference_cp1"
    )
    environment[setting] = value
    path, roots = profile(tmp_path, "7,2,GPU-three", materialize=True)
    arguments = ("doctor", "job") if command == "doctor" else (command,)
    result = run((*arguments, "--profile", path, JOB), environment)
    assert result.returncode == code, result.stderr
    assert not {"build", "create", "start"}.intersection(row[0] for row in calls(log))
    assert not list(roots["runs"].iterdir())
    if setting == "FAKE_DOCKER_PLAN_EXIT":
        assert len(calls(log, "run")) == 2


@pytest.mark.parametrize(
    "variant,preset,gpus,indices,job",
    [
        ("cpu-test", "cpu_test", "7,2", "", "jobs/examples/lifecycle-smoke/run.yaml"),
        (
            "core",
            "training_cp4",
            "GPU-C,GPU-A,GPU-D,GPU-B",
            "0,1,2,3",
            "jobs/waymo/r4c-lora-edm/run.yaml",
        ),
    ],
)
def test_doctor_and_attempt_share_resources_with_real_python_plan(
    fake_docker, tmp_path, capsys, variant, preset, gpus, indices, job
):
    from novel_view.cli.main import main

    environment, log = fake_docker
    environment.update(FAKE_DOCKER_SELECTOR_VARIANT=variant, FAKE_DOCKER_SELECTOR_PRESET=preset)
    path, roots = profile(tmp_path, gpus, materialize=True)
    expected = {"--network": "none"}
    if indices:
        expected.update(
            {
                "--gpus": f'"device={gpus}"',
                "--env": f"CUDA_VISIBLE_DEVICES={indices}",
                "--ipc": "host",
                "--ulimit": "memlock=-1:-1",
            }
        )
    result = run(("doctor", "job", "--profile", path, JOB), environment)
    assert result.returncode == 0 and '{"job"' not in result.stdout
    doctor = next(row for row in calls(log, "run") if "doctor" in row)
    assert not {"--label", "--name", "--attempt-id", "--run-id"}.intersection(doctor)
    assert not calls(log, "create") and not list(roots["runs"].iterdir())
    result = run(("run", "--profile", path, JOB, "--run-id", "run-kept", "--detach"), environment)
    assert result.returncode == 0 and "attempt=lifecycle-smoke/run-kept/" in result.stdout
    assert '{"job"' not in result.stdout
    selector = next(row for row in calls(log, "run") if "select" in row)
    final = next(row for row in calls(log, "run") if "plan" in row)
    assert image(selector) == "distil3d:cuda124-sm80-core"
    chosen = "sha256:fake-cpu-image" if variant == "cpu-test" else "sha256:fake-core-image"
    assert (
        image(final) == chosen
        and image(doctor) == chosen
        and option(final, "--selected-preset") == preset
    )
    (create,) = calls(log, "create")
    assert len(calls(log, "start")) == 1 and not {"build", "logs", "inspect"}.intersection(
        row[0] for row in calls(log)
    )
    assert "--rm" not in create and create[create.index("/opt/envs/core/bin/python") - 1] == chosen
    assert option(create, "--image-id") == chosen
    assert {"io.distil3d.job=lifecycle-smoke", "io.distil3d.run-id=run-kept"}.issubset(create)
    assert all(
        any(value.startswith(f"io.distil3d.{key}=") for value in create)
        for key in ("attempt-id", "attempt-ref")
    )
    mounts = lambda command: [
        command[i + 1] for i, value in enumerate(command) if value == "--mount"
    ]
    assert mounts(create) == mounts(doctor) and len(mounts(create)) == 8
    assert sum(value.endswith(",readonly") for value in mounts(create)) == 5
    assert (
        resources(create) == resources(doctor) == expected
        and "--user" in create
        and "--user" in doctor
    )
    assert len(list((roots["runs"] / "lifecycle-smoke/run-kept/attempts").iterdir())) == 1
    arguments = final[final.index("novel_view.cli.main") + 1 :]
    arguments[arguments.index("--job") + 1] = str(ROOT / job)
    arguments[arguments.index("--config-root") + 1] = str(ROOT)
    assert main(arguments) == 0
    assert resources(json.loads(capsys.readouterr().out)["future_attempt"]["command"]) == expected


@pytest.mark.parametrize("diagnostic", ["ingress", "raster"])
def test_waymo_doctor_preserves_installed_cpu_argv(fake_docker, tmp_path, diagnostic):
    environment, log = fake_docker
    environment["FAKE_DOCKER_RUN_EXIT"] = "23"
    path, roots = profile(tmp_path, "7,2", materialize=True)
    arguments = (
        "--waymo-root",
        "/data/waymo with spaces",
        "--split-config",
        "/project-config/selections/local/split.yaml",
    )
    if diagnostic == "ingress":
        arguments += ("--max-rss-mib", "1024")
    assert run(
        ("doctor", f"waymo-{diagnostic}", "--help"), environment
    ).returncode == 0 and not calls(log)
    assert (
        run(
            ("doctor", f"waymo-{diagnostic}", "--profile", path, "--", *arguments), environment
        ).returncode
        == 23
    )
    (command,) = calls(log, "run")
    assert image(command) == "sha256:fake-core-image"
    assert command[command.index("-m") + 1 :] == [
        "novel_view.cli.main",
        "doctor-waymo",
        diagnostic,
        *arguments,
    ]
    assert (
        resources(command) == {"--network": "none"}
        and command.count("--mount") == 8
        and "--user" in command
    )
    assert not {"create", "start", "build"}.intersection(row[0] for row in calls(log))
    assert not list(roots["runs"].iterdir())


def test_resume_and_repeated_resume_read_exact_records_without_source_container(
    fake_docker, tmp_path
):
    environment, log = fake_docker
    path, roots = profile(tmp_path, materialize=True)
    for index, source in enumerate(("source-attempt", "resume-attempt"), 1):
        checkpoint = f"/models/chosen-{index}.pt"
        reference = f"lifecycle-smoke/run-kept/{source}"
        assert (
            run(
                ("resume", "--profile", path, reference, "--checkpoint", checkpoint, "--detach"),
                environment,
            ).returncode
            == 0
        )
        reader = [row for row in calls(log, "run") if "select-resume" in row][-1]
        final = [row for row in calls(log, "run") if "plan" in row][-1]
        assert (
            reader[1:4] == ["--rm", "--network", "none"]
            and image(reader) == "distil3d:cuda124-sm80-core"
        )
        assert option(reader, "--attempt-id") == source
        assert any(value.endswith("dst=/runs,readonly") for value in reader)
        assert (
            image(final) == "sha256:fake-cpu-image"
            and option(final, "--selected-preset") == "cpu_test"
        )
        assert final.count("--mount") == 1 and option(final, "--mount").endswith(
            "dst=/runs,readonly"
        )
        record = f"/runs/lifecycle-smoke/run-kept/attempts/{source}/attempt.json"
        assert (
            option(final, "--resume-record") == record
            and option(final, "--selected-checkpoint") == checkpoint
        )
        create = calls(log, "create")[-1]
        assert create[create.index("/opt/envs/core/bin/python") - 1] == image(final)
        assert (
            option(create, "--resume-record") == record
            and option(create, "--selected-checkpoint") == checkpoint
        )
        assert option(create, "--attempt-kind") == "resume"
        assert len(list((roots["runs"] / "lifecycle-smoke/run-kept/attempts").iterdir())) == index
    assert not {"build", "ps", "inspect"}.intersection(row[0] for row in calls(log))


def test_premature_log_end_does_not_report_running_container_success(fake_docker, tmp_path):
    environment, log = fake_docker
    environment.update(FAKE_DOCKER_CONTAINER_STATE="running", FAKE_DOCKER_CONTAINER_EXIT="0")
    path, _ = profile(tmp_path, materialize=True)
    assert run(("run", "--profile", path, JOB), environment).returncode == 1
    assert len(calls(log, "logs")) == len(calls(log, "inspect")) == 1 and not calls(log, "stop")


@pytest.mark.parametrize("oom", ["true", "false"])
def test_status_reports_docker_oom_flag_not_exit137(fake_docker, tmp_path, oom):
    environment, log = fake_docker
    environment.update(
        FAKE_DOCKER_EXACT_CONTAINER="exact-container",
        FAKE_DOCKER_SNAPSHOT_REF=REFERENCE,
        FAKE_DOCKER_RECORDED_STATE="running",
        FAKE_DOCKER_CONTAINER_STATE="exited",
        FAKE_DOCKER_CONTAINER_EXIT="137",
        FAKE_DOCKER_CONTAINER_OOM=oom,
    )
    path, _ = profile(tmp_path, materialize=True)
    result = run(("status", "--profile", path, REFERENCE), environment)
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "recorded_state=running",
        "container_state=exited",
        "container_exit_code=137",
        f"container_oom_killed={oom}",
        f"attempt_ref={REFERENCE}",
    ]
    (reader,) = calls(log, "run")
    assert image(reader) == "distil3d:cuda124-sm80-core"
    assert reader.count("--mount") == 1 and option(reader, "--mount").endswith("dst=/runs,readonly")
    assert not any("/project-config/" in value for value in reader)


def test_short_active_reference_needs_record_and_logs_follow_same_attempt(fake_docker, tmp_path):
    environment, log = fake_docker
    path, _ = profile(tmp_path, materialize=True)
    environment.update(
        FAKE_DOCKER_ACTIVE_ROWS=f"container-a {REFERENCE}",
        FAKE_DOCKER_SNAPSHOT_REF=REFERENCE,
        FAKE_DOCKER_RECORDED_STATE="absent",
        FAKE_DOCKER_RECORD_PRESENT="false",
    )
    assert (
        run(("status", "--profile", path, "lifecycle-smoke/run-kept"), environment).returncode == 2
    )
    assert not calls(log, "inspect")
    environment.update(
        FAKE_DOCKER_RECORD_PRESENT="true", FAKE_DOCKER_LOG_CONTENT="synthetic output"
    )
    result = run(("logs", "--profile", path, "lifecycle-smoke/run-kept", "-f"), environment)
    assert result.returncode == 0 and result.stdout == "synthetic output\n"
    assert calls(log, "logs") == [["logs", "--follow", "container-a"]]


def test_stop_reports_natural_exit_race(fake_docker, tmp_path):
    environment, log = fake_docker
    path, _ = profile(tmp_path, materialize=True)
    environment.update(
        FAKE_DOCKER_EXACT_CONTAINER="container-a",
        FAKE_DOCKER_SNAPSHOT_REF=REFERENCE,
        FAKE_DOCKER_CONTAINER_STATE="running",
        FAKE_DOCKER_AFTER_STOP_INSPECT="exited 0 false",
    )
    result = run(("stop", "--profile", path, REFERENCE, "--grace-seconds", "7"), environment)
    assert result.returncode == 0
    assert calls(log, "stop") == [["stop", "--signal", "SIGTERM", "--time", "7", "container-a"]]
    assert result.stdout.splitlines() == [
        "container_state=exited",
        "container_exit_code=0",
        "container_oom_killed=false",
        f"attempt_ref={REFERENCE}",
    ]

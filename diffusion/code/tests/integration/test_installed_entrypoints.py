"""Холодный запуск сохранённых legacy entry points и нового plan из wheel."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return environment


@pytest.mark.parametrize("entry", (
    ("/opt/envs/core/bin/novel-view",), (sys.executable, "-m", "novel_view"),
))
def test_legacy_entrypoint_only_points_to_launcher(tmp_path: Path, entry) -> None:
    completed = subprocess.run(
        (*entry, "--version"),
        cwd=tmp_path,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "novel-view 0.1.0" in completed.stdout and "./distil3d" in completed.stdout
    help_result = subprocess.run(
        (*entry, "--help"), cwd=tmp_path, env=_clean_environment(), capture_output=True, text=True,
    )
    assert help_result.returncode == 0 and "./distil3d" in help_result.stdout
    for removed in ("stages", "check-config"):
        result = subprocess.run(
            (*entry, removed), cwd=tmp_path, env=_clean_environment(), capture_output=True, text=True,
        )
        assert result.returncode == 2


def test_internal_plan_runs_from_the_installed_wheel(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[4]
    mounted_job = repository_root / "jobs/examples/lifecycle-smoke/run.yaml"
    mounted_config = repository_root
    arguments = (
        sys.executable,
        "-m",
        "novel_view.cli.main",
        "plan",
        "--selected-preset",
        "cpu_test",
        "--host-gpu-ids",
        "",
        "--container-gpu-indices",
        "",
        "--ipc=",
        "--memlock=",
        "--job",
        str(mounted_job),
        "--config-root",
        str(mounted_config),
        "--image-variant",
        "cpu-test",
        "--image-id",
        "sha256:integration",
        "--image-source-revision",
        "integration",
        "--image-source-dirty",
        "false",
        "--image-lock-revision",
        "sha256:integration",
        "--host-data-root",
        "/absent/data",
        "--host-models-root",
        "/absent/models",
        "--host-prepared-root",
        "/absent/prepared",
        "--host-runs-root",
        "/absent/runs",
        "--host-cache-root",
        "/absent/cache",
        "--host-jobs-root",
        "/checkout/jobs",
        "--host-recipes-root",
        "/checkout/recipes",
        "--host-selections-root",
        "/checkout/selections",
        "--host-uid",
        "1000",
        "--host-gid",
        "1000",
    )

    completed = subprocess.run(
        arguments,
        cwd=tmp_path,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["job"]["name"] == "lifecycle-smoke"
    assert plan["future_attempt"]["command"][0:2] == ["docker", "create"]

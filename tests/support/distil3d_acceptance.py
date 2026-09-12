"""Small external setup/read helpers for fixed acceptance launcher tests."""

from __future__ import annotations

import csv
import signal
import subprocess
import time
from pathlib import Path


FAILURE_CASES = (
    (
        "FAKE_DOCKER_FAIL_CASE",
        "euvs-source-views-v1",
        "euvs-geometry",
        "euvs-pair-autoregressive",
        "gaussian-full",
    ),
    (
        "FAKE_DOCKER_FAIL_CASE",
        "r4c-lora-lidar-depth",
        "training-fresh-v2",
        "gen3c-ddw-evaluation",
        "checkpoint-selection-v1",
    ),
    (
        "FAKE_DOCKER_ACTIVE_CASE",
        "euvs-source-views-v1",
        "euvs-geometry",
        "euvs-pair-autoregressive",
        "gaussian-full",
    ),
    (
        "FAKE_DOCKER_SELECTION_FAIL_VERSION",
        "1",
        "checkpoint-selection-v1",
        "gen3c-ddw-evaluation",
        "checkpoint-selection-v2",
    ),
)


def accept_inputs(
    environment: dict[str, str], root: Path
) -> tuple[tuple[str, ...], Path, str]:
    source, core, moge = "a" * 40, f"sha256:{'b' * 64}", f"sha256:{'c' * 64}"
    candidate_id = f"{'a' * 12}-{'b' * 12}-{'c' * 12}"
    environment.update(
        FAKE_GIT_REVISION=source,
        FAKE_DOCKER_SOURCE_REVISION=source,
        FAKE_DOCKER_LOCK_REVISION=f"sha256:{'d' * 64}",
        FAKE_DOCKER_CORE_ID=core,
        FAKE_DOCKER_MOGE_ID=moge,
        FAKE_DOCKER_CANDIDATE_ID=candidate_id,
    )
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "candidate.json").write_text('{"synthetic": true}\n')
    (bundle / "images.tar").write_bytes(b"synthetic archive")
    profile, roots = root / "profile", root / "roots"
    profile.mkdir()
    for name in ("data", "models", "prepared", "runs", "cache"):
        (roots / name).mkdir(parents=True)
    (profile / ".env").write_text(
        "\n".join(
            (
                *(
                    f"DISTIL3D_{name.upper()}_ROOT={roots / name}"
                    for name in ("data", "models", "prepared", "runs", "cache")
                ),
                "DISTIL3D_GPU_IDS=0,1,2,3",
                "# SECRET_SHOULD_NOT_RETURN=token-value",
                "",
            )
        )
    )
    output = root / "evidence"
    arguments = (
        "accept",
        "a100-4",
        "--candidate",
        str(bundle / "candidate.json"),
        "--profile",
        str(profile),
        "--output",
        str(output),
    )
    return arguments, output, candidate_id


def summary(output: Path) -> list[dict[str, str]]:
    return list(
        csv.DictReader(
            next(output.iterdir()).joinpath("summary.tsv").open(), delimiter="\t"
        )
    )


def interrupt(
    launcher: Path,
    repository: Path,
    arguments: tuple[str, ...],
    environment: dict[str, str],
    ready: Path,
) -> int:
    process = subprocess.Popen(
        (str(launcher), *arguments), cwd=repository, env=environment
    )
    for _ in range(2000):
        if ready.exists() or process.poll() is not None:
            break
        time.sleep(0.05)
    process.send_signal(signal.SIGINT)
    return process.wait(timeout=10)

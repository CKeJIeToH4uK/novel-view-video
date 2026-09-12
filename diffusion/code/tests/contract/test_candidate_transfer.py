"""Настоящий CLI candidate v1: JSON/host fields и одна ошибка внешнего TSV."""

import hashlib
import json
import subprocess
import sys

import pytest

from novel_view.runtime.candidate import candidate_from_mapping

LOCKS = [
    "core.conda.lock",
    "core.pip.lock",
    "gen3c-eval.conda.lock",
    "gen3c-eval.pip.lock",
    "vggt.conda.lock",
    "vggt.pip.lock",
    "moge-gen3c.pip.lock",
]
INVENTORY = "".join(
    f"containers/diffusion/requirements/{name}\tsha256:{str(i) * 64}\n"
    for i, name in enumerate(LOCKS, 1)
)
REVISION = "sha256:" + hashlib.sha256(INVENTORY.encode()).hexdigest()


def _record():
    return {
        "schema_version": 1,
        "candidate_id": f"{'a' * 12}-{'b' * 12}-{'c' * 12}",
        "source_revision": "a" * 40,
        "source_dirty": False,
        "platform": "linux/amd64",
        "lock_revision": REVISION,
        "locks": [
            {"path": f"containers/diffusion/requirements/{name}", "sha256": str(i) * 64}
            for i, name in enumerate(LOCKS, 1)
        ],
        "transport": "docker-save",
        "images": {
            "core": {"tag": "distil3d:cuda124-sm80-core", "id": f"sha256:{'b' * 64}"},
            "moge": {"tag": "distil3d:cuda124-sm80-moge", "id": f"sha256:{'c' * 64}"},
        },
    }


def _command(*arguments, stdin=""):
    return subprocess.run(
        [sys.executable, "-m", "novel_view.cli.candidate", *arguments],
        input=stdin,
        text=True,
        capture_output=True,
    )


def _write(path, inventory=INVENTORY):
    return _command(
        "write",
        "--source-revision",
        "a" * 40,
        "--lock-revision",
        REVISION,
        "--core-image-id",
        f"sha256:{'b' * 64}",
        "--moge-image-id",
        f"sha256:{'c' * 64}",
        "--output",
        str(path),
        stdin=inventory,
    )


def test_candidate_v1_cli_roundtrip_keeps_image_identity(tmp_path):
    path = tmp_path / "candidate.json"
    result = _write(path)
    assert result.returncode == 0, result.stderr
    assert path.stat().st_mode & 0o777 == 0o644
    assert json.loads(path.read_text()) == _record()
    result = _command("read", "--candidate", str(path))
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        f"candidate_id={'a' * 12}-{'b' * 12}-{'c' * 12}\nsource_revision={'a' * 40}\n"
        "source_dirty=false\nplatform=linux/amd64\n"
        f"lock_revision={REVISION}\ncore_tag=distil3d:cuda124-sm80-core\n"
        f"core_image_id=sha256:{'b' * 64}\nmoge_tag=distil3d:cuda124-sm80-moge\n"
        f"moge_image_id=sha256:{'c' * 64}\ntransport=docker-save\n"
    )


@pytest.mark.parametrize("change", [{"extra": True}, {"schema_version": 2}])
def test_candidate_external_schema_rejects_unknown_or_future_fields(change):
    with pytest.raises(ValueError):
        candidate_from_mapping(_record() | change)


def test_candidate_cli_reports_bad_inventory_without_writing(tmp_path):
    path = tmp_path / "candidate.json"
    result = _write(path, INVENTORY.rstrip("\n"))
    assert result.returncode == 2 and result.stderr
    assert not path.exists()

"""Ordered external selections stay concrete, strict and light to import."""

from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from novel_view.inputs.euvs.spec import load_euvs_selection
from novel_view.inputs.gaussian.spec import load_gaussian_selection


FIXTURE = Path(__file__).with_name("fixtures") / "euvs_selection_v1.yaml"


def test_euvs_order_tags_and_unambiguous_external_identity(tmp_path):
    pairs = load_euvs_selection(FIXTURE).pairs
    assert [pair.name for pair in pairs] == ["pair-z", "pair-a"]
    assert pairs[0].tags == ("primary", "night") and pairs[1].tags == ()
    assert pairs[0].source.image_tokens == ("source-03", "source-01")
    assert pairs[0].target.image_tokens == ("target-02", "target-00")
    path = tmp_path / "selection.yaml"
    for change in ("unknown", "duplicate", "direction"):
        document = yaml.safe_load(FIXTURE.read_text())
        if change == "unknown":
            document["typo"] = True
        elif change == "duplicate":
            document["pairs"][1]["name"] = "pair-z"
        else:
            document["pairs"][0]["direction"] = "target_to_source"
        path.write_text(yaml.safe_dump(document))
        with pytest.raises(ValueError):
            load_euvs_selection(path)


@pytest.mark.parametrize(
    "kind,fields",
    (
        ("full_sequence", {}),
        ("independent_clips", {"clip_indices": [3, 1, 2]}),
        ("dense_independent", {"source_pose_range": {"start": 40, "stop": 955}}),
        ("dense_overlap21", {"source_pose_range": {"start": 955, "stop": 955}}),
        ("dense_handoff", {"dense_camera_ids": [15, 5, 10]}),
    ),
)
def test_five_gaussian_kinds_preserve_full_declared_values(tmp_path, kind, fields):
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "kind": kind, "scene_id": 9, **fields}))
    expected = {"kind": kind, "scene_id": 9, **fields}
    if "clip_indices" in expected:
        expected["clip_indices"] = (3, 1, 2)
    if "dense_camera_ids" in expected:
        expected["dense_camera_ids"] = (15, 5, 10)
    assert asdict(load_gaussian_selection(path)) == expected


@pytest.mark.parametrize(
    "fields",
    (
        {"kind": "unknown"},
        {"kind": "dense_handoff", "dense_camera_ids": [5, 5]},
    ),
)
def test_gaussian_rejects_unknown_route_and_duplicate_camera_id(tmp_path, fields):
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "scene_id": 9, **fields}))
    with pytest.raises(ValueError):
        load_gaussian_selection(path)


def test_installed_selection_and_runner_imports_stay_cold():
    script = (
        "import sys; import novel_view.inputs.euvs.spec, novel_view.inputs.gaussian.spec; "
        "import novel_view.workflows.runner; "
        "assert not {'numpy','cv2','pyarrow','torch'} & sys.modules.keys(); "
        "assert not any(name.endswith('_worker') for name in sys.modules)"
    )
    subprocess.run((sys.executable, "-I", "-c", script), capture_output=True, text=True, check=True)

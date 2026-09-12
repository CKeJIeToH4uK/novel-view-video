"""Настоящий job строит три исторических keyset только по timelines."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from novel_view.config.job import load_resolved_job
from novel_view.inputs.waymo.types import WaymoContractError, WaymoFrameKey
from novel_view.preparation.waymo_depth.keyset import PreparedKeyset, WaymoClipKey
from novel_view.preparation.waymo_depth.spec import load_split_roles
from novel_view.workflows.runner import execute_job, select_job
from tests.support.stage5 import make_runtime


_FIT = ("example-fit-b", "example-fit-a", "example-fit-c")
_DEV = ("example-dev",)


def _document() -> dict:
    return {
        "schema_version": "waymo-segment-split/v1",
        "split_id": "waymo-ddw-lora-v1",
        "dataset": {"name": "synthetic"},
        "required_components": ["vehicle_pose"],
        "ordering": {"digest": "sha256"},
        "assignment": {"debug_subset": "first fit IDs"},
        "splits": {
            "fit": {"source_partition": "training", "count": 3, "segment_ids": list(_FIT)},
            "dev": {"source_partition": "training", "count": 1, "segment_ids": list(_DEV)},
            "debug_subset": {
                "source_partition": "training", "subset_of": "fit",
                "count": 3, "segment_ids": list(_FIT),
            },
            "exposed_debug": {"source_partition": "validation", "segment_ids": ["exposed"]},
            "transfer_test": {"source_partition": "validation", "segment_ids": ["transfer"]},
        },
    }


def test_job_writes_central121_in_order_and_first_two_debug_keys(
    tmp_path: Path, monkeypatch,
) -> None:
    root = Path(__file__).parents[4]
    job = load_resolved_job(root / "jobs/legacy/waymo/keysets/run.yaml", root)
    selected = select_job(job)
    assert (selected.image_variant, selected.preset.name) == ("core", "cpu_test")
    runtime = make_runtime(tmp_path)
    selection = runtime.roots.selections.parent / job.input["selection"]
    selection.parent.mkdir(parents=True)
    selection.write_text(yaml.safe_dump(_document()), encoding="utf-8")
    assert load_split_roles(selection) == ("waymo-ddw-lora-v1", _FIT, _DEV, _FIT)
    calls = []

    def load_index(data, partition, segment, start, count):
        assert data == runtime.roots.data / "waymo"
        assert (partition, start, count) == ("training", 0, 1)
        calls.append(segment)
        return SimpleNamespace(
            timeline=tuple(
                WaymoFrameKey(segment, 1_000_000 + index * 100_000)
                for index in range(125)
            )
        )

    monkeypatch.setattr(
        "novel_view.inputs.waymo.index.build_waymo_v2_window_index", load_index,
    )

    assert execute_job(job, runtime) == 0

    assert calls == list(_FIT + _DEV)
    for name, segments in (
        ("fit-central-v1", _FIT),
        ("dev-central-v1", _DEV),
        ("vertical-debug2-v1", _FIT[:2]),
    ):
        path = runtime.attempt_root / "keysets" / f"{name}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document == {
            "schema_version": "gen3c-waymo-keyset/v1",
            "keyset_id": name,
            "keys": [
                {
                    "split_id": "waymo-ddw-lora-v1",
                    "official_partition": "training",
                    "segment_id": segment,
                    "start_frame_index": 2,
                    "frame_timestamps_micros": [
                        1_000_000 + index * 100_000 for index in range(2, 123)
                    ],
                }
                for segment in segments
            ],
        }
        assert PreparedKeyset.load(path).keyset_id == name
    validation_key = WaymoClipKey(
        split_id="waymo-ddw-lora-v1",
        official_partition="validation",
        segment_id="example-segment",
        start_frame_index=2,
        frame_timestamps_micros=tuple(
            1_000_000 + index * 100_000 for index in range(2, 123)
        ),
    )
    validation = PreparedKeyset("validation-example", (validation_key,))
    path = runtime.attempt_root / "keysets/validation-example.json"
    validation.write(path)
    assert PreparedKeyset.load(path) == validation
    document = json.loads(path.read_text()) | {"alias": "another-file.json"}
    path.write_text(json.dumps(document))
    with pytest.raises(WaymoContractError, match="unexpected fields"):
        PreparedKeyset.load(path)
    attempt = json.loads((runtime.attempt_root / "attempt.json").read_text())
    assert attempt["recorded_state"] == "succeeded"


def test_role_reader_rejects_partition_changes_and_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "roles.yaml"
    document = _document()
    document["splits"]["dev"]["source_partition"] = "validation"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(WaymoContractError, match="training"):
        load_split_roles(path)

    document = _document()
    document["unexpected"] = True
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_split_roles(path)

"""The retained operator-only Waymo diagnostics keep their scientific thresholds."""

import json
from types import SimpleNamespace

import pytest
import yaml

from novel_view.cli.main import main
from novel_view.diagnostics import waymo
from novel_view.inputs.waymo.types import WaymoContractError


def test_frozen_ingress_window_and_exposed_membership(tmp_path):
    waymo._require_frozen_canary(51, 121, 1024.0)
    for settings in ((50, 121, 1024.0), (51, 1, 1024.0), (51, 121, 4096.0)):
        with pytest.raises(WaymoContractError):
            waymo._require_frozen_canary(*settings)
    document = dict(
        schema_version="waymo-segment-split/v1",
        split_id="waymo-ddw-lora-v1",
        splits={
            name: {"segment_ids": [name]}
            for name in ("fit", "dev", "debug_subset", "exposed_debug", "transfer_test")
        },
    )
    document["splits"]["exposed_debug"] = dict(
        source_partition="validation", segment_ids=[waymo._EXPOSED_DEBUG]
    )
    path = tmp_path / "split.yaml"
    path.write_text(yaml.safe_dump(document))
    waymo._require_exposed_debug(path)
    document["split_id"] = "another-experiment"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(WaymoContractError):
        waymo._require_exposed_debug(path)
    document["split_id"] = "waymo-ddw-lora-v1"
    document["splits"]["transfer_test"]["segment_ids"].append(waymo._EXPOSED_DEBUG)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(WaymoContractError):
        waymo._require_exposed_debug(path)


def test_raster_diagnostic_thresholds_and_public_internal_route(tmp_path, monkeypatch, capsys):
    plan = SimpleNamespace(
        valid_roi_xywh=(0, 0, 100, 100),
        crop_xywh=(5, 5, 90, 90),
        roi_retained_fraction=0.75,
        known_fraction=0.995,
    )
    assert waymo._camera_summary(plan, 100, 70)["projection_retained_fraction"] == 0.7
    for roi, known, raw, valid in (
        (0.749, 1.0, 100, 100),
        (0.8, 0.994, 100, 100),
        (0.8, 1.0, 100, 69),
        (0.8, 1.0, 0, 0),
    ):
        changed = SimpleNamespace(
            **(vars(plan) | {"roi_retained_fraction": roi, "known_fraction": known})
        )
        with pytest.raises(WaymoContractError):
            waymo._camera_summary(changed, raw, valid)
    split = tmp_path / "split.yaml"
    seen = []
    monkeypatch.setattr(waymo, "_require_exposed_debug", lambda path: seen.append(path))
    cameras = {name: {"known_fraction": 1.0} for name in ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")}
    monkeypatch.setattr(waymo, "_check_clip", lambda root: cameras if root == tmp_path else {})
    args = ["doctor-waymo", "raster", "--waymo-root", str(tmp_path), "--split-config", str(split)]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert seen == [split] and result["cameras"] == cameras
    assert (result["segment_id"], result["start_frame_index"], result["count"]) == (
        "10203656353524179475_7625_000_7645_000",
        51,
        121,
    )
    assert result["thresholds"] == dict(
        roi_retained_fraction=0.75, known_fraction=0.995, projection_retained_fraction=0.7
    )
    cameras.pop("FRONT_RIGHT")
    with pytest.raises(WaymoContractError):
        main(args)

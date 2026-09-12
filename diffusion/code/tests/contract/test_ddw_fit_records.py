"""Current and historical fit items keep their own payload roots and scientific status."""

import json

import pytest

from novel_view.preparation.waymo_ddw.legacy.assignment import DdwFitAssignment
from novel_view.preparation.waymo_ddw.legacy.axis import DdwVariant
from novel_view.preparation.waymo_ddw.legacy.execution import DdwDepthTelemetry
from novel_view.preparation.waymo_ddw.legacy.fit_record import (
    build_fit_item_record,
    read_fit_audit_source,
    read_fit_item_record,
    write_fit_item_record,
)
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from tests.unit.test_waymo_ddw_fit_workflows import _result


def _assignment(index):
    key = WaymoClipKey("fit-v1", "training", f"segment-{index}", 0, tuple(range(121)))
    return DdwFitAssignment(index, key, DdwVariant(index + 1, 1))


def _item(assignment, accepted):
    return build_fit_item_record(_result(assignment, accepted), DdwDepthTelemetry(4, 5, 6))


@pytest.mark.parametrize("accepted", (True, False))
def test_current_roundtrip_and_only_rejected_legacy_audit(tmp_path, accepted):
    path = tmp_path / "clip.json"
    expected = _item(_assignment(0), accepted)
    write_fit_item_record(path, expected)
    document = json.loads(path.read_text())
    assert document["schema_version"] == "gen3c-waymo-ddw-fit-clip/v2"
    assert read_fit_item_record(path) == read_fit_audit_source(path) == expected
    document["schema_version"] = "gen3c-waymo-ddw-fit-clip/v1"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        read_fit_item_record(path)
    if accepted:
        with pytest.raises(ValueError):
            read_fit_audit_source(path)
    else:
        assert read_fit_audit_source(path) == expected


@pytest.mark.parametrize("change", ("fraction", "reference"))
def test_external_scientific_fields_cannot_contradict_the_decision(tmp_path, change):
    path = tmp_path / "clip.json"
    write_fit_item_record(path, _item(_assignment(0), change == "fraction"))
    document = json.loads(path.read_text())
    if change == "fraction":
        document["observed"]["coverage_q05"] = 1.2
    else:
        document["headroom"] = {
            "gate_id": "waymo-ddw-headroom-gate-v1",
            "metrics": {
                "restoration_pixel_count": 120,
                "mean_absolute_corruption": 0.0,
                "changed_pixel_fraction": 0.0,
            },
            "passed": False,
            "failures": ["mean_absolute_corruption", "changed_pixel_fraction"],
        }
        document["mask_score"] = None
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        read_fit_item_record(path)

"""Runner boundaries for explicit fit, collection and rejected-item audit."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import yaml

from novel_view.config.job import load_resolved_job
from novel_view.preparation.waymo_ddw.legacy.assignment import build_ddw_fit_assignment
from novel_view.preparation.waymo_ddw.legacy.axis import DdwMaskCandidate
from novel_view.preparation.waymo_ddw.legacy.execution import DdwDepthTelemetry
from novel_view.preparation.waymo_ddw.legacy.fit import (
    DdwAcceptedFitCondition, DdwFitRenderTelemetry, DdwPreparedFitClip, decide_ddw_fit,
)
from novel_view.preparation.waymo_ddw.legacy.fit_record import (
    build_fit_item_record, collect_fit_records, write_fit_item_record,
)
from novel_view.preparation.waymo_ddw.legacy.headroom import DdwHeadroomMetrics, score_ddw_headroom
from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors
from novel_view.preparation.waymo_depth.keyset import PreparedKeyset, WaymoClipKey
from novel_view.workflows.runner import execute_job, select_job
from tests.support.stage5 import make_job, make_runtime


def _result(assignment, accepted=True):
    observed = DdwMaskDescriptors(.8 if accepted else .5, .1, 2., .1, .8, .1)
    reference = DdwMaskDescriptors(.8 if accepted else .55, .1, 2., .1, .8, .1)
    headroom = score_ddw_headroom(DdwHeadroomMetrics(120, .1, .1))
    decision = decide_ddw_fit(
        assignment, headroom,
        DdwMaskCandidate(assignment.clip_key, assignment.variant, observed, reference),
    )
    measurement = SimpleNamespace(
        observed=observed, peak_cuda_allocated_bytes=1,
        peak_cuda_reserved_bytes=2, elapsed_seconds=3.,
    )
    ref = SimpleNamespace(descriptors=reference, **vars(measurement))
    condition = DdwAcceptedFitCondition(
        DdwFitRenderTelemetry(4, 5, 6.), (1, 8),
        np.full((121, 1, 8, 3), 128, dtype=np.uint8),
        np.full((121, 1), 255, dtype=np.uint8),
    ) if accepted else None
    return DdwPreparedFitClip(decision, measurement, ref, condition)


def _setup(tmp_path):
    runtime = make_runtime(tmp_path, image="moge")
    keys = tuple(
        WaymoClipKey("fit", "training", f"segment-{i}", 0, tuple(range(121)))
        for i in range(5)
    )
    PreparedKeyset("fit-central-v1", keys).write(runtime.roots.runs / "keys.json")
    return runtime, build_ddw_fit_assignment(keys)


def _selection(runtime, **values):
    path = runtime.roots.selections / "fit.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "keyset": "keys.json", **values}))
    return "selections/fit.yaml"


def _fit_job(selection):
    return make_job(
        "waymo_ddw_fit", 1,
        {"reader": "waymo_v2", "dataset": "waymo", "selection": selection},
        {"moge_checkpoint": "moge/model.pt"}, preset="inference_cp1",
    )


def _measurement(monkeypatch, opened):
    lifetimes = []

    @contextmanager
    def open_clip(key, checkpoint, runtime, logs):
        if lifetimes:
            assert lifetimes[-1]() is None, "previous clip depth survived into the next model"
        depth = np.zeros(1, dtype=np.float32)
        lifetimes.append(weakref.ref(depth))
        opened.append((key, checkpoint, logs))
        logs.mkdir(parents=True)
        yield SimpleNamespace(
            source=SimpleNamespace(clip_key=key, depth=depth), depth=DdwDepthTelemetry(1., 2., 3.),
            warp=object(), reference=object(), logs=logs,
        )

    monkeypatch.setattr(
        "novel_view.preparation.waymo_ddw.legacy.execution.open_legacy_ddw_clip", open_clip
    )


@pytest.mark.parametrize(("folder", "image"), (
    ("ddw-fit", "moge"), ("ddw-fit-collect", "core"), ("ddw-fit-audit", "moge"),
))
def test_tracked_fit_forms_select_real_image(folder, image):
    root = Path(__file__).parents[4]
    job = load_resolved_job(root / f"jobs/legacy/waymo/{folder}/run.yaml", root)
    assert select_job(job).image_variant == image
    with pytest.raises(ValueError, match="unknown fields"):
        select_job(replace(job, input={**job.input, "scan": True}))


def test_fit_uses_full_assignment_then_subset_and_keeps_finished_items_on_error(
    tmp_path, monkeypatch,
):
    runtime, table = _setup(tmp_path)
    opened, assignments = [], []
    _measurement(monkeypatch, opened)

    def prepare(source, assignment, *args):
        assert source.clip_key == assignment.clip_key
        assignments.append(assignment)
        return _result(assignment, accepted=assignment.global_fit_index == 2)

    monkeypatch.setattr(
        "novel_view.preparation.waymo_ddw.legacy.fit.prepare_waymo_ddw_fit_clip", prepare
    )
    job = _fit_job(_selection(runtime, indices=[2, 0]))
    assert execute_job(job, runtime) == 0
    assert assignments == [table.assignments[2], table.assignments[0]]
    assert [entry[0] for entry in opened] == [item.clip_key for item in assignments]
    accepted_root = runtime.attempt_root / "items/clip-0002"
    rejected_root = runtime.attempt_root / "items/clip-0000"
    assert np.load(accepted_root / "condition_rgb.npy").shape == (121, 1, 8, 3)
    assert json.loads((rejected_root / "clip.json").read_text())["status"] == "rejected"
    assert not (rejected_root / "condition_rgb.npy").exists()
    assert len({entry[2] for entry in opened}) == 2

    failure_runtime = make_runtime(tmp_path / "failure", image="moge")
    PreparedKeyset("fit", tuple(item.clip_key for item in table.assignments)).write(
        failure_runtime.roots.runs / "keys.json"
    )

    def fail_second(source, assignment, *args):
        if assignment.global_fit_index == 0:
            raise RuntimeError("model failed")
        return _result(assignment)

    monkeypatch.setattr(
        "novel_view.preparation.waymo_ddw.legacy.fit.prepare_waymo_ddw_fit_clip", fail_second
    )
    failure_job = _fit_job(_selection(failure_runtime, indices=[2, 0, 1]))
    with pytest.raises(RuntimeError, match="model failed"):
        execute_job(failure_job, failure_runtime)
    assert (failure_runtime.attempt_root / "items/clip-0002/clip.json").is_file()
    assert not (failure_runtime.attempt_root / "items/clip-0001").exists()
    assert json.loads((failure_runtime.attempt_root / "attempt.json").read_text())["recorded_state"] == "failed"


def test_collection_explicit_partial_axis_keeps_source_record_payload_base(tmp_path):
    runtime, table = _setup(tmp_path)
    locators = []
    located = []
    for index in (2, 3, 0, 1):
        locator = f"source-{index}/clip.json"
        path = runtime.roots.runs / locator
        path.parent.mkdir()
        item = build_fit_item_record(
            _result(table.assignments[index], index % 2 == 0),
            DdwDepthTelemetry(1., 2., 3.),
        )
        write_fit_item_record(path, item)
        locators.append(locator)
        located.append((locator, item))
    selection = _selection(runtime, item_records=locators)
    job = make_job(
        "waymo_ddw_fit_collection",
        1,
        {"selection": selection},
        {},
        preset="cpu_test",
    )
    assert execute_job(job, runtime) == 0
    result = json.loads((runtime.attempt_root / "collection.json").read_text())
    assert result["schema_version"] == "gen3c-waymo-ddw-fit-collection/v1"
    assert result["status"] == "incomplete" and result["missing_indices"] == [4]
    assert [item["global_fit_index"] for item in result["accepted"]] == [0, 2]
    assert [item["global_fit_index"] for item in result["rejected"]] == [1, 3]
    assert [item["item_record"] for item in result["accepted"]] == [
        "source-0/clip.json", "source-2/clip.json",
    ]
    assert all(
        item["payload"]["condition_rgb"] == "condition_rgb.npy"
        for item in result["accepted"]
    )
    # There are deliberately no NPY files: collection opens only the named JSON.
    foreign = build_fit_item_record(
        _result(replace(table.assignments[0], global_fit_index=9)),
        DdwDepthTelemetry(1., 2., 3.),
    )
    for invalid in ((located[0], located[0]), (("wrong.json", foreign),)):
        with pytest.raises(ValueError):
            collect_fit_records(table.assignments, invalid)


@pytest.mark.parametrize("collection", (False, True))
def test_fit_rejects_validation_keyset_before_consuming_models_or_records(tmp_path, collection):
    runtime, table = _setup(tmp_path)
    PreparedKeyset("validation", tuple(
        replace(item.clip_key, official_partition="validation") for item in table.assignments
    )).write(runtime.roots.runs / "keys.json")
    if collection:
        selection = _selection(runtime, item_records=["not-opened.json"])
        job = make_job("waymo_ddw_fit_collection", 1, {"selection": selection}, {}, preset="cpu_test")
    else:
        job = _fit_job(_selection(runtime, indices=[0]))
    with pytest.raises(ValueError, match="official training"):
        execute_job(job, runtime)


def test_audit_selects_rejected_clip_before_model_and_preserves_source(tmp_path, monkeypatch):
    from novel_view.preparation.waymo_ddw.legacy.audit import DdwFitAuditMaterialization

    runtime, table = _setup(tmp_path)
    path = runtime.roots.runs / "source.json"
    write_fit_item_record(path, build_fit_item_record(_result(table.assignments[1], False),
                                                    DdwDepthTelemetry(1., 2., 3.)))
    original = path.read_bytes()
    opened = []
    _measurement(monkeypatch, opened)

    def materialize(source, selection, *args):
        assert source.clip_key == selection.decision.assignment.clip_key
        return DdwFitAuditMaterialization(
            selection, DdwFitRenderTelemetry(1, 2, 3.), (1, 8),
            np.zeros((121, 1, 8, 3), dtype=np.uint8),
            np.full((121, 1), 255, dtype=np.uint8),
        )

    monkeypatch.setattr(
        "novel_view.preparation.waymo_ddw.legacy.audit.materialize_waymo_ddw_fit_audit", materialize
    )
    monkeypatch.setattr(
        "novel_view.preparation.waymo_ddw.legacy.audit.write_audit_preview",
        lambda path, *_: path.write_bytes(b"preview"),
    )
    job = make_job(
        "waymo_ddw_audit", 1, {"reader": "waymo_v2", "dataset": "waymo", "item_record": "source.json"},
        {"moge_checkpoint": "moge/model.pt"}, preset="inference_cp1",
    )
    assert execute_job(job, runtime) == 0
    assert opened[0][0] == table.assignments[1].clip_key and path.read_bytes() == original
    assert (runtime.attempt_root / "preview.mp4").read_bytes() == b"preview"
    audit = json.loads((runtime.attempt_root / "audit.json").read_text())
    assert audit["schema_version"] == "gen3c-waymo-ddw-fit-visual-audit/v2"
    assert audit["source_status"] == "rejected"
    assert audit["source_item_record"] == "source.json"
    assert audit["payload"] == {
        "condition_rgb": "condition_rgb.npy",
        "condition_known": "condition_known.npy",
        "preview": "preview.mp4",
    }

    write_fit_item_record(path, build_fit_item_record(_result(table.assignments[1]),
                                                    DdwDepthTelemetry(1., 2., 3.)))
    opened.clear()
    from novel_view.workflows.waymo_ddw_audit import run_v1
    with pytest.raises(RuntimeError, match="rejected"):
        run_v1(job, runtime)
    assert opened == []

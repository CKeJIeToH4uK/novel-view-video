"""Historical assignment, fit and rejected-only audit without model weights."""

from collections import Counter
from dataclasses import astuple, replace
import hashlib
import json

import cv2
import numpy as np
import pytest

from novel_view.preparation.waymo_ddw.condition import DdwCondition
from novel_view.preparation.waymo_ddw.legacy import audit, fit
from novel_view.preparation.waymo_ddw.legacy.assignment import (
    DDW_FIT_RECIPE_ID,
    DDW_FIT_VARIANTS,
    build_ddw_fit_assignment,
)
from novel_view.preparation.waymo_ddw.legacy.axis import DdwMaskCandidate
from novel_view.preparation.waymo_ddw.legacy.headroom import DdwHeadroomMetrics, score_ddw_headroom
from novel_view.preparation.waymo_ddw.legacy.local import DdwLocalMeasurement
from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors
from novel_view.preparation.waymo_ddw.legacy.path import build_legacy_ddw_path
from novel_view.preparation.waymo_ddw.legacy.reference import DdwReferenceMeasurement
from novel_view.preparation.waymo_ddw.legacy.render import LegacyDdwRenderResult
from novel_view.preparation.waymo_ddw.legacy.source import LegacyDdwSource
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey


GOOD = DdwMaskDescriptors(0.8, 0.1, 2.0, 0.1, 0.8, 0.1)


def _key(index):
    return WaymoClipKey(
        "fit",
        "training",
        f"segment-{index:03d}",
        index,
        tuple(index * 1_000_000 + frame for frame in range(121)),
    )


def _headroom(passed=True):
    return score_ddw_headroom(
        DdwHeadroomMetrics(10, 0.1 if passed else 0.0, 0.1 if passed else 0.0)
    )


def _decision(assignment, observed):
    return fit.decide_ddw_fit(
        assignment,
        _headroom(),
        DdwMaskCandidate(
            assignment.clip_key,
            assignment.variant,
            observed,
            observed,
        ),
    )


@pytest.fixture
def clip():
    assignment = build_ddw_fit_assignment((_key(0),)).assignments[0]
    rgb = np.broadcast_to(
        np.arange(121, dtype=np.uint8)[:, None, None, None], (121, 16, 16, 3)
    ).copy()
    source = LegacyDdwSource(
        assignment.clip_key,
        rgb,
        np.ones((121, 16, 16), np.float32),
        np.ones((121, 16, 16), bool),
        np.ones((16, 16), bool),
        np.eye(3),
        np.broadcast_to(np.eye(4), (121, 4, 4)).copy(),
    )
    condition = DdwCondition(
        np.zeros((121, 3, 16, 16), np.float32), np.ones((121, 16, 16), bool), 5, 6, 3.0
    )
    rendered = LegacyDdwRenderResult(build_legacy_ddw_path(source, assignment.variant), condition)
    return assignment, source, rendered


def test_assignment_uses_blake2s_rank_and_balances_arbitrary_full_axis():
    keys = tuple(_key(index) for index in range(18))

    def score(key):
        payload = json.dumps(
            dict(
                split_id=key.split_id,
                official_partition=key.official_partition,
                segment_id=key.segment_id,
                start_frame_index=key.start_frame_index,
                frame_timestamps_micros=list(key.frame_timestamps_micros),
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.blake2s(
            DDW_FIT_RECIPE_ID.encode() + b"\0" + payload, digest_size=32
        ).digest()

    ranked = sorted(range(len(keys)), key=lambda index: (score(keys[index]), index))
    expected = {keys[index]: DDW_FIT_VARIANTS[rank % 8] for rank, index in enumerate(ranked)}
    assert (
        score(keys[0]).hex() == "a590e227feb52716f12ba5255c890acf9be0336049a3f1c8305479553bf0744f"
    )
    for order in (keys, keys[::-1]):
        actual = build_ddw_fit_assignment(order).assignments
        assert tuple(item.clip_key for item in actual) == order
        assert {item.clip_key: item.variant for item in actual} == expected
    assert tuple(Counter(expected.values())[variant] for variant in DDW_FIT_VARIANTS) == (
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
    )


def test_frozen_fit_and_separate_rejected_only_audit(clip):
    assignment, _, _ = clip
    assert astuple(fit.DDW_FIT_MASK_GATE)[1:] == (
        0.56,
        0.36,
        31.0,
        0.46,
        0.47,
        0.51,
        (0.13, 0.14, 15.75, 0.17, 0.19, 0.23),
        (0.0,) * 6,
    )
    assert astuple(audit.AUDIT_MASK_GATE)[1:7] == (0.4, 0.6, 40.0, 0.46, 0.47, 0.51)
    stopped = fit.decide_ddw_fit(assignment, _headroom(False), None)
    assert not stopped.accepted and stopped.candidate is None
    accepted = _decision(assignment, GOOD)
    assert accepted.accepted
    rejected = _decision(assignment, replace(GOOD, coverage_q05=0.5))
    assert not rejected.accepted and rejected.mask_score.failures == ("coverage_q05",)
    assert audit.select_waymo_ddw_fit_audit(rejected).audit_mask_score.passed
    assert not rejected.accepted  # A preview never changes the fit decision.
    for ineligible in (stopped, accepted, _decision(assignment, replace(GOOD, coverage_q05=0.39))):
        with pytest.raises(audit.WaymoDdwAuditError):
            audit.select_waymo_ddw_fit_audit(ineligible)


def test_condition_encoding_rounding_layout_and_little_endian():
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 1 / 255, 3 / 255, -3 / 255], np.float32)
    rgb = np.broadcast_to(values, (121, 3, 1, 8)).copy()
    rgb[:, 1] = rgb[:, 1, :, ::-1]
    rgb[:, 2] = 0
    known = np.zeros((121, 1, 8), bool)
    known[:, 0, (0, 2, 7)] = True
    encoded_rgb, encoded_known = fit.encode_waymo_ddw_condition(rgb, known)
    np.testing.assert_array_equal(encoded_rgb[0, 0, :, 0], [0, 0, 128, 255, 255, 128, 129, 126])
    assert encoded_rgb.shape == (121, 1, 8, 3)
    np.testing.assert_array_equal(encoded_rgb[0, 0, 0], [0, 126, 128])
    assert np.all(encoded_known == 0b10000101)


def test_fit_headroom_reference_and_accepted_only_rerender(clip, tmp_path, monkeypatch):
    assignment, source, rendered = clip
    local = DdwLocalMeasurement(rendered.path, GOOD, _headroom(False), 1, 2, 1.0)
    reference = DdwReferenceMeasurement(rendered.path, GOOD, 3, 4, 2.0)
    monkeypatch.setattr(fit, "measure_legacy_ddw_local", lambda *_: local)
    monkeypatch.setattr(
        fit,
        "measure_legacy_ddw_reference",
        lambda *_: pytest.fail("reference after failed headroom"),
    )
    monkeypatch.setattr(
        fit, "render_legacy_ddw", lambda *_: pytest.fail("rerender of rejected item")
    )
    stopped = fit.prepare_waymo_ddw_fit_clip(source, assignment, None, None, tmp_path)
    assert not stopped.accepted and stopped.reference is None
    local = replace(local, headroom=_headroom())
    wrong_reference = replace(reference, descriptors=replace(GOOD, coverage_q05=0.99))
    monkeypatch.setattr(fit, "measure_legacy_ddw_reference", lambda *_: wrong_reference)
    rejected = fit.prepare_waymo_ddw_fit_clip(source, assignment, None, None, tmp_path)
    assert not rejected.accepted and rejected.accepted_condition is None
    monkeypatch.setattr(fit, "measure_legacy_ddw_reference", lambda *_: reference)
    monkeypatch.setattr(fit, "render_legacy_ddw", lambda *_: rendered)
    monkeypatch.setattr(fit, "measure_legacy_ddw_rendered", lambda *_: (GOOD, _headroom()))
    accepted = fit.prepare_waymo_ddw_fit_clip(source, assignment, None, None, tmp_path)
    assert accepted.accepted and accepted.accepted_condition.image_size_hw == (16, 16)
    assert np.all(accepted.accepted_condition.condition_rgb == 128)
    assert np.all(accepted.accepted_condition.condition_known == 255)
    monkeypatch.setattr(
        fit,
        "measure_legacy_ddw_rendered",
        lambda *_: (replace(GOOD, coverage_q05=0.7), _headroom()),
    )
    with pytest.raises(fit.WaymoDdwFitError):
        fit.prepare_waymo_ddw_fit_clip(source, assignment, None, None, tmp_path)


def test_audit_real_video_and_changed_or_failed_rerender(clip, tmp_path, monkeypatch):
    assignment, source, rendered = clip
    observed = replace(GOOD, coverage_q05=0.5)
    selection = audit.select_waymo_ddw_fit_audit(_decision(assignment, observed))
    monkeypatch.setattr(audit, "render_legacy_ddw", lambda *_: rendered)
    monkeypatch.setattr(audit, "measure_legacy_ddw_rendered", lambda *_: (observed, _headroom()))
    result = audit.materialize_waymo_ddw_fit_audit(
        source, selection, None, tmp_path / "out", tmp_path / "back"
    )
    assert result.selection == selection and result.telemetry.elapsed_seconds == 3.0
    assert result.condition_rgb.shape == (121, 16, 16, 3)
    known = np.ones((121, 16, 16), bool)
    known[:, 8:, :8] = False
    result = replace(
        result, condition_known=np.packbits(known.reshape(121, -1), axis=1, bitorder="little")
    )
    output = tmp_path / "preview.mp4"
    audit.write_audit_preview(output, source, result)
    reader = cv2.VideoCapture(str(output))
    frames = []
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        reader.release()
    assert len(frames) == 121 and frames[0].shape == (16, 32, 3)
    np.testing.assert_allclose(frames[0][12, 20], [255, 0, 255], atol=15)
    assert frames[0][:, :16].mean() < 5 and frames[-1][:, :16].mean() > 110
    monkeypatch.setattr(
        audit,
        "measure_legacy_ddw_rendered",
        lambda *_: (replace(observed, coverage_q05=0.49), _headroom()),
    )
    with pytest.raises(audit.WaymoDdwAuditError):
        audit.materialize_waymo_ddw_fit_audit(
            source, selection, None, tmp_path / "out", tmp_path / "back"
        )

    def fail(*_):
        raise OSError("worker failure")

    monkeypatch.setattr(audit, "render_legacy_ddw", fail)
    with pytest.raises(OSError):
        audit.materialize_waymo_ddw_fit_audit(
            source, selection, None, tmp_path / "out", tmp_path / "back"
        )

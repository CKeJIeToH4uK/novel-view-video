"""Scientific ranking, explicit outcomes and historical selection records."""

import json
from dataclasses import asdict, replace

import pytest
import yaml

from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.preparation.waymo_depth.candidate_record import (
    read_candidate_record,
    write_candidate_record,
)
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.preparation.waymo_depth.selection import (
    DEPTH_BACKENDS,
    DepthCandidateFailure,
    DepthCandidateGateSpec,
    DepthCandidateReport,
    DepthClipMetrics,
    DepthGateSpec,
    DepthGateStop,
    normalize_depth_selection_evidence,
    select_depth_backend_v1,
    select_depth_backend_v2,
)
from novel_view.preparation.waymo_depth.selection_record import read_selection_record
from novel_view.workflows.runner import select_job
from novel_view.workflows.waymo_depth_selection import run_v1, run_v2
from tests.support.stage5 import make_job, make_runtime


CAMERAS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")
KEYS = tuple(
    WaymoClipKey(
        "waymo-ddw-lora-v1",
        "training",
        f"segment-{i}",
        9,
        tuple(i * 1_000_000 + 9 + t for t in range(121)),
    )
    for i in range(8)
)
GATES = (
    DepthGateSpec(300_000, 0.7, 0.2, 60.0, 0.005, 16_384.0, 24_576.0),
    DepthCandidateGateSpec(0.7, 0.2, 60.0, 0.005, 16_384.0, 24_576.0),
)


def _report(backend, keys=KEYS, **changes):
    return DepthCandidateReport(
        backend,
        tuple(
            replace(
                DepthClipMetrics(key, camera, 330_000, 0.9, 0.1, 0.2, 10.0, 0.1, 1.0, 50.0, 100.0),
                **changes,
            )
            for key in keys
            for camera in CAMERAS
        ),
    )


def test_gates_apply_to_every_camera_and_keep_inclusive_boundaries():
    gate = DepthGateSpec(50, 0.8, 0.3, 3.0, 0.01, 2000.0, 120.0)
    passing = _report(DEPTH_BACKENDS[1], p95_abs_z_m=1.0, median_abs_rel=0.2)
    for field, value in (
        ("heldout_count", 49),
        ("valid_fraction", 0.79),
        ("median_abs_rel", 0.31),
        ("p95_abs_z_m", 3.01),
        ("peak_ram_mib", 2001.0),
        ("peak_vram_mib", 121.0),
    ):
        base = _report(DEPTH_BACKENDS[0], p95_abs_z_m=1.0)
        rows = list(base.entries)
        rows[17] = replace(rows[17], **{field: value})
        result = select_depth_backend_v1(gate, KEYS, (replace(base, entries=tuple(rows)), passing))
        assert result.selected_backend == DEPTH_BACKENDS[1]
        assert not result.scores[0].passed and result.scores[1].passed
    boundary = _report(
        DEPTH_BACKENDS[0],
        heldout_count=50,
        valid_fraction=0.8,
        median_abs_rel=0.3,
        p95_abs_z_m=3.0,
        peak_ram_mib=2000.0,
        peak_vram_mib=120.0,
    )
    assert select_depth_backend_v1(gate, KEYS, (boundary, passing)).scores[0].passed


@pytest.mark.parametrize("version", (1, 2))
def test_rank_uses_epsilon_then_p95_jitter_vram_time_and_name(version):
    select = (select_depth_backend_v1, select_depth_backend_v2)[version - 1]
    # Each row makes a later criterion prefer the opposite method.
    cases = (
        ({"median_abs_rel": 0.1, "p95_abs_z_m": 20.0}, {"median_abs_rel": 0.105}, 1),
        ({"median_abs_rel": 0.1, "p95_abs_z_m": 20.0}, {"median_abs_rel": 0.11}, 0),
        (
            {"temporal_scale_jitter": 0.01, "peak_vram_mib": 110.0, "elapsed_seconds": 4.0},
            {"temporal_scale_jitter": 0.02, "peak_vram_mib": 90.0},
            0,
        ),
        ({"peak_vram_mib": 90.0, "elapsed_seconds": 4.0}, {"peak_vram_mib": 100.0}, 0),
        ({"elapsed_seconds": 3.0}, {"elapsed_seconds": 2.0}, 1),
        ({}, {}, 0),
    )
    for left, right, expected in cases:
        reports = (_report(DEPTH_BACKENDS[0], **left), _report(DEPTH_BACKENDS[1], **right))
        assert (
            select(GATES[version - 1], KEYS, reports).selected_backend == DEPTH_BACKENDS[expected]
        )
    report = _report(DEPTH_BACKENDS[0])
    first = replace(
        report.entries[0],
        median_abs_rel=0.2,
        p95_abs_z_m=20.0,
        temporal_scale_jitter=0.4,
        elapsed_seconds=5.0,
        peak_vram_mib=110.0,
    )
    score = select(
        GATES[version - 1],
        KEYS,
        (replace(report, entries=(first, *report.entries[1:])), _report(DEPTH_BACKENDS[1])),
    ).scores[0]
    assert (
        score.median_abs_rel,
        score.median_p95_abs_z_m,
        score.median_temporal_scale_jitter,
        score.peak_vram_mib,
        score.total_elapsed_seconds,
    ) == (0.1, 10.0, 0.1, 110.0, 28.0)


def test_selection_requires_ordered_axis_and_keeps_failures_separate():
    reports = tuple(_report(backend) for backend in DEPTH_BACKENDS)
    for rows in (reports[0].entries[:-1], tuple(reversed(reports[0].entries))):
        with pytest.raises(WaymoContractError):
            select_depth_backend_v1(GATES[0], KEYS, (replace(reports[0], entries=rows), reports[1]))
    with pytest.raises(WaymoContractError):
        select_depth_backend_v1(GATES[0], KEYS, tuple(reversed(reports)))
    repeated = KEYS[:-1] + (replace(KEYS[-1], segment_id=KEYS[0].segment_id),)
    with pytest.raises(WaymoContractError):
        select_depth_backend_v1(
            GATES[0], repeated, tuple(_report(backend, repeated) for backend in DEPTH_BACKENDS)
        )
    failure = DepthCandidateFailure(DEPTH_BACKENDS[1], ("candidate_execution_failed: geometry",))
    result = select_depth_backend_v1(GATES[0], KEYS, (reports[0], failure))
    assert result.selected_backend == DEPTH_BACKENDS[0]
    assert result.scores[1].failures == failure.failures
    assert result.scores[1].median_abs_rel is None
    failed = tuple(_report(backend, valid_fraction=0.1) for backend in DEPTH_BACKENDS)
    assert isinstance(select_depth_backend_v1(GATES[0], KEYS, failed), DepthGateStop)
    assert isinstance(
        select_depth_backend_v1(
            GATES[0],
            KEYS,
            (
                DepthCandidateFailure(DEPTH_BACKENDS[0], ("failed",)),
                failure,
            ),
        ),
        DepthGateStop,
    )


@pytest.mark.parametrize("version,count", ((1, 253_517), (1, 330_000), (2, 253_517)))
def test_real_candidate_records_to_selection_v4_and_scientific_stop(tmp_path, version, count):
    runtime = make_runtime(tmp_path)
    candidates = []
    locators = tuple(f"candidate-{index}/result.json" for index in range(8))
    for key, locator in zip(KEYS, locators, strict=True):
        outcomes = tuple(
            _report(backend, (key,), heldout_count=count) for backend in DEPTH_BACKENDS
        )
        candidates.append(outcomes)
        path = runtime.roots.runs / locator
        path.parent.mkdir()
        write_candidate_record(path, key, outcomes)
        record = read_candidate_record(path)
        assert record.clip_key == key and record.outcomes == outcomes
    path = runtime.roots.selections / "reports.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "candidate_results": list(locators)}))
    job = make_job(
        "waymo_depth_selection",
        version,
        {"selection": "selections/reports.yaml"},
        asdict(GATES[version - 1]),
        preset="cpu_test",
    )
    assert select_job(job).image_variant == "core"
    assert (run_v1 if version == 1 else run_v2)(job, runtime) == 0
    result_path = runtime.attempt_root / "depth-selection.json"
    result = read_selection_record(result_path)
    assert result.workflow_version == version and result.candidate_results == locators
    assert result.selection_keys == KEYS
    document = json.loads(result_path.read_text())
    assert document["schema_version"] == "gen3c-waymo-depth-selection/v4"
    assert document["workflow"] == {"name": "waymo_depth_selection", "version": version}
    assert "sha256" not in result_path.read_text()
    if version == 1 and count < 300_000:
        assert isinstance(result.outcome, DepthGateStop)
        assert document["outcome"]["status"] == "scientific_stop"
    else:
        assert result.outcome.selected_backend == DEPTH_BACKENDS[0]
    if version == 2:
        evidence = normalize_depth_selection_evidence(KEYS, tuple(candidates))
        assert result.selection_evidence == evidence
        assert {row.heldout_count for row in evidence.entries} == {count}
        for contradiction in (
            {"passed": False, "failures": ["scientific gate"]},
            {"median_abs_rel": None},
        ):
            changed = json.loads(json.dumps(document))
            changed["outcome"]["scores"][0].update(contradiction)
            result_path.write_text(json.dumps(changed))
            with pytest.raises(WaymoContractError):
                read_selection_record(result_path)
    path.write_text(path.read_text() + "unknown: true\n")
    with pytest.raises(ValueError):
        (run_v1 if version == 1 else run_v2)(job, runtime)


def test_old_selection_v2_v3_reads_without_opening_provenance(tmp_path):
    reports = tuple(_report(backend) for backend in DEPTH_BACKENDS)
    candidates = tuple(
        tuple(_report(backend, (key,)) for backend in DEPTH_BACKENDS) for key in KEYS
    )
    for schema_version, workflow_version in ((2, 1), (3, 2)):
        paths = [str(tmp_path / f"absent-{index}.json") for index in range(8)]
        document = dict(
            schema_version=f"gen3c-waymo-depth-selection/v{schema_version}",
            gate_id=f"waymo-depth-gate-v{workflow_version}",
            gate_spec=asdict(GATES[workflow_version - 1]),
            selection_keys=[asdict(key) for key in KEYS],
            candidate_results=paths,
            decision=asdict(
                (select_depth_backend_v1, select_depth_backend_v2)[workflow_version - 1](
                    GATES[workflow_version - 1], KEYS, reports
                )
            ),
        )
        if schema_version == 3:
            document.update(
                interpretation="exploratory",
                amendment_code_commit="c" * 40,
                amendment=dict(
                    supersedes_gate_id="waymo-depth-gate-v1",
                    v1_outcome="stopped",
                    reason="candidate-neutral-evidence-ownership",
                    interpretation="exploratory",
                    source_candidate_schema="gen3c-waymo-depth-candidate/v2",
                    source_producer_commit="a" * 40,
                    source_results_sha256=["b" * 64] * 8,
                ),
                candidate_results=[{"path": path, "sha256": "d" * 64} for path in paths],
                selection_evidence=asdict(normalize_depth_selection_evidence(KEYS, candidates)),
            )
        path = tmp_path / f"v{schema_version}.json"
        path.write_text(json.dumps(document))
        result = read_selection_record(path)
        assert result.workflow_version == workflow_version
        assert result.candidate_results == tuple(paths)
        assert result.selection_keys == KEYS

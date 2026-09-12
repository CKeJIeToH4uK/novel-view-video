"""Exact scientific outcomes and explicit resource routing, without model execution."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from novel_view.cli.acceptance import main
from novel_view.config.job import resolved_job_to_mapping
from novel_view.diagnostics import _job_worker, job as diagnostics
from novel_view.inputs.waymo.index import _ALL_COMPONENTS
from novel_view.preparation.waymo_depth.candidate_record import write_candidate_record
from novel_view.preparation.waymo_depth.keyset import clip_key_to_mapping
from novel_view.preparation.waymo_depth.selection import DEPTH_BACKENDS, DepthCandidateFailure
from novel_view.workflows.runner import execute_job
from tests.integration.test_ddw_legacy_orders import CLIP, FORMS, _job, _models
from tests.support.stage5 import make_job, make_runtime
from tests.unit.test_waymo_legacy_depth_selection import GATES, KEYS, _report


def _json(path, values):
    path.write_text(json.dumps(values))
    return path


def _depth_inputs(tmp_path, failure):
    runtime = make_runtime(tmp_path)
    locators = tuple(f"candidate-{7 - index}/result.json" for index in range(8))
    for key, locator in zip(KEYS, locators):
        path = runtime.roots.runs / locator
        path.parent.mkdir()
        reports = tuple(
            _report(backend, (key,), median_abs_rel=0.5 if index < failure else 0.1)
            for index, backend in enumerate(DEPTH_BACKENDS)
        )
        write_candidate_record(path, key, reports)
    selection = _json(
        runtime.roots.selections / "reports.json",
        {"schema_version": 1, "candidate_results": list(locators)},
    )
    job = make_job(
        "waymo_depth_selection",
        2,
        {"selection": "selections/reports.json"},
        asdict(GATES[1]),
        preset="cpu_test",
    )
    return runtime, job, selection


@pytest.mark.parametrize(
    "form,failure",
    [(0, False), (1, False), (1, True), (2, False), (2, True), (3, False), (3, True)],
)
def test_ddw_reader_preserves_actual_exit_and_scientific_outcome(
    tmp_path, monkeypatch, capsys, form, failure
):
    directory, _, _, axis, _ = FORMS[form]
    runtime = make_runtime(tmp_path)
    job = _job(directory)
    _models(monkeypatch, runtime, {axis[1].variant_id} if failure else set())
    assert execute_job(job, runtime) == (2 if failure else 0)
    capsys.readouterr()
    clip = _json(tmp_path / "clip.json", {"schema_version": 1, "clip": clip_key_to_mapping(CLIP)})
    kind = ("canary-v1", "canary-v2", "probe", "survey")[form]
    arguments = [
        "waymo-ddw",
        "--kind",
        kind,
        "--path",
        str(runtime.attempt_root / "result.json"),
        "--clip-selection",
        str(clip),
        "--expected-depth-selection",
        job.input["depth_selection"],
    ]
    assert main(arguments) == 0
    expected_format = (
        "gen3c-waymo-ddw-engineering-canary/v2",
        "gen3c-waymo-ddw-v2-canary/v2",
        "gen3c-waymo-ddw-v3-probe/v2",
        "gen3c-waymo-ddw-survey/v2",
    )[form]
    assert capsys.readouterr().out.splitlines() == [
        f"format={expected_format}",
        f"outcome={'scientific_stop' if failure else 'passed'}",
        f"expected_exit={2 if failure else 0}",
        "selected_backend=none",
        "ddw_allowed=false",
    ]
    if form == 0:
        code = """
import sys
from pathlib import Path
from novel_view.preparation.waymo_ddw.legacy.record import read_legacy_ddw_outcome
read_legacy_ddw_outcome(Path(sys.argv[1]), 'canary-v1')
assert not {'torch', 'cv2', 'moge', 'cosmos_predict1', 'vggt_omega'} & set(sys.modules)
"""
        path = runtime.attempt_root / "result.json"
        subprocess.run([sys.executable, "-c", code, str(path)], check=True)
        document = json.loads(path.read_text())
        document["outcome"]["variants"][0]["reference_telemetry"] = None
        _json(path, document)
        with pytest.raises(ValueError):
            main(arguments)


def test_stop_requires_real_headroom_complete_axis_and_recorded_worker(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path)
    job = _job("ddw-probe")
    _models(monkeypatch, runtime, {FORMS[2][3][1].variant_id})
    assert execute_job(job, runtime) == 2
    path = runtime.attempt_root / "result.json"
    original = path.read_text()
    clip = _json(tmp_path / "clip.json", {"schema_version": 1, "clip": clip_key_to_mapping(CLIP)})
    args = [
        "waymo-ddw",
        "--kind",
        "probe",
        "--path",
        str(path),
        "--clip-selection",
        str(clip),
        "--expected-depth-selection",
        job.input["depth_selection"],
    ]
    for change in ("axis", "headroom", "format"):
        document = json.loads(original)
        if change == "axis":
            document["outcome"]["outcomes"].reverse()
        elif change == "headroom":
            document["outcome"]["outcomes"][1]["headroom"]["passed"] = True
        else:
            document["format"] = "gen3c-waymo-ddw-v3-probe/v1"
        _json(path, document)
        with pytest.raises(ValueError):
            main(args)
    path.write_text(original)
    attempt = runtime.attempt_root / "attempt.json"
    record = json.loads(attempt.read_text())
    record["worker_exit_code"] = None
    _json(attempt, record)
    with pytest.raises(ValueError):
        main(args)


@pytest.mark.parametrize(
    "failure,backend", [(0, DEPTH_BACKENDS[0]), (1, DEPTH_BACKENDS[1]), (2, "none")]
)
def test_depth_gate_reads_full_order_and_admits_only_moge(tmp_path, capsys, failure, backend):
    runtime, job, selection = _depth_inputs(tmp_path, failure)
    assert execute_job(job, runtime) == 0
    capsys.readouterr()
    args = [
        "waymo-depth-selection",
        "--path",
        str(runtime.attempt_root / "depth-selection.json"),
        "--reports-selection",
        str(selection),
    ]
    assert main(args) == 0
    assert capsys.readouterr().out.splitlines() == [
        "format=gen3c-waymo-depth-selection/v4",
        f"outcome={'scientific_stop' if backend == 'none' else 'selected'}",
        "expected_exit=0",
        f"selected_backend={backend}",
        f"ddw_allowed={str(backend == DEPTH_BACKENDS[0]).lower()}",
    ]
    values = json.loads(selection.read_text())
    values["candidate_results"].reverse()
    _json(selection, values)
    with pytest.raises(ValueError):
        main(args)


def test_candidate_uses_existing_typed_outcomes_and_matching_clip(tmp_path, capsys):
    path = tmp_path / "result.json"
    write_candidate_record(
        path,
        KEYS[0],
        (
            _report(DEPTH_BACKENDS[0], (KEYS[0],)),
            DepthCandidateFailure(
                DEPTH_BACKENDS[1], ("candidate_execution_failed: invalid geometry",)
            ),
        ),
    )
    _json(tmp_path / "attempt.json", {"worker_exit_code": 0, "recorded_state": "succeeded"})
    clip = _json(
        tmp_path / "clip.json", {"schema_version": 1, "clip": clip_key_to_mapping(KEYS[0])}
    )
    args = ["waymo-candidate", "--path", str(path), "--clip-selection", str(clip)]
    assert main(args) == 0
    assert capsys.readouterr().out.splitlines() == [
        "format=gen3c-waymo-depth-candidate/v2",
        "outcome=completed",
        "expected_exit=0",
        "selected_backend=none",
        "ddw_allowed=false",
    ]
    _json(clip, {"schema_version": 1, "clip": clip_key_to_mapping(KEYS[1])})
    with pytest.raises(ValueError):
        main(args)


def test_doctor_checks_all_reports_and_exact_model_prefixes(tmp_path, monkeypatch):
    runtime, gate, reports = _depth_inputs(tmp_path, 0)
    assert execute_job(gate, runtime) == 0
    calls = []
    monkeypatch.setattr(
        diagnostics.subprocess,
        "run",
        lambda argv, **_: (calls.append(argv) or SimpleNamespace(returncode=0)),
    )

    def doctor(job):
        values = resolved_job_to_mapping(job)
        values.pop("recipe", None)
        path = _json(runtime.roots.jobs / "job.json", values)
        return diagnostics.doctor_job(path, runtime.roots.jobs.parent, roots=runtime.roots)

    assert doctor(gate) == 0 and calls == []
    last = runtime.roots.runs / json.loads(reports.read_text())["candidate_results"][-1]
    contents = last.read_text()
    last.unlink()
    with pytest.raises(FileNotFoundError):
        doctor(gate)
    last.write_text(contents)
    _json(
        runtime.roots.selections / "clip.json",
        {"schema_version": 1, "clip": clip_key_to_mapping(KEYS[0])},
    )
    for component in _ALL_COMPONENTS:
        path = runtime.roots.data / "waymo/training" / component / f"{KEYS[0].segment_id}.parquet"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"parquet")
    checkpoint = _job("ddw-probe").parameters["moge_checkpoint"]
    for relative in (checkpoint, "vggt/model.pt"):
        path = runtime.roots.models / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(b"weights")
    comparison = make_job(
        "waymo_depth_comparison",
        1,
        {
            "reader": "waymo_v2",
            "dataset": "waymo",
            "selection": "selections/clip.json",
        },
        {"moge_checkpoint": checkpoint, "vggt_checkpoint": "vggt/model.pt"},
        preset="inference_cp1",
    )
    assert doctor(comparison) == 0
    for directory, *_ in FORMS:
        job = _job(directory)
        assert (
            doctor(
                replace(
                    job,
                    input={
                        **job.input,
                        "selection": "selections/clip.json",
                        "depth_selection": "attempt/depth-selection.json",
                    },
                )
            )
            == 0
        )
    assert [(call[0], call[4], call[6]) for call in calls] == [
        ("/opt/envs/gen3c/bin/python", "moge", "1"),
        ("/opt/envs/vggt/bin/python", "vggt", "1"),
        *[("/opt/envs/gen3c/bin/python", "legacy_ddw", "1")] * 4,
    ]
    imports = []
    monkeypatch.setattr(_job_worker.importlib, "import_module", imports.append)
    assert _job_worker._imports("moge") == ("moge.model.v1",)
    assert _job_worker._imports("legacy_ddw") == (
        "moge.model.v1",
        "cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch",
        "cosmos_predict1.diffusion.inference.cache_3d",
    )
    assert len(imports) == 4

"""Different historical canary/probe/survey orders through the real runner."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from novel_view.config.job import load_resolved_job
from novel_view.preparation.waymo_ddw.legacy.canary import WaymoDdwCanaryError
from novel_view.preparation.waymo_ddw.legacy.execution import DdwDepthTelemetry
from novel_view.preparation.waymo_ddw.legacy.headroom import DdwHeadroomMetrics, score_ddw_headroom
from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors
from novel_view.preparation.waymo_ddw.legacy.selection import DDW_VARIANTS
from novel_view.preparation.waymo_ddw.legacy.v2 import DDW_V2_VARIANTS
from novel_view.preparation.waymo_ddw.legacy.v2_canary import DDW_V3_PROBE_VARIANTS
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.workflows.runner import execute_job, select_job
from novel_view.workflows.waymo_ddw_canary import parse_job
from tests.support.stage5 import make_runtime


FORMS = (
    ("ddw-canary-v1", "waymo_ddw_canary", 1, DDW_VARIANTS, "gen3c-waymo-ddw-engineering-canary/v2"),
    ("ddw-canary-v2", "waymo_ddw_canary", 2, DDW_V2_VARIANTS, "gen3c-waymo-ddw-v2-canary/v2"),
    ("ddw-probe", "waymo_ddw_probe", 1, DDW_V3_PROBE_VARIANTS, None),
    ("ddw-survey", "waymo_ddw_survey", 1, DDW_VARIANTS, None),
)
DESCRIPTORS = DdwMaskDescriptors(0.8, 0.1, 2.0, 0.1, 0.8, 0.1)
CLIP = WaymoClipKey("split", "training", "segment", 0, tuple(range(121)))


def _job(directory):
    root = Path(__file__).parents[4]
    return load_resolved_job(root / f"jobs/legacy/waymo/{directory}/run.yaml", root)


def _models(monkeypatch, runtime, failed, *, model_error=False):
    calls = []

    @contextmanager
    def measurement(*_):
        yield SimpleNamespace(
            source=SimpleNamespace(clip_key=CLIP),
            depth=DdwDepthTelemetry(1.0, 2.0, 3.0),
            warp=None,
            reference=None,
            logs=runtime.attempt_root,
        )

    def local(_source, variant, *_):
        calls.append(("local", variant.variant_id))
        value = 0.0 if variant.variant_id in failed else 0.1
        return SimpleNamespace(
            path=variant,
            observed=DESCRIPTORS,
            headroom=score_ddw_headroom(DdwHeadroomMetrics(1, value, value)),
            peak_cuda_allocated_bytes=1,
            peak_cuda_reserved_bytes=2,
            elapsed_seconds=3.0,
        )

    def reference(_source, variant, *_):
        calls.append(("reference", variant.variant_id))
        if model_error:
            raise RuntimeError("model failure")
        return SimpleNamespace(
            descriptors=DESCRIPTORS,
            peak_cuda_allocated_bytes=4,
            peak_cuda_reserved_bytes=5,
            elapsed_seconds=6.0,
        )

    monkeypatch.setattr(
        "novel_view.preparation.waymo_ddw.legacy.execution.open_legacy_ddw_measurement", measurement
    )
    for owner in ("canary", "v2_canary"):
        monkeypatch.setattr(
            f"novel_view.preparation.waymo_ddw.legacy.{owner}.measure_legacy_ddw_local", local
        )
        monkeypatch.setattr(
            f"novel_view.preparation.waymo_ddw.legacy.{owner}.measure_legacy_ddw_reference",
            reference,
        )
    return calls


@pytest.mark.parametrize("directory,workflow,version,axis,format_name", FORMS)
@pytest.mark.parametrize("failure", [False, True])
def test_complete_and_scientifically_failed_orders(
    tmp_path,
    monkeypatch,
    directory,
    workflow,
    version,
    axis,
    format_name,
    failure,
):
    job = _job(directory)
    assert (job.workflow.name, job.workflow.version) == (workflow, version)
    assert parse_job(job).recipe.axis == axis
    assert select_job(job).image_variant == "moge"
    runtime = make_runtime(tmp_path)
    failed = {axis[1].variant_id, axis[-1].variant_id} if failure else set()
    calls = _models(monkeypatch, runtime, failed)
    canary = workflow == "waymo_ddw_canary"
    if failure and canary and version == 1:
        with pytest.raises(WaymoDdwCanaryError):
            execute_job(job, runtime)
        result = None
    else:
        assert execute_job(job, runtime) == (2 if failure else 0)
        result = json.loads((runtime.attempt_root / "result.json").read_text())
    visited = axis[:2] if failure and canary else axis
    expected = []
    for variant in visited:
        expected.append(("local", variant.variant_id))
        if variant.variant_id not in failed:
            expected.append(("reference", variant.variant_id))
    assert calls == expected  # Scientific local→reference order, not private helper choreography.
    attempt = json.loads((runtime.attempt_root / "attempt.json").read_text())
    assert attempt["recorded_state"] == ("failed" if failure else "succeeded")
    if result is None:
        return
    outcome = result["outcome"]
    if canary:
        assert result["format"] == format_name
        if failure:
            assert outcome["status"] == "scientific_stop"
            assert outcome["failed_variant"]["variant_id"] == axis[1].variant_id
            assert attempt["worker_exit_code"] == 2
        else:
            assert [item["variant"]["variant_id"] for item in outcome["variants"]] == [
                v.variant_id for v in axis
            ]
    else:
        outcomes = outcome["outcomes"]
        assert [item["variant"]["variant_id"] for item in outcomes] == [v.variant_id for v in axis]
        assert {
            item["variant"]["variant_id"]
            for item in outcomes
            if item["candidate"] is None and item["reference_telemetry"] is None
        } == failed


def test_unknown_external_field_and_changed_scientific_axis():
    job = _job("ddw-canary-v1")
    with pytest.raises(ValueError):
        parse_job(replace(job, input={**job.input, "scan": True}))
    with pytest.raises(ValueError):
        parse_job(
            replace(
                job, parameters={**job.parameters, "axis": list(reversed(job.parameters["axis"]))}
            )
        )


def test_real_model_error_is_not_scientific_stop(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path)
    _models(monkeypatch, runtime, set(), model_error=True)
    with pytest.raises(RuntimeError):
        execute_job(_job("ddw-probe"), runtime)
    assert not (runtime.attempt_root / "result.json").exists()
    assert (
        json.loads((runtime.attempt_root / "attempt.json").read_text())["recorded_state"]
        == "failed"
    )

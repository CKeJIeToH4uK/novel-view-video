"""Explicit job doctor reads named resources without executing the workflow."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from novel_view.config.job import load_resolved_job
from novel_view.diagnostics.job import doctor_job, _generation_weights
from novel_view.generation.gen3c.model_request import build_model_request
from novel_view.workflows.euvs_generation import parse_euvs_generation_job
from novel_view.workflows.gaussian_generation import (
    parse_gaussian_dense_generation_job, parse_gaussian_generation_job,
)
from novel_view.inputs.waymo.index import _ALL_COMPONENTS
from novel_view.runtime.context import CONTAINER_ROOTS

ROOT = Path(__file__).resolve().parents[4]


def test_doctor_reads_named_resources_without_running_workflow(tmp_path, monkeypatch):
    child = Mock(return_value=SimpleNamespace(returncode=47))
    monkeypatch.setattr("novel_view.diagnostics.job.subprocess.run", child)
    roots = replace(
        CONTAINER_ROOTS,
        data=tmp_path / "data",
        models=tmp_path / "models",
        runs=tmp_path / "runs",
        selections=ROOT / "selections",
    )
    cpu_job = ROOT / "jobs/examples/lifecycle-smoke/run.yaml"
    assert doctor_job(cpu_job, ROOT, roots=roots) == 0
    assert not child.called and not roots.runs.exists()
    job = ROOT / "jobs/waymo/r4c-ddw-prepared/run.yaml"
    with pytest.raises(FileNotFoundError):
        doctor_job(job, ROOT, roots=roots)
    for segment in ("synthetic-segment-a", "synthetic-segment-b"):
        for component in _ALL_COMPONENTS:
            path = roots.data / "waymo/training" / component / f"{segment}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"parquet")
    depth = roots.models / "moge-vitl/model.pt"
    depth.parent.mkdir(parents=True)
    depth.write_bytes(b"weights")
    assert doctor_job(job, ROOT, roots=roots) == 47
    assert child.call_count == 1
    (argv,) = child.call_args.args
    expected = (
        "/opt/envs/gen3c/bin/python -m novel_view.diagnostics._job_worker "
        "--backend ddw_preparation --gpu-count 1"
    ).split()
    assert argv[:7] == expected
    assert argv[7:] == [
        "--t5-root",
        str(roots.models / "gen3c/official/checkpoints/google-t5/t5-11b"),
        "--vae-root",
        str(roots.models / "gen3c/official/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p"),
    ]
    assert not roots.runs.exists()


@pytest.mark.parametrize("kind", ["base", "gen3c/base", "full", "lora"])
def test_generation_recipe_preserves_model_request_and_doctor(tmp_path, monkeypatch, kind):
    roots = replace(CONTAINER_ROOTS, models=tmp_path / "models")
    shared = roots.models / "gen3c/official/checkpoints"
    base = shared / "Gen3C-Cosmos-7B/model.pt"
    network = base if kind in ("base", "gen3c/base") else roots.models / "tuned.pt"
    values = dict(backend="gen3c", checkpoint=kind, seed=7)
    if kind in ("full", "lora"):
        values.update(checkpoint=str(network) if kind == "full" else "tuned.pt",
                      model_id=kind, num_steps=11)
    if kind == "lora":
        values["lora"] = dict(working_manifest="run/working.json", evidence_path="run/evidence.json",
                              epochs=2, strength=0.0)
    specs = []
    for name, parse in [("euvs-pair-autoregressive-v1", parse_euvs_generation_job),
                        ("gaussian-dense-independent-v1", parse_gaussian_dense_generation_job)]:
        job = load_resolved_job(ROOT / "jobs/acceptance" / name / "run.yaml", ROOT)
        job = replace(job, parameters={**job.parameters, "generation": values})
        spec = parse(job)
        specs.append(spec if name.startswith("euvs") else spec.generation)
        with pytest.raises(ValueError):
            parse(replace(job, parameters={**job.parameters, "generation": {**values, "typo": 1}}))
        if kind == "lora" and name.startswith("gaussian"):
            with pytest.raises(ValueError):
                parse_gaussian_generation_job(replace(job, parameters=dict(frame_rate=24, generation=values)))
    assert specs[0].generation == specs[1].generation
    model, parameters = build_model_request(roots.models, specs[0].generation)
    assert (model.network_checkpoint, model.shared_checkpoint_root) == (network, shared)
    assert model.model_id == (kind if kind in ("full", "lora") else "gen3c-cosmos-7b-official")
    assert (parameters.seed, parameters.num_steps, parameters.window_seed_policy) == (
        7, 11 if kind in ("full", "lora") else 35, "autoregressive/v1")
    if kind == "lora":
        assert model.lora_weights.strength == 0.0
        assert (model.lora_provenance.working_manifest, model.lora_provenance.evidence_path,
                model.lora_provenance.epochs) == (Path("run/working.json"), Path("run/evidence.json"), 2)
    else:
        assert model.lora_weights is model.lora_provenance is None
    for path in {base, network}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
    child = Mock(return_value=SimpleNamespace(returncode=47))
    monkeypatch.setattr("novel_view.diagnostics.job.subprocess.run", child)
    for spec in specs:
        assert _generation_weights(spec, roots, 2) == 47
        assert child.call_args.args[0] == [
            "/opt/envs/gen3c/bin/python", "-m", "novel_view.diagnostics._job_worker",
            "--backend", "generation", "--gpu-count", "2",
            "--t5-root", str(shared / "google-t5/t5-11b"),
            "--vae-root", str(shared / "Cosmos-Tokenize1-CV8x8x8-720p")]
    if kind == "lora":
        base.unlink()
        with pytest.raises(FileNotFoundError):
            _generation_weights(specs[0], roots, 2)

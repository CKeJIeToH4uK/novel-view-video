"""Saved EUVS versions keep frame identity and lazy model/result locators."""

import json
from pathlib import Path

import pytest

from novel_view.generation.euvs.record import (
    Gen3cRunRecordError,
    build_euvs_gen3c_run_record,
    load_run_record,
    run_record_path,
    write_run_record,
)


FIXTURES = Path(__file__).with_name("fixtures")


@pytest.mark.parametrize("version", (1, 2, 3, 4, 5))
def test_saved_versions_preserve_identity_and_current_roundtrip(tmp_path, version):
    fixture = FIXTURES / f"euvs_generation_run_v{version}.json"
    record = load_run_record(fixture, allow_legacy=version == 1)
    assert record.schema == f"novel-view/euvs-gen3c-run/v{version}"
    assert record.pair.source.image_tokens == ("s0", "s1")
    assert record.pair.target.image_tokens == ("t0", "t1")
    assert record.target_output_index == (1, 6)
    assert record.target_source_sequence_index == (0, 1)
    assert record.output.shape == (121, 704, 1280, 3)
    assert record.model.lora_strength == (0.35 if version == 5 else None)
    if version == 1:
        with pytest.raises(Gen3cRunRecordError):
            load_run_record(fixture)
    else:
        path = tmp_path / "run.json"
        assert write_run_record(path, record) == path
        assert load_run_record(path) == record
        assert json.loads(path.read_text()) == json.loads(fixture.read_text())


def test_builder_preserves_resolved_values_without_opening_payloads(tmp_path):
    base = load_run_record(FIXTURES / "euvs_generation_run_v2.json")
    runs, models = tmp_path / "absent-runs", tmp_path / "absent-models"
    output = runs / "pairs/pair-a/generated_rgb.npy"
    record = build_euvs_gen3c_run_record(
        experiment_name="native-lora",
        pair=base.pair,
        target_output_index=base.target_output_index,
        target_source_sequence_index=base.target_source_sequence_index,
        geometry_backend="vggt-omega",
        geometry_directory=runs / "source-views/s0",
        geometry_provenance="ordered-source-tokens",
        model_id="ddw-step-1309",
        network_checkpoint=models / "gen3c/ddw-step-1309.pt",
        lora_working_manifest=Path("/runs/ddw-working.json"),
        lora_evidence_path=Path("/code/ddw-evidence.json"),
        lora_epochs=20,
        lora_strength=0.35,
        sampling=base.sampling,
        context_parallel_size=2,
        output_rgb_path=output,
        output_frame_count=121,
        runs_root=runs,
        models_root=models,
    )
    assert record.schema == "novel-view/euvs-gen3c-run/v5"
    assert record.geometry.runs_relative_directory == "source-views/s0"
    assert record.model.models_relative_checkpoint == "gen3c/ddw-step-1309.pt"
    assert record.model.lora_strength == 0.35
    assert run_record_path(output) == output.with_suffix(".run.json")
    assert not runs.exists() and not models.exists()


@pytest.mark.parametrize(
    "change", ("unknown", "identity-collision", "direction", "null-lora", "null-strength")
)
def test_record_rejects_external_typo_or_changed_scientific_identity(tmp_path, change):
    version = {"null-lora": 4, "null-strength": 5}.get(change, 2)
    document = json.loads((FIXTURES / f"euvs_generation_run_v{version}.json").read_text())
    if change == "unknown":
        document["typo"] = True
    elif change == "direction":
        document["pair"]["direction"] = "target_to_source"
    elif change == "null-lora":
        document["model"]["lora"] = dict.fromkeys(document["model"]["lora"])
    elif change == "null-strength":
        document["model"]["lora"]["strength"] = None
    else:
        document["pair"]["target"]["image_tokens"][1] = "s0"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document))
    with pytest.raises(Gen3cRunRecordError):
        load_run_record(path)

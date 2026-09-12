"""Versioned epoch/checkpoint loops without a full diffusion model."""

from pathlib import Path
from types import SimpleNamespace as NS

import torch

from novel_view.training.gen3c import data
from novel_view.training.gen3c.lora import method, depth_method
from novel_view.training.gen3c.lora.depth import R4cDepthItem
from novel_view.training.gen3c.lora.depth_spec import DepthObjectiveV2, R4cLoraDepthSpec
from novel_view.training.gen3c.lora.checkpoint import ResumeStateV1
from novel_view.training.gen3c.lora.spec import R4cLoraEdmSpec
from tests.support.gen3c_training import install_training_loop_fakes, local_topology, training_item
from tests.support.gen3c_training_records import depth_facts


def test_v1_epoch_checkpoints_and_resume_start_at_next_item(tmp_path, monkeypatch):
    training = (training_item("train-a", "segment-a"), training_item("train-b", "segment-b"))
    validation = (training_item("validation-a", "segment-v"),)
    fakes = install_training_loop_fakes(method, monkeypatch, version=1)
    arguments = dict(
        spec=R4cLoraEdmSpec(Path("gen3c/base.pt"), 2, 0, 1),
        base_checkpoint=tmp_path / "base.pt",
        prepared_record_reference="waymo/prepared.json",
        artifact_root=tmp_path,
        training_items=training,
        validation_items=validation,
        output=tmp_path,
        source_attempt="edm/run/fresh",
        topology=local_topology(),
        device="cpu",
    )
    before = torch.get_rng_state().clone()
    fresh = tuple(method.run_training_v1(**arguments, resume=None))
    assert [row.completed_step for row in fresh] == [1, 2, 3, 4]
    assert [row.completed_step for row in fresh if row.checkpoint] == [2, 4]
    assert [state.next_step for state in fakes.states] == [2, 4]
    assert all(record.validation_loss == -0.25 for record in fakes.records)
    assert torch.equal(torch.get_rng_state(), before)
    fakes.states.clear()
    fakes.records.clear()
    restored = ResumeStateV1(
        2, data.r4c_data_cursor(training, training_seed=0, completed_steps=2), ()
    )
    monkeypatch.setattr(method, "restore_lora_v1", lambda *_: None)
    monkeypatch.setattr(method, "restore_training_state_v1", lambda *_args, **_kwargs: restored)
    resumed = tuple(
        method.run_training_v1(
            **{**arguments, "source_attempt": "edm/run/resumed"}, resume=object()
        )
    )
    assert [row.completed_step for row in resumed] == [3, 4]
    assert [state.next_step for state in fakes.states] == [4]
    assert fakes.records[0].source_attempt == "edm/run/resumed"


def test_v2_fits_before_model_and_resume_restores_frozen_observer_without_refit(
    tmp_path, monkeypatch
):
    prepared = (
        training_item("sample-train-a", "segment-a"),
        training_item("sample-train-b", "segment-b"),
    )
    training = tuple(R4cDepthItem(item, item.base_latent) for item in prepared)
    validation = (R4cDepthItem(training_item("sample-validation", "segment-v"), Path("v")),)
    fakes = install_training_loop_fakes(depth_method, monkeypatch, version=2)
    observer = torch.nn.Linear(1, 1).requires_grad_(False)
    _, fit_identity, fit_result = depth_facts()
    monkeypatch.setattr(depth_method, "LatentDepthObserver", lambda: observer)
    monkeypatch.setattr(
        depth_method,
        "fit_depth_observer",
        lambda *_args, **_kw: fakes.events.append("fit") or fit_result,
    )
    arguments = dict(
        spec=R4cLoraDepthSpec(Path("gen3c/base.pt"), 2, 0, 1, 0.1, fit_identity.spec),
        base_checkpoint=tmp_path / "base.pt",
        prepared_record_reference="waymo/prepared.json",
        artifact_root=tmp_path,
        training_items=training,
        validation_items=validation,
        output=tmp_path,
        source_attempt="depth/run/fresh",
        topology=local_topology(),
        device="cpu",
    )
    fresh = tuple(depth_method.run_training_v2(**arguments, resume=None))
    assert fakes.events[:2] == ["fit", "model"]
    assert [row.completed_step for row in fresh if row.checkpoint] == [2, 4]
    assert [state.observer_lineage for state in fakes.states] == ["depth/run/fresh"] * 2
    resume = NS(
        objective=DepthObjectiveV2(0.2),
        observer_fit_identity=fakes.states[0].observer_fit_identity,
        observer_fit_result=fit_result,
        observer_lineage="depth/run/fresh",
    )
    fakes.events.clear()
    fakes.states.clear()
    fakes.records.clear()
    restored = ResumeStateV1(
        2, data.r4c_data_cursor(prepared, training_seed=0, completed_steps=2), ()
    )
    monkeypatch.setattr(
        depth_method,
        "restore_depth_observer_v2",
        lambda *_args, **_kw: fakes.events.append("observer") or observer,
    )
    monkeypatch.setattr(depth_method, "restore_lora_v2", lambda *_: fakes.events.append("lora"))
    monkeypatch.setattr(depth_method, "restore_training_state_v2", lambda *_args, **_kw: restored)
    resumed = tuple(
        depth_method.run_training_v2(
            **{**arguments, "source_attempt": "depth/run/resumed"}, resume=resume
        )
    )
    assert fakes.events == ["observer", "model", "lora"]
    assert [row.completed_step for row in resumed] == [3, 4]
    assert fakes.states[0].objective == DepthObjectiveV2(0.2)
    assert (fakes.records[0].observer_lineage, fakes.records[0].source_attempt) == (
        "depth/run/fresh",
        "depth/run/resumed",
    )

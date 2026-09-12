"""Actual checkpoint bytes restore a two-update CPU training state."""

import copy
import json
from unittest.mock import patch

import pytest
import torch

from novel_view.training.gen3c.lora import checkpoint, depth_checkpoint
from novel_view.training.gen3c.lora.depth import LatentDepthObserver
from tests.support.gen3c_training_records import checkpoint_record, depth_facts
from tests.support.gen3c_training import (
    CheckpointModel,
    assert_nested_equal,
    checkpoint_training_state,
    legacy_index_document,
    fake_cursor,
    fake_method,
    legacy_checkpoint_payload,
    optimization_step,
)


@pytest.mark.parametrize("version", [1, 2])
def test_checkpoint_restores_two_step_parameters_moments_master_and_rng(tmp_path, version):
    initial = CheckpointModel()
    continuous, interrupted = copy.deepcopy(initial), copy.deepcopy(initial)
    full, first = fake_method(continuous), fake_method(interrupted)
    torch.manual_seed(37)
    optimization_step(continuous, full.optimizer, full.scheduler, torch.randn(5, 2))
    expected_loss = optimization_step(
        continuous, full.optimizer, full.scheduler, 1 + torch.randn(5, 2)
    )
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(37)
    optimization_step(interrupted, first.optimizer, first.scheduler, torch.randn(5, 2))
    path = tmp_path / "checkpoint.pt"
    baseline = checkpoint_training_state(interrupted, first, torch.get_rng_state())
    fields = {"format", "adapter", "optimizer", "scheduler", "progress", "topology", "rng_states"}
    if version == 1:
        checkpoint.save_checkpoint_v1(path, baseline)
    else:
        observer = LatentDepthObserver()
        with torch.no_grad():
            for index, parameter in enumerate(observer.parameters(), 1):
                parameter.fill_(index)
        objective, fit_identity, fit_result = depth_facts()
        depth_checkpoint.save_checkpoint_v2(
            path,
            depth_checkpoint.TrainingStateV2(
                baseline, objective, observer, fit_identity, fit_result, "depth/run/fresh"
            ),
        )
        fields |= {
            "objective",
            "observer",
            "observer_fit_identity",
            "observer_fit_result",
            "observer_lineage",
        }
    raw = torch.load(path, weights_only=True, map_location="cpu")
    assert raw["format"] == (
        checkpoint.R4C_CHECKPOINT_FORMAT_V1
        if version == 1
        else depth_checkpoint.R4C_CHECKPOINT_FORMAT_V2
    )
    assert set(raw) == fields
    if version == 2:
        assert {name: tuple(value.shape) for name, value in raw["observer"].items()} == {
            "input.weight": (32, 16, 1, 1, 1),
            "input.bias": (32,),
            "output.weight": (1, 32, 1, 1, 1),
            "output.bias": (1,),
        }
        assert all(
            value.device.type == "cpu" and value.dtype == torch.float32 and value.is_contiguous()
            for value in raw["observer"].values()
        )
        torch.save({key: value for key, value in raw.items() if key != "format"}, path)
        with pytest.raises(ValueError):
            depth_checkpoint.load_checkpoint_v2(path)
    assert [row["global_rank"] for row in raw["rng_states"]] == [0, 1]
    raw["rng_states"].reverse()
    torch.save(raw, path)
    loaded = (
        checkpoint.load_checkpoint_v1(path, legacy_index_map=tmp_path / "unused.json")
        if version == 1
        else depth_checkpoint.load_checkpoint_v2(path)
    )
    resumed = copy.deepcopy(initial)
    (checkpoint.restore_lora_v1 if version == 1 else depth_checkpoint.restore_lora_v2)(
        loaded, resumed
    )
    next_method = fake_method(resumed)
    with patch("torch.cuda.current_device", return_value=0), patch(
        "torch.cuda.set_rng_state"
    ) as cuda:
        restore = (
            checkpoint.restore_training_state_v1
            if version == 1
            else depth_checkpoint.restore_training_state_v2
        )
        progress = restore(loaded, next_method, global_rank=0)
    assert cuda.call_args.args[0].tolist() == [7]
    assert progress.next_step == 1 and progress.data_cursor == fake_cursor()
    actual_loss = optimization_step(
        resumed, next_method.optimizer, next_method.scheduler, 1 + torch.randn(5, 2)
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert_nested_equal(continuous.state_dict(), resumed.state_dict())
    assert_nested_equal(
        checkpoint.pack_r4c_fused_adam(continuous, full.optimizer),
        checkpoint.pack_r4c_fused_adam(resumed, next_method.optimizer),
    )
    assert next_method.scheduler.steps == full.scheduler.steps == 2
    if version == 2:
        with patch("novel_view.training.gen3c.lora.depth.fit_depth_observer") as fit:
            restored = depth_checkpoint.restore_depth_observer_v2(loaded, device="cpu")
        fit.assert_not_called()
        assert_nested_equal(observer.state_dict(), restored.state_dict())
        assert not restored.training and all(
            not value.requires_grad for value in restored.parameters()
        )
        assert (loaded.objective, loaded.observer_fit_identity, loaded.observer_fit_result) == (
            objective,
            fit_identity,
            fit_result,
        )
        assert loaded.observer_lineage == "depth/run/fresh"


def test_old_six_field_checkpoint_translates_85_row_cursor(tmp_path):
    model = CheckpointModel()
    method = fake_method(model)
    optimization_step(model, method.optimizer, method.scheduler, torch.ones(5, 2))
    path, mapping = tmp_path / "checkpoint.pt", tmp_path / "map.json"
    checkpoint.save_checkpoint_v1(
        path, checkpoint_training_state(model, method, torch.get_rng_state())
    )
    tagged = torch.load(path, weights_only=True, map_location="cpu")
    mapping.write_text(json.dumps(legacy_index_document()))
    for completed, epoch in [(77, 1), (154, 2)]:
        payload = legacy_checkpoint_payload(copy.deepcopy(tagged), step=completed)
        torch.save(payload, path)
        loaded = checkpoint.load_checkpoint_v1(path, legacy_index_map=mapping)
        assert loaded.legacy and loaded.resume.next_step == completed
        assert loaded.resume.data_cursor.training_sample_ids == tuple(
            f"sample-{i:03d}" for i in range(77)
        )
        assert (loaded.resume.data_cursor.epoch, loaded.resume.data_cursor.offset) == (epoch, 0)
    payload["format"] = "future"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint_v1(path, legacy_index_map=mapping)

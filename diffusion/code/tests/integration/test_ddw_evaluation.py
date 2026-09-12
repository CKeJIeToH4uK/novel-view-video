"""Matched LoRA/RNG, heldout numerics and actual 121-frame JSON/MP4 output."""

import json
import random
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from novel_view.evaluation.ddw import execute, metrics, sampling, spec
from novel_view.evaluation.ddw.record import (
    DdwItemResult,
    EvaluationCheckpoint,
    read_evaluation_record,
)
from novel_view.models.gen3c.lora_weights import gen3c_lora_state_by_name
from novel_view.training.gen3c.data import R4cTrainingSplit
from tests.support.ddw_evaluation import evaluation_spec, matched_items, prepared_item
from tests.support.gen3c_training import prepared_record
from tests.support.gen3c_training_records import checkpoint_record
from tests.support.stage5 import make_runtime


def test_matched_step_identity_and_binary_progress_are_the_same_experiment():
    values = dict(
        prepared_record="prepared.json",
        split="split.yaml",
        v1_checkpoint_record="v1.json",
        v2_checkpoint_record="v2.json",
    )
    recipe = dict(
        model_contract=evaluation_spec().recipe.model_contract,
        comparison_step=77,
        sampling=dict(seed=0, steps=35, guidance=1.0, condition_augment_sigma=0.001),
        gen3c=dict(base_checkpoint="gen3c/official/model.pt", tokenizer="gen3c/tokenizer"),
        depth=dict(backend="moge", checkpoint="moge/model.pt"),
    )
    options = dict(
        workflow_name="ddw_evaluation", workflow_version=1, execution_preset="training_cp4"
    )
    assert spec.parse_ddw_evaluation_spec(values, recipe, **options).recipe.comparison_step == 77
    with pytest.raises(ValueError):
        spec.parse_ddw_evaluation_spec(values | {"typo": True}, recipe, **options)
    v1, v2 = checkpoint_record(77, -0.2, "train/run/v1", 1), checkpoint_record(
        77, -0.1, "train/run/v2", 2
    )
    training, validation = matched_items()
    options = dict(
        prepared_record_reference=v1.identity.prepared_record,
        base_checkpoint_reference=v1.identity.base_checkpoint,
        comparison_step=77,
    )
    spec.require_matched_checkpoint_records(training, validation, v1, v2, **options)
    for different in [
        replace(v2, identity=replace(v2.identity, training_seed=9)),
        replace(v2, completed_step=78),
    ]:
        with pytest.raises(ValueError):
            spec.require_matched_checkpoint_records(training, validation, v1, different, **options)
    binary1 = NS(resume=NS(next_step=77))
    binary2 = NS(
        baseline=NS(resume=NS(next_step=77)),
        objective=v2.objective,
        observer_fit_identity=v2.observer_fit_identity,
        observer_fit_result=v2.observer_fit_result,
        observer_lineage=v2.observer_lineage,
    )
    spec.require_binary_record_facts(binary1, binary2, v1, v2)
    with pytest.raises(ValueError):
        spec.require_binary_record_facts(NS(resume=NS(next_step=78)), binary2, v1, v2)


def test_actual_lora_replacement_and_rng_make_v2_independent_of_previous_variant(monkeypatch):
    net = torch.nn.Module()
    net.q_lora = torch.nn.Linear(2, 1, bias=False)
    net.v_lora = torch.nn.Linear(2, 1, bias=False)
    net.peft_lora_enabled = True
    initial = {
        name: torch.zeros_like(value) for name, value in gen3c_lora_state_by_name(net).items()
    }
    v1 = {name: torch.tensor([[i + 1.0, i + 2.0]]) for i, name in enumerate(initial)}
    v2 = {name: torch.tensor([[i + 3.0, i + 5.0]]) for i, name in enumerate(initial)}
    shape = (1, 1, 1, 1, 1)
    noise = torch.zeros(shape)
    state = sampling.R4cSamplingState(
        None, None, noise, noise.clone(), sampling._capture_rng(torch, noise.device), initial
    )
    draws, enabled, copied = [], [], []

    def sample(_torch, model, _condition, _uncondition, initial_noise, condition_noise, _spec):
        draws.append((random.random(), np.random.random(), torch.rand(1).item()))
        enabled.append(model.model.peft_lora_enabled)
        copied.append((initial_noise.clone(), condition_noise.clone()))
        initial_noise.add_(10)
        condition_noise.add_(20)
        value = (
            net.q_lora(torch.tensor([[1.0, 2.0]])).item()
            + net.v_lora(torch.tensor([[3.0, 4.0]])).item()
        )
        return torch.full(shape, value if net.peft_lora_enabled else 0.0)

    monkeypatch.setattr(sampling, "_sample_once", sample)
    monkeypatch.setattr(sampling, "R4C_GEN3C_MODEL_CONTRACT", NS(base_latent_shape=shape))
    model, schedule = NS(model=net), evaluation_spec().recipe.sampling
    alone = sampling.sample_r4c_variant(model, state, schedule, v2)
    variants = sampling.sample_r4c_variants(model, state, schedule, v1, v2)
    assert (variants.base.item(), variants.v1.item(), variants.v2.item()) == (0.0, 23.0, 49.0)
    torch.testing.assert_close(alone, variants.v2, rtol=0, atol=0)
    assert draws == [draws[0]] * 4 and enabled == [True, False, True, True]
    assert all(not left.any() and not right.any() for left, right in copied)
    assert (
        not state.initial_noise.any() and not state.condition_noise.any() and net.peft_lora_enabled
    )
    for name, value in gen3c_lora_state_by_name(net).items():
        torch.testing.assert_close(value, v2[name], rtol=0, atol=0)


def test_heldout_math_all_rgb_frames_and_real_workflow_json_mp4(tmp_path, monkeypatch):
    import cv2

    target = np.broadcast_to(
        np.arange(121, dtype=np.uint8)[:, None, None, None], (121, 64, 64, 3)
    ).copy()
    errors = np.full((121, 1, 1, 1), 10, dtype=np.uint8)
    errors[0], errors[-1] = 30, 60
    decoded = dict(base=target + errors, v1=target.copy(), v2=target + 2)
    depths = {name: np.full(target.shape[:3], 2.0, np.float32) for name in decoded}
    for value in depths.values():
        value[:, 1:3, 1:3] = 3.0
    valid = {name: np.ones(target.shape[:3], bool) for name in decoded}
    valid["v1"][1, 2, 2] = False
    lidar = metrics.SparseEvaluationLidar(
        np.arange(0, 243, 2, dtype=np.int64),
        np.tile(np.array([[0.0, 0.0], [1.5, 1.5]], np.float32), (121, 1)),
        np.tile(np.array([4.0, 9.0], np.float32), 121),
        np.tile([False, True], 121),
    )
    measured = metrics.evaluate_ddw_item("validation-a", target, decoded, depths, valid, lidar)
    base, v1, v2 = measured.variants
    assert [row.frame_index for row in base.depth.frames] == list(range(1, 121))
    assert {row.scale for row in base.depth.frames} == {2.0}
    assert (base.depth.median_abs_rel, base.depth.median_abs_z_m) == pytest.approx((1 / 3, 3.0))
    assert v1.depth.valid_fraction == pytest.approx(119 / 120)
    assert (base.rgb.mae, v1.rgb.mae, v2.rgb.mae) == pytest.approx((1280 / 121, 0.0, 2.0))
    assert v1.rgb.mae_improvement_vs_base == pytest.approx(1280 / 121)
    assert v2.rgb.mean_abs_change_vs_base == pytest.approx(1280 / 121 - 2)
    runtime = make_runtime(tmp_path, attempt="evaluation")
    prepared = prepared_record(
        tuple(
            prepared_item(name, partition, segment)
            for name, partition, segment in [
                ("train-a", "training", "segment-a"),
                ("train-b", "training", "segment-b"),
                ("validation-a", "validation", "segment-v"),
            ]
        )
    )
    run_spec = evaluation_spec()
    run_spec = replace(
        run_spec,
        input=replace(run_spec.input, prepared_record="waymo-ddw/r4c-ddw-prepared/prepared.json"),
    )
    monkeypatch.setattr(execute, "read_prepared_record", lambda _: prepared)
    monkeypatch.setattr(
        execute,
        "load_r4c_training_split",
        lambda _: R4cTrainingSplit(("train-a", "train-b"), ("validation-a",)),
    )
    monkeypatch.setattr(
        execute,
        "read_checkpoint_record_v1",
        lambda _: checkpoint_record(77, -0.2, "train/run/v1", 1),
    )
    monkeypatch.setattr(
        execute,
        "read_checkpoint_record_v2",
        lambda _: checkpoint_record(77, -0.1, "train/run/v2", 2),
    )
    generated = []
    monkeypatch.setattr(execute, "_run_generation", lambda *_: generated.append(True))

    def evaluate(
        _spec, _runtime, _prepared, sample, index, _latents, _scratch, video_root, _logs, _env
    ):
        assert generated == [True] and sample.prepared.sample_id == measured.sample_id
        name = f"item-{index:04d}.mp4"
        metrics.write_four_panel_video(video_root / name, target, decoded, lidar)
        return DdwItemResult(measured.sample_id, f"videos/{name}", measured)

    monkeypatch.setattr(execute, "_evaluate_sample", evaluate)
    path = execute.run_evaluation_v1(run_spec, runtime)
    record = read_evaluation_record(path)
    assert (record.prepared_record, record.split) == (
        run_spec.input.prepared_record,
        run_spec.input.split,
    )
    for field in ("model_contract", "base_checkpoint", "tokenizer", "depth_checkpoint", "sampling"):
        assert getattr(record, field) == getattr(run_spec.recipe, field)
    assert record.v1_checkpoint == EvaluationCheckpoint(
        "v1.json", "step-000000077.pt", "train/run/v1", 77, "gen3c-kendall-edm", 1, None, None
    )
    assert record.v2_checkpoint == EvaluationCheckpoint(
        "v2.json",
        "step-000000077.pt",
        "train/run/v2",
        77,
        "gen3c-kendall-edm-lidar-depth",
        1,
        0.1,
        "train/run/fresh",
    )
    assert (
        record.comparison_step
        == record.v1_checkpoint.completed_step
        == record.v2_checkpoint.completed_step
        == 77
    )
    assert record.items[0].metrics == measured
    assert record.aggregates == metrics.aggregate_ddw_items((measured,))
    document = json.loads(path.read_text())
    assert document["kind"] == "heldout-point-validation" and "request" not in document
    assert "temporary" not in path.read_text()
    capture = cv2.VideoCapture(str(runtime.attempt_root / record.items[0].video))
    frames = []
    while (result := capture.read())[0]:
        frames.append(result[1])
    capture.release()
    assert len(frames) == 121 and frames[0].shape == (64, 256, 3)
    for index in (0, 120):
        observed = [
            frames[index][50:60, panel * 64 + 40 : panel * 64 + 55].mean() for panel in range(4)
        ]
        expected = [index, index + (30 if index == 0 else 60), index, index + 2]
        np.testing.assert_allclose(observed, expected, atol=5, rtol=0)


def test_evaluation_parent_and_v1_import_no_heavy_or_v2_model():
    code = """
import sys
import novel_view.workflows.ddw_evaluation
import novel_view.evaluation.ddw.record
import novel_view.training.gen3c.data
import novel_view.training.gen3c.topology
import novel_view.training.gen3c.lora.method
import novel_view.training.gen3c.lora.objective
import novel_view.training.gen3c.lora.step
assert not {'torch', 'cv2', 'novel_view.training.gen3c.lora.depth'} & sys.modules.keys()
assert not any(name.startswith(('cosmos_predict1', 'moge')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)

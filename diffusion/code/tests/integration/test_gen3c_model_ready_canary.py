"""Компактные CPU-доказательства явной Gen3C model-ready диагностики."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from novel_view.diagnostics import _gen3c_model_ready_worker as worker
from novel_view.diagnostics import gen3c as controller


class _Encoder:
    def encode_prompts(self, prompts, *, max_length):
        assert (prompts, max_length) == ([""], 512)
        embedding = torch.zeros((1, 512, 1024))
        embedding[:, 0, 0] = 1.25
        mask = torch.zeros((1, 512), dtype=torch.int64)
        mask[:, 0] = 1
        return embedding, mask


def _patch_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda _device: torch.tensor([7]))
    for name, value in (
        ("memory_allocated", 11),
        ("memory_reserved", 12),
        ("max_memory_allocated", 21),
        ("max_memory_reserved", 22),
    ):
        monkeypatch.setattr(torch.cuda, name, lambda _device, value=value: value)


def test_controller_uses_new_sample_identity_and_installed_workers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        controller, "run_process", lambda *args: calls.append(args)
    )
    arguments = argparse.Namespace(
        sample_id="sample-a",
        magnitude_m=2.0,
        sign=-1,
        target_rgb=Path("/input/target.npy"),
        condition_rgb=Path("/input/condition.npy"),
        condition_known=Path("/input/known.npy"),
        text_encoder=Path("/models/t5"),
        tokenizer=Path("/models/tokenizer"),
        scratch_root=tmp_path,
        run_dir=tmp_path / "run",
        cuda_visible_device="0",
        vae_only=False,
    )

    controller._run(arguments)

    assert len(calls) == 2
    assert calls[0][0][2:] == [
        "prompt",
        "--text-encoder",
        "/models/t5",
        "--scratch-root",
        str(tmp_path),
    ]
    assert calls[1][0][2:9] == [
        "vae",
        "--sample-id",
        "sample-a",
        "--magnitude-m",
        "2.0",
        "--sign",
        "-1",
    ]
    assert "--working-manifest" not in calls[1][0]
    assert "--upstream-root" not in calls[1][0]
    assert calls[0][0][0] == calls[1][0][0] == str(controller.GEN3C_PYTHON)
    assert calls[0][1] == arguments.run_dir / "prompt.log"
    assert calls[1][1] == arguments.run_dir / "vae.log"
    assert calls[0][3] is calls[1][3]


def test_prompt_is_repeatable_and_round_trips_new_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_cuda(monkeypatch)
    monkeypatch.setattr(worker, "_load_encoder", lambda _path: (torch, _Encoder()))
    monkeypatch.setattr(worker.time, "monotonic", iter((1.0, 2.0, 4.5)).__next__)
    output = io.StringIO()

    with redirect_stdout(output):
        worker._run_prompt(
            argparse.Namespace(text_encoder=Path("/models/t5"), scratch_root=tmp_path)
        )

    result = json.loads(output.getvalue())
    assert result["status"] == "pass"
    assert result["phase"] == "prompt"
    assert result["load_seconds"] == 1.0
    assert result["check_seconds"] == 2.5
    assert tuple(tmp_path.iterdir()) == ()


def test_vae_checks_repeat_target_barrier_pose_and_new_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_cuda(monkeypatch)
    target = np.arange(18, dtype=np.uint8).reshape(3, 2, 3)
    stored: dict[str, tuple[torch.Tensor, ...]] = {}

    class _Model:
        def encode(self, value):
            return value.sum().reshape(1, 1)

        def encode_warped_frames(self, _condition, _known, _dtype):
            return torch.cat((torch.ones((1, 32)), torch.zeros((1, 32))), dim=1)

    def target_tensor(_torch, rgb, _device):
        per_frame = rgb.reshape(rgb.shape[0], -1).sum(axis=1, dtype=np.float32)
        return torch.from_numpy(per_frame).reshape(1, 1, -1)

    def source(_model, value, *, num_frames_condition):
        assert num_frames_condition == 1
        return value.sum().reshape(1, 1), None

    monkeypatch.setattr(worker, "_load_model", lambda _path: (torch, _Model(), source))
    monkeypatch.setattr(worker, "_read_target", lambda _path: target.copy())
    monkeypatch.setattr(worker, "_target_tensor", target_tensor)
    monkeypatch.setattr(worker, "_cpu_latent", lambda _torch, value, *_args: value)
    monkeypatch.setattr(
        worker,
        "_condition_tensors",
        lambda *_args: (torch.ones(1), torch.ones(1)),
    )
    monkeypatch.setattr(
        worker,
        "write_base_latents",
        lambda _path, _sample, clean, source: stored.update(
            base=(clean.clone(), source.clone())
        ),
    )
    monkeypatch.setattr(
        worker,
        "read_base_latents",
        lambda _path, _sample: SimpleNamespace(
            clean_latent=stored["base"][0], source_latent=stored["base"][1]
        ),
    )
    monkeypatch.setattr(
        worker,
        "write_pose_latent",
        lambda _path, _sample, _magnitude, _sign, value: stored.update(
            pose=(value.clone(),)
        ),
    )
    monkeypatch.setattr(
        worker,
        "read_pose_latent",
        lambda *_args: SimpleNamespace(pose_latent=stored["pose"][0]),
    )
    monkeypatch.setattr(worker.time, "monotonic", iter((1.0, 2.0, 4.0)).__next__)
    output = io.StringIO()

    with redirect_stdout(output):
        worker._run_vae(
            argparse.Namespace(
                sample_id="sample-a",
                magnitude_m=2.0,
                sign=-1,
                target_rgb=Path("/input/target.npy"),
                condition_rgb=Path("/input/condition.npy"),
                condition_known=Path("/input/known.npy"),
                tokenizer=Path("/models/tokenizer"),
                scratch_root=tmp_path,
            )
        )

    result = json.loads(output.getvalue())
    assert (result["status"], result["sample_id"], result["sign"]) == (
        "pass",
        "sample-a",
        -1,
    )
    assert result["metrics"]["clean_target_effect"]["mismatch_count"] == 1
    assert result["metrics"]["source_target_effect"]["mismatch_count"] == 0
    assert tuple(tmp_path.iterdir()) == ()


def test_diagnostic_parents_remain_cold() -> None:
    code = """
import sys
import novel_view.diagnostics.gen3c
import novel_view.diagnostics._gen3c_model_ready_worker
assert 'torch' not in sys.modules
assert not any(name.startswith('cosmos_predict1') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)

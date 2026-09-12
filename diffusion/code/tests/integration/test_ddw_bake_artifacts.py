"""Scratch pixels → prompt/VAE worker → читаемые BF16/sparse артефакты."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from novel_view.preparation.waymo_ddw import (
    _prompt_worker as prompt_worker,
    _vae_worker as vae,
    artifacts,
    bake,
)
from novel_view.preparation.waymo_ddw.condition import DdwCondition
from novel_view.preparation.waymo_ddw.raster import FrontLidarDepth
from novel_view.preparation.waymo_ddw.selection import SelectedDdwSample


def test_target_and_condition_keep_distinct_all_code_rounding():
    codes = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 1, 1, 256)
    known = torch.tensor([[[[[False, True]]]]])
    target = vae._normalize_target(torch, codes, "cpu")
    condition, mask = vae._normalize_condition(torch, codes, known, "cpu")
    expected_target = codes.to(torch.bfloat16).div(127.5).sub(1.0)
    expected_condition = codes.to(torch.float32).div(127.5).sub(1.0).to(torch.bfloat16)
    assert torch.equal(target, expected_target)
    assert torch.equal(condition[:, :, 0].permute(0, 2, 1, 3, 4), expected_condition)
    assert torch.count_nonzero(expected_target != expected_condition).item() == 128
    assert mask.flatten().tolist() == [0.0, 1.0]


def test_real_staging_worker_and_all_four_artifacts(tmp_path, monkeypatch):
    target = (np.arange(121 * 2 * 7 * 3) % 256).astype(np.uint8).reshape(121, 2, 7, 3)
    boundary = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], np.float32)
    rgb = np.broadcast_to(boundary, (121, 3, 2, 7)).copy()
    rgb[:, 1] = rgb[:, 1, :, ::-1]
    rgb[:, 2] = 0
    known = np.zeros((121, 2, 7), bool)
    known[:, 0, (0, 2, 6)] = True
    known[:, 1, (1, 4)] = True
    K = np.broadcast_to(np.array([[3.0, 0, 1.0], [0, 4.0, 0.5], [0, 0, 1.0]]), (121, 3, 3)).copy()
    K[:, 0, 0] += np.arange(121) / 100
    offsets = np.array([0, 2, *([4] * 120)], np.int64)
    lidar = FrontLidarDepth(
        offsets,
        np.array([[1, 0], [2, 1], [3, 0], [4, 1]], np.float32),
        np.array([2.0, 3.0, 5.0, 7.0], np.float32),
        np.array([True, False, False, True]),
    )
    front = SimpleNamespace(
        selected=SelectedDdwSample("a", "training", "s", 0, 1.5, -1),
        rgb_thwc=target,
        K_canvas=K,
        lidar=lidar,
    )
    task = bake.stage_bake_input(front, DdwCondition(rgb, known, 0, 0, 0.0), tmp_path, tmp_path)
    actual_rgb, actual_known = bake.read_condition_pixels(
        task.condition_rgb_path, task.condition_known_path, (2, 7)
    )
    np.testing.assert_array_equal(actual_rgb[0, 0, :, 0], [0, 0, 64, 128, 191, 255, 255])
    np.testing.assert_array_equal(actual_rgb[..., 1], actual_rgb[..., 0][..., ::-1])
    np.testing.assert_array_equal(actual_rgb[..., 2], 128)
    np.testing.assert_array_equal(actual_known, known)
    np.testing.assert_array_equal(np.load(task.condition_known_path)[0], [0b01000101, 0b00001001])
    np.testing.assert_array_equal(np.load(task.target_rgb_path), target)
    target_tensor = torch.from_numpy(target).permute(3, 0, 1, 2)[None]
    target_expected = target_tensor.to(torch.bfloat16).div(127.5).sub(1.0)
    condition_expected = (
        torch.from_numpy(actual_rgb)
        .permute(3, 0, 1, 2)[None]
        .float()
        .div(127.5)
        .sub(1.0)
        .bfloat16()
    )

    class Model:
        def encode(self, pixels):
            assert torch.equal(pixels, target_expected)
            return torch.zeros(artifacts.BASE_LATENT_SHAPE)

        def encode_warped_frames(self, pixels, mask, dtype):
            assert dtype == torch.bfloat16
            assert torch.equal(pixels[:, :, 0].permute(0, 2, 1, 3, 4), condition_expected)
            assert torch.equal(mask[0, :, 0, 0], torch.from_numpy(known).bfloat16())
            return torch.full(artifacts.POSE_LATENT_SHAPE, 2.0)

    def source(model, pixels, *, num_frames_condition):
        assert num_frames_condition == 1
        assert torch.equal(pixels, target_expected[:, :, :1])
        return torch.ones(artifacts.BASE_LATENT_SHAPE), None

    class Encoder:
        failure = None

        def encode_prompts(self, prompts, *, max_length):
            assert prompts == [""] and max_length == 512
            embedding = torch.zeros((1, 512, 1024))
            embedding[:, 0] = torch.arange(1024)
            mask = torch.zeros((1, 512), dtype=torch.int64)
            mask[:, 0] = 1
            if self.failure == "mask":
                mask[:, 1] = 1
            if self.failure == "tail":
                embedding[:, 1, 0] = 1
            return embedding, mask

    encoder = Encoder()
    monkeypatch.setattr(prompt_worker, "_load_encoder", lambda _: (torch, encoder))
    # Keep 121 frames and the actual readers; only the CPU spatial canvas is small.
    monkeypatch.setattr(vae, "_SIZE_HW", (2, 7))
    monkeypatch.setattr(vae, "_RGB_SHAPE", (121, 2, 7, 3))

    def process(command, log, description, environment):
        if "--text-encoder" in command:
            prompt_worker._run(
                SimpleNamespace(
                    text_encoder=Path(command[command.index("--text-encoder") + 1]),
                    output=Path(command[command.index("--output") + 1]),
                )
            )
        else:
            for item in vae._load_tasks(Path(command[command.index("--tasks") + 1])):
                vae._write_lidar_for_task(item)
                vae._bake_task(item, torch, Model(), source, "cpu")

    monkeypatch.setattr(bake, "run_process", process)
    prompt_path = bake.bake_prompt(tmp_path / "t5", tmp_path / "prompt.pt", tmp_path / "prompt.log")
    bake.bake_items((task,), tmp_path / "tokenizer", tmp_path / "tasks.json", tmp_path / "vae.log")
    base = artifacts.read_base_latents(task.base_latent_path, "a")
    pose = artifacts.read_pose_latent(task.pose_latent_path, "a", 1.5, -1)
    prompt = artifacts.read_empty_prompt(prompt_path)
    sparse = artifacts.read_lidar_depth(task.lidar_depth_path, "a")
    assert base.clean_latent.shape == (1, 16, 16, 88, 160)
    assert pose.pose_latent.shape == (1, 64, 16, 88, 160)
    assert (
        base.clean_latent.dtype
        == base.source_latent.dtype
        == pose.pose_latent.dtype
        == torch.bfloat16
    )
    assert not base.clean_latent.any()
    assert torch.all(base.source_latent == 1) and torch.all(pose.pose_latent == 2)
    assert prompt.t5_text_embeddings.dtype == torch.bfloat16
    assert torch.equal(prompt.t5_text_embeddings[0, 0], torch.arange(1024).bfloat16())
    assert not prompt.t5_text_embeddings[:, 1:].any()
    assert (sparse.camera_name, sparse.raster_size_hw) == ("FRONT", (704, 1280))
    for name, expected in (
        ("K_canvas", K),
        ("frame_offsets", offsets),
        ("canvas_xy", lidar.canvas_xy),
        ("depth_z_m", lidar.depth_z_m),
        ("evaluation_holdout", lidar.evaluation_holdout),
    ):
        np.testing.assert_array_equal(getattr(sparse, name), expected)
    raw = torch.load(task.lidar_depth_path, weights_only=True, map_location="cpu")
    assert set(raw) == {
        "format",
        "sample_id",
        "camera_name",
        "raster_size_hw",
        "K_canvas",
        "frame_offsets",
        "canvas_xy",
        "depth_z_m",
        "evaluation_holdout",
    }
    for failure in ("mask", "tail"):
        encoder.failure = failure
        with pytest.raises(ValueError):
            prompt_worker._encode_empty_prompt(torch, encoder)


def test_eight_plus_two_batch_order_and_first_failure(tmp_path, monkeypatch):
    task = bake.VaeBakeTask(
        "a",
        1.0,
        -1,
        *(
            tmp_path / name
            for name in ("target", "rgb", "known", "lidar", "base", "pose", "sparse")
        ),
    )
    tasks = tuple(replace(task, sample_id=f"sample-{i}") for i in range(10))
    batches = tuple(bake.iter_vae_batches(tasks))
    rows = []

    def process(command, *_):
        rows.append(vae._load_tasks(Path(command[command.index("--tasks") + 1])))

    monkeypatch.setattr(bake, "run_process", process)
    for index, batch in enumerate(batches):
        bake.bake_items(
            batch, tmp_path / "tokenizer", tmp_path / f"tasks-{index}.json", tmp_path / "log"
        )
    assert [len(batch) for batch in rows] == [8, 2]
    assert [item.sample_id for batch in rows for item in batch] == [
        f"sample-{i}" for i in range(10)
    ]
    loads, completed = [], []

    def load(tokenizer):
        loads.append(tokenizer)
        return SimpleNamespace(device=lambda *_: "cpu"), None, None

    def fail_second(item, *_):
        if item.sample_id == "sample-1":
            raise RuntimeError("second item failed")
        completed.append(item.sample_id)

    monkeypatch.setattr(vae, "_load_model", load)
    monkeypatch.setattr(vae, "_write_lidar_for_task", lambda _: None)
    monkeypatch.setattr(vae, "_bake_task", fail_second)
    with pytest.raises(RuntimeError):
        vae._run(SimpleNamespace(tasks=tmp_path / "tasks-0.json", tokenizer=tmp_path / "tokenizer"))
    assert loads == [tmp_path / "tokenizer"]
    assert completed == ["sample-0"]

"""Value seams for Gen3C windows and the resident model session."""

from __future__ import annotations

import contextlib
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from novel_view.generation.gen3c._worker import _generate_windows
from novel_view.generation.gen3c import windows as window_policy
from novel_view.models.gen3c.lora_weights import Gen3cLoraWeights
from novel_view.models.gen3c.request import Gen3cModelSpec, Gen3cSampling
from novel_view.models.gen3c.session import (
    _build_pipeline,
    _capture_rng_state,
    _pipeline_class,
    _restore_rng_state,
)


class _Cache:
    def __init__(self) -> None:
        self.input_image = torch.zeros((1, 1, 1, 3, 2, 3))
        self.starts: list[int] = []
        self.w2c: list[np.ndarray] = []
    def render_cache(self, w2c, _intrinsics, *, start_frame_idx):
        self.starts.append(start_frame_idx)
        self.w2c.append(w2c.numpy().copy())
        return (
            torch.zeros((1, 121, 1, 3, 2, 3)),
            torch.ones((1, 121, 1, 1, 2, 3)),
        )


class _Model:
    torch = torch
    device = torch.device("cpu")
    def __init__(self) -> None:
        self.seeds: list[np.ndarray] = []

    def encode_seed(self, rgb):
        return torch.from_numpy(np.asarray(rgb)).float() * (2 / 255) - 1

    def generate_window(self, _rgb, _mask, seed):
        self.seeds.append(seed.numpy().copy())
        call = len(self.seeds) - 1
        return np.stack(
            [np.full((2, 3, 3), call * 100 + frame, np.uint8) for frame in range(121)]
        )


def _inputs() -> dict[str, np.ndarray]:
    eye4 = np.repeat(np.eye(4)[None], 241, axis=0)
    eye3 = np.repeat(np.eye(3)[None], 241, axis=0)
    return {
        "source_index": np.r_[np.zeros(120, np.int64), np.ones(121, np.int64)],
        "source_rgb": np.stack(
            [np.full((2, 3, 3), value, np.uint8) for value in (73, 200)]
        ),
        "source_w2c": eye4[:2].copy(),
        "source_intrinsics": eye3[:2].copy(),
        "query_w2c": eye4,
        "query_intrinsics": eye3,
    }


@pytest.mark.parametrize(
    ("policy", "second_seed", "reseed_w2c"),
    (
        (window_policy.WINDOW_SEED_AUTOREGRESSIVE, 120, 0.0),
        (window_policy.WINDOW_SEED_SOURCE_RESEED, 200, 7.0),
    ),
)
def test_two_windows_keep_seam_and_selected_seed(policy, second_seed, reseed_w2c):
    values = _inputs()
    values["source_w2c"][1, 0, 3] = 7
    cache, model = _Cache(), _Model()
    output = np.empty((241, 2, 3, 3), np.uint8)

    _generate_windows(cache, model, values, output, policy)

    assert cache.starts == [0, 120]
    assert output[[0, 120, 121, 240], 0, 0, 0].tolist() == [0, 120, 101, 220]
    np.testing.assert_allclose(model.seeds[0], 73 * (2 / 255) - 1, atol=1e-7)
    np.testing.assert_allclose(model.seeds[1], second_seed * (2 / 255) - 1, atol=2e-7)
    assert float(cache.w2c[1][0, 0, 0, 3]) == reseed_w2c


class _Pipeline:
    def __init__(self, **arguments):
        self.arguments = arguments
        self.model = SimpleNamespace(
            set_up_model=Mock(),
            model=SimpleNamespace(load_state_dict=Mock()),
            net=SimpleNamespace(enable_context_parallel=Mock()),
            chunk_size=121,
            cuda=Mock(),
        )
        self._load_network()


def test_model_construction_keeps_fixed_facts_and_full_checkpoint():
    spec = Gen3cModelSpec(Path("shared"), Path("network.pt"))
    sampling = Gen3cSampling("road", "bad", 1.5, 35, 7)
    modules = {
        "cosmos_predict1.diffusion.inference.gen3c_pipeline": SimpleNamespace(
            Gen3cPipeline=_Pipeline
        ),
        "cosmos_predict1.diffusion.inference.inference_utils": SimpleNamespace(
            skip_init_linear=lambda: contextlib.nullcontext()
        ),
    }
    fake_torch = SimpleNamespace(load=Mock(return_value={"weight": object()}))
    with patch(
        "novel_view.models.gen3c.session.importlib.import_module",
        side_effect=modules.__getitem__,
    ):
        pipeline = _build_pipeline(spec, sampling, fake_torch, None)

    assert (pipeline.arguments["height"], pipeline.arguments["width"]) == (704, 1280)
    assert (
        pipeline.arguments["fps"],
        pipeline.arguments["num_video_frames"],
    ) == (24, 121)
    pipeline.model.model.load_state_dict.assert_called_once_with(
        fake_torch.load.return_value, strict=False
    )


def test_lora_construction_receives_only_model_ready_weights():
    weights = Gen3cLoraWeights(Path("adapter.pt"), 0.25)
    spec = Gen3cModelSpec(Path("shared"), Path("unused.pt"), weights)
    assembled = SimpleNamespace(cuda=Mock())
    pipeline_type = _pipeline_class(
        SimpleNamespace(Gen3cPipeline=_Pipeline),
        SimpleNamespace(),
        SimpleNamespace(),
        spec,
        "cp",
    )
    with patch(
        "novel_view.models.gen3c.session.load_gen3c_lora_model",
        return_value=assembled,
    ) as load:
        pipeline_type()

    assert load.call_args.kwargs == {
        "upstream_root": None,
        "base_checkpoint": Path("shared/Gen3C-Cosmos-7B/model.pt"),
        "weights": weights,
        "context_parallel_group": "cp",
    }


def test_request_rng_boundary_replays_python_numpy_torch_and_cuda():
    class Cuda:
        state = torch.tensor([4, 2], dtype=torch.uint8)

        def get_rng_state(self, _device):
            return self.state.clone()

        def set_rng_state(self, state, _device):
            self.state = state.clone()

    fake = SimpleNamespace(
        cuda=Cuda(),
        get_rng_state=torch.get_rng_state,
        set_rng_state=torch.set_rng_state,
    )
    saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        boundary = _capture_rng_state(fake, "cuda:0")
        expected = random.random(), float(np.random.random()), float(torch.rand(()))
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        fake.cuda.state[:] = 0
        _restore_rng_state(boundary, fake, "cuda:0")
        actual = random.random(), float(np.random.random()), float(torch.rand(()))
        assert actual == expected
        torch.testing.assert_close(
            fake.cuda.state,
            torch.tensor([4, 2], dtype=torch.uint8),
        )
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])

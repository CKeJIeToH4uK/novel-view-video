"""Lifecycle seams for the direct-output Gen3C generation session."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from novel_view.generation.gen3c import session as module
from novel_view.generation.gen3c.session import (
    Gen3cGenerationError,
    Gen3cGenerationModel,
    Gen3cGenerationParameters,
    Gen3cGenerationResources,
    Gen3cGenerationSession,
)
from novel_view.models.gen3c.request import Gen3cModelSpec
from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import ProcessError


class _Input:
    image_size_hw = (704, 1280)
    model_frame_count = unpadded_frame_count = 121
    context_depth_output_slots = None

    def build_conditioning(self):
        return object()


class _Process:
    def __init__(self, root: Path, fail: bool) -> None:
        self.root, self.fail, self.requests = root, fail, {}
        self.terminated = False

    def poll(self):
        if (self.root / "stop").is_file():
            return 0
        for request, value in tuple(self.requests.items()):
            if not (request / "request.ready").is_file():
                continue
            if self.fail:
                return 17
            np.save(value.output_rgb_path, np.zeros(1, np.uint8))
            np.save(request / "cuda_bytes_0.npy", np.arange(6, dtype=np.int64))
            np.save(request / "elapsed_seconds_0.npy", np.asarray((0.5,)))
            (request / "request.done").touch()
            del self.requests[request]
        return None

    def wait(self):
        if self.fail:
            raise ProcessError("worker exited with code 17")
        return None

    def terminate(self, grace_seconds=10.0):
        self.terminated = True


def _open(tmp_path: Path, monkeypatch, fail=False):
    process = None

    def start(root, *_args, **_kwargs):
        nonlocal process
        process = _Process(root, fail)
        np.save(root / "startup_peak_cuda_bytes_0.npy", np.asarray((10, 20)))
        (root / "worker.ready").touch()
        return process

    def write(root, request):
        process.requests[root] = request

    monkeypatch.setattr(module, "_start_worker", start)
    monkeypatch.setattr(module, "write_generation_request", write)
    model = Gen3cGenerationModel("base", Gen3cModelSpec(Path("shared"), Path("net")))
    resources = Gen3cGenerationResources(
        tmp_path, {"CUDA_VISIBLE_DEVICES": "0"}, Path("upstream"), 1
    )
    return Gen3cGenerationSession(
        model, Gen3cGenerationParameters(seed=7), resources
    ), process


def test_one_worker_serves_two_direct_outputs(tmp_path, monkeypatch):
    session, process = _open(tmp_path, monkeypatch)
    first = session.generate(_Input(), tmp_path / "first.npy")
    second = session.generate(_Input(), tmp_path / "second.npy")
    session.close()
    assert (first.generated_rgb_path, second.generated_rgb_path) == (
        tmp_path / "first.npy", tmp_path / "second.npy"
    )
    assert first.per_rank_peak_cuda_allocated_bytes == (10,)
    assert not process.terminated


def test_worker_exit_breaks_and_stops_session(tmp_path, monkeypatch):
    session, process = _open(tmp_path, monkeypatch, fail=True)
    with pytest.raises(Gen3cGenerationError, match="worker exited with code 17"):
        session.generate(_Input(), tmp_path / "x.npy")
    assert process.terminated
    with pytest.raises(Gen3cGenerationError, match="broken"):
        session.generate(_Input(), tmp_path / "y.npy")


def test_worker_uses_fixed_prefix_and_composition_entrypoint():
    resources = Gen3cGenerationResources(Path("scratch"), {}, Path("upstream"), 1)
    model = Gen3cGenerationModel("base", Gen3cModelSpec(Path("shared"), Path("net")))
    with patch.object(module, "start_torchrun", return_value=Mock()) as start:
        module._start_worker(
            Path("session"), model, Gen3cGenerationParameters(seed=0), resources
        )
        legacy_arguments = start.call_args.args[2]
        module._start_worker(
            Path("session"),
            model,
            Gen3cGenerationParameters(seed=0),
            Gen3cGenerationResources(Path("scratch"), {}, None, 1),
        )
    assert start.call_args.args[0] == GEN3C_PYTHON
    assert start.call_args.args[1].name == "_worker.py"
    assert "--timeout" not in start.call_args.args[2]
    assert legacy_arguments[-2:] == ["--upstream-root", "upstream"]
    assert "--upstream-root" not in start.call_args.args[2]

"""Real CPU tensors through MoGe's raw NPY worker and Cosmos context slot."""

import sys
from types import SimpleNamespace as NS

import numpy as np

from novel_view.models.moge import _worker as worker
from novel_view.models.moge.gen3c import predict_gen3c_context_depth
from novel_view.models.moge.protocol import write_moge_request
from novel_view.models.moge.request import MogeRequest


def test_worker_loads_once_and_preserves_fov_rgb_and_raw_arrays(tmp_path, monkeypatch):
    import torch

    rgb = np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)
    fovs, calls, loads = [60.0, 70.0], [], []

    class Model:
        @classmethod
        def from_pretrained(cls, checkpoint):
            loads.append(checkpoint)
            return cls()

        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

        def infer(self, image, **kwargs):
            calls.append((image.numpy().copy(), kwargs))
            return {
                "depth": torch.full((2, 3), kwargs["fov_x"]),
                "mask": torch.tensor([[True, False, True], [False, True, True]]),
                "intrinsics": torch.diag(torch.tensor([kwargs["fov_x"], 2.0, 1.0])),
            }

    for name in ["moge", "moge.model"]:
        monkeypatch.setitem(sys.modules, name, NS())
    monkeypatch.setitem(sys.modules, "moge.model.v1", NS(MoGeModel=Model))
    real_to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor, "to", lambda value, *, device, dtype: real_to(value, dtype=dtype)
    )
    for name in ["reset_peak_memory_stats", "synchronize"]:
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 128 * 1024**2)
    write_moge_request(tmp_path, MogeRequest(iter(rgb), np.array(fovs), 2, (2, 3)))
    worker._run(NS(exchange=tmp_path, checkpoint=tmp_path / "model.pt"))
    assert loads == [tmp_path / "model.pt"] and len(calls) == 2
    for index, (image, kwargs) in enumerate(calls):
        np.testing.assert_array_equal(image, rgb[index].transpose(2, 0, 1).astype(np.float32) / 255)
        assert image.dtype == np.float32
        assert kwargs == dict(
            fov_x=fovs[index],
            num_tokens=2500,
            apply_mask=False,
            force_projection=True,
            use_fp16=True,
        )
    np.testing.assert_array_equal(
        np.load(tmp_path / "depth_model_units.npy"),
        np.broadcast_to(np.array(fovs)[:, None, None], (2, 2, 3)),
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "model_valid.npy"), [[[True, False, True], [False, True, True]]] * 2
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "model_intrinsics.npy"), [np.diag([fov, 2, 1]) for fov in fovs]
    )
    assert np.load(tmp_path / "peak_vram_mib.npy") == 128
    assert np.load(tmp_path / "peak_ram_mib.npy") > 0


def test_cosmos_context_extracts_first_slot_and_float_depth():
    import torch

    rgb, model = np.zeros((2, 3, 3), np.uint8), object()

    def predictor(image, height, width, device, loaded):
        assert image is rgb and loaded is model
        assert (height, width, device) == (2, 3, "cuda:0")
        depth = torch.arange(12, dtype=torch.float64).reshape(2, 1, 1, 2, 3)
        return None, depth, depth % 2 == 0, None, None

    result = predict_gen3c_context_depth(predictor, rgb, 2, 3, "cuda:0", model)
    np.testing.assert_array_equal(result.depth_model_units, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(result.valid, [[True, False, True], [False, True, False]])
    assert result.depth_model_units.dtype == np.float32

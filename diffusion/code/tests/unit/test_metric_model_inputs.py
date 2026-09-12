"""LPIPS range and DINO normalization use real numeric inputs, no weights."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np

from novel_view.models.dinov2._inference import Dinov2Vitb14
from novel_view.models.lpips._inference import LpipsAlexNet


def test_lpips_constructor_and_ordered_pair_range(monkeypatch):
    calls = {}

    class Model:
        def eval(self):
            return self

        def cuda(self):
            return self

        def __call__(self, first, second, *, normalize):
            assert not normalize
            np.testing.assert_array_equal(first, -np.ones((1, 3, 2, 4)))
            np.testing.assert_array_equal(second, np.ones((1, 3, 2, 4)))
            return first - second

    def factory(**kwargs):
        calls.update(kwargs)
        assert os.environ["TORCH_HOME"] == "/cache/torch"
        return Model()

    monkeypatch.setenv("TORCH_HOME", "previous")
    monkeypatch.setitem(sys.modules, "lpips", NS(LPIPS=factory))
    model = LpipsAlexNet(Path("/cache/torch"))
    pair = np.stack([np.zeros((3, 2, 4), np.float32), np.ones((3, 2, 4), np.float32)])
    np.testing.assert_array_equal(model.spatial_map(pair), np.full((1, 3, 2, 4), -2.0))
    assert calls == dict(net="alex", version="0.1", spatial=True, eval_mode=True)


def test_dino_checkpoint_and_numeric_bicubic_preprocessing(tmp_path, monkeypatch):
    import torch
    from PIL import Image

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(0.0))

        def cuda(self):
            return self

        def get_intermediate_layers(self, image, *, n, norm):
            assert n == 1 and norm
            return [image * self.scale, torch.zeros_like(image)]

    def factory(*, pretrained):
        assert not pretrained
        return Model()

    for name in ["dinov2", "dinov2.hub"]:
        monkeypatch.setitem(sys.modules, name, NS())
    monkeypatch.setitem(sys.modules, "dinov2.hub.backbones", NS(dinov2_vitb14=factory))
    checkpoint = tmp_path / "dino.pt"
    torch.save({"scale": torch.tensor(2.0)}, checkpoint)
    before = sys.path.copy()
    model = Dinov2Vitb14(tmp_path / "upstream", checkpoint)
    assert sys.path == before
    # A 2x3 nonconstant grid exposes resize mode, half-pixel mapping and channel normalization.
    pair = torch.arange(36, dtype=torch.float32).reshape(2, 3, 2, 3) / 35
    actual = model.patch_features(pair)
    # Independent Pillow float-channel bicubic oracle, not the same torch resize call.
    resized = np.stack(
        [
            [
                np.asarray(
                    Image.fromarray(channel.numpy()).resize((840, 462), Image.Resampling.BICUBIC)
                )
                for channel in frame
            ]
            for frame in pair
        ]
    )
    mean = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
    np.testing.assert_allclose(actual.numpy(), (resized - mean) / std * 2, rtol=0, atol=4e-6)

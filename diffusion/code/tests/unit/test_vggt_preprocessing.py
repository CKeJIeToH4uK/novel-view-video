"""CPU pixel conversion and conservative support, without a VGGT model."""

import numpy as np
import pytest

from novel_view.models.vggt._worker import _preprocess_images, _target_size
from novel_view.preparation.waymo_depth.vggt import build_conservative_model_mask


def test_direct_pixels_are_exact_chw_float_conversion():
    import torch

    rgb = np.random.default_rng(13).integers(0, 256, (2, 704, 1280, 3), np.uint8)
    actual = _preprocess_images(rgb, _target_size((704, 1280), "direct-704x1280"))
    expected = torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy()).float() / 255
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("mode", ["balanced-512", "balanced-896"])
def test_unknown_pixels_cannot_influence_conservative_bicubic_support(mode):
    import torch

    rgb = np.random.default_rng(17).integers(0, 256, (2, 704, 1280, 3), np.uint8)
    known = np.ones((704, 1280), np.bool_)
    known[220:360, 470:690] = False
    changed = rgb.copy()
    changed[:, ~known] = 255 - changed[:, ~known]
    target = _target_size((704, 1280), mode)
    support = torch.from_numpy(build_conservative_model_mask(known, target).copy())
    assert support[10, 10]  # A known region must survive, not an all-empty mask.
    difference = torch.any(
        _preprocess_images(rgb, target) != _preprocess_images(changed, target),
        dim=(0, 1),
    )
    assert torch.any(difference[~support])
    assert not torch.any(difference[support])

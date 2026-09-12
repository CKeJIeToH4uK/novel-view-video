"""CPU value test for Gen3C context-depth science."""

from __future__ import annotations

import numpy as np
import torch

from novel_view.generation.gen3c.context_depth import prepare_context_depth


def test_context_depth_keeps_metric_fit_and_neighbour_filter() -> None:
    frame_count, height, width = 11, 24, 32
    y, x = np.mgrid[:height, :width]
    source = 4.0 + 0.002 * x + 0.001 * y
    moge = np.repeat(source[None], frame_count, axis=0).astype(np.float32)
    moge[10, 9:15, 12:20] += 1.0
    reference = 1.0 / (1.5 / moge + 0.02)
    reference[10] = 1.0 / (1.5 / source + 0.02)
    reference = reference.astype(np.float32)
    moge_valid = np.ones_like(moge, dtype=np.bool_)
    reference_valid = np.ones_like(moge, dtype=np.bool_)
    moge_valid[:, 0, 0] = False
    reference_valid[:, 0, 0] = False
    w2c = np.repeat(
        np.eye(4, dtype=np.float64)[None],
        frame_count,
        axis=0,
    )
    intrinsics = np.repeat(
        np.array(
            ((20.0, 0.0, 16.0), (0.0, 20.0, 12.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )[None],
        frame_count,
        axis=0,
    )

    def align_depth(source_depth, _target_depth, _target_mask, **_kwargs):
        aligned = 1.0 / (1.5 / source_depth + 0.02)
        aligned = aligned.clone()
        aligned[1, 1] = float("inf")
        return aligned

    def reliable_depth(depth, **_kwargs):
        return torch.ones_like(depth, dtype=torch.bool)

    def forward_warp(_image, mask, depth, *_args, **_kwargs):
        return _image, mask, depth[:, 0], None

    result = prepare_context_depth(
        moge,
        moge_valid,
        reference,
        reference_valid,
        w2c,
        intrinsics,
        torch=torch,
        align_depth=align_depth,
        reliable_depth_mask_range_batch=reliable_depth,
        forward_warp=forward_warp,
        min_alignment_pixels=128,
    )

    np.testing.assert_allclose(
        np.delete(result.alignment_scale, 10),
        1.5,
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        np.delete(result.alignment_bias, 10),
        0.02,
        rtol=2e-4,
        atol=2e-4,
    )
    assert np.all(np.isfinite(result.alignment_mae_m))
    assert np.all(~result.valid[:, 1, 1])
    assert not result.valid[10, 12, 16]
    assert result.valid[0, 12, 16]
    assert np.all(result.depth_z_m[~result.valid] == 0.0)

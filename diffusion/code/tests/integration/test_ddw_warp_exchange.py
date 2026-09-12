"""NPY → CPU worker → NPY: подменяется только Gen3C splat, не transport."""

from pathlib import Path

import numpy as np
import pytest
import torch

from novel_view.preparation.waymo_ddw import _warp_worker as worker, warp
from novel_view.runtime.process import ProcessError


class RecordingWarp:
    def __init__(self):
        self.depth = []
        self.mask = []
        self.batch_sizes = []

    def reliable_depth_mask_range_batch(self, depth, **parameters):
        assert parameters == {"window_size": 5, "ratio_thresh": 0.05}
        self.depth.append(depth.clone())
        return depth > 0

    def unproject_points(self, depth, w2c, K, *, is_depth, mask):
        assert is_depth
        self.mask.append(mask.clone())
        points = torch.zeros((*depth.shape[:1], *depth.shape[2:], 3))
        points[..., 2] = depth[:, 0]
        return points

    def forward_warp(self, **values):
        assert (values["depth1"], values["transformation1"], values["boundary_mask"]) == (
            None,
            None,
            None,
        )
        assert values["render_depth"] and not values["foreground_masking"]
        self.batch_sizes.append(len(values["frame1"]))
        rgb = values["frame1"] + 0.25
        return rgb, values["mask1"].to(rgb), values["world_points1"][..., 2] + 1, None


@pytest.mark.parametrize("layout", ["uint8_thwc", "normalized_tchw"])
def test_both_layouts_real_exchange_masks_holes_and_chronological_batches(
    tmp_path, monkeypatch, layout
):
    rgb = np.arange(54, dtype=np.uint8).reshape(3, 2, 3, 3)
    rgb[0, 0, :, 0] = (0, 127, 255)
    normalized = np.moveaxis(rgb.astype(np.float32) * np.float32(2 / 255) - 1, -1, 1)
    if layout == "normalized_tchw":
        normalized += np.float32(0.000123)  # Detect an unwanted uint8 round-trip.
    depths = np.full((3, 2, 3), 2.0, np.float32)
    depths[0] = [[np.nan, np.inf, -np.inf], [101.0, 2.0, 3.0]]
    valid = np.ones_like(depths, bool)
    valid[0, 1, 1] = False
    rect = np.ones((2, 3), bool)
    rect[1, 2] = False
    K = np.broadcast_to(np.diag([2.0, 3.0, 1.0]), (3, 3, 3)).copy()
    W2C = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    W2C[:, 0, 3] = (1, 2, 3)
    target_K, target_W2C = K.copy(), W2C.copy()
    target_K[:, :2, :2] *= 2
    target_W2C[:, 1, 3] = 4
    request = warp.WarpRequest(
        rgb if layout == "uint8_thwc" else normalized,
        layout,
        depths,
        valid,
        rect,
        K,
        W2C,
        target_K,
        target_W2C,
    )
    module = RecordingWarp()
    exchanges = []

    def process(command, log, description, environment):
        root = Path(command[command.index("--exchange") + 1])
        exchanges.append(root)
        values = worker._load_inputs(root)
        try:
            np.testing.assert_array_equal(values["rgb"][:, 0], normalized)
            for name, expected in (("source_k", K), ("source_w2c", W2C)):
                np.testing.assert_array_equal(values[name][:, 0], expected)
            np.testing.assert_array_equal(values["target_k"], target_K)
            np.testing.assert_array_equal(values["target_w2c"], target_W2C)
            worker._execute_forward_warp(module, torch, values, root, torch.device("cpu"))
        finally:
            for value in values.values():
                value._mmap.close()

    monkeypatch.setattr(warp, "run_process", process)
    result = warp.run_forward_warp(request, warp.WarpResources(tmp_path), tmp_path / "warp.log")
    assert module.batch_sizes == [2, 1]
    safe = np.full_like(depths, 2.0)
    safe[0] = [[100.0, 100.0, 0.0], [100.0, 2.0, 3.0]]
    expected_known = valid & rect & (safe > 0)
    np.testing.assert_array_equal(torch.cat(module.depth)[:, 0], safe)
    np.testing.assert_array_equal(torch.cat(module.mask)[:, 0], expected_known)
    expected_rgb = np.where(expected_known[:, None], normalized + 0.25, -1.0)
    np.testing.assert_array_equal(result.rgb_minus_one_to_one[:, 0], expected_rgb)
    np.testing.assert_array_equal(result.depth_z_m[:, 0], np.where(expected_known, safe + 1, 0))
    np.testing.assert_array_equal(result.known[:, 0], expected_known)
    assert (result.peak_cuda_allocated_bytes, result.peak_cuda_reserved_bytes) == (0, 0)
    assert result.elapsed_seconds >= 0
    assert len(exchanges) == 1 and not exchanges[0].exists()

    def fail(*_):
        raise ProcessError("device process failed")

    monkeypatch.setattr(warp, "run_process", fail)
    with pytest.raises(ProcessError):
        warp.run_forward_warp(request, warp.WarpResources(tmp_path), tmp_path / "failed.log")

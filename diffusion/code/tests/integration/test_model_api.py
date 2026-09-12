"""Small no-weight proofs on the installed production Python prefixes."""

import inspect
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


def _cache_inputs(frame_count=121):
    rgb = np.empty((2, 7, 9, 3), np.uint8)
    rgb[0], rgb[1] = (16, 32, 48), (192, 208, 224)
    depth = np.full((2, 7, 9), 5, np.float32)
    depth[0, 3, 4] = 101
    w2c = np.repeat(np.eye(4)[None], 2, axis=0)
    w2c[1, 0, 3] = -0.25
    intrinsics = np.array([[[4, 0, 4], [0, 4, 3], [0, 0, 1]],
                           [[5, 0, 3.5], [0, 6, 2.5], [0, 0, 1]]], float)
    indices = np.arange(frame_count, dtype=np.int64) % 2
    return dict(
        source_rgb=rgb, source_depth=depth, source_valid=np.ones_like(depth, bool),
        source_w2c=w2c, source_intrinsics=intrinsics, source_index=indices,
        query_w2c=w2c[indices].copy(), query_intrinsics=intrinsics[indices].copy(),
        selected_slots=np.array(sorted({0, 1, 120, frame_count - 1}), np.int64),
    )


def test_cache4d_buffers_and_chunked_unprojection():
    import torch
    from novel_view.models.gen3c.cache4d import runtime
    from novel_view.models.gen3c.cache4d._worker import _execute_cache4d

    module, device = runtime.import_cache4d(None), torch.device("cpu")
    values = _cache_inputs()
    for context in (False, True):
        if context:
            values.update(
                context_rgb=np.full((2, 7, 9, 3), 96, np.uint8),
                context_depth=np.full((2, 7, 9), 4, np.float32),
                context_valid=np.ones((2, 7, 9), bool),
                context_w2c=values["source_w2c"].copy(),
                context_intrinsics=values["source_intrinsics"].copy(),
                context_index=values["source_index"].copy(),
            )
        dense = runtime.dense_cache_inputs(values, 17)
        cache = runtime.create_cache4d(module, torch, dense, 17, device)
        assert isinstance(cache, module.Cache4D)
        assert cache.filter_points_threshold == 0.05
        assert not cache.foreground_masking
        assert cache.input_frame_count() == 121
        if context:
            assert dense.normalized_rgb.shape == (121, 2, 3, 7, 9)
            assert dense.depth_z_m.shape == (121, 2, 1, 7, 9)
            assert cache.input_image.shape[:3] == (1, 121, 2)
            rendered = _execute_cache4d(module, torch, values, 17, device)
            assert rendered["coverage"].shape == (121,)
            assert np.all(rendered["coverage"] > 0)
            assert rendered["rgb"].shape == (3, 7, 9, 3)
            assert not rendered["mask"][0, 3, 4]
            assert rendered["coverage"][0] > rendered["mask"][0].mean()
        else:
            chunked = runtime._unproject_points(
                module, torch, dense.depth_z_m, dense.source_w2c,
                dense.source_intrinsics, 17, device,
            )
            direct = module.unproject_points(
                torch.from_numpy(dense.depth_z_m),
                torch.from_numpy(dense.source_w2c),
                torch.from_numpy(dense.source_intrinsics), is_depth=True,
            )
            np.testing.assert_allclose(chunked.numpy(), direct.numpy(),
                                       rtol=1e-6, atol=1e-6)


def test_cache4d_identity_global_windows_and_empty_support():
    import torch
    from novel_view.models.gen3c.cache4d.runtime import import_cache4d
    from novel_view.models.gen3c.cache4d._worker import _execute_cache4d

    module = import_cache4d(None)
    for frame_count in (121, 241):
        values = _cache_inputs(frame_count)
        if frame_count == 241:
            indices = (np.arange(241, dtype=np.int64) // 73) % 2
            values["source_index"] = indices
            values["query_w2c"] = values["source_w2c"][indices].copy()
            values["query_intrinsics"] = values["source_intrinsics"][indices].copy()
            values["selected_slots"] = np.array([0, 1, 120, 121, 240], np.int64)
        result = _execute_cache4d(module, torch, values, 17, torch.device("cpu"))
        assert result["coverage"].shape == (frame_count,)
        assert np.all((result["coverage"] > 0) & (result["coverage"] <= 1))
        for row, slot in enumerate(values["selected_slots"]):
            supported = result["mask"][row]
            expected = values["source_rgb"][values["source_index"][slot]]
            np.testing.assert_array_equal(result["rgb"][row][supported],
                                          expected[supported])
        np.testing.assert_array_equal(result["overlap"], [0, 0])
        np.testing.assert_array_equal(result["peaks"], [0, 0])
        expected_far = np.array([1 / 63, 0])[values["source_index"]]
        np.testing.assert_array_equal(result["far_depth"], expected_far)
    values = _cache_inputs()
    values["source_valid"][0] = False
    result = _execute_cache4d(module, torch, values, 17, torch.device("cpu"))
    np.testing.assert_array_equal(result["coverage"][::2], 0)
    assert np.all(result["coverage"][1::2] > 0)


def test_warp_identity_and_subpixel_splat():
    import torch
    from novel_view.preparation.waymo_ddw import _warp_worker as worker

    module = worker._official_module()
    rgb = torch.linspace(-1, 1, 2 * 3 * 7 * 9).reshape(2, 3, 7, 9)
    depth = torch.full((2, 1, 7, 9), 2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)
    k, w2c = torch.eye(3).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1)
    actual, rendered_depth, known = worker._render_chunk(
        module, torch, rgb, depth, valid, valid, k, w2c, k, w2c,
    )
    torch.testing.assert_close(actual, rgb, rtol=0, atol=1e-6)
    torch.testing.assert_close(rendered_depth, depth[:, 0])
    assert known.all()
    rgb = torch.full((1, 3, 7, 9), -1.0)
    rgb[0, :, 3, 4], rgb[0, :, 3, 5] = 0.2, 0.8
    valid = torch.zeros((1, 1, 7, 9), dtype=torch.bool)
    valid[0, 0, 3, 4:6] = True
    depth = torch.full_like(valid, 2, dtype=torch.float32)
    target = w2c[:1].clone()
    target[:, 0, 3] = 0.5
    actual, rendered_depth, known = worker._render_chunk(
        module, torch, rgb, depth, valid, torch.ones_like(valid),
        k[:1], w2c[:1], k[:1], target,
    )
    assert torch.nonzero(known).tolist() == [[0, 3, 4], [0, 3, 5], [0, 3, 6]]
    torch.testing.assert_close(actual[0, :, 3, 4:7],
                               torch.tensor([[0.2, 0.65, 0.8]]).repeat(3, 1))
    torch.testing.assert_close(rendered_depth[0, 3, 4:7], torch.full((3,), 2.0))


def test_warp_collision_depth_weights_and_batch_maximum(monkeypatch):
    import torch
    from novel_view.preparation.waymo_ddw import _warp_worker as worker

    module = worker._official_module()
    rgb = torch.tensor([[[[0.0, 1.0, -1.0]]]]).repeat(2, 1, 1, 1)
    mask = torch.tensor([[[[True, True, False]]]]).repeat(2, 1, 1, 1)
    depth = torch.tensor([[[[1.0, 2.0, 2.0]]], [[[100.0, 100.0, 100.0]]]])
    k, w2c = torch.eye(3).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1)
    points = torch.zeros((2, 1, 3, 3))
    points[..., 2] = depth[:, 0]
    # Isolate splat collision from unprojection/filtering; splat itself is real.
    monkeypatch.setattr(module, "reliable_depth_mask_range_batch",
                        lambda value, **_: torch.ones_like(value, dtype=torch.bool))
    monkeypatch.setattr(module, "unproject_points",
                        lambda value, *_, **__: points[:value.shape[0]])
    actual, rendered_depth, known = worker._render_chunk(
        module, torch, rgb, depth, mask, mask, k, w2c, k, w2c,
    )
    alone = worker._render_chunk(
        module, torch, rgb[:1], depth[:1], mask[:1], mask[:1],
        k[:1], w2c[:1], k[:1], w2c[:1],
    )[0]
    weights = np.array([math.exp(math.log1p(z) / math.log1p(100) * 50) + 1e-7
                        for z in (1, 2)])
    inverse = 1 / weights
    assert float(actual[0, 0, 0, 0]) == pytest.approx(inverse[1] / inverse.sum(), abs=1e-6)
    assert float(rendered_depth[0, 0, 0]) == pytest.approx(
        inverse @ [1, 2] / inverse.sum(), abs=1e-6,
    )
    assert known[0, 0, 0]
    assert actual[0, 0, 0, 0] > alone[0, 0, 0, 0] + 0.01


def test_warp_sanitization_and_both_masks(monkeypatch):
    import torch
    from novel_view.preparation.waymo_ddw import _warp_worker as worker

    module = worker._official_module()
    depth = torch.full((1, 1, 7, 9), 2.0)
    depth[0, 0, 0, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), 101])
    original, rectification = (torch.ones_like(depth, dtype=torch.bool) for _ in range(2))
    original[0, 0, 3, 3], rectification[0, 0, 4, 4] = False, False
    observed = {}
    reliable, unproject, forward = (module.reliable_depth_mask_range_batch,
                                    module.unproject_points, module.forward_warp)

    def filter_depth(value, **kwargs):
        observed["depth"] = value.clone()
        observed["reliable"] = reliable(value, **kwargs)
        return observed["reliable"]

    def unproject_mask(value, *args, **kwargs):
        observed["mask"] = kwargs["mask"].clone()
        return unproject(value, *args, **kwargs)

    def splat_mask(**kwargs):
        observed["forward_mask"] = kwargs["mask1"].clone()
        return forward(**kwargs)

    monkeypatch.setattr(module, "reliable_depth_mask_range_batch", filter_depth)
    monkeypatch.setattr(module, "unproject_points", unproject_mask)
    monkeypatch.setattr(module, "forward_warp", splat_mask)
    k, w2c = torch.eye(3)[None], torch.eye(4)[None]
    worker._render_chunk(module, torch, torch.zeros((1, 3, 7, 9)), depth,
                         original, rectification, k, w2c, k, w2c)
    assert observed["depth"][0, 0, 0, :4].tolist() == [100, 100, 0, 100]
    assert not observed["reliable"][0, 0, 0, 0]
    assert not observed["mask"][0, 0, 0, 0]
    assert not observed["mask"][0, 0, 3, 3]
    assert not observed["mask"][0, 0, 4, 4]
    torch.testing.assert_close(observed["mask"], observed["forward_mask"])


def _vggt_loader_parity(root):
    """Runs in VGGT's Python without importing pytest into that prefix."""
    import torch
    from PIL import Image
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from novel_view.models.vggt._worker import _preprocess_images, _target_size

    rgb = np.random.default_rng(20260730).integers(0, 256, (2, 704, 1280, 3), np.uint8)
    paths = [root / "1.png", root / "0.png"]
    for path, pixels in zip(paths, rgb):
        Image.fromarray(pixels).save(path)
    for mode, resolution, size in (("balanced-512", 512, (384, 688)),
                                    ("balanced-896", 896, (672, 1216))):
        official = load_and_preprocess_images(paths, mode="balanced", image_resolution=resolution)
        assert _target_size((704, 1280), mode) == size
        actual = _preprocess_images(rgb, size)
        assert actual.dtype == torch.float32
        assert torch.equal(actual, official)


def test_vggt_official_loader(tmp_path):
    from novel_view.models.vggt import _worker
    from novel_view.runtime.executables import VGGT_PYTHON

    subprocess.run([str(VGGT_PYTHON), str(Path(__file__).resolve()), str(tmp_path)],
                   cwd=Path(_worker.__file__).parent, check=True)


def test_lora_zero_delta_and_nonzero_forward():
    import torch
    from cosmos_predict1.diffusion.training.utils.peft.lora_net import LoRALinearLayer

    x = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 32
    base = torch.nn.Linear(16, 16, bias=False)
    lora = LoRALinearLayer(16, 16, rank=8, linear=True)
    baseline = base(x)
    assert torch.equal(lora(x), torch.zeros_like(baseline))
    assert torch.equal(baseline + lora(x), baseline)
    with torch.no_grad():
        lora.net[0].weight.fill_(0.125)
        lora.net[1].weight.fill_(0.25)
    expected = x.sum(-1, keepdim=True).expand(-1, 16) / 4
    torch.testing.assert_close(lora(x), expected, rtol=0, atol=0)
    assert not torch.equal(baseline + lora(x), baseline)


def test_te_recompute_preserves_result_and_gradients():
    import copy
    import torch
    from novel_view.training.gen3c.lora.method import _apply_activation_recompute

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.dropout = torch.nn.Dropout(0.25)
            self.calls = 0

        def forward(self, value, *, scale, nested):
            self.calls += 1
            return self.dropout(self.linear(value + nested["offset"])) * scale

    class TinyDit(torch.nn.Module):
        fsdp_wrap_block_cls = Block

        def __init__(self):
            super().__init__()
            self.first = Block()
            self.untouched = torch.nn.ReLU()
            self.second = Block()

        def forward(self, value, *, scale, nested):
            value = self.first(value, scale=scale, nested=nested)
            return self.second(self.untouched(value), scale=scale, nested=nested)

    reference = TinyDit()
    checkpointed = copy.deepcopy(reference)
    expected_keys = tuple(reference.state_dict())
    _apply_activation_recompute(checkpointed)
    assert tuple(checkpointed.state_dict()) == expected_keys
    reference_input = torch.randn(2, 3, requires_grad=True)
    checkpointed_input = reference_input.detach().clone().requires_grad_()
    reference_offset = torch.randn(2, 3, requires_grad=True)
    checkpointed_offset = reference_offset.detach().clone().requires_grad_()
    scale = torch.tensor(0.75)
    torch.manual_seed(17)
    reference_output = reference(
        reference_input, scale=scale, nested={"offset": reference_offset},
    )
    torch.manual_seed(17)
    checkpointed_output = checkpointed(
        checkpointed_input, scale=scale, nested={"offset": checkpointed_offset},
    )
    torch.testing.assert_close(checkpointed_output, reference_output)
    reference_output.square().mean().backward()
    checkpointed_output.square().mean().backward()
    torch.testing.assert_close(checkpointed_input.grad, reference_input.grad)
    torch.testing.assert_close(checkpointed_offset.grad, reference_offset.grad)
    for expected, actual in zip(reference.parameters(), checkpointed.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)
    assert (reference.first.calls, reference.second.calls) == (1, 1)
    assert (checkpointed.first.calls, checkpointed.second.calls) == (2, 2)


def test_grounding_sam2_signatures():
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import GroundingDinoProcessor

    grounding = inspect.signature(GroundingDinoProcessor.post_process_grounded_object_detection)
    assert {"threshold", "text_threshold"} <= grounding.parameters.keys()
    assert "box_threshold" not in grounding.parameters
    sam = inspect.signature(SAM2ImagePredictor.predict)
    assert {"box", "multimask_output", "return_logits"} <= sam.parameters.keys()


def test_moge_v1_signature():
    from moge.model.v1 import MoGeModel

    inspect.signature(MoGeModel.from_pretrained).bind(Path("unused-checkpoint.pt"))
    inspect.signature(MoGeModel.infer).bind(
        None, None, fov_x=70.0, num_tokens=2500, apply_mask=False,
        force_projection=True, use_fp16=True,
    )


def test_lpips_dinov2_signatures(monkeypatch):
    monkeypatch.syspath_prepend("/opt/upstream/dinov2")
    import lpips
    from dinov2.hub.backbones import dinov2_vitb14
    from dinov2.models.vision_transformer import DinoVisionTransformer

    inspect.signature(lpips.LPIPS).bind(net="alex", version="0.1", spatial=True, eval_mode=True)
    inspect.signature(lpips.LPIPS.forward).bind(None, None, None, normalize=False)
    inspect.signature(dinov2_vitb14).bind(pretrained=False)
    inspect.signature(DinoVisionTransformer.get_intermediate_layers).bind(None, None, n=1, norm=True)


if __name__ == "__main__":
    _vggt_loader_parity(Path(sys.argv[1]))
else:
    import pytest

    pytestmark = pytest.mark.native_api

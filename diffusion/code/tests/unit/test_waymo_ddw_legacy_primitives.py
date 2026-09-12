"""Numerical seams of the historical one-source DDW and global reference."""

import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from novel_view.models.gen3c.cache4d.backend import Gen3cCache4dResources
from novel_view.models.gen3c.cache4d.protocol import write_cache4d_conditioning
from novel_view.preparation.waymo_ddw.condition import DdwCondition
from novel_view.preparation.waymo_ddw.legacy import local, reference, render
from novel_view.preparation.waymo_ddw.legacy.axis import DdwVariant
from novel_view.preparation.waymo_ddw.legacy.masks import measure_ddw_masks
from novel_view.preparation.waymo_ddw.legacy.path import build_legacy_ddw_path
from novel_view.preparation.waymo_ddw.legacy.source import (
    LegacyDdwSource, open_legacy_ddw_source,
)
from novel_view.preparation.waymo_ddw.warp import WarpResources, WarpResult
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey


@pytest.fixture
def source():
    shape = (121, 3, 4)
    rgb = (np.arange(np.prod(shape) * 3) % 256).astype(np.uint8).reshape(*shape, 3)
    valid = np.ones(shape, np.bool_)
    valid[0, 0, 0] = False
    rect = np.ones(shape[1:], np.bool_)
    rect[0, -1] = False
    return LegacyDdwSource(
        WaymoClipKey("waymo-ddw-lora-v1", "training", "test-segment", 0, tuple(range(121))),
        rgb, np.full(shape, 4.0, np.float32), valid, rect,
        np.repeat(np.eye(3)[None], 121, axis=0),
        np.repeat(np.eye(4)[None], 121, axis=0),
    )


def test_source_borrows_rgb_and_closes_mapping(source, tmp_path):
    path = tmp_path / "front.npy"
    np.save(path, source.rgb_thwc)
    camera = SimpleNamespace(rgb_path=path, raster_plan=SimpleNamespace(
        rectification_known=source.rectification_known
    ))
    depth = SimpleNamespace(
        clip_key=source.clip_key, depth_z_m=source.depth_z_m, valid=source.depth_valid,
        K_canvas=source.K_canvas, world_to_camera_cv=source.world_to_camera_cv,
    )
    with open_legacy_ddw_source(camera, depth) as opened:
        np.testing.assert_array_equal(opened.rgb_thwc, source.rgb_thwc)
        assert opened.depth_z_m is source.depth_z_m
        assert isinstance(opened.rgb_thwc, np.memmap)
        mapping = opened.rgb_thwc._mmap
    assert mapping.closed


def test_two_passes_use_actual_outward_float_result(source, monkeypatch, tmp_path):
    path = build_legacy_ddw_path(source, DdwVariant(2, -1))
    np.testing.assert_allclose(path.target.displacement_m[[0, 60, 120]], [0, -1, -2], atol=1e-12)
    np.testing.assert_array_equal(source.world_to_camera_cv, np.repeat(np.eye(4)[None], 121, axis=0))
    arrays = (121, 1, 3, 4)
    outward = WarpResult(
        np.full((121, 1, 3, 3, 4), .123456, np.float32),
        np.full(arrays, 5.5, np.float32), np.ones(arrays, np.bool_), 7, 11, 1.25,
    )
    returning = WarpResult(
        np.full_like(outward.rgb_minus_one_to_one, .333333),
        np.full_like(outward.depth_z_m, 6), outward.known.copy(), 9, 10, 2.5,
    )
    requests = []

    def warp(request, resources, log_path):
        requests.append(request)
        log_path.write_text("permanent log")
        return outward if len(requests) == 1 else returning

    monkeypatch.setattr(render, "run_forward_warp", warp)
    result = render.render_legacy_ddw(
        source, path, WarpResources(tmp_path), tmp_path / "out.log", tmp_path / "back.log"
    )
    first, second = requests
    assert first.source_rgb_layout == "uint8_thwc"
    assert first.source_rgb is source.rgb_thwc
    assert second.source_rgb_layout == "normalized_tchw"
    for value, original in ((second.source_rgb, outward.rgb_minus_one_to_one),
                            (second.source_depth_z_m, outward.depth_z_m),
                            (second.source_depth_valid, outward.known)):
        assert np.shares_memory(value, original)
    assert second.source_rectification_known.all()
    np.testing.assert_array_equal(second.source_world_to_camera_cv, path.target.virtual_world_to_camera_cv)
    np.testing.assert_array_equal(second.target_world_to_camera_cv, source.world_to_camera_cv)
    assert result.condition.rgb_minus_one_to_one.shape == (121, 3, 3, 4)
    assert np.shares_memory(result.condition.rgb_minus_one_to_one, returning.rgb_minus_one_to_one)
    assert (result.condition.peak_cuda_allocated_bytes, result.condition.peak_cuda_reserved_bytes,
            result.condition.elapsed_seconds) == (9, 11, 3.75)
    assert (tmp_path / "out.log").read_text() == "permanent log"


def test_global_reference_serializes_all_rows_and_releases_dense(source, monkeypatch, tmp_path):
    path = build_legacy_ddw_path(source, DdwVariant(3, 1))
    known = np.ones((121, 3, 4), np.bool_)
    known[::2, 0, 0] = False
    dense_refs = []

    def cache4d(conditioning, resources, log_path):
        selected = write_cache4d_conditioning(tmp_path, conditioning)
        np.testing.assert_array_equal(selected, np.arange(121))
        for name, expected in {
            "source_rgb": source.rgb_thwc,
            "source_depth_z_m": source.depth_z_m,
            "source_geometry_valid": source.depth_valid & source.rectification_known,
            "source_intrinsics": source.K_canvas,
            "anchor_to_source_camera": source.world_to_camera_cv,
            "anchor_to_query_camera": path.target.virtual_world_to_camera_cv,
            "query_intrinsics": source.K_canvas,
        }.items():
            np.testing.assert_array_equal(np.load(tmp_path / f"{name}.npy"), expected)
        mask = known.copy()
        dense_refs.append(weakref.ref(mask))
        return SimpleNamespace(selected_coverage_mask=mask, peak_cuda_allocated_bytes=3,
                               peak_cuda_reserved_bytes=5, elapsed_seconds=1.5)

    monkeypatch.setattr(reference, "render_gen3c_cache4d", cache4d)
    result = reference.measure_legacy_ddw_reference(
        source, path, Gen3cCache4dResources(tmp_path, Path("/opt/upstream/gen3c")),
        tmp_path / "reference.log",
    )
    assert result.descriptors == measure_ddw_masks(known, np.broadcast_to(source.rectification_known, known.shape))
    assert dense_refs[0]() is None
    assert (result.peak_cuda_allocated_bytes, result.elapsed_seconds) == (3, 1.5)


def test_local_reduces_and_releases_condition_before_reference(source, monkeypatch, tmp_path):
    refs = []

    def render_local(source, path, *_):
        rgb = np.moveaxis(source.rgb_thwc.astype(np.float32) * (2 / 255) - 1, -1, 1).copy()
        rgb[:, :, 1, 1] += .1
        known = np.ones((121, 3, 4), np.bool_)
        refs.extend((weakref.ref(rgb), weakref.ref(known)))
        return render.LegacyDdwRenderResult(path, DdwCondition(rgb, known, 7, 9, 1.0))

    monkeypatch.setattr(local, "render_legacy_ddw", render_local)
    measured = local.measure_legacy_ddw_local(
        source, DdwVariant(1, 1), WarpResources(tmp_path), tmp_path / "a", tmp_path / "b"
    )
    assert all(ref() is None for ref in refs)
    assert measured.headroom.passed


def test_real_process_error_is_not_a_headroom_failure(source, monkeypatch, tmp_path):
    def fail(*_):
        raise OSError("worker failed")

    monkeypatch.setattr(render, "run_forward_warp", fail)
    with pytest.raises(OSError):
        render.render_legacy_ddw(
            source, build_legacy_ddw_path(source, DdwVariant(1, 1)),
            WarpResources(tmp_path), tmp_path / "a", tmp_path / "b",
        )


@pytest.mark.parametrize("backend", ["moge-v1-lidar-scale", "vggt-omega", None])
def test_explicit_depth_selection_precedes_model_and_borrows_front(
    backend, source, monkeypatch, tmp_path,
):
    from novel_view.preparation.waymo_ddw.legacy import execution
    from novel_view.preparation.waymo_ddw.legacy.spec import (
        LegacyDdwInput, LegacyDdwRecipe, LegacyDdwSpec,
    )
    from novel_view.preparation.waymo_depth.selection import DepthGateDecision, DepthGateStop
    from tests.support.stage5 import make_runtime

    runtime = make_runtime(tmp_path)
    spec = LegacyDdwSpec(
        LegacyDdwInput("waymo", "selections/local/clip.yaml", "depth/choice.json"),
        LegacyDdwRecipe("moge/model.pt", (DdwVariant(1, 1),)),
    )
    outcome = DepthGateStop(()) if backend is None else DepthGateDecision(backend, ())
    calls = []

    def selection(path):
        assert path == runtime.roots.runs / "depth/choice.json"
        return SimpleNamespace(outcome=outcome)

    def reader(*args):
        calls.append("reader")
        assert args == (runtime.roots.data / "waymo", "training", "test-segment", 0, 121)
        return object()

    camera = SimpleNamespace(
        rgb_path=tmp_path / "front.npy", evidence=object(),
        raster_plan=SimpleNamespace(rectification_known=source.rectification_known),
        K_canvas=source.K_canvas, world_to_camera_cv=source.world_to_camera_cv,
    )
    np.save(camera.rgb_path, source.rgb_thwc)

    def moge(rgb, evidence, rect, K, w2c, resources, log):
        calls.append("moge")
        np.testing.assert_array_equal(np.stack([frame.copy() for frame in rgb]), source.rgb_thwc)
        assert rect is source.rectification_known and K is source.K_canvas
        metric = SimpleNamespace(
            clip_key=source.clip_key, depth_z_m=source.depth_z_m, valid=source.depth_valid,
            K_canvas=K, world_to_camera_cv=w2c,
        )
        return SimpleNamespace(metric_depth=SimpleNamespace(clip=metric),
                               elapsed_seconds=1., worker_peak_ram_mib=2., worker_peak_vram_mib=3.)

    monkeypatch.setattr(execution, "read_selection_record", selection)
    monkeypatch.setattr(execution, "load_depth_clip_selection", lambda _: source.clip_key)
    monkeypatch.setattr(execution, "WaymoV2FrameReader", reader)
    monkeypatch.setattr(execution, "build_depth_comparison_clip",
                        lambda *_: SimpleNamespace(camera=lambda name: camera if name == "FRONT" else None))
    monkeypatch.setattr(execution, "run_moge_metric_depth", moge)
    if backend != "moge-v1-lidar-scale":
        with pytest.raises(ValueError), execution.open_legacy_ddw_measurement(spec, runtime):
            pytest.fail("non-MoGe selection reached model input")
        assert calls == []
    else:
        with execution.open_legacy_ddw_measurement(spec, runtime) as measurement:
            assert measurement.source.depth_z_m is source.depth_z_m
            assert measurement.depth.elapsed_seconds == 1.
            assert measurement.reference.upstream_root == Path("/opt/upstream/gen3c")
        assert calls == ["reader", "moge"]

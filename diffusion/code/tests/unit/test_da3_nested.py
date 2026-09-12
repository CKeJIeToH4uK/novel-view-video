"""Numerical proof for the frozen, checkout-only DA3 prototype."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from novel_view.generation.euvs.source_views import SourceGeometryInput
from research.da3_nested import adapter
from research.da3_nested.adapter import (
    Da3NestedConditionedPrediction,
    Da3NestedError,
    build_da3_nested_input_plan,
)
from research.da3_nested.io import write_source_rgb


@pytest.fixture
def source_input():
    # The measured pack is already built here; EUVS decoding has its own proof.
    valid = np.ones((28, 56), dtype=np.bool_)
    valid[0, 0] = False
    raster_plan = SimpleNamespace(output_size_hw=(28, 56), valid_mask=valid)
    source = SimpleNamespace(
        sequence_id=("source-9", "source-2"),
        frames=tuple(SimpleNamespace(raster=SimpleNamespace(
            plan=raster_plan,
            rgb=np.full((28, 56, 3), value, dtype=np.uint8),
        )) for value in (9, 2)),
    )
    intrinsics = np.repeat(np.array([
        [80.0, 0.0, 28.0], [0.0, 82.0, 14.0], [0.0, 0.0, 1.0],
    ])[None], 2, axis=0)
    w2c = np.repeat(np.eye(4)[None], 2, axis=0)
    w2c[1, :2, 3] = (-2.0, -0.5)
    return SourceGeometryInput(source, intrinsics, w2c)


def raw_prediction(source_input):
    plan = build_da3_nested_input_plan((28, 56), 28)
    depth = np.arange(1, 785, dtype=np.float32).reshape(2, 14, 28)
    return plan, dict(
        depth_z_m=depth,
        confidence=depth / 100,
        conditioned_w2c=source_input.reference_to_camera[:, :3].astype(np.float32),
        conditioned_intrinsics=np.einsum(
            "ij,njk->nik", plan.source_to_model_pixels, source_input.intrinsics,
        ).astype(np.float32),
        sky_mask=None,
    )


@pytest.mark.parametrize("resolution,model_size", [
    (504, (280, 504)), (896, (490, 896)), (1280, (700, 1274)),
])
def test_official_grid_and_measured_intrinsics_scale(resolution, model_size):
    plan = build_da3_nested_input_plan((704, 1280), resolution)
    assert plan.model_size_hw == model_size
    np.testing.assert_array_equal(
        plan.source_to_model_pixels,
        np.diag([model_size[1] / 1280, model_size[0] / 704, 1.0]),
    )
    # The official nearest-patch rule rounds an exact tie upward.
    assert build_da3_nested_input_plan((35, 35), 35).model_size_hw == (42, 42)


@pytest.mark.parametrize("with_sky", [False, True])
def test_nearest_depth_confidence_and_masks_preserve_measured_pack(source_input, with_sky):
    plan, values = raw_prediction(source_input)
    sky = np.zeros((2, 14, 28), dtype=np.bool_)
    sky[0, 0, 1] = True
    values["sky_mask"] = sky if with_sky else None
    prediction = Da3NestedConditionedPrediction(source_input, plan, **values)
    result = prediction.to_posed_depth_sequence()

    expected_valid = np.ones((2, 28, 56), dtype=np.bool_)
    expected_valid[:, 0, 0] = False
    if with_sky:
        expected_valid &= ~sky.repeat(2, axis=1).repeat(2, axis=2)
    expected_depth = values["depth_z_m"].repeat(2, axis=1).repeat(2, axis=2)
    expected_confidence = values["confidence"].repeat(2, axis=1).repeat(2, axis=2)
    expected_depth[~expected_valid] = 0
    expected_confidence[~expected_valid] = 0

    assert result.sequence_id == ("source-9", "source-2")
    np.testing.assert_array_equal(result.geometry_valid_mask, expected_valid)
    np.testing.assert_array_equal(result.depth_z_m, expected_depth)
    np.testing.assert_array_equal(result.backend_confidence, expected_confidence)
    np.testing.assert_array_equal(result.intrinsics, source_input.intrinsics)
    np.testing.assert_array_equal(result.reference_to_camera, source_input.reference_to_camera)
    values["depth_z_m"][:] = 999
    assert prediction.depth_z_m[0, 0, 0] == 1  # Own data beyond worker scratch lifetime.


@pytest.mark.parametrize("field,index", [
    ("conditioned_w2c", (1, 0, 3)),
    ("conditioned_intrinsics", (0, 0, 2)),
    ("depth_z_m", (0, 0, 0)),
])
def test_rejects_camera_substitution_or_nonmetric_depth(source_input, field, index):
    plan, values = raw_prediction(source_input)
    values[field][index] += 1 if field != "depth_z_m" else -1
    with pytest.raises(Da3NestedError):
        Da3NestedConditionedPrediction(source_input, plan, **values)


def test_transient_rgb_keeps_source_order(source_input, tmp_path):
    destination = tmp_path / "rgb.npy"
    write_source_rgb(source_input.source, destination)
    rgb = np.load(destination, allow_pickle=False)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[:, 0, 0, 0], (9, 2))


def test_worker_help_runs_outside_checkout_without_model_import(tmp_path):
    worker = Path(adapter.__file__).with_name("_worker.py")
    result = subprocess.run(
        [sys.executable, "-I", str(worker), "--help"], cwd=tmp_path,
        text=True, capture_output=True, check=True,
    )
    assert "--input-extrinsics" in result.stdout
    assert "--process-resolution" in result.stdout

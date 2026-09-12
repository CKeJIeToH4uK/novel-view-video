"""Independent camera-pack Sim(3) and metric source-grid depth targets."""

from dataclasses import replace

import numpy as np
import pytest

from novel_view.geometry.sim3 import CameraPackAlignment, CameraPackAlignmentError
from novel_view.models.vggt.alignment import align_vggt_cameras
from novel_view.models.vggt.posed_depth import build_vggt_posed_depth, VggtOmegaPosedDepthError
from novel_view.models.vggt.request import VggtOmegaRawPrediction
from novel_view.models.vggt.spec import VggtOmegaInputMode, VggtOmegaInputPlan


def w2c(rotations, centers):
    matrices = np.tile(np.eye(4), (len(centers), 1, 1))
    matrices[:, :3, :3] = rotations.transpose(0, 2, 1)
    matrices[:, :3, 3] = -np.einsum("nij,nj->ni", matrices[:, :3, :3], centers)
    return matrices


def test_sim3_with_noncommuting_cameras_large_translation_and_rotation_projection():
    centers = np.array([[0, 0, 0], [1.1, 0.2, -0.1], [2.4, -0.3, 0.2], [4, 0.1, 0.4]])
    rz = np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1]])
    ry = np.array([[12 / 13, 0, -5 / 13], [0, 1, 0], [5 / 13, 0, 12 / 13]])
    rotations = np.stack([np.eye(3), rz, ry, rz @ ry])
    model = w2c(rotations, centers)[:, :3].astype(np.float32)
    model[0, :3, :3] = [[1.001, 0.0002, 0], [0.0002, 0.999, 0], [0, 0, 1]]
    rotation, translation, scale = rz @ ry, np.array([1000003.0, -2000004.0, 51.0]), 2.75
    measured_centers = scale * (centers @ rotation.T) + translation
    measured = w2c(rotation @ rotations, measured_centers)
    result = align_vggt_cameras(model, measured)
    assert result.scale_m_per_model_unit == pytest.approx(scale, abs=1e-6)
    np.testing.assert_allclose(result.model_to_world_rotation, rotation, rtol=0, atol=2e-7)
    np.testing.assert_allclose(result.model_to_world_translation_m, translation, rtol=0, atol=2e-6)
    aligned_r = result.aligned_world_to_camera[:, :3, :3]
    aligned_t = result.aligned_world_to_camera[:, :3, 3]
    # W2C t=-R*C amplifies rounded model rotations at UTM-scale centers.
    # Compare physical rotation and recovered center instead of a blanket t tolerance.
    np.testing.assert_allclose(aligned_r, measured[:, :3, :3], rtol=0, atol=2e-7)
    recovered_centers = -np.einsum("nji,nj->ni", aligned_r, aligned_t)
    np.testing.assert_allclose(recovered_centers, measured_centers, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(result.aligned_world_to_camera[:, 3], [[0, 0, 0, 1]] * 4)
    assert max(result.center_residual_m) < 2e-6
    assert max(result.rotation_residual_deg) < 2e-5
    assert result.model_rotation_projection_residual_fro[0] > 0


@pytest.mark.parametrize("unobservable", ["rotation", "translation"])
def test_sim3_rejects_unidentifiable_geometry(unobservable):
    centers = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    rotations = np.tile(np.eye(3), (2, 1, 1))
    model = w2c(rotations, centers)[:, :3].astype(np.float32)
    if unobservable == "rotation":
        rotations[1] = np.diag([1, -1, -1])
    else:
        model[:, :3, 3] = 0
    with pytest.raises(CameraPackAlignmentError):
        align_vggt_cameras(model, w2c(rotations, centers))


@pytest.mark.parametrize("resized", [False, True])
def test_metric_depth_confidence_k_and_validity_on_source_grid(resized):
    size = (4, 6) if resized else (2, 3)
    pixel_scale = 0.5 if resized else 1.0
    plan = VggtOmegaInputPlan(
        VggtOmegaInputMode.BALANCED_512,
        size,
        (2, 3),
        np.diag([pixel_scale, pixel_scale, 1]),
    )
    first = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    depth, confidence = np.stack([first, first + 10]), np.stack([first + 20, first + 30])
    k = np.tile(np.array([[3.0, 0.25, 1.5], [0, 4, 1], [0, 0, 1]], np.float32), (2, 1, 1))
    cameras = np.tile(np.eye(4), (2, 1, 1))
    cameras[1, 0, 3] = -4
    alignment = CameraPackAlignment(2.0, np.eye(3), np.zeros(3), cameras, *(np.zeros(2),) * 3)
    raw = VggtOmegaRawPrediction(plan, depth, confidence, np.zeros((2, 9)), cameras[:, :3], k)
    valid = np.ones((2, *size), np.bool_)
    valid[0, 1, 2] = valid[1, -1, -1] = False
    result = build_vggt_posed_depth(raw, alignment, valid, ("a", "b"))
    # Literal half-pixel bilinear grid: no call to the production resizer in the oracle.
    expected = (
        np.array(
            [
                [1, 1.25, 1.75, 2.25, 2.75, 3],
                [1.75, 2, 2.5, 3, 3.5, 3.75],
                [3.25, 3.5, 4, 4.5, 5, 5.25],
                [4, 4.25, 4.75, 5.25, 5.75, 6],
            ],
            np.float32,
        )
        if resized
        else first
    )
    expected_depth = np.stack([expected, expected + 10]) * 2
    expected_confidence = np.stack([expected + 20, expected + 30])
    expected_depth[~valid] = expected_confidence[~valid] = 0
    np.testing.assert_array_equal(result.depth_z_m, expected_depth)
    np.testing.assert_array_equal(result.backend_confidence, expected_confidence)
    np.testing.assert_array_equal(result.geometry_valid_mask, valid)
    expected_k = (
        np.tile(np.array([[6.0, 0.5, 3], [0, 8, 2], [0, 0, 1]]), (2, 1, 1)) if resized else k
    )
    np.testing.assert_array_equal(result.intrinsics, expected_k)
    np.testing.assert_array_equal(result.reference_to_camera, cameras)
    assert result.sequence_id == ("a", "b")
    assert result.frame_count == 2
    assert result.image_size_hw == size
    with pytest.raises(VggtOmegaPosedDepthError):
        build_vggt_posed_depth(
            raw, replace(alignment, scale_m_per_model_unit=float("inf")), valid, ("a", "b")
        )

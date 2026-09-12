"""Independent projection/mask mathematics and a real token-ordered PNG cache."""

from pathlib import Path

import cv2
import numpy as np

from novel_view.evaluation.euvs.masks import (
    EuvsDynamicMaskFrame, GroundedSam2Resources, load_or_predict_masks, rasterize_dynamic_mask,
)
from novel_view.evaluation.euvs.support_projection import project_source_dynamic
from novel_view.evaluation.euvs.support_views import build_support_views
from novel_view.inputs.euvs.pair import EuvsFrameInput, EuvsFrameSequence
from novel_view.inputs.nuplan.camera import GlobalCameraGeometry
from novel_view.inputs.nuplan.db import RawCameraCalibration, RawEgoPose
from novel_view.inputs.nuplan.raster import NuPlanRasterPlan
from novel_view.inputs.types import FrameRef
from novel_view.models.grounded_sam2.request import GroundedSam2Prediction


def _frame(root: Path, token: int) -> EuvsFrameInput:
    image_token = f"{token:016x}"
    return EuvsFrameInput(
        FrameRef(root / f"{image_token}.png", root / "traversal.db", token,
                 "CAM_F0", image_token, f"pose-{token}", "camera-6"),
        RawEgoPose(token, float(token), 0., 0., 1., 0., 0., 0., 32611),
        RawCameraCalibration("pinhole", (0., 0., 0.), (1., 0., 0., 0.),
            ((8., 0., 2.), (0., 8., 1.5), (0., 0., 1.)), (0.,) * 5, (4, 5)),
        GlobalCameraGeometry(np.eye(4), 32611),
    )


def test_png_cache_uses_token_order_binary_polarity_and_only_missing_calls_model(tmp_path, monkeypatch):
    first, second = _frame(tmp_path, 1), _frame(tmp_path, 2)
    for i, frame in enumerate((first, second), 1):
        cv2.imwrite(str(frame.ref.image_path), np.full((4, 5, 3), i, np.uint8))
    sequence = EuvsFrameSequence((second, first, second), "6")
    calls = []

    def predict(request, _resources, log_path):
        assert log_path == tmp_path / "source-masks.log"
        calls.append([int(rgb[0, 0, 0]) for rgb in request.rgb_frames])
        masks = np.zeros((request.frame_count, 4, 5), bool)
        masks[:, 0, 0] = True
        return GroundedSam2Prediction(masks)

    monkeypatch.setattr("novel_view.evaluation.euvs.masks.run_grounded_sam2", predict)
    resources = GroundedSam2Resources(tmp_path, tmp_path / "source-masks.log",
                                      tmp_path / "grounding", tmp_path / "sam2.pt")
    cache = tmp_path / "cache"
    predicted = load_or_predict_masks(sequence, cache, resources)
    cached = load_or_predict_masks(sequence, cache)
    assert calls == [[2, 1]]
    assert [f.input.ref.image_token for f in cached.frames] == list(sequence.sequence_id)
    for actual, saved in zip(predicted.frames, cached.frames):
        np.testing.assert_array_equal(actual.dynamic_mask, saved.dynamic_mask)
        assert saved.dynamic_mask.sum() == 1 and saved.dynamic_mask[0, 0]
    encoded = cv2.imread(str(cache / "native" / f"{first.ref.image_token}.png"), cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(np.unique(encoded), np.array([0, 255], np.uint8))


def test_four_neighbours_use_nearest_z_and_or_only_at_exact_tie():
    target_k = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
    for depth, expected in (([[2., 1.]], [[1, 0, 0], [1, 0, 0]]),
                            ([[1., 1.]], [[1, 1, 0], [1, 1, 0]])):
        result = project_source_dynamic(np.array(depth, np.float32), np.ones((1, 2), bool),
            np.array([[True, False]]), np.eye(3), np.eye(4), target_k, np.eye(4), (2, 3))
        np.testing.assert_array_equal(result.visible_label_mask, np.array(expected, bool))


def test_projection_uses_full_w2c_and_target_camera_z():
    source = np.array([[0., -1., 0., 1.], [1., 0., 0., 2.], [0., 0., 1., .5], [0., 0., 0., 1.]])
    target = np.array([[0., 1., 0., 3.], [-1., 0., 0., -1.], [0., 0., 1., 1.5], [0., 0., 0., 1.]])
    result = project_source_dynamic(np.array([[0., 0., 2.]], np.float32),
        np.array([[False, False, True]]), np.ones((1, 3), bool),
        np.array([[2., 0., 1.], [0., 1., 0.], [0., 0., 1.]]), source, np.eye(3), target, (3, 4))
    expected = np.zeros((3, 4), bool)
    expected[:2, 1] = True
    np.testing.assert_array_equal(result.coverage_mask, expected)
    np.testing.assert_array_equal(result.visible_label_mask, expected)


def test_native_mask_and_support_tracks_keep_validity_and_disocclusion():
    native = np.zeros((4, 5), bool)
    native[0, 0] = True
    map_x = np.tile(np.arange(5, dtype=np.float32), (4, 1)) - 1
    map_y = np.tile(np.arange(4, dtype=np.float32)[:, None], (1, 5))
    valid = np.ones((4, 5), bool)
    valid[0, 1] = False
    plan = NuPlanRasterPlan((4, 5), (4, 5), np.eye(3), map_x, map_y, valid)
    raster = rasterize_dynamic_mask(EuvsDynamicMaskFrame(_frame(Path("/dataset"), 1), native), plan)
    expected = np.zeros((4, 5), bool)
    expected[0, 1] = True
    np.testing.assert_array_equal(raster.dynamic_mask, expected)
    assert not raster.raster_valid_mask[0, 1]
    target_dynamic = np.array([[False, False, True], [False, False, False]])
    source_dynamic = np.array([[True, False, False], [False, False, False]])
    views = build_support_views(np.array([[True, False, True], [True, True, True]]),
                                target_dynamic, source_dynamic)
    np.testing.assert_array_equal(views.target_static, [[True, False, False], [True, True, True]])
    np.testing.assert_array_equal(views.source_aware, [[False, False, False], [True, True, True]])

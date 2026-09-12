"""The producer-v1 boundary decodes native camera values without frame payloads."""

import json
from pathlib import Path

import numpy as np
import pytest

from novel_view.inputs.gaussian.reader import GaussianExportError, read_gaussian_export


FIXTURE = Path(__file__).with_name("fixtures") / "gaussian_depth_export_v1.json"


def test_producer_v1_preserves_fields_order_cameras_and_lazy_paths(tmp_path):
    export = read_gaussian_export(FIXTURE)
    assert export.info_file == FIXTURE
    assert (export.scene_id, export.splats_variant) == (9, "v3-refined")
    assert (export.camera_name, export.target_name) == ("/camera/inner/frontal/middle", "left_1.0")
    assert export.target_shift_camera_xyz_m == (-1.0, 0.0, 0.0)
    assert export.source_size_hw == (1080, 1920)
    np.testing.assert_array_equal(
        export.intrinsics, [[1200, 0, 959.5], [0, 1200, 539.5], [0, 0, 1]]
    )
    assert [frame.pose_index for frame in export.frames] == [2, 5]
    assert [frame.timestamp_ns for frame in export.frames] == [1_000_000_000, 1_125_000_000]
    for frame, index in zip(export.frames, (2, 5), strict=True):
        directory = FIXTURE.parent / f"frames/pose_{index:05d}"
        assert (frame.rgb_path, frame.depth_path, frame.mask_path) == (
            directory / "rgb.png",
            directory / "depth.npy",
            directory / "mask.png",
        )
    expected = np.tile(np.eye(4), (4, 1, 1))
    expected[:, :2, 3] = [[2, 0], [3, 0], [5, 0.5], [6, 0.5]]
    actual = [pose for frame in export.frames for pose in (frame.source_w2c, frame.target_w2c)]
    np.testing.assert_array_equal(actual, expected)
    assert not (FIXTURE.parent / "frames").exists()
    document = json.loads(FIXTURE.read_text())
    document["camera"]["pose"] = "camera_to_world"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document))
    with pytest.raises(GaussianExportError):
        read_gaussian_export(path)

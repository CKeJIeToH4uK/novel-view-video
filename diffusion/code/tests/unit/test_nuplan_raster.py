"""Independent camera/map, interpolation and stored-image raster examples."""

import cv2
import numpy as np
from PIL import Image
import pytest

from novel_view.inputs.nuplan.db import RawCameraCalibration
from novel_view.inputs.nuplan.raster import (
    NuPlanRasterError, NuPlanRasterPlan, build_nuplan_output_intrinsics, build_nuplan_raster_plan,
    read_native_rgb, read_rasterized_rgb,
)


def _calibration(size=(100, 100), intrinsics=None, distortion=(0., 0., 0., 0., 0.)):
    return RawCameraCalibration(
        model="pinhole", translation_xyz=(0., 0., 0.), rotation_wxyz=(1., 0., 0., 0.),
        intrinsics=intrinsics or ((50., 0., 50.), (0., 50., 50.), (0., 0., 1.)),
        distortion=distortion, image_size_hw=size,
    )


def test_output_camera_and_direct_map_share_cover_crop_geometry():
    calibration = _calibration(
        (108, 192), ((154.5, 0., 96.), (0., 154.5, 56.), (0., 0., 1.)),
    )
    plan = build_nuplan_raster_plan(calibration, (70, 128))
    # Scale 2/3 gives 72x128, then crop one row on each side.
    expected_k = [[103., 0., 64.], [0., 103., 109 / 3], [0., 0., 1.]]
    np.testing.assert_allclose(plan.output_intrinsics, expected_k, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        plan.output_intrinsics, build_nuplan_output_intrinsics(calibration, (70, 128)),
    )
    np.testing.assert_array_equal([plan.map_x[20, 40], plan.map_y[20, 40]], [60., 31.5])


def test_valid_mask_is_exact_bilinear_zero_border_support():
    plan = build_nuplan_raster_plan(_calibration(distortion=(0.5, 0., 0., 0., 0.)), (100, 100))
    source = np.ones(plan.source_size_hw, dtype=np.float32)
    linear = cv2.remap(source, plan.map_x, plan.map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.) == 1
    nearest = cv2.remap(source, plan.map_x, plan.map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.) == 1
    np.testing.assert_array_equal(plan.valid_mask, linear)
    assert np.all(~plan.valid_mask | nearest)
    assert np.any(nearest & ~plan.valid_mask)  # Distinguish the two interpolation supports.


def test_opencv_raster_rejects_nonzero_measured_source_skew():
    calibration = _calibration(intrinsics=((50., 2., 50.), (0., 50., 50.), (0., 0., 1.)))
    with pytest.raises(NuPlanRasterError):
        build_nuplan_raster_plan(calibration, (100, 100))


def test_stored_png_and_direct_remap_produce_expected_rgb(tmp_path):
    rgb = np.array([
        [[101, 11, 1], [102, 12, 2], [103, 13, 3]],
        [[104, 14, 4], [105, 15, 5], [106, 16, 6]],
    ], dtype=np.uint8)
    path = tmp_path / "stored.png"
    Image.fromarray(rgb).save(path)
    plan = NuPlanRasterPlan(
        source_size_hw=(2, 3), output_size_hw=(2, 2), output_intrinsics=np.eye(3),
        map_x=np.array([[2., 0.], [1., 2.]], np.float32),
        map_y=np.array([[0., 1.], [1., 1.]], np.float32), valid_mask=np.ones((2, 2), bool),
    )
    np.testing.assert_array_equal(read_native_rgb(path, (2, 3)), rgb)
    np.testing.assert_array_equal(
        read_rasterized_rgb(path, plan).rgb,
        [[[103, 13, 3], [104, 14, 4]], [[105, 15, 5], [106, 16, 6]]],
    )


def test_exif_rotation_does_not_change_the_measured_sensor_grid(tmp_path):
    y, x = np.indices((6, 8))
    rgb = np.stack((x * 20, y * 30, np.full_like(x, 50)), axis=-1).astype(np.uint8)
    path = tmp_path / "orientation.jpg"
    exif = Image.Exif()
    exif[274] = 6  # A viewer would rotate this non-square image by 90 degrees.
    Image.fromarray(rgb).save(path, exif=exif, quality=100, subsampling=0)
    with Image.open(path) as stored:
        expected = np.asarray(stored)
    native = read_native_rgb(path, (6, 8))
    assert native.shape == (6, 8, 3)
    np.testing.assert_allclose(native.astype(int), expected.astype(int), rtol=0, atol=2)

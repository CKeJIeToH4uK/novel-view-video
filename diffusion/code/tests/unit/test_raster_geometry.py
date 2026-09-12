"""Integer cover/crop and full pinhole transformation in pixel-edge coordinates."""

import numpy as np
from numpy.testing import assert_allclose as close, assert_array_equal as equal
import pytest

from novel_view.geometry.raster import CoverCropTransform


@pytest.mark.parametrize("source,output,resized,crop,intrinsics,expected_pixels,expected_k,atol", [
    (
        (1080, 1920), (704, 1280), (720, 1280), (8, 0),
        [[1545, 0, 960], [0, 1545, 560], [0, 0, 1]],
        [[2 / 3, 0, 0], [0, 2 / 3, -8], [0, 0, 1]],
        [[1030, 0, 640], [0, 1030, 1096 / 3], [0, 0, 1]], 2e-13,
    ),
    (
        (3, 5), (4, 4), (4, 7), (0, 1),
        [[10, 2, 2.5], [0, 12, 1.5], [0, 0, 1]],
        [[7 / 5, 0, -1], [0, 4 / 3, 0], [0, 0, 1]],
        [[14, 2.8, 2.5], [0, 16, 2], [0, 0, 1]], 5e-16,
    ),
])
def test_cover_crop_scales_full_intrinsics(source, output, resized, crop,
                                          intrinsics, expected_pixels, expected_k, atol):
    transform = CoverCropTransform(source, output)
    assert transform.resized_size_hw == resized
    assert (transform.crop_top, transform.crop_left) == crop
    assert (transform.scale_y, transform.scale_x) == (
        resized[0] / source[0], resized[1] / source[1],
    )
    equal(transform.pixel_transform, expected_pixels)
    close(transform.transform_intrinsics(intrinsics), expected_k, rtol=0, atol=atol)

    # A(KX/Z) == (AK)X/Z, including skew and unequal integer-rounded axis scales.
    point = np.array([.25, -.5, 2])
    source_pixel = np.asarray(intrinsics) @ point
    source_pixel /= source_pixel[2]
    target_pixel = transform.transform_intrinsics(intrinsics) @ point
    target_pixel /= target_pixel[2]
    close(target_pixel, np.asarray(expected_pixels) @ source_pixel, rtol=0, atol=atol)

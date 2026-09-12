import math

import numpy as np
from skimage.metrics import structural_similarity

from novel_view.metrics.aggregate import mean_columns, summarize_cameras
from novel_view.metrics.image import (
    MetricStatus,
    MetricValue,
    UndefinedReason,
    fractional_patch_support,
    masked_average,
    masked_psnr,
    masked_ssim,
    reduce_rgb_mse_psnr,
    weighted_average,
)


def test_psnr_preserves_empty_perfect_and_finite_states() -> None:
    target = np.zeros((12, 13, 3), dtype=np.uint8)
    prediction = target.copy()
    empty = np.zeros((12, 13), dtype=np.bool_)
    full = np.ones((12, 13), dtype=np.bool_)

    assert masked_psnr(prediction, target, empty) == MetricValue.undefined(
        UndefinedReason.EMPTY_SUPPORT
    )
    assert masked_psnr(prediction, target, full) == MetricValue.positive_infinity()
    prediction[0, 0] = 255
    finite = masked_psnr(prediction, target, full)
    expected = 10.0 * math.log10(
        255.0**2 / (3.0 * 255.0**2 / (12 * 13 * 3))
    )

    assert finite == MetricValue.finite(expected)
    mse, psnr = reduce_rgb_mse_psnr(prediction, target)
    assert mse == 3.0 * 255.0**2 / (12 * 13 * 3)
    assert psnr == expected


def test_ssim_uses_only_fully_supported_windows() -> None:
    target = np.random.default_rng(3).integers(0, 256, (12, 13, 3), np.uint8)
    support = np.ones((12, 13), dtype=np.bool_)
    perfect, count = masked_ssim(target, target, support)

    assert perfect.status is MetricStatus.FINITE
    assert perfect.value is not None
    assert math.isclose(perfect.value, 1.0, abs_tol=1e-12)
    assert count == 6

    support[5, 5] = False
    undefined, count = masked_ssim(target, target, support)
    assert undefined == MetricValue.undefined(UndefinedReason.NO_FULL_WINDOWS)
    assert count == 0


def test_equal_camera_pair_and_location_reductions_do_not_weight_by_size() -> None:
    cameras = summarize_cameras(
        (
            MetricValue.finite(2.0),
            MetricValue.positive_infinity(),
            MetricValue.undefined(UndefinedReason.EMPTY_SUPPORT),
            MetricValue.finite(4.0),
        )
    )
    location_one = mean_columns(((1.0,), (3.0,)))
    location_two = mean_columns(((10.0,),))
    global_value = mean_columns((location_one, location_two))

    assert cameras.finite_mean == 3.0
    assert (
        cameras.finite_frame_count,
        cameras.positive_infinity_frame_count,
        cameras.undefined_frame_count,
    ) == (2, 1, 1)
    assert location_one == (2.0,)
    assert global_value == (6.0,)


def test_fractional_dino_patches_have_independent_area_and_weight_oracle():
    support = np.zeros((3, 5), dtype=np.bool_)
    support[0, 0] = support[1, 2] = True
    # Patch area 15/4: one whole pixel plus one quarter in the first patch,
    # and one quarter pixel in each remaining patch.
    weights = fractional_patch_support(support, (2, 2))
    np.testing.assert_allclose(weights, [[1 / 3, 1 / 15], [1 / 15, 1 / 15]], atol=1e-15)
    value, mass = weighted_average(np.array([1., 2., 3., 4.]), weights.ravel())
    assert math.isclose(value, 7 / 4, abs_tol=1e-15)
    assert math.isclose(mass, 8 / 15, abs_tol=1e-15)
    lpips, pixels = masked_average(np.arange(15).reshape(3, 5), support)
    assert (lpips, pixels) == (3.5, 2.)


def test_ssim_matches_independent_population_covariance_oracle():
    rng = np.random.default_rng(20260805)
    target = rng.integers(0, 256, size=(31, 43, 3), dtype=np.uint8)
    prediction = target.copy()
    prediction[8:22, 11:29] = rng.integers(0, 256, size=(14, 18, 3), dtype=np.uint8)
    result, count = masked_ssim(prediction, target, np.ones((31, 43), dtype=np.bool_))
    expected = structural_similarity(prediction, target, data_range=255, channel_axis=2,
                                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    assert count == (31 - 10) * (43 - 10)
    assert math.isclose(result.value, expected, abs_tol=1e-12)

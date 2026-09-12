"""Optional research-only parity with the installed official DA3 processor."""

import pytest


@pytest.mark.native_api
def test_sizes_and_intrinsics_match_all_planned_resolutions():
    import numpy as np
    from depth_anything_3.utils.io.input_processor import InputProcessor

    from research.da3_nested.adapter import build_da3_nested_input_plan

    generator = np.random.default_rng(20260730)
    source_rgb = generator.integers(0, 256, size=(2, 704, 1_280, 3), dtype=np.uint8)
    source_intrinsics = np.repeat(
        np.array(
            [[1_030.0, 0.0, 640.0], [0.0, 1_030.0, 352.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )[None],
        2,
        axis=0,
    )
    source_extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    source_extrinsics[1, 0, 3] = -2.0

    processor = InputProcessor()
    for resolution in (504, 896, 1_280):
        plan = build_da3_nested_input_plan((704, 1_280), resolution)
        tensor, extrinsics, intrinsics = processor(
            list(source_rgb),
            source_extrinsics,
            source_intrinsics,
            process_res=resolution,
            process_res_method="upper_bound_resize",
            num_workers=1,
            sequential=True,
            print_progress=False,
        )

        assert tuple(tensor.shape) == (2, 3, *plan.model_size_hw)
        np.testing.assert_allclose(
            extrinsics.numpy(), source_extrinsics.astype(np.float32), rtol=0.0, atol=0.0
        )
        expected_intrinsics = np.einsum(
            "ij,njk->nik", plan.source_to_model_pixels, source_intrinsics
        ).astype(np.float32)
        np.testing.assert_allclose(intrinsics.numpy(), expected_intrinsics, rtol=1e-6, atol=1e-5)

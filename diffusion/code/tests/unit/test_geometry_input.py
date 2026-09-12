"""Numerical proof for the measured EUVS source camera pack."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from novel_view.inputs.nuplan.camera import build_global_camera_geometry
from novel_view.inputs.types import FrameRef
from novel_view.inputs.euvs.pair import EuvsFrameInput, EuvsFrameSequence
from novel_view.inputs.euvs.rgb import (
    EuvsRasterizedFrame,
    EuvsRasterizedSequence,
)
from novel_view.generation.euvs.source_views import (
    build_source_geometry_input,
)
from novel_view.inputs.nuplan.db import RawCameraCalibration, RawEgoPose
from novel_view.inputs.nuplan.raster import NuPlanRasterPlan, RasterizedRgb


def test_rebase_avoids_utm_scale_inverse_cancellation():
    size = (8, 12)
    intrinsics = np.array([[20., 0., 6.], [0., 21., 4.], [0., 0., 1.]])
    calibration = RawCameraCalibration(
        model="pinhole", translation_xyz=(0., 0., 0.),
        rotation_wxyz=(1., 0., 0., 0.),
        intrinsics=tuple(tuple(row) for row in intrinsics),
        distortion=(0.,) * 5, image_size_hw=size,
    )
    plan = NuPlanRasterPlan(
        source_size_hw=size, output_size_hw=size, output_intrinsics=intrinsics,
        map_x=np.zeros(size, np.float32), map_y=np.zeros(size, np.float32),
        valid_mask=np.ones(size, np.bool_),
    )
    measured = (
        ((650_000., 4_100_000., 20.),
         (0.10940056490727669, 0.8147623423285729,
          0.3011754438512965, -0.4832051261547221)),
        ((650_003., 4_100_002., 20.5), (1., 0., 0., 0.)),
    )
    frames = []
    for index, (translation, rotation) in enumerate(measured, 1):
        token = f"{index:016x}"
        pose = RawEgoPose(index * 100, *translation, *rotation, 32611)
        frame = EuvsFrameInput(
            FrameRef(
                image_path=Path(f"/dataset/{token}.jpg"),
                db_path=Path("/dataset/source.db"), timestamp_us=index * 100,
                channel="CAM_F0", image_token=token,
                ego_pose_token=f"{index + 100:016x}", camera_token="0000000000000200",
            ),
            pose, calibration, build_global_camera_geometry(pose, calibration),
        )
        frames.append(EuvsRasterizedFrame(
            frame, RasterizedRgb(plan, np.full((*size, 3), index - 1, np.uint8)),
        ))
    source = EuvsRasterizedSequence(
        EuvsFrameSequence(tuple(frame.input for frame in frames), "2"), tuple(frames),
    )
    first, second = (frame.input.geometry for frame in source.frames)
    assert np.max(np.abs(first.global_to_camera @ first.camera_to_global - np.eye(4))) > 2e-9

    result = build_source_geometry_input(source)

    np.testing.assert_array_equal(result.reference_to_camera[0], np.eye(4))
    np.testing.assert_array_equal(result.intrinsics[0], intrinsics)
    second_rotation = second.global_to_camera[:3, :3]
    np.testing.assert_allclose(
        result.reference_to_camera[1, :3, :3],
        second_rotation @ first.global_to_camera[:3, :3].T, rtol=0, atol=1e-12,
    )
    # Recovering a global centre from W2C retains nanometre UTM roundoff.
    np.testing.assert_allclose(
        result.reference_to_camera[1, :3, 3],
        second_rotation @ np.array([-3., -2., -0.5]), rtol=0, atol=1e-8,
    )

"""Independent numeric camera conventions, rigid transforms and trajectories."""

from dataclasses import replace
from math import sqrt

import numpy as np
from numpy.testing import assert_allclose as close, assert_array_equal as equal
import pytest

from novel_view.geometry.camera import (
    camera_axes_in_world, camera_centre, invert_rigid,
    quaternion_wxyz_to_rotation, rigid_transform,
)
from novel_view.geometry.trajectory import (
    CameraInterpolationError, interpolate_w2c_shortest,
    translation_quaternion_xyzw_to_w2c, w2c_to_translation_quaternion_xyzw,
)
from novel_view.inputs.nuplan.camera import (
    GlobalCameraGeometry, build_global_camera_geometry,
    build_reference_to_camera_transforms,
)
from novel_view.inputs.nuplan.db import RawCameraCalibration, RawEgoPose


def _rz_input(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _w2c_input(centre, angle):
    result = np.eye(4)
    result[:3, :3] = _rz_input(angle).T
    result[:3, 3] = -result[:3, :3] @ centre
    return result


def test_nuplan_camera_composition_and_rigid_semantics():
    q = sqrt(0.5)
    pose = RawEgoPose(1000, 10.0, 0.0, 0.0, q, 0.0, 0.0, q, 32611)
    calibration = RawCameraCalibration(
        "pinhole", (2.0, 0.0, 0.0), (q, q, 0.0, 0.0),
        ((1000.0, 0.0, 960.0), (0.0, 1000.0, 560.0), (0.0, 0.0, 1.0)),
        (0.0,) * 5, (1080, 1920),
    )
    geometry = build_global_camera_geometry(pose, calibration)
    # Rz(90) @ Rx(90), not the reverse; the ego rotation also moves the lever arm.
    c2w = np.array([
        [0, 0, 1, 10], [1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 0, 1],
    ])
    w2c = np.array([
        [0, 1, 0, -2], [0, 0, 1, 0], [1, 0, 0, -10], [0, 0, 0, 1],
    ])
    close(geometry.camera_to_global, c2w, rtol=0, atol=2e-12)
    close(geometry.global_to_camera, w2c, rtol=0, atol=2e-12)
    close(rigid_transform(c2w[:3, :3], c2w[:3, 3]), c2w, rtol=0, atol=0)
    close(invert_rigid(w2c), c2w, rtol=0, atol=0)
    equal(camera_centre(w2c), (10, 2, 0))
    close(geometry.camera_center_global, (10, 2, 0), rtol=0, atol=2e-12)
    axes = np.column_stack(camera_axes_in_world(w2c))
    equal(axes, c2w[:3, :3])
    equal(np.cross(axes[:, 0], axes[:, 1]), axes[:, 2])
    actual_axes = np.column_stack((
        geometry.camera_right_global, geometry.camera_down_global,
        geometry.camera_forward_global,
    ))
    close(actual_axes, axes, rtol=0, atol=2e-12)
    points = np.vstack((np.array([[10], [2], [0]]) + 7 * axes, np.ones(3)))
    close(
        geometry.global_to_camera @ points,
        np.vstack((7 * np.eye(3), np.ones(3))), rtol=0, atol=3e-12,
    )


@pytest.mark.parametrize("scale", [1.0, -1.0, 2.0, 1e300, 1e-300])
def test_quaternion_wxyz_scale_sign_and_components(scale):
    # Literal normalized matrix for wxyz=(.7,.1,-.2,.3); not another converter.
    expected = np.array([
        [.37, -.46, -.22], [.38, .43, -.26], [.34, .02, .53],
    ]) / .63
    close(
        quaternion_wxyz_to_rotation(scale * np.array([.7, .1, -.2, .3])),
        expected, rtol=0, atol=2e-12,
    )
    close(quaternion_wxyz_to_rotation((0, 1, 0, 0)),
          np.diag([1, -1, -1]), rtol=0, atol=1e-12)


def test_relative_w2c_is_stable_and_accepts_equal_reference_copy():
    centre = np.array([664425.123456789, 3997674.987654321, 612.125])
    other_centre = centre + [.003, -.002, .0005]
    reference = GlobalCameraGeometry(_w2c_input(centre, -17), 32611)
    other = GlobalCameraGeometry(_w2c_input(other_centre, 37), 32611)
    same = replace(reference, global_to_camera=reference.global_to_camera.copy())
    result = build_reference_to_camera_transforms(reference, (other, same))
    expected = np.eye(4)
    expected[:3, :3] = _rz_input(37).T @ _rz_input(-17)
    expected[:3, 3] = _rz_input(37).T @ (centre - other_centre)
    close(result[0], expected, rtol=0, atol=1e-9)
    point = np.array([2.0, -1.0, 5.0, 1.0])
    close(
        result[0] @ point,
        other.global_to_camera @ reference.camera_to_global @ point,
        rtol=0, atol=3e-9,
    )
    equal(result[1], np.eye(4))


@pytest.mark.parametrize("left,right,middle", [
    (170, -170, 180), (0, 0, 0), (0, 0.00001, 0.000005),
])
def test_shortest_so3_interpolates_centres(left, right, middle):
    result = interpolate_w2c_shortest(
        _w2c_input((0, 0, 0), left), _w2c_input((2, -4, 6), right), 0.5,
    )
    close(camera_centre(result), (1, -2, 3), rtol=0, atol=1e-12)
    close(result[:3, :3], _rz_input(middle).T, rtol=0, atol=1e-12)


def test_so3_pi_is_ambiguous():
    with pytest.raises(CameraInterpolationError):
        interpolate_w2c_shortest(
            _w2c_input((0, 0, 0), 0), _w2c_input((1, 0, 0), 180), .5,
        )


@pytest.mark.parametrize("quaternion,rotation", [
    ((.2, -.3, .4, .5), np.array([
        [.04, -.52, -.14], [.28, .14, -.44], [.46, -.04, .28],
    ]) / .54),
    ((1, 0, 0, 0), np.diag([1, -1, -1])),
    ((0, 1, 0, 0), np.diag([-1, 1, -1])),
    ((0, 0, 1, 0), np.diag([-1, -1, 1])),
])
def test_reconstruction_xyzw_encodes_measured_w2c(quaternion, rotation):
    expected = np.eye(4)
    expected[:3, :3], expected[:3, 3] = rotation, (1.5, -2, .25)
    original = translation_quaternion_xyzw_to_w2c((1.5, -2, .25), quaternion)
    close(original, expected, rtol=0, atol=1e-12)
    translation, encoded = w2c_to_translation_quaternion_xyzw(expected)
    equal(translation, (1.5, -2, .25))
    assert encoded[3] >= 0
    close(
        translation_quaternion_xyzw_to_w2c(translation, encoded),
        expected, rtol=0, atol=1e-12,
    )

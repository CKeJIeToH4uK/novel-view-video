"""Construct role-neutral camera geometry from raw nuPlan metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from novel_view.geometry.camera import (
    CameraGeometryError,
    camera_axes_in_world,
    camera_centre,
    invert_rigid,
    quaternion_wxyz_to_rotation,
    relative_world_to_camera_transforms,
    rigid_transform,
)
from novel_view.inputs.nuplan.db import RawCameraCalibration, RawEgoPose


@dataclass(frozen=True, slots=True, eq=False)
class GlobalCameraGeometry:
    """A camera pose expressed in one projected global coordinate system.

    ``global_to_camera`` follows the OpenCV camera convention and acts on
    homogeneous column vectors. Camera coordinates use ``+x`` right, ``+y``
    down, and ``+z`` forward.

    Attributes:
        global_to_camera: Rigid ``float64`` transform from global coordinates
            to camera coordinates, with shape ``(4, 4)``.
        epsg: Positive EPSG code of the global coordinate system.
    """

    global_to_camera: npt.NDArray[np.float64]
    epsg: int

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Take immutable ownership of the transform built by this module."""
        object.__setattr__(
            self,
            "global_to_camera",
            _readonly_copy(self.global_to_camera),
        )

    def __eq__(self, other: object) -> bool:
        """Compare exact transform values and coordinate systems."""
        if not isinstance(other, GlobalCameraGeometry):
            return NotImplemented
        return self.epsg == other.epsg and bool(
            np.array_equal(self.global_to_camera, other.global_to_camera)
        )

    @property
    def camera_to_global(self) -> npt.NDArray[np.float64]:
        """Return the analytically inverted camera-to-global transform."""
        return _readonly_copy(invert_rigid(self.global_to_camera))

    @property
    def camera_center_global(self) -> npt.NDArray[np.float64]:
        """Return the camera optical centre in global coordinates."""
        return _readonly_copy(camera_centre(self.global_to_camera))

    @property
    def camera_right_global(self) -> npt.NDArray[np.float64]:
        """Return the global unit vector along camera ``+x`` (right)."""
        right, _, _ = camera_axes_in_world(self.global_to_camera)
        return _readonly_copy(right)

    @property
    def camera_down_global(self) -> npt.NDArray[np.float64]:
        """Return the global unit vector along camera ``+y`` (down)."""
        _, down, _ = camera_axes_in_world(self.global_to_camera)
        return _readonly_copy(down)

    @property
    def camera_forward_global(self) -> npt.NDArray[np.float64]:
        """Return the global unit vector along camera ``+z`` (forward)."""
        _, _, forward = camera_axes_in_world(self.global_to_camera)
        return _readonly_copy(forward)


def build_global_camera_geometry(
    pose: RawEgoPose,
    calibration: RawCameraCalibration,
) -> GlobalCameraGeometry:
    """Compose linked nuPlan ego pose and camera calibration.

    nuPlan stores the ego-to-global pose and camera-to-ego calibration. With
    ``T_A_B`` denoting a transform from frame ``B`` to frame ``A``, this
    function computes::

        T_global_camera = T_global_ego @ T_ego_camera
        T_camera_global = inverse_rigid(T_global_camera)

    Args:
        pose: Raw linked ego pose for one frame.
        calibration: Raw calibration for that frame's camera token.

    Returns:
        Validated role-neutral global camera geometry.

    Raises:
        CameraGeometryError: A translation or quaternion is invalid, or the
            composed transform violates the output contract.
    """
    global_ego_translation = np.asarray((pose.x, pose.y, pose.z), dtype=np.float64)
    ego_camera_translation = np.asarray(
        calibration.translation_xyz, dtype=np.float64
    )
    global_ego_rotation = quaternion_wxyz_to_rotation(
        (pose.qw, pose.qx, pose.qy, pose.qz)
    )
    ego_camera_rotation = quaternion_wxyz_to_rotation(
        calibration.rotation_wxyz
    )

    global_camera_rotation = global_ego_rotation @ ego_camera_rotation
    global_camera_translation = (
        global_ego_translation
        + global_ego_rotation @ ego_camera_translation
    )
    global_to_camera = invert_rigid(
        rigid_transform(
            global_camera_rotation,
            global_camera_translation,
        )
    )
    return GlobalCameraGeometry(global_to_camera, pose.epsg)


def build_reference_to_camera_transforms(
    reference: GlobalCameraGeometry,
    cameras: Sequence[GlobalCameraGeometry],
) -> npt.NDArray[np.float64]:
    """Express ordered global cameras relative to one reference camera.

    For a global-to-camera rotation ``R_i`` and global camera centre ``C_i``,
    the transform from the reference camera frame to camera ``i`` is:

    ``R_i_reference = R_i @ R_reference.T``
    ``t_i_reference = R_i @ (C_reference - C_i)``

    Camera centres are subtracted before applying a rotation. This avoids
    cancelling two homogeneous transforms whose projected global
    translations can be millions of metres.

    Args:
        reference: Camera whose coordinate frame becomes the common reference.
        cameras: Cameras to express in caller-provided order.

    Returns:
        Read-only contiguous ``float64 [N,4,4]`` OpenCV W2C transforms.
        A camera exactly equal to ``reference`` receives an exact identity.

    Raises:
        CameraGeometryError: An input is not a validated camera or uses a
            different EPSG coordinate system.
    """
    camera_sequence = tuple(cameras)
    for index, camera in enumerate(camera_sequence):
        if camera.epsg != reference.epsg:
            raise CameraGeometryError(
                f"cameras[{index}] has EPSG {camera.epsg}, "
                f"expected {reference.epsg}"
            )

    camera_matrices = (
        np.stack([camera.global_to_camera for camera in camera_sequence])
        if camera_sequence
        else np.empty((0, 4, 4), dtype=np.float64)
    )
    transforms = relative_world_to_camera_transforms(
        reference.global_to_camera,
        camera_matrices,
    )
    return _readonly_copy(transforms)


def _readonly_copy(value: object) -> npt.NDArray[np.float64]:
    """Return a C-contiguous ``float64`` view of an immutable bytes buffer."""
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    immutable_buffer = array.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=np.float64).reshape(array.shape)

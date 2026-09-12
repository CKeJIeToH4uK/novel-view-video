"""Чистые преобразования rigid-камер в соглашении OpenCV W2C."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class CameraGeometryError(ValueError):
    """Камерное преобразование математически не определено."""


def rigid_transform(
    rotation: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Собрать однородное rigid-преобразование из вращения и переноса."""
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def invert_rigid(
    transform: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Аналитически обратить rigid-преобразование."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rigid_transform(rotation.T, -rotation.T @ translation)


def camera_centre(
    world_to_camera: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Вернуть оптический центр W2C-камеры в исходной системе."""
    return -world_to_camera[:3, :3].T @ world_to_camera[:3, 3]


def camera_axes_in_world(
    world_to_camera: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Вернуть глобальные OpenCV-оси камеры: вправо, вниз и вперёд."""
    rotation = world_to_camera[:3, :3]
    return (
        np.array(rotation[0], copy=True),
        np.array(rotation[1], copy=True),
        np.array(rotation[2], copy=True),
    )


def quaternion_wxyz_to_rotation(
    quaternion_wxyz: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Преобразовать конечный ненулевой quaternion ``wxyz`` во вращение."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise CameraGeometryError(
            "quaternion_wxyz must contain four finite values"
        )
    scale = float(np.max(np.abs(quaternion)))
    if scale == 0.0:
        raise CameraGeometryError("quaternion_wxyz must be non-zero")
    unit = quaternion / scale
    unit /= np.linalg.norm(unit)
    w, x, y, z = unit
    return np.array(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def relative_world_to_camera_transforms(
    reference_world_to_camera: npt.NDArray[np.float64],
    world_to_camera: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Выразить упорядоченные глобальные W2C относительно одной камеры."""
    cameras = np.asarray(world_to_camera, dtype=np.float64)
    reference_rotation = reference_world_to_camera[:3, :3]
    reference_center = camera_centre(reference_world_to_camera)
    rotations = cameras[:, :3, :3]
    centers = -np.einsum(
        "nji,nj->ni",
        rotations,
        cameras[:, :3, 3],
    )

    result = np.repeat(
        np.eye(4, dtype=np.float64)[None],
        cameras.shape[0],
        axis=0,
    )
    result[:, :3, :3] = rotations @ reference_rotation.T
    result[:, :3, 3] = np.einsum(
        "nij,nj->ni",
        rotations,
        reference_center - centers,
    )
    exact_reference = np.all(
        cameras == reference_world_to_camera,
        axis=(1, 2),
    )
    result[exact_reference] = np.eye(4, dtype=np.float64)
    return result

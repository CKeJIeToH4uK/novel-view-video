"""Rigid W2C-интерполяция и кодирование reconstruction camera table."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from novel_view.geometry.camera import camera_centre

_PI_AMBIGUITY_ATOL_RAD = 1e-7
_SO3_SERIES_THRESHOLD_RAD = 1e-4


class CameraInterpolationError(ValueError):
    """Две камеры не имеют единственной кратчайшей SO(3)-дуги."""


def interpolate_w2c_shortest(
    left_w2c: npt.NDArray[np.float64],
    right_w2c: npt.NDArray[np.float64],
    alpha: float,
) -> npt.NDArray[np.float64]:
    """Линейно интерполировать центры и кратчайшие C2W-вращения."""
    left_c2w = left_w2c[:3, :3].T
    right_c2w = right_w2c[:3, :3].T
    rotation_vector = _so3_log(left_c2w.T @ right_c2w)
    if rotation_vector is None:
        raise CameraInterpolationError(
            "shortest camera rotation is ambiguous at pi"
        )
    interpolated_c2w = left_c2w @ _so3_exp(rotation_vector * alpha)
    centre = (
        (1.0 - alpha) * camera_centre(left_w2c)
        + alpha * camera_centre(right_w2c)
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = interpolated_c2w.T
    result[:3, 3] = -result[:3, :3] @ centre
    return result


def translation_quaternion_xyzw_to_w2c(
    translation: npt.ArrayLike,
    quaternion_xyzw: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Декодировать rigid-преобразование reconstruction JSON."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion_xyzw must contain four finite values")
    x, y, z, w = quaternion / norm
    rotation = np.array(
        (
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ),
            (
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ),
            (
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    if result[:3, 3].shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("translation must contain three finite values")
    return result


def w2c_to_translation_quaternion_xyzw(
    w2c: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Закодировать rigid W2C в reconstruction JSON camera schema."""
    matrix = np.asarray(w2c, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("w2c must be finite 4x4")
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        first, second = ((1, 2), (2, 0), (0, 1))[index]
        scale = 2.0 * np.sqrt(
            max(
                0.0,
                1.0
                + rotation[index, index]
                - rotation[first, first]
                - rotation[second, second],
            )
        )
        if scale == 0.0:
            raise ValueError("w2c rotation cannot be encoded as a quaternion")
        quaternion = np.empty(4, dtype=np.float64)
        quaternion[index] = 0.25 * scale
        quaternion[first] = (
            rotation[first, index] + rotation[index, first]
        ) / scale
        quaternion[second] = (
            rotation[second, index] + rotation[index, second]
        ) / scale
        quaternion[3] = (
            rotation[second, first] - rotation[first, second]
        ) / scale
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return np.array(matrix[:3, 3], copy=True), quaternion


def _so3_log(
    rotation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | None:
    """Вернуть устойчивый главный rotation vector или ``None`` при pi."""
    skew_vector = 0.5 * np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ),
        dtype=np.float64,
    )
    sine = float(np.linalg.norm(skew_vector))
    cosine = float(
        np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    )
    angle = float(np.arctan2(sine, cosine))
    if np.pi - angle <= _PI_AMBIGUITY_ATOL_RAD:
        return None
    if angle < _SO3_SERIES_THRESHOLD_RAD:
        angle_squared = angle * angle
        scale = (
            1.0
            + angle_squared / 6.0
            + 7.0 * angle_squared**2 / 360.0
        )
    else:
        scale = angle / sine
    return skew_vector * scale


def _so3_exp(
    rotation_vector: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Перевести rotation vector в SO(3), не теряя малые повороты."""
    angle_squared = float(rotation_vector @ rotation_vector)
    angle = float(np.sqrt(angle_squared))
    if angle < _SO3_SERIES_THRESHOLD_RAD:
        sinc = 1.0 - angle_squared / 6.0 + angle_squared**2 / 120.0
        cosc = 0.5 - angle_squared / 24.0 + angle_squared**2 / 720.0
    else:
        sinc = float(np.sin(angle) / angle)
        cosc = float((1.0 - np.cos(angle)) / angle_squared)
    x, y, z = rotation_vector
    cross = np.array(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + sinc * cross
        + cosc * (cross @ cross)
    )

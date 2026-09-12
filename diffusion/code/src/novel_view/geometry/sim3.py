"""Orientation-first Sim(3) alignment of two OpenCV camera packs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt


class CameraPackAlignmentError(ValueError):
    """The camera packs do not define one positive metric similarity."""


@dataclass(frozen=True, slots=True, eq=False)
class CameraPackAlignment:
    """One orientation-first Sim(3) fit of predicted to measured cameras.

    The returned transform follows
    ``X_world = scale * rotation @ X_model + translation``. Both inputs are
    OpenCV world-to-camera packs in their respective worlds. Model rotations
    are projected onto SO(3) before fitting; the projection residual is
    reported as a diagnostic and never used as a quality threshold.
    """

    scale_m_per_model_unit: float
    model_to_world_rotation: npt.NDArray[np.float64]
    model_to_world_translation_m: npt.NDArray[np.float64]
    aligned_world_to_camera: npt.NDArray[np.float64]
    center_residual_m: npt.NDArray[np.float64]
    rotation_residual_deg: npt.NDArray[np.float64]
    model_rotation_projection_residual_fro: npt.NDArray[np.float64]

    __hash__: ClassVar[None] = None


def align_camera_packs(
    model_world_to_camera: npt.NDArray[np.float32],
    measured_world_to_camera: npt.NDArray[np.float64],
) -> CameraPackAlignment:
    """Fit one positive Sim(3), giving camera orientation priority."""
    model_rotations, model_centers, projection_residual = (
        _canonical_model_cameras(model_world_to_camera)
    )
    measured_rotations = measured_world_to_camera[:, :3, :3]
    measured_centers = _camera_centers(
        measured_rotations,
        measured_world_to_camera[:, :3, 3],
    )
    relative_rotations = measured_rotations.transpose(0, 2, 1) @ model_rotations
    rotation = _unique_chordal_mean(relative_rotations)
    scale, translation = _fit_scale_and_translation(
        model_centers,
        measured_centers,
        rotation,
    )
    aligned_world_to_camera, aligned_centers = _align_model_cameras(
        model_rotations,
        model_centers,
        scale,
        rotation,
        translation,
    )
    return CameraPackAlignment(
        scale_m_per_model_unit=scale,
        model_to_world_rotation=_read_only(rotation),
        model_to_world_translation_m=_read_only(translation),
        aligned_world_to_camera=_read_only(aligned_world_to_camera),
        center_residual_m=_read_only(
            np.linalg.norm(aligned_centers - measured_centers, axis=1)
        ),
        rotation_residual_deg=_read_only(
            _rotation_residuals_deg(
                aligned_world_to_camera[:, :3, :3],
                measured_rotations,
            )
        ),
        model_rotation_projection_residual_fro=_read_only(projection_residual),
    )


def _canonical_model_cameras(
    model_world_to_camera: npt.NDArray[np.float32],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    rotations = np.empty((len(model_world_to_camera), 3, 3), dtype=np.float64)
    residuals = np.empty(len(model_world_to_camera), dtype=np.float64)
    for index, raw_rotation in enumerate(model_world_to_camera[:, :3, :3]):
        raw_float64 = raw_rotation.astype(np.float64)
        determinant = float(np.linalg.det(raw_float64))
        if not np.isfinite(determinant) or determinant <= 0.0:
            raise CameraPackAlignmentError(
                "model camera rotations must preserve orientation"
            )
        rotations[index] = _project_to_so3(raw_float64)
        residuals[index] = np.linalg.norm(
            rotations[index] - raw_float64,
            ord="fro",
        )
    translations = model_world_to_camera[:, :3, 3].astype(np.float64)
    return rotations, _camera_centers(rotations, translations), residuals


def _project_to_so3(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    left, _, right_transpose = np.linalg.svd(matrix)
    if np.linalg.det(left @ right_transpose) < 0.0:
        left[:, -1] *= -1.0
    return left @ right_transpose


def _camera_centers(
    rotations: npt.NDArray[np.float64],
    translations: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return -np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), translations)


def _unique_chordal_mean(
    rotations: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    matrix = np.sum(rotations, axis=0)
    left, singular_values, right_transpose = np.linalg.svd(matrix)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * len(rotations)
    if singular_values[1] <= tolerance:
        raise CameraPackAlignmentError(
            "relative rotations have no unique chordal mean"
        )
    orientation = -1.0 if np.linalg.det(left @ right_transpose) < 0.0 else 1.0
    if orientation < 0.0 and singular_values[1] - singular_values[2] <= tolerance:
        raise CameraPackAlignmentError(
            "relative rotations have no unique proper chordal mean"
        )
    left[:, -1] *= orientation
    return left @ right_transpose


def _fit_scale_and_translation(
    model_centers: npt.NDArray[np.float64],
    measured_centers: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
) -> tuple[float, npt.NDArray[np.float64]]:
    model_mean = np.mean(model_centers, axis=0)
    measured_mean = np.mean(measured_centers, axis=0)
    model_centered = model_centers - model_mean
    measured_centered = measured_centers - measured_mean
    rotated_model = (rotation @ model_centered.T).T
    denominator = float(np.sum(model_centered * model_centered))
    if denominator <= 0.0:
        raise CameraPackAlignmentError(
            "model camera translation variance must be positive"
        )
    scale = float(np.sum(measured_centered * rotated_model)) / denominator
    if not np.isfinite(scale) or scale <= 0.0:
        raise CameraPackAlignmentError(
            "camera alignment scale must be finite and positive"
        )
    translation = measured_mean - scale * (rotation @ model_mean)
    return scale, translation


def _align_model_cameras(
    model_rotations: npt.NDArray[np.float64],
    model_centers: npt.NDArray[np.float64],
    scale: float,
    rotation: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    aligned_rotations = model_rotations @ rotation.T
    aligned_centers = scale * (rotation @ model_centers.T).T + translation
    aligned = np.repeat(
        np.eye(4, dtype=np.float64)[None],
        len(model_rotations),
        axis=0,
    )
    aligned[:, :3, :3] = aligned_rotations
    aligned[:, :3, 3] = -np.einsum(
        "nij,nj->ni",
        aligned_rotations,
        aligned_centers,
    )
    return aligned, aligned_centers


def _rotation_residuals_deg(
    aligned_rotations: npt.NDArray[np.float64],
    measured_rotations: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    relative = aligned_rotations @ measured_rotations.transpose(0, 2, 1)
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _read_only(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result

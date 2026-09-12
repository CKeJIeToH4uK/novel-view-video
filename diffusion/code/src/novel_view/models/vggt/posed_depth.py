"""Convert one aligned raw VGGT prediction to common metric depth."""

from __future__ import annotations

from typing import TypeVar, cast

import cv2
import numpy as np
import numpy.typing as npt

from novel_view.geometry.depth import PosedDepthSequence
from novel_view.geometry.sim3 import CameraPackAlignment
from novel_view.models.vggt.request import VggtOmegaRawPrediction

_ArrayScalar = TypeVar("_ArrayScalar", bound=np.generic)


class VggtOmegaPosedDepthError(ValueError):
    """An aligned VGGT prediction cannot form coherent metric depth."""


def build_vggt_posed_depth(
    prediction: VggtOmegaRawPrediction,
    alignment: CameraPackAlignment,
    source_valid: npt.NDArray[np.bool_],
    sequence_id: tuple[str, ...],
) -> PosedDepthSequence:
    """Transfer raw depth/K to the source grid and apply one Sim(3) scale."""
    source_size_hw = (int(source_valid.shape[1]), int(source_valid.shape[2]))
    depth, confidence = _transfer_dense_maps(
        prediction,
        alignment.scale_m_per_model_unit,
        source_size_hw,
    )
    intrinsics = np.linalg.solve(
        prediction.input_plan.source_to_model_pixels,
        prediction.model_intrinsics.astype(np.float64),
    )
    if not np.all(np.isfinite(intrinsics)):
        raise VggtOmegaPosedDepthError(
            "source-grid predicted intrinsics must remain finite"
        )

    valid = np.array(source_valid, dtype=np.bool_, order="C", copy=True)
    valid &= np.isfinite(depth) & (depth > 0.0)
    depth[~valid] = 0.0
    confidence[~valid] = 0.0
    return PosedDepthSequence(
        sequence_id=sequence_id,
        depth_z_m=_readonly_copy(depth),
        geometry_valid_mask=_readonly_copy(valid),
        intrinsics=_readonly_copy(intrinsics),
        reference_to_camera=_readonly_copy(alignment.aligned_world_to_camera),
        backend_confidence=_readonly_copy(confidence),
    )


def _transfer_dense_maps(
    prediction: VggtOmegaRawPrediction,
    scale_m_per_model_unit: float,
    source_size_hw: tuple[int, int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Transfer depth and confidence once, then convert depth to metres."""
    frame_count = len(prediction.depth_model_units)
    output_shape = (frame_count, *source_size_hw)
    depth = np.empty(output_shape, dtype=np.float32)
    confidence = np.empty(output_shape, dtype=np.float32)
    same_grid = prediction.input_plan.model_size_hw == source_size_hw
    for index in range(frame_count):
        if same_grid:
            depth[index] = prediction.depth_model_units[index]
            confidence[index] = prediction.depth_confidence[index]
        else:
            depth[index] = _resize(
                prediction.depth_model_units[index], source_size_hw
            )
            confidence[index] = _resize(
                prediction.depth_confidence[index], source_size_hw
            )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        np.multiply(depth, np.float32(scale_m_per_model_unit), out=depth)
    if not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
        raise VggtOmegaPosedDepthError(
            "metric VGGT depth must remain finite and positive after scaling"
        )
    return depth, confidence


def _resize(
    values: npt.NDArray[np.float32],
    output_size_hw: tuple[int, int],
) -> npt.NDArray[np.float32]:
    height, width = output_size_hw
    return cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)


def _readonly_copy(
    value: npt.NDArray[_ArrayScalar],
) -> npt.NDArray[_ArrayScalar]:
    owned = np.frombuffer(
        np.ascontiguousarray(value).tobytes(),
        dtype=value.dtype,
    ).reshape(value.shape)
    return cast(npt.NDArray[_ArrayScalar], owned)


__all__ = [
    "VggtOmegaPosedDepthError",
    "build_vggt_posed_depth",
]

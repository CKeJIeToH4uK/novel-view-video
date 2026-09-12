"""Project source dynamic labels through metric camera-Z depth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

SOURCE_DYNAMIC_PROJECTION_PROTOCOL = "four-neighbour-zbuffer/v1"


class SourceProjectionError(ValueError):
    """Source depth cannot define a target-grid dynamic projection."""


@dataclass(frozen=True, slots=True, eq=False)
class SourceDynamicProjection:
    """Visible source coverage and dynamic labels on one target grid."""

    coverage_mask: npt.NDArray[np.bool_]
    visible_label_mask: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        coverage = _readonly(self.coverage_mask)
        visible = _readonly(self.visible_label_mask)
        if np.any(visible & ~coverage):
            raise SourceProjectionError("visible dynamic labels require coverage")
        object.__setattr__(self, "coverage_mask", coverage)
        object.__setattr__(self, "visible_label_mask", visible)


def project_source_dynamic(
    depth_z_m: npt.NDArray[np.float32],
    geometry_valid_mask: npt.NDArray[np.bool_],
    source_dynamic_mask: npt.NDArray[np.bool_],
    source_intrinsics: npt.NDArray[np.float64],
    reference_to_source_camera: npt.NDArray[np.float64],
    target_intrinsics: npt.NDArray[np.float64],
    reference_to_target_camera: npt.NDArray[np.float64],
    target_size_hw: tuple[int, int],
) -> SourceDynamicProjection:
    """Use four floor/ceil neighbours, nearest z and OR only at exact ties."""
    if np.any(
        geometry_valid_mask & (~np.isfinite(depth_z_m) | (depth_z_m <= 0.0))
    ):
        raise SourceProjectionError("valid source depth must be finite and positive")
    height, width = target_size_hw
    empty = np.zeros((height, width), dtype=np.bool_)
    source_y, source_x = np.nonzero(geometry_valid_mask)
    if source_x.size == 0:
        return SourceDynamicProjection(empty, empty)

    pixels = np.stack(
        (
            source_x.astype(np.float64),
            source_y.astype(np.float64),
            np.ones(source_x.size, dtype=np.float64),
        )
    )
    source_points = np.linalg.solve(source_intrinsics, pixels)
    source_points *= depth_z_m[source_y, source_x].astype(np.float64)
    source_rotation = reference_to_source_camera[:3, :3]
    target_rotation = reference_to_target_camera[:3, :3]
    rotation = target_rotation @ source_rotation.T
    translation = reference_to_target_camera[:3, 3] - (
        rotation @ reference_to_source_camera[:3, 3]
    )
    target_points = rotation @ source_points + translation[:, None]
    target_z = target_points[2]
    projected = target_intrinsics @ target_points
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        target_x = projected[0] / target_z
        target_y = projected[1] / target_z
    inside = (
        (target_z > 0.0)
        & np.isfinite(target_z)
        & np.isfinite(target_x)
        & np.isfinite(target_y)
        & (target_x >= 0.0)
        & (target_x <= width - 1)
        & (target_y >= 0.0)
        & (target_y <= height - 1)
    )
    if not np.any(inside):
        return SourceDynamicProjection(empty, empty)
    return _splat(
        target_x[inside],
        target_y[inside],
        target_z[inside],
        source_dynamic_mask[source_y, source_x][inside],
        target_size_hw,
    )


def _splat(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    dynamic: npt.NDArray[np.bool_],
    size_hw: tuple[int, int],
) -> SourceDynamicProjection:
    height, width = size_hw
    x0, x1 = np.floor(x).astype(np.int64), np.ceil(x).astype(np.int64)
    y0, y1 = np.floor(y).astype(np.int64), np.ceil(y).astype(np.int64)
    selected = np.ones(x.shape, dtype=np.bool_)
    contributions = (
        (x0, y0, selected),
        (x1, y0, x1 != x0),
        (x0, y1, y1 != y0),
        (x1, y1, (x1 != x0) & (y1 != y0)),
    )
    minimum_z = np.full(height * width, np.inf, dtype=np.float64)
    for target_x, target_y, keep in contributions:
        pixel = target_y[keep] * width + target_x[keep]
        np.minimum.at(minimum_z, pixel, z[keep])
    visible = np.zeros(height * width, dtype=np.bool_)
    for target_x, target_y, keep in contributions:
        pixel = target_y[keep] * width + target_x[keep]
        winners = z[keep] == minimum_z[pixel]
        np.logical_or.at(visible, pixel[winners], dynamic[keep][winners])
    coverage = np.isfinite(minimum_z).reshape(size_hw)
    return SourceDynamicProjection(coverage, visible.reshape(size_hw))


def _readonly(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    return np.frombuffer(mask.tobytes(order="C"), dtype=np.bool_).reshape(mask.shape)


__all__ = [
    "SOURCE_DYNAMIC_PROJECTION_PROTOCOL",
    "SourceDynamicProjection",
    "SourceProjectionError",
    "project_source_dynamic",
]

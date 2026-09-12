"""Decode stored nuPlan images onto a rectified pinhole raster."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import numpy.typing as npt

from novel_view.geometry.raster import CoverCropTransform
from novel_view.inputs.nuplan.db import RawCameraCalibration

_IMREAD_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


class NuPlanRasterError(ValueError):
    """A calibration or derived OpenCV raster plan violates its contract."""


@dataclass(frozen=True, slots=True, eq=False)
class NuPlanRasterPlan:
    """A final pinhole grid and inverse map into one distorted source raster.

    ``map_x`` and ``map_y`` follow the OpenCV inverse-remap convention:
    output pixel ``(x, y)`` samples the distorted source at
    ``(map_x[y, x], map_y[y, x])``. The maps combine lens undistortion with
    the chosen final virtual pinhole grid. The builder decides how that grid
    is derived; the current native-``K`` builder uses resize-to-cover and a
    central crop.

    ``valid_mask`` is tied to bilinear sampling with a zero constant border.
    A true element means that remapping a unit-valued source raster with
    ``cv2.INTER_LINEAR`` receives no contribution from outside that raster.

    Normal pipeline code should obtain instances from
    :func:`build_nuplan_raster_plan`. Direct construction only takes
    immutable ownership of the supplied arrays; their internal numeric
    contract is trusted.

    Attributes:
        source_size_hw: Distorted source raster size as ``(height, width)``.
        output_size_hw: Final pinhole raster size as ``(height, width)``.
        output_intrinsics: Final read-only ``float64`` pinhole matrix.
        map_x: Read-only ``float32`` source-column map with output shape.
        map_y: Read-only ``float32`` source-row map with output shape.
        valid_mask: Read-only boolean bilinear-support mask with output shape.
    """

    source_size_hw: tuple[int, int]
    output_size_hw: tuple[int, int]
    output_intrinsics: npt.NDArray[np.float64]
    map_x: npt.NDArray[np.float32]
    map_y: npt.NDArray[np.float32]
    valid_mask: npt.NDArray[np.bool_]

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Take immutable ownership of the arrays created by OpenCV."""
        object.__setattr__(
            self,
            "output_intrinsics",
            _readonly_copy(self.output_intrinsics, np.float64),
        )
        object.__setattr__(self, "map_x", _readonly_copy(self.map_x, np.float32))
        object.__setattr__(self, "map_y", _readonly_copy(self.map_y, np.float32))
        object.__setattr__(
            self, "valid_mask", _readonly_copy(self.valid_mask, np.bool_)
        )


def build_nuplan_output_intrinsics(
    calibration: RawCameraCalibration,
    output_size_hw: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Express one measured nuPlan pinhole camera on an output grid.

    This is the lightweight camera-only half of
    :func:`build_nuplan_raster_plan`. Both public builders use the same
    validated native-``K`` and cover-crop calculation, so the returned matrix
    is exactly the one that a complete rectification plan would use. Lens
    distortion does not change this virtual output camera and is therefore
    neither interpreted nor used to build OpenCV maps here.

    Args:
        calibration: Raw nuPlan intrinsics and native image size. Camera pose,
            hardware label, and distortion coefficients are not used.
        output_size_hw: Final pinhole raster size as ``(height, width)``.

    Returns:
        Read-only C-contiguous ``float64 [3,3]`` output intrinsics.

    Raises:
        NuPlanRasterError: The calibration, native intrinsics, requested size,
            or source skew violates the supported raster contract.
    """
    _, _, output_intrinsics = _validated_output_camera(
        calibration,
        output_size_hw,
    )
    return output_intrinsics


def build_nuplan_raster_plan(
    calibration: RawCameraCalibration,
    output_size_hw: tuple[int, int],
) -> NuPlanRasterPlan:
    """Build a one-pass native-``K`` rectification plan.

    The first baseline keeps the native focal lengths and principal point for
    the undistorted camera. :class:`CoverCropTransform` then expresses that
    camera directly on the requested output grid::

        K_output = A_cover_crop @ K_native

    ``cv2.initUndistortRectifyMap`` receives ``K_output`` and the final output
    size, so no intermediate rectified raster or separate resize operation is
    materialized.

    Args:
        calibration: Raw nuPlan intrinsics, five OpenCV/Caltech distortion
            coefficients, and native raster size. Camera pose fields and the
            hardware model label are intentionally ignored.
        output_size_hw: Final raster size as ``(height, width)``.

    Returns:
        Immutable final intrinsics, float source-coordinate maps, and a
        bilinear-support validity mask.

    Raises:
        NuPlanRasterError: The calibration, requested size, source skew, or a
            derived OpenCV result violates the supported contract.
    """
    layout, source_intrinsics, output_intrinsics = _validated_output_camera(
        calibration,
        output_size_hw,
    )
    distortion = np.asarray(calibration.distortion, dtype=np.float64)

    output_height, output_width = layout.output_size_hw
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_intrinsics,
        distortion,
        np.eye(3, dtype=np.float64),
        output_intrinsics,
        (output_width, output_height),
        cv2.CV_32FC1,
    )
    valid_mask = _bilinear_valid_mask(
        layout.source_size_hw,
        map_x,
        map_y,
    )

    return NuPlanRasterPlan(
        source_size_hw=layout.source_size_hw,
        output_size_hw=layout.output_size_hw,
        output_intrinsics=output_intrinsics,
        map_x=map_x,
        map_y=map_y,
        valid_mask=valid_mask,
    )


class RasterRgbError(ValueError):
    """A stored image cannot be decoded with its nuPlan calibration."""


@dataclass(frozen=True, slots=True, eq=False)
class RasterizedRgb:
    """One immutable RGB image tied to the plan that produced its grid."""

    plan: NuPlanRasterPlan
    rgb: npt.NDArray[np.uint8]

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rgb", _readonly_rgb(self.rgb))


def read_native_rgb(
    image_path: Path,
    expected_size_hw: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """Decode the stored sensor orientation as immutable RGB."""
    decoded_bgr = _read_native_bgr(image_path, expected_size_hw)
    return _readonly_rgb(cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB))


def read_rasterized_rgb(
    image_path: Path,
    plan: NuPlanRasterPlan,
) -> RasterizedRgb:
    """Decode one image and apply the direct OpenCV remap exactly once."""
    decoded_bgr = _read_native_bgr(image_path, plan.source_size_hw)
    raster_bgr = cv2.remap(
        decoded_bgr,
        plan.map_x,
        plan.map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    raster_rgb = cv2.cvtColor(raster_bgr, cv2.COLOR_BGR2RGB)
    return RasterizedRgb(plan=plan, rgb=raster_rgb)


def _validated_output_camera(
    calibration: RawCameraCalibration,
    output_size_hw: tuple[int, int],
) -> tuple[
    CoverCropTransform,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Return the one canonical native and output pinhole calculation."""
    layout = CoverCropTransform(
        calibration.image_size_hw,
        output_size_hw,
    )
    source_intrinsics = np.asarray(calibration.intrinsics, dtype=np.float64)
    if source_intrinsics[0, 1] != 0.0:
        raise NuPlanRasterError(
            "calibration.intrinsics must have zero skew for OpenCV "
            "undistortion"
        )
    output_intrinsics = layout.transform_intrinsics(source_intrinsics)
    return layout, source_intrinsics, output_intrinsics


def _bilinear_valid_mask(
    source_size_hw: tuple[int, int],
    map_x: npt.NDArray[np.float32],
    map_y: npt.NDArray[np.float32],
) -> npt.NDArray[np.bool_]:
    """Mark output samples whose linear footprint avoids the zero border."""
    unit_source = np.ones(source_size_hw, dtype=np.float32)
    coverage = cv2.remap(
        unit_source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return coverage == np.float32(1.0)


def _readonly_copy(
    value: object,
    dtype: type[np.float32] | type[np.float64] | type[np.bool_],
) -> npt.NDArray[np.generic]:
    """Return a C-contiguous array backed by immutable bytes."""
    array = np.array(value, dtype=dtype, copy=True, order="C")
    immutable_buffer = array.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=dtype).reshape(array.shape)


def _read_native_bgr(
    image_path: Path,
    expected_size_hw: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    decoded_bgr = cv2.imread(str(image_path), _IMREAD_FLAGS)
    if decoded_bgr is None:
        raise RasterRgbError(f"cannot decode image {image_path}")
    expected_shape = (*expected_size_hw, 3)
    if decoded_bgr.shape != expected_shape:
        raise RasterRgbError(
            f"decoded image has shape {decoded_bgr.shape}, expected {expected_shape}"
        )
    return decoded_bgr


def _readonly_rgb(value: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    immutable_buffer = np.ascontiguousarray(value, dtype=np.uint8).tobytes()
    return np.frombuffer(immutable_buffer, dtype=np.uint8).reshape(value.shape)

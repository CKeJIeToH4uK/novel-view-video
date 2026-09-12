"""Measured Waymo camera raster shared by three-camera depth and FRONT DDW."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from novel_view.inputs.waymo._arrays import readonly_owned
from novel_view.inputs.waymo.camera import WaymoCameraCalibration, WaymoCameraFrame
from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT


GEN3C_CANVAS_SIZE_HW = R4C_GEN3C_MODEL_CONTRACT.raster_size_hw
_ASPECT_UNITS_HW = (11, 20)
_PROJECTION_ZERO_ATOL = np.float32(1e-5)


@dataclass(frozen=True, slots=True, eq=False)
class WaymoGen3cRasterPlan:
    """One measured camera with an alpha-zero, central 20:11 direct remap."""

    calibration: WaymoCameraCalibration
    valid_roi_xywh: tuple[int, int, int, int]
    crop_xywh: tuple[int, int, int, int]
    canvas_intrinsics: npt.NDArray[np.float64]
    map_x: npt.NDArray[np.float32]
    map_y: npt.NDArray[np.float32]
    rectification_known: npt.NDArray[np.bool_]

    @property
    def K_canvas(self) -> npt.NDArray[np.float64]:
        """Expose the same measured K under the existing FRONT field name."""
        return self.canvas_intrinsics

    @property
    def source_size_hw(self) -> tuple[int, int]:
        return self.calibration.height, self.calibration.width

    @property
    def roi_retained_fraction(self) -> float:
        _, _, roi_width, roi_height = self.valid_roi_xywh
        _, _, crop_width, crop_height = self.crop_xywh
        return (crop_width * crop_height) / (roi_width * roi_height)

    @property
    def known_fraction(self) -> float:
        return float(np.mean(self.rectification_known, dtype=np.float64))


def build_waymo_gen3c_raster_plan(
    calibration: WaymoCameraCalibration,
) -> WaymoGen3cRasterPlan:
    """Build the fixed alpha-zero, centre-crop, one-remap raster plan."""
    source_intrinsics = np.asarray(calibration.intrinsics, dtype=np.float64)
    distortion = np.asarray(calibration.distortion_k1_k2_p1_p2_k3, dtype=np.float64)
    source_size_wh = (calibration.width, calibration.height)
    rectified_intrinsics, valid_roi = cv2.getOptimalNewCameraMatrix(
        source_intrinsics,
        distortion,
        source_size_wh,
        0.0,
        source_size_wh,
        centerPrincipalPoint=False,
    )
    valid_roi = tuple(int(value) for value in valid_roi)
    crop = _central_20_by_11_crop(valid_roi)
    canvas_intrinsics = _canvas_intrinsics(rectified_intrinsics, crop)
    output_height, output_width = GEN3C_CANVAS_SIZE_HW
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_intrinsics,
        distortion,
        np.eye(3, dtype=np.float64),
        canvas_intrinsics,
        (output_width, output_height),
        cv2.CV_32FC1,
    )
    coverage = cv2.remap(
        np.ones((calibration.height, calibration.width), dtype=np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return WaymoGen3cRasterPlan(
        calibration=calibration,
        valid_roi_xywh=valid_roi,
        crop_xywh=crop,
        canvas_intrinsics=readonly_owned(canvas_intrinsics),
        map_x=readonly_owned(np.asarray(map_x, dtype=np.float32)),
        map_y=readonly_owned(np.asarray(map_y, dtype=np.float32)),
        rectification_known=readonly_owned(coverage == np.float32(1.0)),
    )


def rectify_waymo_rgb(
    plan: WaymoGen3cRasterPlan,
    frame: WaymoCameraFrame,
) -> npt.NDArray[np.uint8]:
    """Rectify measured RGB directly into the final canvas without channel swap."""
    result = cv2.remap(
        frame.rgb,
        plan.map_x,
        plan.map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    result[~plan.rectification_known] = 0
    return readonly_owned(result)


def rectify_waymo_projection_xy(
    plan: WaymoGen3cRasterPlan,
    source_xy: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Keep the historical three-camera projection's near-zero convention."""
    canvas_xy = cv2.undistortPoints(
        source_xy.reshape(-1, 1, 2),
        np.asarray(plan.calibration.intrinsics, dtype=np.float64),
        np.asarray(plan.calibration.distortion_k1_k2_p1_p2_k3, dtype=np.float64),
        R=np.eye(3, dtype=np.float64),
        P=plan.canvas_intrinsics,
    ).reshape(-1, 2)
    canvas_xy = np.asarray(canvas_xy, dtype=np.float32)
    canvas_xy[np.abs(canvas_xy) <= _PROJECTION_ZERO_ATOL] = 0.0
    height, width = GEN3C_CANVAS_SIZE_HW
    valid = (
        (source_xy[:, 0] >= 0.0)
        & (source_xy[:, 0] < plan.calibration.width)
        & (source_xy[:, 1] >= 0.0)
        & (source_xy[:, 1] < plan.calibration.height)
        & (canvas_xy[:, 0] >= 0.0)
        & (canvas_xy[:, 0] < width)
        & (canvas_xy[:, 1] >= 0.0)
        & (canvas_xy[:, 1] < height)
    )
    return readonly_owned(canvas_xy), readonly_owned(valid)


def _central_20_by_11_crop(
    roi_xywh: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left, top, width, height = roi_xywh
    units = min(width // _ASPECT_UNITS_HW[1], height // _ASPECT_UNITS_HW[0])
    if units <= 0:
        raise WaymoContractError("valid ROI is too small for a 20:11 crop")
    crop_width = units * _ASPECT_UNITS_HW[1]
    crop_height = units * _ASPECT_UNITS_HW[0]
    return (
        left + (width - crop_width) // 2,
        top + (height - crop_height) // 2,
        crop_width,
        crop_height,
    )


def _canvas_intrinsics(
    rectified_intrinsics: npt.NDArray[np.float64],
    crop_xywh: tuple[int, int, int, int],
) -> npt.NDArray[np.float64]:
    left, top, crop_width, _ = crop_xywh
    scale = GEN3C_CANVAS_SIZE_HW[1] / crop_width
    result = np.asarray(rectified_intrinsics, dtype=np.float64).copy()
    result[0, :2] *= scale
    result[1, :2] *= scale
    result[0, 2] = scale * (result[0, 2] - left + 0.5) - 0.5
    result[1, 2] = scale * (result[1, 2] - top + 0.5) - 0.5
    return result

"""Run the frozen pose-conditioned DA3 Nested research prototype."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, TypeVar, cast

import cv2
import numpy as np
import numpy.typing as npt

from novel_view.geometry.depth import PosedDepthSequence
from novel_view.generation.euvs.source_views import SourceGeometryInput
from novel_view.runtime.process import run_process
from research.da3_nested.io import write_source_rgb

_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
_PATCH_SIZE = 14
_CAMERA_RTOL = 2e-5
_CAMERA_ATOL = 2e-5
_OUTPUT_FILES = {
    "depth_z_m": "depth_z_m.npy",
    "confidence": "confidence.npy",
    "sky_mask": "sky_mask.npy",
    "has_sky": "has_sky.npy",
    "conditioned_w2c": "conditioned_w2c.npy",
    "conditioned_intrinsics": "conditioned_intrinsics.npy",
    "is_metric": "is_metric.npy",
}
_ArrayScalar = TypeVar("_ArrayScalar", bound=np.generic)


class Da3NestedError(RuntimeError):
    """DA3 Nested input preparation, execution, or output is invalid."""


@dataclass(frozen=True, slots=True, eq=False)
class Da3NestedInputPlan:
    """Exact upper-bound-resize relation between source and DA3 grids."""

    source_size_hw: tuple[int, int]
    process_resolution: int
    model_size_hw: tuple[int, int]
    source_to_model_pixels: npt.NDArray[np.float64]

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validate the official patch-14 resize geometry."""
        source_size = _validated_size(
            self.source_size_hw,
            "source_size_hw",
        )
        process_resolution = _validated_resolution(
            self.process_resolution
        )
        model_size = _validated_size(
            self.model_size_hw,
            "model_size_hw",
        )
        expected_size = _model_size(source_size, process_resolution)
        if model_size != expected_size:
            raise Da3NestedError(
                f"model_size_hw must be {expected_size} for "
                f"process_resolution={process_resolution}"
            )
        transform = self.source_to_model_pixels
        expected_transform = _pixel_scale(source_size, model_size)
        if (
            not isinstance(transform, np.ndarray)
            or transform.dtype != np.dtype(np.float64)
            or transform.shape != (3, 3)
            or not np.array_equal(transform, expected_transform)
        ):
            raise Da3NestedError(
                "source_to_model_pixels must exactly scale the declared grids"
            )
        object.__setattr__(
            self,
            "source_to_model_pixels",
            _readonly_copy(transform),
        )


@dataclass(frozen=True, slots=True)
class Da3NestedResources:
    """Machine-local DA3 model directory and isolated process resources."""

    python_executable: Path
    scratch_root: Path
    log_path: Path
    model_directory: Path
    environment_overrides: Mapping[str, str] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True, eq=False)
class Da3NestedConditionedPrediction:
    """Metric DA3 depth conditioned on the measured source camera pack.

    The camera arrays are the measured nuPlan cameras returned by the
    official aligned API, not independent model predictions. ``confidence``
    remains an opaque backend quantity. ``sky_mask`` is optional because the
    public DA3 result may omit it even though the Nested model internally uses
    a sky branch.
    """

    input: SourceGeometryInput
    input_plan: Da3NestedInputPlan
    depth_z_m: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    conditioned_w2c: npt.NDArray[np.float32]
    conditioned_intrinsics: npt.NDArray[np.float32]
    sky_mask: npt.NDArray[np.bool_] | None = None

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validate metric depth, camera identity, order, and ownership."""
        if not isinstance(self.input_plan, Da3NestedInputPlan):
            raise Da3NestedError(
                "input_plan must be a Da3NestedInputPlan"
            )
        source_size = self.input.source.frames[
            0
        ].raster.plan.output_size_hw
        if self.input_plan.source_size_hw != source_size:
            raise Da3NestedError(
                "input_plan source size must match the source raster"
            )

        frame_count = len(self.input.source.frames)
        height, width = self.input_plan.model_size_hw
        depth_shape = (frame_count, height, width)
        depth = _require_array(
            self.depth_z_m,
            "depth_z_m",
            np.float32,
            depth_shape,
        )
        confidence = _require_array(
            self.confidence,
            "confidence",
            np.float32,
            depth_shape,
        )
        w2c = _require_array(
            self.conditioned_w2c,
            "conditioned_w2c",
            np.float32,
            (frame_count, 3, 4),
        )
        intrinsics = _require_array(
            self.conditioned_intrinsics,
            "conditioned_intrinsics",
            np.float32,
            (frame_count, 3, 3),
        )
        sky = self.sky_mask
        if sky is not None:
            sky = _require_array(
                sky,
                "sky_mask",
                np.bool_,
                depth_shape,
            )

        for name, value in (
            ("depth_z_m", depth),
            ("confidence", confidence),
            ("conditioned_w2c", w2c),
            ("conditioned_intrinsics", intrinsics),
        ):
            if not np.all(np.isfinite(value)):
                raise Da3NestedError(
                    f"{name} must contain only finite values"
                )
        if np.any(depth <= 0.0):
            raise Da3NestedError(
                "depth_z_m must be strictly positive"
            )
        _validate_conditioned_cameras(
            self.input,
            self.input_plan,
            w2c,
            intrinsics,
        )

        object.__setattr__(
            self,
            "depth_z_m",
            _readonly_copy(depth),
        )
        object.__setattr__(
            self,
            "confidence",
            _readonly_copy(confidence),
        )
        object.__setattr__(
            self,
            "conditioned_w2c",
            _readonly_copy(w2c),
        )
        object.__setattr__(
            self,
            "conditioned_intrinsics",
            _readonly_copy(intrinsics),
        )
        if sky is not None:
            object.__setattr__(
                self,
                "sky_mask",
                _readonly_copy(sky),
            )

    def to_posed_depth_sequence(self) -> PosedDepthSequence:
        """Return the common source-grid result without inventing detail.

        DA3's official evaluation path uses nearest-neighbour depth resize
        back to the requested evaluation grid. The same rule is used here for
        depth, confidence, and the optional sky mask.
        """
        source = self.input.source
        output_size = source.frames[0].raster.plan.output_size_hw
        frame_count = len(source.frames)
        height, width = output_size
        depth = np.empty(
            (frame_count, height, width),
            dtype=np.float32,
        )
        confidence = np.empty_like(depth)
        sky = (
            np.empty((frame_count, height, width), dtype=np.bool_)
            if self.sky_mask is not None
            else None
        )
        for index in range(frame_count):
            depth[index] = cv2.resize(
                self.depth_z_m[index],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            confidence[index] = cv2.resize(
                self.confidence[index],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            if sky is not None:
                sky[index] = cv2.resize(
                    self.sky_mask[index].astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.bool_)

        valid = np.stack(
            [
                frame.raster.plan.valid_mask
                for frame in source.frames
            ]
        ).astype(np.bool_, copy=True)
        if sky is not None:
            valid &= ~sky
        valid &= np.isfinite(depth) & (depth > 0.0)
        depth[~valid] = 0.0
        confidence[~valid] = 0.0

        return PosedDepthSequence(
            sequence_id=source.sequence_id,
            depth_z_m=_readonly_copy(depth),
            geometry_valid_mask=_readonly_copy(valid),
            intrinsics=_readonly_copy(
                np.array(
                    self.input.intrinsics,
                    dtype=np.float64,
                    copy=True,
                )
            ),
            reference_to_camera=_readonly_copy(
                np.array(
                    self.input.reference_to_camera,
                    dtype=np.float64,
                    copy=True,
                )
            ),
            backend_confidence=_readonly_copy(confidence),
        )


def build_da3_nested_input_plan(
    source_size_hw: tuple[int, int],
    process_resolution: int,
) -> Da3NestedInputPlan:
    """Build the deterministic official upper-bound-resize input plan."""
    source_size = _validated_size(
        source_size_hw,
        "source_size_hw",
    )
    resolution = _validated_resolution(process_resolution)
    model_size = _model_size(source_size, resolution)
    return Da3NestedInputPlan(
        source_size_hw=source_size,
        process_resolution=resolution,
        model_size_hw=model_size,
        source_to_model_pixels=_pixel_scale(source_size, model_size),
    )


def predict_da3_nested_conditioned(
    input: SourceGeometryInput,
    process_resolution: int,
    resources: Da3NestedResources,
) -> Da3NestedConditionedPrediction:
    """Run one joint pose-conditioned DA3 Nested source prediction."""
    if not isinstance(resources, Da3NestedResources):
        raise Da3NestedError(
            "resources must be Da3NestedResources"
        )
    source_size = input.source.frames[0].raster.plan.output_size_hw
    plan = build_da3_nested_input_plan(
        source_size,
        process_resolution,
    )

    outputs: dict[str, npt.NDArray[np.generic]] = {}
    with tempfile.TemporaryDirectory(
        prefix="da3-nested-",
        dir=resources.scratch_root,
    ) as temporary_directory:
        exchange = Path(temporary_directory)
        write_source_rgb(
            input.source,
            exchange / "source_rgb.npy",
        )
        np.save(
            exchange / "reference_to_camera.npy",
            input.reference_to_camera,
            allow_pickle=False,
        )
        np.save(
            exchange / "intrinsics.npy",
            input.intrinsics,
            allow_pickle=False,
        )
        _run_worker(exchange, plan, resources)
        try:
            for name, filename in _OUTPUT_FILES.items():
                outputs[name] = np.load(
                    exchange / filename,
                    mmap_mode="r",
                    allow_pickle=False,
                )
            if not _scalar_bool(outputs["is_metric"], "is_metric"):
                raise Da3NestedError(
                    f"{_MODEL_ID} did not report metric depth"
                )
            sky_mask = (
                cast(
                    npt.NDArray[np.bool_],
                    outputs["sky_mask"],
                )
                if _scalar_bool(outputs["has_sky"], "has_sky")
                else None
            )
            return Da3NestedConditionedPrediction(
                input=input,
                input_plan=plan,
                depth_z_m=cast(
                    npt.NDArray[np.float32],
                    outputs["depth_z_m"],
                ),
                confidence=cast(
                    npt.NDArray[np.float32],
                    outputs["confidence"],
                ),
                conditioned_w2c=cast(
                    npt.NDArray[np.float32],
                    outputs["conditioned_w2c"],
                ),
                conditioned_intrinsics=cast(
                    npt.NDArray[np.float32],
                    outputs["conditioned_intrinsics"],
                ),
                sky_mask=sky_mask,
            )
        finally:
            for value in outputs.values():
                if isinstance(value, np.memmap):
                    value._mmap.close()


def predict_da3_nested_geometry(
    input: SourceGeometryInput,
    process_resolution: int,
    resources: Da3NestedResources,
) -> PosedDepthSequence:
    """Run DA3 Nested and return the backend-neutral geometry result."""
    return predict_da3_nested_conditioned(
        input,
        process_resolution,
        resources,
    ).to_posed_depth_sequence()


def _run_worker(
    exchange: Path,
    plan: Da3NestedInputPlan,
    resources: Da3NestedResources,
) -> None:
    """Invoke the DA3-only worker through the shared process boundary."""
    worker = Path(__file__).with_name("_worker.py")
    arguments = [
        "--input-rgb",
        str(exchange / "source_rgb.npy"),
        "--input-extrinsics",
        str(exchange / "reference_to_camera.npy"),
        "--input-intrinsics",
        str(exchange / "intrinsics.npy"),
        "--output-directory",
        str(exchange),
        "--model-directory",
        str(resources.model_directory),
        "--process-resolution",
        str(plan.process_resolution),
        "--expected-height",
        str(plan.model_size_hw[0]),
        "--expected-width",
        str(plan.model_size_hw[1]),
    ]
    run_process(
        [str(resources.python_executable), str(worker), *arguments],
        resources.log_path,
        (
            f"DA3 Nested worker at {plan.model_size_hw} "
            f"for {_MODEL_ID}"
        ),
        resources.environment_overrides,
    )


def _validate_conditioned_cameras(
    input: SourceGeometryInput,
    plan: Da3NestedInputPlan,
    actual_w2c: npt.NDArray[np.float32],
    actual_intrinsics: npt.NDArray[np.float32],
) -> None:
    """Require the aligned API to return, not silently replace, measured K/E."""
    expected_w2c = input.reference_to_camera[:, :3].astype(np.float32)
    expected_intrinsics = np.einsum(
        "ij,njk->nik",
        plan.source_to_model_pixels,
        input.intrinsics,
    ).astype(np.float32)
    if not np.allclose(
        actual_w2c,
        expected_w2c,
        rtol=_CAMERA_RTOL,
        atol=_CAMERA_ATOL,
    ):
        raise Da3NestedError(
            "conditioned_w2c must equal the measured local camera pack"
        )
    if not np.allclose(
        actual_intrinsics,
        expected_intrinsics,
        rtol=_CAMERA_RTOL,
        atol=_CAMERA_ATOL,
    ):
        raise Da3NestedError(
            "conditioned_intrinsics must equal measured K on the model grid"
        )


def _model_size(
    source_size_hw: tuple[int, int],
    process_resolution: int,
) -> tuple[int, int]:
    """Mirror official longest-side resize and nearest patch-14 rounding."""
    height, width = source_size_hw
    scale = process_resolution / max(height, width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    model_height = _nearest_patch_multiple(resized_height)
    model_width = _nearest_patch_multiple(resized_width)
    if model_height < _PATCH_SIZE or model_width < _PATCH_SIZE:
        raise Da3NestedError(
            "process_resolution produces a model grid smaller than one patch"
        )
    return model_height, model_width


def _nearest_patch_multiple(value: int) -> int:
    """Use the official tie-up nearest-multiple rule."""
    down = (value // _PATCH_SIZE) * _PATCH_SIZE
    up = down + _PATCH_SIZE
    return up if abs(up - value) <= abs(value - down) else down


def _pixel_scale(
    source_size_hw: tuple[int, int],
    model_size_hw: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Return the project's edge-coordinate source-to-model transform."""
    source_height, source_width = source_size_hw
    model_height, model_width = model_size_hw
    return np.array(
        [
            [model_width / source_width, 0.0, 0.0],
            [0.0, model_height / source_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _validated_size(
    value: object,
    name: str,
) -> tuple[int, int]:
    """Require two positive built-in integers."""
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(component, bool)
            or not isinstance(component, int)
            or component <= 0
            for component in value
        )
    ):
        raise Da3NestedError(
            f"{name} must be two positive integers"
        )
    return cast(tuple[int, int], value)


def _validated_resolution(value: object) -> int:
    """Require a practical positive processing resolution."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < _PATCH_SIZE
    ):
        raise Da3NestedError(
            f"process_resolution must be an integer >= {_PATCH_SIZE}"
        )
    return value


def _require_array(
    value: object,
    name: str,
    dtype: type[_ArrayScalar],
    shape: tuple[int, ...],
) -> npt.NDArray[_ArrayScalar]:
    """Require one exact raw-model array without implicit conversion."""
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(dtype)
    ):
        raise Da3NestedError(
            f"{name} must be a {np.dtype(dtype).name} NumPy array"
        )
    if value.shape != shape:
        raise Da3NestedError(
            f"{name} must have shape {shape}, got {value.shape}"
        )
    return cast(npt.NDArray[_ArrayScalar], value)


def _scalar_bool(value: object, name: str) -> bool:
    """Read an exact zero-dimensional bool array from the worker."""
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.bool_)
        or value.shape != ()
    ):
        raise Da3NestedError(
            f"{name} must be a scalar bool NumPy array"
        )
    return bool(value)


def _readonly_copy(
    value: npt.NDArray[_ArrayScalar],
) -> npt.NDArray[_ArrayScalar]:
    """Return an owned C-contiguous array whose buffer cannot be reopened."""
    owned = np.frombuffer(
        np.ascontiguousarray(value).tobytes(),
        dtype=value.dtype,
    ).reshape(value.shape)
    return cast(npt.NDArray[_ArrayScalar], owned)

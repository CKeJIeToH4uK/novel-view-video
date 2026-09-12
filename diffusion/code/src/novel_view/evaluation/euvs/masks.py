"""Load or predict exact native EUVS dynamic masks through a direct PNG cache."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from novel_view.inputs.euvs.pair import EuvsFrameInput, EuvsFrameSequence
from novel_view.inputs.nuplan.raster import NuPlanRasterPlan, read_native_rgb
from novel_view.models.grounded_sam2.backend import (
    GroundedSam2Resources as RawGroundedSam2Resources,
    run_grounded_sam2,
)
from novel_view.models.grounded_sam2.request import GroundedSam2Request
from novel_view.models.grounded_sam2.spec import GroundedSam2ModelError


class EuvsMaskError(RuntimeError):
    """One native mask cache value or prediction is unusable."""


@dataclass(frozen=True, slots=True)
class DynamicMaskRecipe:
    """Stable identity and raw model arguments for one mask procedure."""

    recipe_id: str
    grounding_model_id: str
    sam2_model_id: str
    sam2_model_config: str
    prompt: str
    box_threshold: float
    text_threshold: float
    numeric_mode: str
    mask_polarity: str = "255=dynamic"
    instance_combination: str = "logical_or"


EUVS_GROUNDED_SAM2_RECIPE = DynamicMaskRecipe(
    recipe_id="grounded-sam2-euvs-v1",
    grounding_model_id="IDEA-Research/grounding-dino-tiny",
    sam2_model_id="facebook/sam2.1-hiera-large",
    sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
    prompt="person. rider. car. truck. bus. train. motorcycle. bicycle.",
    box_threshold=0.4,
    text_threshold=0.3,
    numeric_mode="cuda-bfloat16-tf32",
)


@dataclass(frozen=True, slots=True)
class GroundedSam2Resources:
    """Machine-local assets for the Stage 4 raw model owner."""

    scratch_root: Path
    log_path: Path
    grounding_model_directory: Path
    sam2_checkpoint_path: Path
    environment_overrides: Mapping[str, str] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True, eq=False)
class EuvsDynamicMaskFrame:
    """One native-grid mask where true means dynamic."""

    input: EuvsFrameInput
    dynamic_mask: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        expected = self.input.calibration.image_size_hw
        if self.dynamic_mask.dtype != np.bool_ or self.dynamic_mask.shape != expected:
            raise EuvsMaskError(f"dynamic mask must be bool {expected}")
        object.__setattr__(self, "dynamic_mask", _readonly(self.dynamic_mask))


@dataclass(frozen=True, slots=True, eq=False)
class EuvsDynamicMaskSequence:
    """Native masks in exact positional correspondence with one sequence."""

    input_sequence: EuvsFrameSequence
    recipe_id: str
    frames: tuple[EuvsDynamicMaskFrame, ...]

    def __post_init__(self) -> None:
        tokens = tuple(frame.input.ref.image_token for frame in self.frames)
        if tokens != self.input_sequence.sequence_id:
            raise EuvsMaskError("mask frames must preserve sequence token order")


@dataclass(frozen=True, slots=True, eq=False)
class RasterizedDynamicMask:
    """One native dynamic mask remapped onto a pinhole raster."""

    native: EuvsDynamicMaskFrame
    plan: NuPlanRasterPlan
    dynamic_mask: npt.NDArray[np.bool_]

    @property
    def raster_valid_mask(self) -> npt.NDArray[np.bool_]:
        return self.plan.valid_mask


def rasterize_dynamic_mask(
    frame: EuvsDynamicMaskFrame,
    plan: NuPlanRasterPlan,
) -> RasterizedDynamicMask:
    """Remap one native dynamic mask with nearest-neighbour sampling."""
    remapped = cv2.remap(
        frame.dynamic_mask.astype(np.uint8),
        plan.map_x,
        plan.map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.bool_)
    return RasterizedDynamicMask(frame, plan, _readonly(remapped))


def load_or_predict_masks(
    sequence: EuvsFrameSequence,
    recipe_cache_directory: Path,
    resources: GroundedSam2Resources | None = None,
    recipe: DynamicMaskRecipe = EUVS_GROUNDED_SAM2_RECIPE,
) -> EuvsDynamicMaskSequence:
    """Read exact token PNGs; only FileNotFoundError triggers one model call."""
    native = recipe_cache_directory / "native"
    masks: dict[str, npt.NDArray[np.bool_]] = {}
    missing: list[EuvsFrameInput] = []
    missing_tokens: set[str] = set()
    for frame in sequence.frames:
        token = frame.ref.image_token
        if token in masks or token in missing_tokens:
            continue
        try:
            masks[token] = _read_png(
                native / f"{token}.png",
                frame.calibration.image_size_hw,
            )
        except FileNotFoundError:
            missing.append(frame)
            missing_tokens.add(token)

    if missing:
        if resources is None:
            raise EuvsMaskError(
                "resources are required for missing mask "
                f"{missing[0].ref.image_token!r}"
            )
        try:
            predicted = run_grounded_sam2(
                GroundedSam2Request(
                    rgb_frames=_native_rgb_frames(tuple(missing)),
                    frame_count=len(missing),
                    image_size_hw=missing[0].calibration.image_size_hw,
                    sam2_model_config=recipe.sam2_model_config,
                    prompt=recipe.prompt,
                    box_threshold=recipe.box_threshold,
                    text_threshold=recipe.text_threshold,
                ),
                RawGroundedSam2Resources(
                    scratch_root=resources.scratch_root,
                    grounding_model_directory=resources.grounding_model_directory,
                    sam2_checkpoint_path=resources.sam2_checkpoint_path,
                    environment_overrides=resources.environment_overrides,
                ),
                resources.log_path,
            ).dynamic_mask
        except GroundedSam2ModelError as error:
            raise EuvsMaskError(str(error)) from error
        native.mkdir(parents=True, exist_ok=True)
        for frame, mask in zip(missing, predicted, strict=True):
            token = frame.ref.image_token
            _write_png(native / f"{token}.png", mask)
            masks[token] = mask

    return EuvsDynamicMaskSequence(
        sequence,
        recipe.recipe_id,
        tuple(
            EuvsDynamicMaskFrame(frame, masks[frame.ref.image_token])
            for frame in sequence.frames
        ),
    )


def _native_rgb_frames(
    frames: tuple[EuvsFrameInput, ...],
) -> Iterator[npt.NDArray[np.uint8]]:
    for frame in frames:
        yield read_native_rgb(
            frame.ref.image_path,
            frame.calibration.image_size_hw,
        )


def _read_png(
    path: Path,
    expected_size_hw: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    payload = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    encoded = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
    if encoded is None or encoded.shape != expected_size_hw:
        raise EuvsMaskError(f"cached mask {path} must use size {expected_size_hw}")
    values = np.unique(encoded)
    if not np.all(np.isin(values, (0, 255))):
        raise EuvsMaskError(f"cached mask {path} must contain only 0 and 255")
    return _readonly(encoded == np.uint8(255))


def _write_png(path: Path, mask: npt.NDArray[np.bool_]) -> None:
    encoded = mask.astype(np.uint8) * np.uint8(255)
    if not cv2.imwrite(str(path), encoded):
        raise EuvsMaskError(f"cannot encode cached mask {path}")


def _readonly(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    return np.frombuffer(mask.tobytes(order="C"), dtype=np.bool_).reshape(mask.shape)


__all__ = [
    "DynamicMaskRecipe",
    "EUVS_GROUNDED_SAM2_RECIPE",
    "EuvsDynamicMaskFrame",
    "EuvsDynamicMaskSequence",
    "EuvsMaskError",
    "GroundedSam2Resources",
    "RasterizedDynamicMask",
    "load_or_predict_masks",
    "rasterize_dynamic_mask",
]

"""Prepare and bind reusable EUVS source views around source-neutral VGGT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast
from urllib.parse import quote

import numpy as np
import numpy.typing as npt

from novel_view.geometry.depth import PosedDepthSequence
from novel_view.geometry.sim3 import CameraPackAlignment
from novel_view.inputs.euvs.index import FramesIndex
from novel_view.inputs.euvs.pair import EuvsPairInput, load_euvs_pair_input
from novel_view.inputs.euvs.rgb import EuvsRasterizedSequence, rasterize_euvs_source
from novel_view.inputs.euvs.spec import EuvsPairSelection, EuvsSelection
from novel_view.inputs.nuplan.camera import build_reference_to_camera_transforms
from novel_view.models.vggt.alignment import align_vggt_cameras
from novel_view.models.vggt.posed_depth import (
    VggtOmegaPosedDepthError,
    build_vggt_posed_depth,
)
from novel_view.models.vggt.record import (
    SavedVggtOmegaPrediction,
    load_saved_vggt_omega_prediction,
)
from novel_view.models.vggt.request import VggtOmegaRequest
from novel_view.models.vggt.spec import (
    VggtOmegaInputMode,
    build_vggt_omega_input_plan,
)

_ArrayScalar = TypeVar("_ArrayScalar", bound=np.generic)


class EuvsSourceViewError(ValueError):
    """A selected source or saved VGGT record has conflicting identity."""


@dataclass(frozen=True, slots=True, eq=False)
class SourceGeometryInput:
    """Ordered EUVS RGB with measured source-grid K and local OpenCV W2C.

    Camera zero defines the reference frame; translations remain in metres.
    The builder owns the read-only float64 arrays without reordering frames.
    """

    source: EuvsRasterizedSequence
    intrinsics: npt.NDArray[np.float64]
    reference_to_camera: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedEuvsVggtGeometry:
    """One source-matched saved prediction in metric source coordinates."""

    alignment: CameraPackAlignment
    posed_depth: PosedDepthSequence
    provenance: str


@dataclass(frozen=True, slots=True, eq=False)
class EuvsSourceView:
    """One selected physical pair and its measured source-only model input."""

    selection: EuvsPairSelection
    physical_pair: EuvsPairInput
    geometry_input: SourceGeometryInput


def build_source_geometry_input(
    source: EuvsRasterizedSequence,
) -> SourceGeometryInput:
    """Collect measured K and stably rebase nuPlan W2C into source camera zero."""
    intrinsics = np.stack([
        frame.raster.plan.output_intrinsics for frame in source.frames
    ]).astype(np.float64, copy=False)
    cameras = tuple(frame.input.geometry for frame in source.frames)
    transforms = build_reference_to_camera_transforms(cameras[0], cameras)
    return SourceGeometryInput(
        source,
        _readonly_copy(intrinsics),
        _readonly_copy(transforms),
    )


def unique_source_views(
    selection: EuvsSelection,
) -> tuple[EuvsPairSelection, ...]:
    """Keep the first pair for each exact ordered source-token tuple."""
    unique: dict[tuple[str, ...], EuvsPairSelection] = {}
    first_tokens: dict[str, tuple[str, ...]] = {}
    for pair in selection.pairs:
        sequence_id = pair.source.image_tokens
        if sequence_id in unique:
            continue
        first_token = sequence_id[0]
        previous = first_tokens.get(first_token)
        if previous is not None and previous != sequence_id:
            raise EuvsSourceViewError(
                f"different source sequences start with {first_token!r}"
            )
        first_tokens[first_token] = sequence_id
        unique[sequence_id] = pair
    return tuple(unique.values())


def source_view_directory_name(source_tokens: tuple[str, ...]) -> str:
    """Use the first immutable source token as the direct directory ID."""
    return quote(source_tokens[0], safe="")


def read_source_view(
    selection: EuvsPairSelection,
    frames_index: FramesIndex,
    output_size_hw: tuple[int, int],
) -> EuvsSourceView:
    """Read one physical pair but rasterize only its selected source RGB."""
    pair = load_euvs_pair_input(
        name=selection.name,
        location=selection.location,
        channel=selection.channel,
        source_traversal=selection.source.traversal,
        source_image_tokens=selection.source.image_tokens,
        target_traversal=selection.target.traversal,
        target_image_tokens=selection.target.image_tokens,
        frames_index=frames_index,
    )
    source = rasterize_euvs_source(pair, output_size_hw)
    return EuvsSourceView(
        selection=selection,
        physical_pair=pair,
        geometry_input=build_source_geometry_input(source),
    )


def build_vggt_source_request(
    source_view: EuvsSourceView,
    mode: VggtOmegaInputMode,
) -> VggtOmegaRequest:
    """Build the source-neutral VGGT request without model or process state."""
    source = source_view.geometry_input.source
    return VggtOmegaRequest(
        rgb_frames=(frame.raster.rgb for frame in source.frames),
        frame_count=len(source.frames),
        input_plan=build_vggt_omega_input_plan(source.image_size_hw, mode),
    )


def bind_source_view(
    source_view: EuvsSourceView,
    directory: Path,
) -> ResolvedEuvsVggtGeometry:
    """Bind one current model-owned record by its full ordered identity."""
    return resolve_euvs_vggt_geometry(
        source_view.geometry_input,
        directory,
        expected_provenance="ordered-source-tokens",
    )


def load_euvs_vggt_prediction(
    source_input: SourceGeometryInput,
    directory: Path,
) -> SavedVggtOmegaPrediction:
    """Load a current or historical record under the old EUVS policy."""
    saved = load_saved_vggt_omega_prediction(directory)
    _require_source_match(
        saved,
        source_input.source.sequence_id,
        source_input.source.image_size_hw,
    )
    return saved


def resolve_euvs_vggt_geometry(
    source_input: SourceGeometryInput,
    directory: Path,
    *,
    expected_provenance: str | None = None,
) -> ResolvedEuvsVggtGeometry:
    """Load, align and adapt one saved result without rebinding raw arrays."""
    saved = load_euvs_vggt_prediction(source_input, directory)
    if (
        expected_provenance is not None
        and saved.provenance != expected_provenance
    ):
        raise EuvsSourceViewError(
            "saved VGGT provenance disagrees with the required provenance"
        )
    alignment = align_vggt_cameras(
        saved.prediction.model_w2c,
        source_input.reference_to_camera,
    )
    source = source_input.source
    source_valid = np.stack([
        frame.raster.plan.valid_mask for frame in source.frames
    ])
    try:
        posed_depth = build_vggt_posed_depth(
            saved.prediction,
            alignment,
            source_valid,
            source.sequence_id,
        )
    except VggtOmegaPosedDepthError as error:
        raise EuvsSourceViewError(str(error)) from error
    return ResolvedEuvsVggtGeometry(alignment, posed_depth, saved.provenance)


def _require_source_match(
    saved: SavedVggtOmegaPrediction,
    source_tokens: tuple[str, ...],
    source_size_hw: tuple[int, int],
) -> None:
    """Accept exact ordered identity or the historical edge attestation."""
    attested = (
        saved.prediction.input_plan.source_size_hw == source_size_hw
        and saved.source_frame_count == len(source_tokens)
        and saved.first_source_token == source_tokens[0]
        and saved.last_source_token == source_tokens[-1]
        and saved.frame_order_preserved
    )
    if not attested:
        raise EuvsSourceViewError(
            "saved VGGT source attestation does not match EUVS input"
        )
    if saved.source_tokens is not None and saved.source_tokens != source_tokens:
        raise EuvsSourceViewError(
            "saved VGGT ordered source tokens do not match EUVS input"
        )


def _readonly_copy(
    value: npt.NDArray[_ArrayScalar],
) -> npt.NDArray[_ArrayScalar]:
    owned = np.frombuffer(
        np.ascontiguousarray(value).tobytes(),
        dtype=value.dtype,
    ).reshape(value.shape)
    return cast(npt.NDArray[_ArrayScalar], owned)


__all__ = [
    "EuvsSourceView",
    "EuvsSourceViewError",
    "ResolvedEuvsVggtGeometry",
    "SourceGeometryInput",
    "bind_source_view",
    "build_source_geometry_input",
    "build_vggt_source_request",
    "load_euvs_vggt_prediction",
    "read_source_view",
    "resolve_euvs_vggt_geometry",
    "source_view_directory_name",
    "unique_source_views",
]

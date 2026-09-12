"""Build the complete CPU-only Gen3C plan for one EUVS pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from novel_view.generation.gen3c.conditioning import (
    Gen3cProjectedSingleSourcePlan,
    build_direct_gen3c_query_trajectory,
    build_projected_single_source_plan,
)
from novel_view.generation.gen3c.euvs_input import EuvsGen3cConditioningInput
from novel_view.generation.gen3c.timeline import (
    Gen3cTimeline,
    build_direct_gen3c_timeline,
)
from novel_view.geometry.camera import invert_rigid
from novel_view.geometry.depth import PosedDepthSequence
from novel_view.geometry.polyline import (
    PolylineProjection,
    project_points_to_polyline,
    select_nearest_polyline_vertex_indices,
)
from novel_view.inputs.euvs.pair import EuvsFrameInput, EuvsPairInput
from novel_view.inputs.nuplan.camera import build_reference_to_camera_transforms

if TYPE_CHECKING:
    from novel_view.generation.euvs.source_views import ResolvedEuvsVggtGeometry
    from novel_view.inputs.euvs.rgb import EuvsRasterizedSequence


class EuvsGenerationPlanError(ValueError):
    """A physical pair and its bound source geometry disagree."""


@dataclass(frozen=True, slots=True, eq=False)
class EuvsGenerationInput:
    """One exact pair, source evidence and measured target cameras."""

    pair: EuvsPairInput
    source_rgb: EuvsRasterizedSequence
    source_geometry: PosedDepthSequence
    target_intrinsics: npt.NDArray[np.float64]
    reference_to_target_camera: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class EuvsAlignmentCandidate:
    """Target projection and signed excess outside the source domain."""

    projection: PolylineProjection
    source_domain_excess: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class EuvsSourceMapping:
    """Target-ordered nearest endpoints of already selected source segments."""

    alignment: EuvsAlignmentCandidate
    source_vertex_progress: npt.NDArray[np.float64]
    target_source_progress: npt.NDArray[np.float64]
    source_sequence_index: npt.NDArray[np.int64]
    source_progress_residual: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class EuvsGenerationPlan:
    """All lightweight science needed for one EUVS Gen3C request."""

    geometry: ResolvedEuvsVggtGeometry
    input: EuvsGenerationInput
    mapping: EuvsSourceMapping
    timeline: Gen3cTimeline
    conditioning: Gen3cProjectedSingleSourcePlan
    request: EuvsGen3cConditioningInput


def build_euvs_generation_plan(
    pair: EuvsPairInput,
    source_rgb: EuvsRasterizedSequence,
    geometry: ResolvedEuvsVggtGeometry,
) -> EuvsGenerationPlan:
    """Compose one physical pair through mapping, timeline and conditioning."""
    generation_input = build_euvs_generation_input(
        pair,
        source_rgb,
        geometry.posed_depth,
    )
    mapping = build_euvs_source_mapping(generation_input.pair)
    timeline = build_direct_gen3c_timeline(
        tuple(
            frame.ref.timestamp_us
            for frame in generation_input.pair.target.frames
        ),
        mapping.source_sequence_index,
    )
    trajectory = build_direct_gen3c_query_trajectory(
        timeline,
        generation_input.source_geometry.reference_to_camera,
        generation_input.reference_to_target_camera,
        generation_input.source_geometry.intrinsics,
        generation_input.target_intrinsics,
    )
    source_xy, anchor_to_source_xy = _conditioning_source_frame(
        generation_input,
        trajectory.anchor_source_sequence_index,
    )
    conditioning = build_projected_single_source_plan(
        trajectory,
        source_xy,
        anchor_to_source_xy,
    )
    request = EuvsGen3cConditioningInput(
        conditioning,
        generation_input.source_rgb,
        generation_input.source_geometry,
    )
    return EuvsGenerationPlan(
        geometry=geometry,
        input=generation_input,
        mapping=mapping,
        timeline=timeline,
        conditioning=conditioning,
        request=request,
    )


def build_euvs_generation_input(
    pair: EuvsPairInput,
    source_rgb: EuvsRasterizedSequence,
    source_geometry: PosedDepthSequence,
) -> EuvsGenerationInput:
    """Join exact source identity once and derive target K/W2C on its grid."""
    from novel_view.inputs.nuplan.raster import build_nuplan_output_intrinsics

    if not (
        pair.source.sequence_id
        == source_rgb.sequence_id
        == source_geometry.sequence_id
    ):
        raise EuvsGenerationPlanError(
            f"pair {pair.name!r} source RGB and geometry identities differ"
        )
    if source_rgb.image_size_hw != source_geometry.image_size_hw:
        raise EuvsGenerationPlanError(
            f"pair {pair.name!r} source RGB and geometry raster sizes differ"
        )

    output_size_hw = source_rgb.image_size_hw
    target_intrinsics = np.stack(
        [
            build_nuplan_output_intrinsics(frame.calibration, output_size_hw)
            for frame in pair.target.frames
        ]
    )
    target_camera = build_reference_to_camera_transforms(
        pair.source.frames[0].geometry,
        tuple(frame.geometry for frame in pair.target.frames),
    )
    return EuvsGenerationInput(
        pair=pair,
        source_rgb=source_rgb,
        source_geometry=source_geometry,
        target_intrinsics=_readonly_float64(target_intrinsics),
        reference_to_target_camera=_readonly_float64(target_camera),
    )


def build_euvs_source_mapping(pair: EuvsPairInput) -> EuvsSourceMapping:
    """Project targets and select the nearest endpoint of each chosen segment."""
    source_xy = _camera_centres_xy(pair.source.frames)
    target_xy = _camera_centres_xy(pair.target.frames)
    projection = project_points_to_polyline(source_xy, target_xy)
    excess = _endpoint_excess(source_xy, target_xy, projection)
    source_progress = _path_progress(source_xy)
    target_progress = projection.source_progress + excess
    source_index = select_nearest_polyline_vertex_indices(projection)
    residual = source_progress[source_index] - target_progress
    alignment = EuvsAlignmentCandidate(
        projection=projection,
        source_domain_excess=_readonly_float64(excess),
    )
    return EuvsSourceMapping(
        alignment=alignment,
        source_vertex_progress=_readonly_float64(source_progress),
        target_source_progress=_readonly_float64(target_progress),
        source_sequence_index=_readonly_int64(source_index),
        source_progress_residual=_readonly_float64(residual),
    )


def _camera_centres_xy(
    frames: tuple[EuvsFrameInput, ...],
) -> npt.NDArray[np.float64]:
    return np.stack(
        [frame.geometry.camera_center_global[:2] for frame in frames]
    )


def _path_progress(points_xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    progress = np.zeros(points_xy.shape[0], dtype=np.float64)
    steps = np.diff(points_xy, axis=0)
    progress[1:] = np.cumsum(np.hypot(steps[:, 0], steps[:, 1]))
    return progress


def _endpoint_excess(
    source_xy: npt.NDArray[np.float64],
    target_xy: npt.NDArray[np.float64],
    projection: PolylineProjection,
) -> npt.NDArray[np.float64]:
    excess = np.zeros(target_xy.shape[0], dtype=np.float64)
    first_delta = source_xy[1] - source_xy[0]
    first_direction = first_delta / np.hypot(*first_delta)
    before = (
        (projection.segment_index == 0)
        & (projection.segment_fraction == 0.0)
    )
    excess[before] = np.minimum(
        (target_xy[before] - source_xy[0]) @ first_direction,
        0.0,
    )

    last_segment = source_xy.shape[0] - 2
    last_delta = source_xy[-1] - source_xy[-2]
    last_direction = last_delta / np.hypot(*last_delta)
    after = (
        (projection.segment_index == last_segment)
        & (projection.segment_fraction == 1.0)
    )
    excess[after] = np.maximum(
        (target_xy[after] - source_xy[-1]) @ last_direction,
        0.0,
    )
    return excess


def _conditioning_source_frame(
    generation_input: EuvsGenerationInput,
    anchor_source_sequence_index: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    source_frames = generation_input.pair.source.frames
    physical_reference = source_frames[0].geometry
    reference_center = physical_reference.camera_center_global
    source_xy = np.stack(
        [
            frame.geometry.camera_center_global[:2] - reference_center[:2]
            for frame in source_frames
        ]
    )
    anchor_w2c = generation_input.source_geometry.reference_to_camera[
        anchor_source_sequence_index
    ]
    anchor_to_reference = invert_rigid(anchor_w2c)
    reference_to_global_rotation = physical_reference.camera_to_global[:3, :3]
    anchor_to_source_xy = np.eye(4, dtype=np.float64)
    anchor_to_source_xy[:3, :3] = (
        reference_to_global_rotation @ anchor_to_reference[:3, :3]
    )
    anchor_to_source_xy[:3, 3] = (
        reference_to_global_rotation @ anchor_to_reference[:3, 3]
    )
    return source_xy, anchor_to_source_xy


def _readonly_float64(value: object) -> npt.NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )


def _readonly_int64(value: object) -> npt.NDArray[np.int64]:
    array = np.array(value, dtype=np.int64, copy=True, order="C")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.int64).reshape(
        array.shape
    )


__all__ = [
    "EuvsAlignmentCandidate",
    "EuvsGenerationInput",
    "EuvsGenerationPlan",
    "EuvsGenerationPlanError",
    "EuvsSourceMapping",
    "build_euvs_generation_input",
    "build_euvs_generation_plan",
    "build_euvs_source_mapping",
]

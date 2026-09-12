"""Build compact direct-camera conditioning for pinned Gen3C windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from novel_view.generation.gen3c.timeline import Gen3cTimeline, Gen3cWindow
from novel_view.geometry.camera import camera_centre, invert_rigid
from novel_view.geometry.polyline import (
    project_points_to_polyline,
    select_nearest_polyline_vertex_indices,
)
from novel_view.geometry.trajectory import (
    CameraInterpolationError,
    interpolate_w2c_shortest,
)


class Gen3cConditioningError(ValueError):
    """A requested camera trajectory is scientifically ambiguous."""


@dataclass(frozen=True, slots=True, eq=False)
class Gen3cDirectQueryTrajectory:
    """Source and measured target cameras in one source-anchor frame."""

    timeline: Gen3cTimeline
    anchor_source_sequence_index: int
    anchor_to_source_camera: npt.NDArray[np.float64]
    anchor_to_target_camera: npt.NDArray[np.float64]
    source_intrinsics: npt.NDArray[np.float64]
    target_intrinsics: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class Gen3cProjectedSingleSourcePlan:
    """Projected-nearest single-source conditioning baseline."""

    query_trajectory: Gen3cDirectQueryTrajectory
    source_polyline_xy: npt.NDArray[np.float64]
    anchor_to_source_polyline_frame: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class Gen3cSingleSourceConditioningWindow:
    """One materialized 121-slot camera and source schedule."""

    window: Gen3cWindow
    anchor_to_query_camera: npt.NDArray[np.float64]
    query_intrinsics: npt.NDArray[np.float64]
    source_sequence_index: npt.NDArray[np.int64]


def build_direct_gen3c_query_trajectory(
    timeline: Gen3cTimeline,
    source_reference_to_camera: npt.ArrayLike,
    reference_to_target_camera: npt.ArrayLike,
    source_intrinsics: npt.ArrayLike,
    target_intrinsics: npt.ArrayLike,
) -> Gen3cDirectQueryTrajectory:
    """Rebase source and target cameras together at the first target source."""
    source_camera = np.asarray(source_reference_to_camera, dtype=np.float64)
    target_camera = np.asarray(reference_to_target_camera, dtype=np.float64)
    anchor_index = int(timeline.source_sequence_index[0])
    anchor_to_reference = invert_rigid(source_camera[anchor_index])
    rebased_source = source_camera @ anchor_to_reference
    rebased_target = target_camera @ anchor_to_reference
    rebased_source[anchor_index] = np.eye(4, dtype=np.float64)
    return Gen3cDirectQueryTrajectory(
        timeline=timeline,
        anchor_source_sequence_index=anchor_index,
        anchor_to_source_camera=_readonly_float64(rebased_source),
        anchor_to_target_camera=_readonly_float64(rebased_target),
        source_intrinsics=_readonly_float64(source_intrinsics),
        target_intrinsics=_readonly_float64(target_intrinsics),
    )


def build_projected_single_source_plan(
    query_trajectory: Gen3cDirectQueryTrajectory,
    source_polyline_xy: npt.ArrayLike,
    anchor_to_source_polyline_frame: npt.ArrayLike,
) -> Gen3cProjectedSingleSourcePlan:
    """Bind one measured source polyline to a direct query trajectory."""
    return Gen3cProjectedSingleSourcePlan(
        query_trajectory=query_trajectory,
        source_polyline_xy=_readonly_float64(source_polyline_xy),
        anchor_to_source_polyline_frame=_readonly_float64(
            anchor_to_source_polyline_frame
        ),
    )


def materialize_gen3c_conditioning_window(
    plan: Gen3cProjectedSingleSourcePlan,
    window: Gen3cWindow,
) -> Gen3cSingleSourceConditioningWindow:
    """Materialize W2C, K and one source index for a canonical window."""
    slots = np.arange(window.start, window.stop, dtype=np.int64)
    query_camera, query_intrinsics = _materialize_query_cameras(
        plan.query_trajectory, slots
    )
    query_xy = _query_centres_source_xy(plan, query_camera)
    projection = project_points_to_polyline(plan.source_polyline_xy, query_xy)
    source_index = select_nearest_polyline_vertex_indices(projection)
    _restore_exact_source_choices(plan.query_trajectory, slots, source_index)
    return Gen3cSingleSourceConditioningWindow(
        window=window,
        anchor_to_query_camera=_readonly_float64(query_camera),
        query_intrinsics=_readonly_float64(query_intrinsics),
        source_sequence_index=_readonly_int64(source_index),
    )


def _materialize_query_cameras(
    trajectory: Gen3cDirectQueryTrajectory,
    slots: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    timeline = trajectory.timeline
    target_slots = timeline.target_output_index
    target_camera = trajectory.anchor_to_target_camera
    target_intrinsics = trajectory.target_intrinsics
    anchor_intrinsics = trajectory.source_intrinsics[
        trajectory.anchor_source_sequence_index
    ]

    cameras = np.empty((slots.size, 4, 4), dtype=np.float64)
    intrinsics = np.empty((slots.size, 3, 3), dtype=np.float64)
    for local_index, slot_value in enumerate(slots):
        slot = int(slot_value)
        if slot == 0:
            cameras[local_index] = np.eye(4, dtype=np.float64)
            intrinsics[local_index] = anchor_intrinsics
            continue

        right = int(np.searchsorted(target_slots, slot, side="left"))
        if right == target_slots.size:
            cameras[local_index] = target_camera[-1]
            intrinsics[local_index] = target_intrinsics[-1]
        elif int(target_slots[right]) == slot:
            cameras[local_index] = target_camera[right]
            intrinsics[local_index] = target_intrinsics[right]
        else:
            left = right - 1
            denominator = int(target_slots[right] - target_slots[left])
            alpha = (slot - int(target_slots[left])) / denominator
            try:
                cameras[local_index] = interpolate_w2c_shortest(
                    target_camera[left], target_camera[right], alpha
                )
            except CameraInterpolationError as error:
                raise Gen3cConditioningError(
                    "shortest target-camera rotation is ambiguous at pi "
                    f"between target indices {left} and {right}"
                ) from error
            intrinsics[local_index] = (
                (1.0 - alpha) * target_intrinsics[left]
                + alpha * target_intrinsics[right]
            )
            intrinsics[local_index, 0, 1] = 0.0
            intrinsics[local_index, 1, 0] = 0.0
            intrinsics[local_index, 2] = (0.0, 0.0, 1.0)
    return cameras, intrinsics


def _query_centres_source_xy(
    plan: Gen3cProjectedSingleSourcePlan,
    query_camera: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    centres_anchor = np.stack([camera_centre(value) for value in query_camera])
    homogeneous = np.concatenate(
        [centres_anchor, np.ones((centres_anchor.shape[0], 1))], axis=1
    )
    centres_source_frame = (
        plan.anchor_to_source_polyline_frame @ homogeneous.T
    ).T
    return centres_source_frame[:, :2]


def _restore_exact_source_choices(
    trajectory: Gen3cDirectQueryTrajectory,
    slots: npt.NDArray[np.int64],
    source_index: npt.NDArray[np.int64],
) -> None:
    timeline = trajectory.timeline
    target_slots = timeline.target_output_index
    right = np.searchsorted(target_slots, slots, side="left")
    within = right < target_slots.size
    safe_right = np.minimum(right, target_slots.size - 1)
    exact = within & (target_slots[safe_right] == slots)
    source_index[exact] = timeline.source_sequence_index[right[exact]]
    source_index[slots == 0] = trajectory.anchor_source_sequence_index
    source_index[slots > target_slots[-1]] = timeline.source_sequence_index[-1]


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

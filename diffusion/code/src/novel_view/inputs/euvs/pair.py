"""Read one ordered physical EUVS pair without experiment configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from novel_view.inputs.euvs.index import FramesIndex
from novel_view.inputs.nuplan.camera import (
    GlobalCameraGeometry,
    build_global_camera_geometry,
)
from novel_view.inputs.nuplan.db import (
    RawCameraCalibration,
    RawEgoPose,
    read_camera_calibrations,
    read_ego_poses,
)
from novel_view.inputs.types import FrameRef

_Role = Literal["source", "target"]


class EuvsPairInputError(ValueError):
    """The ordered pair violates an EUVS temporal or coordinate invariant."""


@dataclass(frozen=True, slots=True)
class EuvsFrameInput:
    ref: FrameRef
    raw_ego_pose: RawEgoPose
    calibration: RawCameraCalibration
    geometry: GlobalCameraGeometry

    @property
    def image_token(self) -> str:
        return self.ref.image_token


@dataclass(frozen=True, slots=True)
class EuvsFrameSequence:
    frames: tuple[EuvsFrameInput, ...]
    traversal: str

    @property
    def sequence_id(self) -> tuple[str, ...]:
        return tuple(frame.image_token for frame in self.frames)


@dataclass(frozen=True, slots=True)
class EuvsPairInput:
    name: str
    location: str
    channel: str
    source: EuvsFrameSequence
    target: EuvsFrameSequence


def load_euvs_pair_input(
    *,
    name: str,
    location: str,
    channel: str,
    source_traversal: str,
    source_image_tokens: Sequence[str],
    target_traversal: str,
    target_image_tokens: Sequence[str],
    frames_index: FramesIndex,
) -> EuvsPairInput:
    source_refs = frames_index.resolve(
        source_image_tokens,
        location=location,
        traversal=source_traversal,
        channel=channel,
    )
    target_refs = frames_index.resolve(
        target_image_tokens,
        location=location,
        traversal=target_traversal,
        channel=channel,
    )
    _check_timestamps(name, "source", "image", source_refs)
    _check_timestamps(name, "target", "image", target_refs)

    refs = source_refs + target_refs
    poses = read_ego_poses(refs)
    source_count = len(source_refs)
    _check_timestamps(name, "source", "pose", poses[:source_count])
    _check_timestamps(name, "target", "pose", poses[source_count:])
    _check_common_epsg(name, refs, poses, source_count)

    calibrations = read_camera_calibrations(refs)
    frames = tuple(
        EuvsFrameInput(
            ref=ref,
            raw_ego_pose=pose,
            calibration=calibration,
            geometry=build_global_camera_geometry(pose, calibration),
        )
        for ref, pose, calibration in zip(refs, poses, calibrations, strict=True)
    )
    return EuvsPairInput(
        name=name,
        location=location,
        channel=channel,
        source=EuvsFrameSequence(frames[:source_count], source_traversal),
        target=EuvsFrameSequence(frames[source_count:], target_traversal),
    )


def _check_timestamps(
    pair_name: str,
    role: _Role,
    timestamp_kind: str,
    values: Sequence[FrameRef] | Sequence[RawEgoPose],
) -> None:
    for index in range(1, len(values)):
        previous = values[index - 1].timestamp_us
        current = values[index].timestamp_us
        if current <= previous:
            raise EuvsPairInputError(
                f"pair {pair_name!r} {role} {timestamp_kind} timestamps "
                f"must be strictly increasing: index {index - 1} has "
                f"{previous}, index {index} has {current}"
            )


def _check_common_epsg(
    pair_name: str,
    refs: tuple[FrameRef, ...],
    poses: tuple[RawEgoPose, ...],
    source_count: int,
) -> None:
    expected = poses[0].epsg
    for index, pose in enumerate(poses[1:], start=1):
        if pose.epsg != expected:
            role, role_index = _role_index(index, source_count)
            raise EuvsPairInputError(
                f"pair {pair_name!r} {role}[{role_index}] image_token "
                f"{refs[index].image_token!r} has EPSG {pose.epsg}, "
                f"expected {expected}"
            )


def _role_index(index: int, source_count: int) -> tuple[_Role, int]:
    if index < source_count:
        return "source", index
    return "target", index - source_count

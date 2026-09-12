"""Strict external schemas for the five Gaussian selections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, TypeAlias

from novel_view.config.job import ResolvedJob
from novel_view.config.load import load_yaml_mapping, reject_unknown_fields


_BASE_FIELDS = frozenset({"schema_version", "kind", "scene_id"})
_RANGE_FIELDS = frozenset({"start", "stop"})
_JOB_INPUT_FIELDS = frozenset({"reader", "info_file", "selection"})
_DENSE_JOB_INPUT_FIELDS = _JOB_INPUT_FIELDS | {
    "source_transforms",
    "source_intrinsics",
}
_HANDOFF_JOB_INPUT_FIELDS = frozenset(
    {"independent_attempt", "overlap_attempt", "selection"}
)


@dataclass(frozen=True, slots=True)
class GaussianInputSpec:
    """Exact producer JSON and selection references of one Gaussian job."""

    info_file: str
    selection: str


@dataclass(frozen=True, slots=True)
class GaussianDenseInputSpec:
    """Exact producer, reconstruction camera-table and selection references."""

    info_file: str
    source_transforms: str
    source_intrinsics: str
    selection: str


@dataclass(frozen=True, slots=True)
class GaussianHandoffInputSpec:
    """Two exact dense attempts and one ordered publication selection."""

    independent_attempt: str
    overlap_attempt: str
    selection: str


@dataclass(frozen=True, slots=True)
class GaussianPoseRange:
    """Inclusive producer pose range."""

    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class GaussianFullSequenceSelection:
    """Use the full producer-ordered sequence of one scene."""

    kind: Literal["full_sequence"]
    scene_id: int


@dataclass(frozen=True, slots=True)
class GaussianIndependentClipsSelection:
    """Run exact 1-based independent clip IDs in declared order."""

    kind: Literal["independent_clips"]
    scene_id: int
    clip_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GaussianDenseIndependentSelection:
    """Densify one inclusive source-pose range with fresh chunk seeds."""

    kind: Literal["dense_independent"]
    scene_id: int
    source_pose_range: GaussianPoseRange


@dataclass(frozen=True, slots=True)
class GaussianDenseOverlapSelection:
    """Densify one inclusive source-pose range with overlap21 context."""

    kind: Literal["dense_overlap21"]
    scene_id: int
    source_pose_range: GaussianPoseRange


@dataclass(frozen=True, slots=True)
class GaussianDenseHandoffSelection:
    """Publish exact dense camera IDs in declared order."""

    kind: Literal["dense_handoff"]
    scene_id: int
    dense_camera_ids: tuple[int, ...]


GaussianSelection: TypeAlias = (
    GaussianFullSequenceSelection
    | GaussianIndependentClipsSelection
    | GaussianDenseIndependentSelection
    | GaussianDenseOverlapSelection
    | GaussianDenseHandoffSelection
)


def parse_gaussian_job_input(job: ResolvedJob) -> GaussianInputSpec:
    """Parse the portable job input without opening `/data` or selection."""
    reject_unknown_fields(job.input, _JOB_INPUT_FIELDS)
    _require_fields(job.input, _JOB_INPUT_FIELDS)
    if _string(job.input, "reader") != "gaussian_depth_export":
        raise ValueError("gaussian_generation requires gaussian_depth_export")
    return GaussianInputSpec(
        info_file=_string(job.input, "info_file"),
        selection=_string(job.input, "selection"),
    )


def parse_gaussian_dense_job_input(job: ResolvedJob) -> GaussianDenseInputSpec:
    """Parse the exact dense inputs without probing their filesystem paths."""
    reject_unknown_fields(job.input, frozenset(_DENSE_JOB_INPUT_FIELDS))
    _require_fields(job.input, frozenset(_DENSE_JOB_INPUT_FIELDS))
    if _string(job.input, "reader") != "gaussian_depth_export":
        raise ValueError("gaussian_generation requires gaussian_depth_export")
    return GaussianDenseInputSpec(
        info_file=_string(job.input, "info_file"),
        source_transforms=_string(job.input, "source_transforms"),
        source_intrinsics=_string(job.input, "source_intrinsics"),
        selection=_string(job.input, "selection"),
    )


def parse_gaussian_handoff_job_input(job: ResolvedJob) -> GaussianHandoffInputSpec:
    """Parse exact runs-relative handoff inputs without opening them."""
    reject_unknown_fields(job.input, _HANDOFF_JOB_INPUT_FIELDS)
    _require_fields(job.input, _HANDOFF_JOB_INPUT_FIELDS)
    return GaussianHandoffInputSpec(
        independent_attempt=_string(job.input, "independent_attempt"),
        overlap_attempt=_string(job.input, "overlap_attempt"),
        selection=_string(job.input, "selection"),
    )


def load_gaussian_selection(source: Path) -> GaussianSelection:
    """Parse one tagged selection without scanning the producer tree."""
    values = load_yaml_mapping(source)
    _require_fields(values, _BASE_FIELDS)
    if _integer(values, "schema_version") != 1:
        raise ValueError("unsupported Gaussian selection schema_version")
    kind = _string(values, "kind")
    scene_id = _nonnegative_integer(values, "scene_id")

    if kind == "full_sequence":
        reject_unknown_fields(values, _BASE_FIELDS)
        return GaussianFullSequenceSelection(
            kind="full_sequence",
            scene_id=scene_id,
        )
    if kind == "independent_clips":
        fields = _BASE_FIELDS | {"clip_indices"}
        reject_unknown_fields(values, frozenset(fields))
        _require_fields(values, frozenset(fields))
        return GaussianIndependentClipsSelection(
            kind="independent_clips",
            scene_id=scene_id,
            clip_indices=_ordered_ids(values, "clip_indices", minimum=1),
        )
    if kind in ("dense_independent", "dense_overlap21"):
        fields = _BASE_FIELDS | {"source_pose_range"}
        reject_unknown_fields(values, frozenset(fields))
        _require_fields(values, frozenset(fields))
        pose_range = _pose_range(_mapping(values, "source_pose_range"))
        if kind == "dense_independent":
            return GaussianDenseIndependentSelection(
                "dense_independent",
                scene_id,
                pose_range,
            )
        return GaussianDenseOverlapSelection(
            "dense_overlap21",
            scene_id,
            pose_range,
        )
    if kind == "dense_handoff":
        fields = _BASE_FIELDS | {"dense_camera_ids"}
        reject_unknown_fields(values, frozenset(fields))
        _require_fields(values, frozenset(fields))
        return GaussianDenseHandoffSelection(
            kind="dense_handoff",
            scene_id=scene_id,
            dense_camera_ids=_ordered_ids(values, "dense_camera_ids", minimum=0),
        )
    raise ValueError(f"unsupported Gaussian selection kind: {kind}")


def _pose_range(values: Mapping[str, object]) -> GaussianPoseRange:
    reject_unknown_fields(values, _RANGE_FIELDS)
    _require_fields(values, _RANGE_FIELDS)
    start = _nonnegative_integer(values, "start")
    stop = _nonnegative_integer(values, "stop")
    if stop < start:
        raise ValueError("source_pose_range stop precedes start")
    return GaussianPoseRange(start=start, stop=stop)


def _ordered_ids(
    values: Mapping[str, object],
    field: str,
    *,
    minimum: int,
) -> tuple[int, ...]:
    value = values[field]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    identifiers = tuple(value)
    if any(
        not isinstance(identifier, int)
        or isinstance(identifier, bool)
        or identifier < minimum
        for identifier in identifiers
    ):
        raise ValueError(f"{field} contains an invalid identifier")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{field} contains duplicate identifiers")
    return identifiers


def _require_fields(
    values: Mapping[str, object],
    required: frozenset[str],
) -> None:
    missing = required - values.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")


def _mapping(values: Mapping[str, object], field: str) -> dict[str, object]:
    value = values[field]
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _string(values: Mapping[str, object], field: str) -> str:
    value = values[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _integer(values: Mapping[str, object], field: str) -> int:
    value = values[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _nonnegative_integer(values: Mapping[str, object], field: str) -> int:
    value = _integer(values, field)
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


__all__ = [
    "GaussianDenseInputSpec",
    "GaussianDenseHandoffSelection",
    "GaussianDenseIndependentSelection",
    "GaussianDenseOverlapSelection",
    "GaussianFullSequenceSelection",
    "GaussianHandoffInputSpec",
    "GaussianInputSpec",
    "GaussianIndependentClipsSelection",
    "GaussianPoseRange",
    "GaussianSelection",
    "load_gaussian_selection",
    "parse_gaussian_dense_job_input",
    "parse_gaussian_handoff_job_input",
    "parse_gaussian_job_input",
]

"""Read and write the two-method Waymo depth candidate v2 result."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.preparation.waymo_depth.depth import DEPTH_CAMERA_ORDER
from novel_view.preparation.waymo_depth.keyset import (
    WaymoClipKey,
    clip_key_from_mapping,
    clip_key_to_mapping,
)
from novel_view.preparation.waymo_depth.selection import (
    CANDIDATE_EXECUTION_FAILURE,
    DEPTH_BACKENDS,
    DepthCandidateFailure,
    DepthCandidateOutcome,
    DepthCandidateReport,
    DepthClipMetrics,
)


CANDIDATE_RECORD_VERSION = "gen3c-waymo-depth-candidate/v2"


@dataclass(frozen=True, slots=True)
class DepthCandidateRecord:
    clip_key: WaymoClipKey
    outcomes: tuple[DepthCandidateOutcome, ...]


def write_depth_outcome(path: Path, outcome: DepthCandidateOutcome) -> None:
    """Write one completed method before the next method starts."""
    _write_json(path, _outcome_to_mapping(outcome))


def write_candidate_record(
    path: Path,
    clip_key: WaymoClipKey,
    outcomes: tuple[DepthCandidateOutcome, ...],
) -> None:
    """Write the complete two-method candidate record in existing v2 format."""
    _write_json(
        path,
        {
            "schema_version": CANDIDATE_RECORD_VERSION,
            "clip_key": clip_key_to_mapping(clip_key),
            "outcomes": [_outcome_to_mapping(outcome) for outcome in outcomes],
        },
    )


def read_candidate_record(path: Path) -> DepthCandidateRecord:
    """Read one strict v2 record without probing any neighbouring files."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "clip_key",
        "outcomes",
    }:
        raise WaymoContractError("candidate record has unexpected fields")
    if document["schema_version"] != CANDIDATE_RECORD_VERSION:
        raise WaymoContractError("unsupported candidate record version")
    clip_key = clip_key_from_mapping(document["clip_key"])
    raw_outcomes = document["outcomes"]
    if not isinstance(raw_outcomes, list) or len(raw_outcomes) != 2:
        raise WaymoContractError("candidate record requires two outcomes")
    outcomes = tuple(_outcome_from_mapping(value) for value in raw_outcomes)
    if tuple(outcome.backend for outcome in outcomes) != DEPTH_BACKENDS:
        raise WaymoContractError("candidate outcomes are not in backend order")
    expected_axis = tuple((clip_key, camera) for camera in DEPTH_CAMERA_ORDER)
    for outcome in outcomes:
        if isinstance(outcome, DepthCandidateReport) and tuple(
            (entry.clip_key, entry.camera_name) for entry in outcome.entries
        ) != expected_axis:
            raise WaymoContractError("candidate report has the wrong clip/camera axis")
    return DepthCandidateRecord(clip_key, outcomes)


def _outcome_to_mapping(outcome: DepthCandidateOutcome) -> dict[str, object]:
    if isinstance(outcome, DepthCandidateFailure):
        return {
            "backend": outcome.backend,
            "status": "failed",
            "failures": list(outcome.failures),
        }
    return {
        "backend": outcome.backend,
        "status": "passed",
        "entries": [_metric_to_mapping(entry) for entry in outcome.entries],
    }


def _outcome_from_mapping(value: object) -> DepthCandidateOutcome:
    if not isinstance(value, dict):
        raise WaymoContractError("candidate outcome must be an object")
    backend = value.get("backend")
    if backend not in DEPTH_BACKENDS:
        raise WaymoContractError("candidate outcome has an unknown backend")
    if value.get("status") == "failed":
        failures = value.get("failures")
        if set(value) != {"backend", "status", "failures"} or not isinstance(
            failures, list
        ) or len(failures) != 1 or not isinstance(failures[0], str) or not (
            failures[0].startswith(CANDIDATE_EXECUTION_FAILURE)
            and failures[0][len(CANDIDATE_EXECUTION_FAILURE):].strip()
        ):
            raise WaymoContractError("candidate failure is malformed")
        return DepthCandidateFailure(backend, tuple(failures))
    entries = value.get("entries")
    if value.get("status") != "passed" or set(value) != {
        "backend",
        "status",
        "entries",
    } or not isinstance(entries, list) or not entries:
        raise WaymoContractError("candidate report is malformed")
    return DepthCandidateReport(
        backend,
        tuple(_metric_from_mapping(entry) for entry in entries),
    )


def _metric_to_mapping(entry: DepthClipMetrics) -> dict[str, object]:
    return {
        "clip_key": clip_key_to_mapping(entry.clip_key),
        "camera_name": entry.camera_name,
        "heldout_count": entry.heldout_count,
        "valid_fraction": entry.valid_fraction,
        "median_abs_rel": entry.median_abs_rel,
        "median_abs_z_m": entry.median_abs_z_m,
        "p95_abs_z_m": entry.p95_abs_z_m,
        "temporal_scale_jitter": entry.temporal_scale_jitter,
        "elapsed_seconds": entry.elapsed_seconds,
        "peak_ram_mib": entry.peak_ram_mib,
        "peak_vram_mib": entry.peak_vram_mib,
    }


def _metric_from_mapping(value: object) -> DepthClipMetrics:
    fields = {
        "clip_key",
        "camera_name",
        "heldout_count",
        "valid_fraction",
        "median_abs_rel",
        "median_abs_z_m",
        "p95_abs_z_m",
        "temporal_scale_jitter",
        "elapsed_seconds",
        "peak_ram_mib",
        "peak_vram_mib",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise WaymoContractError("candidate metric has unexpected fields")
    camera = value["camera_name"]
    count = value["heldout_count"]
    if camera not in DEPTH_CAMERA_ORDER or type(count) is not int or count < 0:
        raise WaymoContractError("candidate metric axis/count is invalid")
    numbers = {
        name: _nonnegative(value[name], name)
        for name in fields - {"clip_key", "camera_name", "heldout_count"}
    }
    if numbers["valid_fraction"] > 1.0:
        raise WaymoContractError("valid_fraction must not exceed one")
    return DepthClipMetrics(
        clip_key=clip_key_from_mapping(value["clip_key"]),
        camera_name=camera,
        heldout_count=count,
        **numbers,
    )


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise WaymoContractError(f"{name} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise WaymoContractError(f"{name} must be a non-negative number")
    return result


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CANDIDATE_RECORD_VERSION",
    "DepthCandidateRecord",
    "read_candidate_record",
    "write_candidate_record",
    "write_depth_outcome",
]

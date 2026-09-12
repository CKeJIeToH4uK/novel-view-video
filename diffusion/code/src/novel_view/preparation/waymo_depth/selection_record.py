"""The current Waymo depth selection v4 result: selected or scientific stop."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import cast

from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.preparation.waymo_depth.depth import DEPTH_CAMERA_ORDER
from novel_view.preparation.waymo_depth.keyset import (
    WaymoClipKey,
    clip_key_from_mapping,
    clip_key_to_mapping,
)
from novel_view.preparation.waymo_depth.selection import (
    DEPTH_BACKENDS,
    DepthCandidateGateSpec,
    DepthCandidateScore,
    DepthGateDecision,
    DepthGateOutcome,
    DepthGateSpec,
    DepthGateStop,
    DepthSelectionEvidence,
    DepthSelectionEvidenceEntry,
)


SELECTION_RECORD_VERSION = "gen3c-waymo-depth-selection/v4"


@dataclass(frozen=True, slots=True)
class DepthSelectionRecord:
    """One old or current outcome over eight ordered candidate records."""

    schema_version: str
    workflow_version: int
    gate_id: str
    gate_spec: DepthGateSpec | DepthCandidateGateSpec
    selection_keys: tuple[WaymoClipKey, ...]
    candidate_results: tuple[str, ...]
    outcome: DepthGateOutcome
    selection_evidence: DepthSelectionEvidence | None


def write_selection_record(path: Path, record: DepthSelectionRecord) -> None:
    """Write the provenance-free v4 shape used by the new workflow."""
    document: dict[str, object] = {
        "schema_version": SELECTION_RECORD_VERSION,
        "workflow": {
            "name": "waymo_depth_selection",
            "version": record.workflow_version,
        },
        "gate_id": record.gate_id,
        "gate_spec": _gate_spec_to_mapping(record.gate_spec),
        "selection_keys": [clip_key_to_mapping(key) for key in record.selection_keys],
        "candidate_results": list(record.candidate_results),
        "outcome": _selection_outcome_to_mapping(record.outcome),
    }
    if record.selection_evidence is not None:
        document["selection_evidence"] = _evidence_to_mapping(
            record.selection_evidence
        )
    _write_json(path, document)


def read_selection_record(path: Path) -> DepthSelectionRecord:
    """Read v4, adapting an old v2/v3 envelope only when one is encountered."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise WaymoContractError("selection record must be an object")
    schema = document.get("schema_version")
    if schema != SELECTION_RECORD_VERSION:
        from novel_view.preparation.waymo_depth.legacy_selection_record import (
            normalize_legacy_selection_record,
        )

        document = normalize_legacy_selection_record(document)
    workflow = document.get("workflow")
    if not isinstance(workflow, dict) or set(workflow) != {"name", "version"} or (
        workflow.get("name") != "waymo_depth_selection"
    ) or workflow.get("version") not in (1, 2):
        raise WaymoContractError("selection v4 workflow identity is invalid")
    workflow_version = cast(int, workflow["version"])
    expected = {
        "schema_version", "workflow", "gate_id", "gate_spec",
        "selection_keys", "candidate_results", "outcome",
    }
    if workflow_version == 2:
        expected.add("selection_evidence")
    gate_id = f"waymo-depth-gate-v{workflow_version}"
    if set(document) != expected or document.get("gate_id") != gate_id:
        raise WaymoContractError("selection v4 fields or gate identity are invalid")
    gate_spec = _gate_spec_from_mapping(document["gate_spec"], workflow_version)
    candidates = _candidate_locators(document["candidate_results"])
    evidence = (
        _evidence_from_mapping(document["selection_evidence"])
        if workflow_version == 2 else None
    )
    outcome = _selection_outcome_from_mapping(document["outcome"])
    keys = _selection_keys_from_mapping(document["selection_keys"])
    if evidence is not None:
        expected_axis = tuple(
            (index, key, camera)
            for index, key in enumerate(keys)
            for camera in DEPTH_CAMERA_ORDER
        )
        actual_axis = tuple(
            (entry.selection_index, entry.clip_key, entry.camera_name)
            for entry in evidence.entries
        )
        if actual_axis != expected_axis:
            raise WaymoContractError("selection evidence has the wrong 8x3 axis")
    return DepthSelectionRecord(
        schema_version=cast(str, schema),
        workflow_version=workflow_version,
        gate_id=cast(str, document["gate_id"]),
        gate_spec=gate_spec,
        selection_keys=keys,
        candidate_results=candidates,
        outcome=outcome,
        selection_evidence=evidence,
    )


def _gate_spec_to_mapping(
    spec: DepthGateSpec | DepthCandidateGateSpec,
) -> dict[str, object]:
    values = {
        "minimum_valid_fraction": spec.minimum_valid_fraction,
        "maximum_median_abs_rel": spec.maximum_median_abs_rel,
        "maximum_p95_abs_z_m": spec.maximum_p95_abs_z_m,
        "epsilon_depth": spec.epsilon_depth,
        "maximum_peak_ram_mib": spec.maximum_peak_ram_mib,
        "maximum_peak_vram_mib": spec.maximum_peak_vram_mib,
    }
    if isinstance(spec, DepthGateSpec):
        values = {"minimum_heldout_count": spec.minimum_heldout_count, **values}
    return values


def _gate_spec_from_mapping(
    value: object,
    workflow_version: int,
) -> DepthGateSpec | DepthCandidateGateSpec:
    common = {
        "minimum_valid_fraction", "maximum_median_abs_rel",
        "maximum_p95_abs_z_m", "epsilon_depth", "maximum_peak_ram_mib",
        "maximum_peak_vram_mib",
    }
    fields = common | ({"minimum_heldout_count"} if workflow_version == 1 else set())
    if not isinstance(value, dict) or set(value) != fields:
        raise WaymoContractError("selection gate spec has unexpected fields")
    numbers = {name: _nonnegative(value[name], name) for name in common}
    if workflow_version == 1:
        count = value["minimum_heldout_count"]
        if type(count) is not int or count <= 0:
            raise WaymoContractError("minimum_heldout_count must be positive int")
        if numbers["minimum_valid_fraction"] > 1.0:
            raise WaymoContractError("minimum_valid_fraction must not exceed one")
        return DepthGateSpec(minimum_heldout_count=count, **numbers)
    if numbers["minimum_valid_fraction"] > 1.0:
        raise WaymoContractError("minimum_valid_fraction must not exceed one")
    return DepthCandidateGateSpec(**numbers)


def _selection_keys_from_mapping(value: object) -> tuple[WaymoClipKey, ...]:
    if not isinstance(value, list) or len(value) != 8:
        raise WaymoContractError("selection record requires eight clip keys")
    keys = tuple(clip_key_from_mapping(item) for item in value)
    identities = {(key.official_partition, key.segment_id) for key in keys}
    if len(identities) != 8:
        raise WaymoContractError("selection record repeats a clip identity")
    return keys


def _candidate_locators(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != 8:
        raise WaymoContractError("selection record requires eight candidate locators")
    locators = tuple(_text(item, "candidate locator") for item in value)
    if len(set(locators)) != 8:
        raise WaymoContractError("candidate locators must be distinct")
    return locators


def _selection_outcome_to_mapping(outcome: DepthGateOutcome) -> dict[str, object]:
    if isinstance(outcome, DepthGateStop):
        return {
            "status": "scientific_stop",
            "scores": [_score_to_mapping(score) for score in outcome.scores],
        }
    return {
        "status": "selected",
        "selected_backend": outcome.selected_backend,
        "scores": [_score_to_mapping(score) for score in outcome.scores],
    }


def _score_to_mapping(score: DepthCandidateScore) -> dict[str, object]:
    return {
        "backend": score.backend,
        "passed": score.passed,
        "failures": list(score.failures),
        "median_abs_rel": score.median_abs_rel,
        "median_p95_abs_z_m": score.median_p95_abs_z_m,
        "median_temporal_scale_jitter": score.median_temporal_scale_jitter,
        "peak_vram_mib": score.peak_vram_mib,
        "total_elapsed_seconds": score.total_elapsed_seconds,
    }


def _selection_outcome_from_mapping(value: object) -> DepthGateOutcome:
    if not isinstance(value, dict) or value.get("status") not in {
        "selected",
        "scientific_stop",
    }:
        raise WaymoContractError("selection outcome is malformed")
    if value["status"] == "selected":
        if set(value) != {"status", "selected_backend", "scores"}:
            raise WaymoContractError("selected outcome is malformed")
        return _decision_from_mapping(
            {
                "selected_backend": value["selected_backend"],
                "scores": value["scores"],
            }
        )
    if set(value) != {"status", "scores"}:
        raise WaymoContractError("scientific stop outcome is malformed")
    scores = _scores_from_mapping(value["scores"])
    if any(score.passed for score in scores):
        raise WaymoContractError("scientific stop contains a passing backend")
    return DepthGateStop(scores)


def _decision_from_mapping(value: object) -> DepthGateDecision:
    if not isinstance(value, dict) or set(value) != {"selected_backend", "scores"} or (
        value["selected_backend"] not in DEPTH_BACKENDS
    ):
        raise WaymoContractError("selection decision is malformed")
    scores = _scores_from_mapping(value["scores"])
    selected_score = next(
        score for score in scores if score.backend == value["selected_backend"]
    )
    if not selected_score.passed:
        raise WaymoContractError("selected depth backend did not pass its gate")
    return DepthGateDecision(value["selected_backend"], scores)


def _scores_from_mapping(value: object) -> tuple[DepthCandidateScore, ...]:
    if not isinstance(value, list) or len(value) != 2:
        raise WaymoContractError("selection outcome requires two scores")
    score_fields = {
        "backend", "passed", "failures", "median_abs_rel",
        "median_p95_abs_z_m", "median_temporal_scale_jitter", "peak_vram_mib",
        "total_elapsed_seconds",
    }
    scores = []
    optional_numbers = score_fields - {"backend", "passed", "failures"}
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != score_fields or (
            raw["backend"] not in DEPTH_BACKENDS or type(raw["passed"]) is not bool
        ):
            raise WaymoContractError("selection score is malformed")
        failures = raw["failures"]
        if not isinstance(failures, list) or any(not isinstance(item, str) for item in failures):
            raise WaymoContractError("selection score failures are malformed")
        if raw["passed"] != (not failures):
            raise WaymoContractError("selection score status contradicts its failures")
        numbers = {
            name: None if raw[name] is None else _nonnegative(raw[name], name)
            for name in optional_numbers
        }
        if raw["passed"] and any(number is None for number in numbers.values()):
            raise WaymoContractError("passing depth score requires measured metrics")
        scores.append(
            DepthCandidateScore(
                backend=raw["backend"],
                passed=raw["passed"],
                failures=tuple(failures),
                **numbers,
            )
        )
    if tuple(score.backend for score in scores) != DEPTH_BACKENDS:
        raise WaymoContractError("selection score backend order is invalid")
    return tuple(scores)


def _evidence_to_mapping(evidence: DepthSelectionEvidence) -> dict[str, object]:
    return {
        "entries": [
            {
                "selection_index": entry.selection_index,
                "clip_key": clip_key_to_mapping(entry.clip_key),
                "camera_name": entry.camera_name,
                "heldout_count": entry.heldout_count,
            }
            for entry in evidence.entries
        ]
    }


def _evidence_from_mapping(value: object) -> DepthSelectionEvidence:
    if not isinstance(value, dict) or set(value) != {"entries"} or not isinstance(
        value["entries"], list
    ):
        raise WaymoContractError("selection evidence is malformed")
    entries = []
    for raw in value["entries"]:
        fields = {"selection_index", "clip_key", "camera_name", "heldout_count"}
        if not isinstance(raw, dict) or set(raw) != fields:
            raise WaymoContractError("selection evidence entry is malformed")
        index, camera, count = raw["selection_index"], raw["camera_name"], raw["heldout_count"]
        if type(index) is not int or index < 0 or camera not in DEPTH_CAMERA_ORDER or (
            type(count) is not int or count <= 0
        ):
            raise WaymoContractError("selection evidence entry values are invalid")
        entries.append(
            DepthSelectionEvidenceEntry(
                index, clip_key_from_mapping(raw["clip_key"]), camera, count
            )
        )
    return DepthSelectionEvidence(tuple(entries))


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise WaymoContractError(f"{name} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise WaymoContractError(f"{name} must be a non-negative number")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WaymoContractError(f"{name} must be non-empty text")
    return value


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DepthSelectionRecord",
    "SELECTION_RECORD_VERSION",
    "read_selection_record",
    "write_selection_record",
]

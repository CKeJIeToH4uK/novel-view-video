"""Read-only adaptation of historical selection v2/v3 envelopes.

Historical hash strings are parsed as fields only. Referenced files, Git and
digests are never opened, compared or recomputed; no old-format writer exists.
"""

from __future__ import annotations

from novel_view.inputs.waymo.types import WaymoContractError


def normalize_legacy_selection_record(document: dict[str, object]) -> dict[str, object]:
    """Translate only the old envelope; scientific fields are read once by v4."""
    schema = document.get("schema_version")
    if schema == "gen3c-waymo-depth-selection/v2":
        expected = {
            "schema_version", "gate_id", "gate_spec", "selection_keys",
            "candidate_results", "decision",
        }
        if set(document) != expected or document.get("gate_id") != "waymo-depth-gate-v1":
            raise WaymoContractError("selection v2 fields or gate identity are invalid")
        workflow_version = 1
        candidates = document["candidate_results"]
    elif schema == "gen3c-waymo-depth-selection/v3":
        expected = {
            "schema_version", "gate_id", "interpretation", "amendment",
            "amendment_code_commit", "gate_spec", "selection_keys",
            "candidate_results", "selection_evidence", "decision",
        }
        if set(document) != expected or document.get("gate_id") != "waymo-depth-gate-v2" or (
            document.get("interpretation") != "exploratory"
        ):
            raise WaymoContractError("selection v3 fields or identity are invalid")
        _read_historical_amendment(
            document["amendment"], document["amendment_code_commit"]
        )
        workflow_version = 2
        candidates = _legacy_candidate_paths(document["candidate_results"])
    else:
        raise WaymoContractError("unsupported selection record version")
    decision = document["decision"]
    if not isinstance(decision, dict) or set(decision) != {"selected_backend", "scores"}:
        raise WaymoContractError("selection decision is malformed")
    current = {
        "schema_version": "gen3c-waymo-depth-selection/v4",
        "workflow": {"name": "waymo_depth_selection", "version": workflow_version},
        "gate_id": document["gate_id"],
        "gate_spec": document["gate_spec"],
        "selection_keys": document["selection_keys"],
        "candidate_results": candidates,
        "outcome": {"status": "selected", **decision},
    }
    if workflow_version == 2:
        current["selection_evidence"] = document["selection_evidence"]
    return current


def _legacy_candidate_paths(value: object) -> list[object]:
    if not isinstance(value, list):
        raise WaymoContractError("historical candidate locators must be a list")
    paths = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise WaymoContractError("historical candidate locator is malformed")
        _hex(item["sha256"], 64, "candidate sha256")
        paths.append(item["path"])
    return paths


def _read_historical_amendment(value: object, commit: object) -> None:
    fields = {
        "supersedes_gate_id", "v1_outcome", "reason", "interpretation",
        "source_candidate_schema", "source_producer_commit", "source_results_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields or (
        value["supersedes_gate_id"] != "waymo-depth-gate-v1"
        or value["v1_outcome"] != "stopped"
        or value["interpretation"] != "exploratory"
        or value["source_candidate_schema"] != "gen3c-waymo-depth-candidate/v2"
    ):
        raise WaymoContractError("historical selection amendment is malformed")
    _text(value["reason"], "amendment reason")
    _hex(value["source_producer_commit"], 40, "source producer commit")
    _hex(commit, 40, "amendment code commit")
    hashes = value["source_results_sha256"]
    if not isinstance(hashes, list) or len(hashes) != 8:
        raise WaymoContractError("historical source hashes are malformed")
    for digest in hashes:
        _hex(digest, 64, "source result sha256")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WaymoContractError(f"{name} must be non-empty text")
    return value


def _hex(value: object, length: int, name: str) -> str:
    text = _text(value, name)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise WaymoContractError(f"{name} is malformed")
    return text


__all__ = ["normalize_legacy_selection_record"]

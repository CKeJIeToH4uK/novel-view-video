"""Deterministic v1/v2 selection of one historical Waymo depth method."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal, TypeAlias

from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.preparation.waymo_depth.depth import DEPTH_CAMERA_ORDER, DepthCameraName
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey


DepthBackendName: TypeAlias = Literal[
    "moge-v1-lidar-scale",
    "vggt-omega-sim3",
]
DEPTH_BACKENDS: tuple[DepthBackendName, ...] = (
    "moge-v1-lidar-scale",
    "vggt-omega-sim3",
)
CANDIDATE_EXECUTION_FAILURE = "candidate_execution_failed: "


@dataclass(frozen=True, slots=True)
class DepthClipMetrics:
    clip_key: WaymoClipKey
    camera_name: DepthCameraName
    heldout_count: int
    valid_fraction: float
    median_abs_rel: float
    median_abs_z_m: float
    p95_abs_z_m: float
    temporal_scale_jitter: float
    elapsed_seconds: float
    peak_ram_mib: float
    peak_vram_mib: float


@dataclass(frozen=True, slots=True)
class DepthCandidateReport:
    backend: DepthBackendName
    entries: tuple[DepthClipMetrics, ...]


@dataclass(frozen=True, slots=True)
class DepthCandidateFailure:
    backend: DepthBackendName
    failures: tuple[str, ...]


DepthCandidateOutcome: TypeAlias = DepthCandidateReport | DepthCandidateFailure


@dataclass(frozen=True, slots=True)
class DepthGateSpec:
    minimum_heldout_count: int
    minimum_valid_fraction: float
    maximum_median_abs_rel: float
    maximum_p95_abs_z_m: float
    epsilon_depth: float
    maximum_peak_ram_mib: float
    maximum_peak_vram_mib: float


@dataclass(frozen=True, slots=True)
class DepthCandidateGateSpec:
    """V2 gates that measure a candidate, not shared LiDAR evidence."""

    minimum_valid_fraction: float
    maximum_median_abs_rel: float
    maximum_p95_abs_z_m: float
    epsilon_depth: float
    maximum_peak_ram_mib: float
    maximum_peak_vram_mib: float


@dataclass(frozen=True, slots=True)
class DepthCandidateScore:
    backend: DepthBackendName
    passed: bool
    failures: tuple[str, ...]
    median_abs_rel: float | None
    median_p95_abs_z_m: float | None
    median_temporal_scale_jitter: float | None
    peak_vram_mib: float | None
    total_elapsed_seconds: float | None


@dataclass(frozen=True, slots=True)
class DepthGateDecision:
    selected_backend: DepthBackendName
    scores: tuple[DepthCandidateScore, ...]


@dataclass(frozen=True, slots=True)
class DepthGateStop:
    """Expected scientific result when neither candidate passes its gates."""

    scores: tuple[DepthCandidateScore, ...]


DepthGateOutcome: TypeAlias = DepthGateDecision | DepthGateStop


@dataclass(frozen=True, slots=True)
class DepthSelectionEvidenceEntry:
    selection_index: int
    clip_key: WaymoClipKey
    camera_name: DepthCameraName
    heldout_count: int


@dataclass(frozen=True, slots=True)
class DepthSelectionEvidence:
    entries: tuple[DepthSelectionEvidenceEntry, ...]


def select_depth_backend_v1(
    spec: DepthGateSpec,
    selection_keys: tuple[WaymoClipKey, ...],
    reports: tuple[DepthCandidateOutcome, ...],
) -> DepthGateOutcome:
    """Apply the original candidate and per-backend heldout-count gates."""
    return _select_depth_backend(spec, selection_keys, reports)


def select_depth_backend_v2(
    spec: DepthCandidateGateSpec,
    selection_keys: tuple[WaymoClipKey, ...],
    reports: tuple[DepthCandidateOutcome, ...],
) -> DepthGateOutcome:
    """Apply candidate-owned gates after shared evidence was normalized."""
    return _select_depth_backend(spec, selection_keys, reports)


def combine_candidate_outcomes(
    selection_keys: tuple[WaymoClipKey, ...],
    candidates: tuple[tuple[DepthCandidateOutcome, ...], ...],
) -> tuple[DepthCandidateOutcome, ...]:
    """Flatten eight ordered two-backend results into two 8x3 reports."""
    if len(candidates) != len(selection_keys) or len(selection_keys) != 8:
        raise WaymoContractError("selection requires eight candidate results")
    combined: list[DepthCandidateOutcome] = []
    for backend_index, backend in enumerate(DEPTH_BACKENDS):
        outcomes = tuple(candidate[backend_index] for candidate in candidates)
        if tuple(outcome.backend for outcome in outcomes) != (backend,) * 8:
            raise WaymoContractError("candidate backends do not match selection order")
        failures = tuple(
            f"selection[{index}].{failure}"
            for index, outcome in enumerate(outcomes)
            if isinstance(outcome, DepthCandidateFailure)
            for failure in outcome.failures
        )
        if failures:
            combined.append(DepthCandidateFailure(backend, failures))
        else:
            combined.append(
                DepthCandidateReport(
                    backend,
                    tuple(
                        entry
                        for outcome in outcomes
                        if isinstance(outcome, DepthCandidateReport)
                        for entry in outcome.entries
                    ),
                )
            )
    return tuple(combined)


def normalize_depth_selection_evidence(
    selection_keys: tuple[WaymoClipKey, ...],
    candidates: tuple[tuple[DepthCandidateOutcome, ...], ...],
) -> DepthSelectionEvidence:
    """Extract one positive, backend-neutral heldout count per clip/camera."""
    if len(selection_keys) != 8 or len(candidates) != 8:
        raise WaymoContractError("selection evidence requires eight clip outcomes")
    normalized: list[DepthSelectionEvidenceEntry] = []
    for selection_index, (clip_key, outcomes) in enumerate(
        zip(selection_keys, candidates, strict=True)
    ):
        if len(outcomes) != len(DEPTH_BACKENDS) or tuple(
            outcome.backend for outcome in outcomes
        ) != DEPTH_BACKENDS:
            raise WaymoContractError("selection evidence backend order is invalid")
        reports = tuple(
            outcome for outcome in outcomes if isinstance(outcome, DepthCandidateReport)
        )
        expected = tuple((clip_key, name) for name in DEPTH_CAMERA_ORDER)
        for report in reports:
            actual = tuple((entry.clip_key, entry.camera_name) for entry in report.entries)
            if actual != expected:
                raise WaymoContractError("selection evidence report axes are invalid")
        for camera_index, camera_name in enumerate(DEPTH_CAMERA_ORDER):
            counts = tuple(report.entries[camera_index].heldout_count for report in reports)
            if not counts or any(count <= 0 for count in counts):
                raise WaymoContractError("selection evidence lacks a positive count")
            if len(set(counts)) != 1:
                raise WaymoContractError("selection evidence counts disagree")
            normalized.append(
                DepthSelectionEvidenceEntry(
                    selection_index, clip_key, camera_name, counts[0]
                )
            )
    return DepthSelectionEvidence(tuple(normalized))


def _select_depth_backend(
    spec: DepthGateSpec | DepthCandidateGateSpec,
    selection_keys: tuple[WaymoClipKey, ...],
    reports: tuple[DepthCandidateOutcome, ...],
) -> DepthGateOutcome:
    identities = {(key.official_partition, key.segment_id) for key in selection_keys}
    if len(selection_keys) != 8 or len(identities) != 8:
        raise WaymoContractError("selection must contain eight distinct segments")
    if tuple(report.backend for report in reports) != DEPTH_BACKENDS:
        raise WaymoContractError("reports must contain both backends in order")
    required = tuple(
        (key, camera_name)
        for key in selection_keys
        for camera_name in DEPTH_CAMERA_ORDER
    )
    scores = tuple(_score_candidate(spec, required, report) for report in reports)
    passed = tuple(score for score in scores if score.passed)
    if not passed:
        return DepthGateStop(scores)
    best_abs_rel = min(
        score.median_abs_rel for score in passed if score.median_abs_rel is not None
    )
    finalists = tuple(
        score
        for score in passed
        if score.median_abs_rel is not None
        and score.median_abs_rel <= best_abs_rel + spec.epsilon_depth
    )
    selected = min(
        finalists,
        key=lambda score: (
            score.median_p95_abs_z_m,
            score.median_temporal_scale_jitter,
            score.peak_vram_mib,
            score.total_elapsed_seconds,
            score.backend,
        ),
    )
    return DepthGateDecision(selected.backend, scores)


def _score_candidate(
    spec: DepthGateSpec | DepthCandidateGateSpec,
    required: tuple[tuple[WaymoClipKey, DepthCameraName], ...],
    report: DepthCandidateOutcome,
) -> DepthCandidateScore:
    if isinstance(report, DepthCandidateFailure):
        return DepthCandidateScore(
            report.backend, False, report.failures, None, None, None, None, None
        )
    actual = tuple((entry.clip_key, entry.camera_name) for entry in report.entries)
    if actual != required:
        raise WaymoContractError(
            f"{report.backend} metrics do not match the ordered 8x3 selection"
        )
    failures: list[str] = []
    for index, entry in enumerate(report.entries):
        if isinstance(spec, DepthGateSpec) and (
            entry.heldout_count < spec.minimum_heldout_count
        ):
            failures.append(f"entry[{index}].heldout_count")
        if entry.valid_fraction < spec.minimum_valid_fraction:
            failures.append(f"entry[{index}].valid_fraction")
        if entry.median_abs_rel > spec.maximum_median_abs_rel:
            failures.append(f"entry[{index}].median_abs_rel")
        if entry.p95_abs_z_m > spec.maximum_p95_abs_z_m:
            failures.append(f"entry[{index}].p95_abs_z_m")
        if entry.peak_ram_mib > spec.maximum_peak_ram_mib:
            failures.append(f"entry[{index}].peak_ram_mib")
        if entry.peak_vram_mib > spec.maximum_peak_vram_mib:
            failures.append(f"entry[{index}].peak_vram_mib")
    return DepthCandidateScore(
        backend=report.backend,
        passed=not failures,
        failures=tuple(failures),
        median_abs_rel=statistics.median(entry.median_abs_rel for entry in report.entries),
        median_p95_abs_z_m=statistics.median(
            entry.p95_abs_z_m for entry in report.entries
        ),
        median_temporal_scale_jitter=statistics.median(
            entry.temporal_scale_jitter for entry in report.entries
        ),
        peak_vram_mib=max(entry.peak_vram_mib for entry in report.entries),
        total_elapsed_seconds=sum(entry.elapsed_seconds for entry in report.entries),
    )


__all__ = [
    "CANDIDATE_EXECUTION_FAILURE",
    "DEPTH_BACKENDS",
    "DepthBackendName",
    "DepthCandidateFailure",
    "DepthCandidateGateSpec",
    "DepthCandidateOutcome",
    "DepthCandidateReport",
    "DepthCandidateScore",
    "DepthClipMetrics",
    "DepthGateDecision",
    "DepthGateOutcome",
    "DepthGateSpec",
    "DepthGateStop",
    "DepthSelectionEvidence",
    "DepthSelectionEvidenceEntry",
    "combine_candidate_outcomes",
    "normalize_depth_selection_evidence",
    "select_depth_backend_v1",
    "select_depth_backend_v2",
]

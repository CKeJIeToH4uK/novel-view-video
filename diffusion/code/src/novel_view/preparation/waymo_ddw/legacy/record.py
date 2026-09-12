"""Provenance-free v2 records for the four historical DDW orders."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from novel_view.preparation.waymo_depth.keyset import (
    WaymoClipKey,
    clip_key_from_mapping,
    clip_key_to_mapping,
)


ENGINEERING_CANARY_FORMAT = "gen3c-waymo-ddw-engineering-canary/v2"
V2_CANARY_FORMAT = "gen3c-waymo-ddw-v2-canary/v2"
V3_PROBE_FORMAT = "gen3c-waymo-ddw-v3-probe/v2"
SURVEY_FORMAT = "gen3c-waymo-ddw-survey/v2"
MOGE_DEPTH_BACKEND = "moge-v1-lidar-scale"


@dataclass(frozen=True, slots=True)
class LegacyDdwOutcome:
    """Only the scientific result needed by explicit acceptance."""

    format: str
    clip_key: WaymoClipKey
    depth_selection: str
    status: str


def read_legacy_ddw_outcome(path: Path, kind: str) -> LegacyDdwOutcome:
    """Read the existing outcome, not a second telemetry or array contract."""
    from novel_view.preparation.waymo_ddw.legacy.axis import DdwVariant
    from novel_view.preparation.waymo_ddw.legacy.v2 import DDW_V2_VARIANTS

    full_axis = (DdwVariant(1, -1), DdwVariant(1, 1)) + DDW_V2_VARIANTS

    if kind == "canary-v1":
        format_name, workflow, version, axis = (
            ENGINEERING_CANARY_FORMAT,
            "waymo_ddw_canary",
            1,
            full_axis,
        )
    elif kind == "canary-v2":
        format_name, workflow, version, axis = (
            V2_CANARY_FORMAT,
            "waymo_ddw_canary",
            2,
            DDW_V2_VARIANTS,
        )
    elif kind == "probe":
        format_name, workflow, version, axis = (
            V3_PROBE_FORMAT,
            "waymo_ddw_probe",
            1,
            DDW_V2_VARIANTS[2:],
        )
    elif kind == "survey":
        format_name, workflow, version, axis = (SURVEY_FORMAT, "waymo_ddw_survey", 1, full_axis)
    else:
        raise ValueError("unknown historical DDW outcome")

    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document["format"] != format_name
        or document["workflow"] != {"name": workflow, "version": version}
        or document["depth"]["backend"] != MOGE_DEPTH_BACKEND
    ):
        raise ValueError("DDW outcome has a different scientific identity")
    clip_key = clip_key_from_mapping(document["input"]["clip"])
    outcome = document["outcome"]
    expected_axis = tuple(_variant(item) for item in axis)
    if kind == "canary-v2" and outcome["status"] == "scientific_stop":
        if (
            outcome["failed_variant"] not in expected_axis
            or _read_headroom(outcome["headroom"]).passed
        ):
            raise ValueError("v2 STOP requires an actual failed axis headroom")
    else:
        canary = kind in ("canary-v1", "canary-v2")
        rows = outcome["variants" if canary else "outcomes"]
        if tuple(row["variant"] for row in rows) != expected_axis:
            raise ValueError("DDW outcome does not contain its full ordered axis")
        headrooms = tuple(_read_headroom(row["headroom"]) for row in rows)
        passed = all(score.passed for score in headrooms)
        if outcome["status"] != ("passed" if passed else "scientific_stop") or (
            canary and not passed
        ):
            raise ValueError("DDW outcome contradicts its measured headroom")
        if kind == "canary-v1" and outcome["canary_id"] != "waymo-ddw-engineering-canary-v1":
            raise ValueError("unsupported engineering canary identity")
        if kind == "canary-v2" and (
            outcome["calibration"]["calibration_id"] != "waymo-ddw-mask-calibration-v2"
        ):
            raise ValueError("unsupported v2 calibration identity")
        for row, score in zip(rows, headrooms):
            if (row["candidate"] is not None) != score.passed or (
                row["reference_telemetry"] is not None
            ) != score.passed:
                raise ValueError("DDW reference presence contradicts its headroom")
    return LegacyDdwOutcome(
        format_name, clip_key, document["input"]["depth_selection"], outcome["status"]
    )


def _read_headroom(value: dict[str, object]) -> DdwHeadroomScore:
    from novel_view.preparation.waymo_ddw.legacy.headroom import (
        DdwHeadroomMetrics,
        score_ddw_headroom,
    )

    metrics = DdwHeadroomMetrics(**value["metrics"])
    if metrics.restoration_pixel_count <= 0 or (
        not math.isfinite(metrics.mean_absolute_corruption)
        or metrics.mean_absolute_corruption < 0
        or not 0 <= metrics.changed_pixel_fraction <= 1
    ):
        raise ValueError("DDW headroom metrics are not physical")
    score = score_ddw_headroom(metrics)
    if (
        value["gate_id"] != score.gate_id
        or value["passed"] is not score.passed
        or (value["failures"] != list(score.failures))
    ):
        raise ValueError("DDW headroom verdict disagrees with its metrics")
    return score


if TYPE_CHECKING:
    from novel_view.preparation.waymo_ddw.legacy.headroom import DdwHeadroomScore
    from novel_view.preparation.waymo_ddw.legacy.execution import DdwDepthTelemetry
    from novel_view.preparation.waymo_ddw.legacy.axis import DdwVariant
    from novel_view.preparation.waymo_ddw.legacy.canary import (
        DdwCanaryVariantEvidence,
        DdwEngineeringCanaryEvidence,
    )
    from novel_view.preparation.waymo_ddw.legacy.v2 import DdwV2ScientificStop
    from novel_view.preparation.waymo_ddw.legacy.v2_canary import (
        DdwSurveyEvidence,
        DdwV2CanaryEvidence,
        DdwV2VariantEvidence,
        DdwV3ProbeEvidence,
        DdwV3ProbeOutcome,
    )


def write_engineering_canary_record(
    path: Path,
    depth_selection: str,
    depth: DdwDepthTelemetry,
    evidence: DdwEngineeringCanaryEvidence,
) -> None:
    """Write the successful original canary; v1 has no structured STOP."""
    document = _envelope(
        ENGINEERING_CANARY_FORMAT,
        "waymo_ddw_canary",
        1,
        depth_selection,
        evidence.clip_key,
        depth,
    )
    document["outcome"] = {
        "status": "passed",
        "canary_id": evidence.canary_id,
        "variants": [_passed_variant(item) for item in evidence.variants],
    }
    _write(path, document)


def write_v2_canary_record(
    path: Path,
    depth_selection: str,
    depth: DdwDepthTelemetry,
    evidence: DdwV2CanaryEvidence,
) -> None:
    """Write a fully passed six-variant canary and its v2 calibration."""
    document = _envelope(
        V2_CANARY_FORMAT,
        "waymo_ddw_canary",
        2,
        depth_selection,
        evidence.clip_key,
        depth,
    )
    document["outcome"] = {
        "status": "passed",
        "variants": [_passed_variant(item) for item in evidence.variants],
        "calibration": {
            "calibration_id": evidence.calibration.calibration_id,
            "spec": asdict(evidence.calibration.spec),
        },
    }
    _write(path, document)


def write_v2_canary_stop_record(
    path: Path,
    depth_selection: str,
    clip_key: WaymoClipKey,
    depth: DdwDepthTelemetry,
    stop: DdwV2ScientificStop,
) -> None:
    """Write only the first failed v2 variant; the command returns code 2."""
    document = _envelope(
        V2_CANARY_FORMAT,
        "waymo_ddw_canary",
        2,
        depth_selection,
        clip_key,
        depth,
    )
    document["outcome"] = {
        "status": "scientific_stop",
        "failed_variant": _variant(stop.variant),
        "headroom": asdict(stop.headroom),
    }
    _write(path, document)


def write_probe_record(
    path: Path,
    depth_selection: str,
    depth: DdwDepthTelemetry,
    evidence: DdwV3ProbeEvidence,
) -> None:
    """Write the full four-outcome probe before its final verdict."""
    document = _envelope(
        V3_PROBE_FORMAT,
        "waymo_ddw_probe",
        1,
        depth_selection,
        evidence.clip_key,
        depth,
    )
    document["outcome"] = {
        "status": "passed" if evidence.passed else "scientific_stop",
        "outcomes": [_complete_outcome(item) for item in evidence.outcomes],
    }
    _write(path, document)


def write_survey_record(
    path: Path,
    depth_selection: str,
    depth: DdwDepthTelemetry,
    evidence: DdwSurveyEvidence,
) -> None:
    """Write the full eight-outcome survey before its final verdict."""
    document = _envelope(
        SURVEY_FORMAT,
        "waymo_ddw_survey",
        1,
        depth_selection,
        evidence.clip_key,
        depth,
    )
    document["outcome"] = {
        "status": "passed" if evidence.passed else "scientific_stop",
        "outcomes": [_complete_outcome(item) for item in evidence.outcomes],
    }
    _write(path, document)


def _envelope(
    format_name: str,
    workflow_name: str,
    workflow_version: int,
    depth_selection: str,
    clip_key: WaymoClipKey,
    depth: DdwDepthTelemetry,
) -> dict[str, object]:
    return {
        "format": format_name,
        "workflow": {"name": workflow_name, "version": workflow_version},
        "input": {
            "depth_selection": depth_selection,
            "clip": clip_key_to_mapping(clip_key),
        },
        "depth": {
            "backend": MOGE_DEPTH_BACKEND,
            "telemetry": asdict(depth),
        },
    }


def _passed_variant(
    item: DdwCanaryVariantEvidence | DdwV2VariantEvidence,
) -> dict[str, object]:
    return {
        "variant": _variant(item.candidate.variant),
        "headroom": asdict(item.headroom),
        "local_telemetry": {
            "peak_cuda_allocated_bytes": item.local_peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": item.local_peak_cuda_reserved_bytes,
            "elapsed_seconds": item.local_elapsed_seconds,
        },
        "candidate": {
            "observed": asdict(item.candidate.observed),
            "reference": asdict(item.candidate.reference),
        },
        "reference_telemetry": {
            "peak_cuda_allocated_bytes": item.reference_peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": item.reference_peak_cuda_reserved_bytes,
            "elapsed_seconds": item.reference_elapsed_seconds,
        },
    }


def _complete_outcome(item: DdwV3ProbeOutcome) -> dict[str, object]:
    candidate = (
        None
        if item.candidate is None
        else {
            "observed": asdict(item.candidate.observed),
            "reference": asdict(item.candidate.reference),
        }
    )
    reference = (
        None
        if item.candidate is None
        else {
            "peak_cuda_allocated_bytes": item.reference_peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": item.reference_peak_cuda_reserved_bytes,
            "elapsed_seconds": item.reference_elapsed_seconds,
        }
    )
    return {
        "variant": _variant(item.variant),
        "headroom": asdict(item.headroom),
        "local_telemetry": {
            "peak_cuda_allocated_bytes": item.local_peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": item.local_peak_cuda_reserved_bytes,
            "elapsed_seconds": item.local_elapsed_seconds,
        },
        "candidate": candidate,
        "reference_telemetry": reference,
    }


def _variant(variant: DdwVariant) -> dict[str, object]:
    return {
        "variant_id": variant.variant_id,
        "magnitude_m": variant.magnitude_m,
        "sign": variant.sign,
    }


def _write(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ENGINEERING_CANARY_FORMAT",
    "LegacyDdwOutcome",
    "MOGE_DEPTH_BACKEND",
    "SURVEY_FORMAT",
    "V2_CANARY_FORMAT",
    "V3_PROBE_FORMAT",
    "read_legacy_ddw_outcome",
    "write_engineering_canary_record",
    "write_probe_record",
    "write_survey_record",
    "write_v2_canary_record",
    "write_v2_canary_stop_record",
]

"""Current item, collection and audit records for the historical DDW fit."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from novel_view.preparation.waymo_depth.keyset import (
    clip_key_from_mapping,
    clip_key_to_mapping,
)


FIT_ITEM_SCHEMA = "gen3c-waymo-ddw-fit-clip/v2"
FIT_COLLECTION_SCHEMA = "gen3c-waymo-ddw-fit-collection/v1"
FIT_AUDIT_SCHEMA = "gen3c-waymo-ddw-fit-visual-audit/v2"
_ITEM_FIELDS = {
    "schema_version",
    "status",
    "global_fit_index",
    "clip_key",
    "variant",
    "headroom",
    "observed",
    "reference",
    "mask_score",
    "measurement_telemetry",
    "reference_telemetry",
    "recompute_telemetry",
    "depth_telemetry",
    "payload",
}
_DESCRIPTOR_FIELDS = {
    "coverage_q05",
    "largest_hole_fraction_p95",
    "hole_component_count_p95",
    "boundary_fraction_p95",
    "temporal_iou_q05",
    "temporal_flicker_p95",
}
_GPU_TELEMETRY_FIELDS = {
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
    "elapsed_seconds",
}
_DEPTH_TELEMETRY_FIELDS = {
    "elapsed_seconds",
    "worker_peak_ram_mib",
    "worker_peak_vram_mib",
}

if TYPE_CHECKING:
    from novel_view.preparation.waymo_ddw.legacy.assignment import DdwFitAssignment
    from novel_view.preparation.waymo_ddw.legacy.audit import (
        DdwFitAuditMaterialization,
    )
    from novel_view.preparation.waymo_ddw.legacy.axis import (
        DdwMaskCandidate,
        DdwVariant,
    )
    from novel_view.preparation.waymo_ddw.legacy.execution import DdwDepthTelemetry
    from novel_view.preparation.waymo_ddw.legacy.fit import (
        DdwFitDecision,
        DdwFitRenderTelemetry,
        DdwPreparedFitClip,
    )
    from novel_view.preparation.waymo_ddw.legacy.gate import (
        DdwMaskGateScore,
        DdwMaskGateSpec,
    )
    from novel_view.preparation.waymo_ddw.legacy.headroom import DdwHeadroomScore
    from novel_view.preparation.waymo_ddw.legacy.local import DdwLocalMeasurement
    from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors
    from novel_view.preparation.waymo_ddw.legacy.reference import (
        DdwReferenceMeasurement,
    )


@dataclass(frozen=True, slots=True)
class DdwFitPayload:
    """Locators relative to the parent of the current v2 item record."""

    condition_rgb: str
    condition_known: str


@dataclass(frozen=True, slots=True)
class DdwFitItemRecord:
    decision: DdwFitDecision
    observed: DdwMaskDescriptors
    measurement_telemetry: DdwFitRenderTelemetry
    reference_telemetry: DdwFitRenderTelemetry | None
    recompute_telemetry: DdwFitRenderTelemetry | None
    depth_telemetry: DdwDepthTelemetry
    payload: DdwFitPayload | None

    @property
    def accepted(self) -> bool:
        return self.decision.accepted


@dataclass(frozen=True, slots=True)
class DdwFitCollectionEntry:
    item_record: str
    item: DdwFitItemRecord


@dataclass(frozen=True, slots=True)
class DdwFitCollectionRecord:
    total: int
    accepted: tuple[DdwFitCollectionEntry, ...]
    rejected: tuple[DdwFitCollectionEntry, ...]
    missing_indices: tuple[int, ...]

    @property
    def status(self) -> str:
        return "complete" if not self.missing_indices else "incomplete"


def build_fit_item_record(
    result: DdwPreparedFitClip,
    depth: DdwDepthTelemetry,
) -> DdwFitItemRecord:
    """Bind one typed fit result to its directly written v2 record."""
    measurement = result.measurement
    reference = result.reference
    condition = result.accepted_condition
    return DdwFitItemRecord(
        decision=result.decision,
        observed=measurement.observed,
        measurement_telemetry=_render_telemetry(measurement),
        reference_telemetry=(
            None if reference is None else _render_telemetry(reference)
        ),
        recompute_telemetry=(None if condition is None else condition.telemetry),
        depth_telemetry=depth,
        payload=(
            None
            if condition is None
            else DdwFitPayload("condition_rgb.npy", "condition_known.npy")
        ),
    )


def write_fit_item_record(path: Path, record: DdwFitItemRecord) -> None:
    """Write one v2 item after any accepted payload has been written."""
    _write_json(path, _item_document(record))


def read_fit_item_record(path: Path) -> DdwFitItemRecord:
    """Read one strict v2 item; payload locators use ``path.parent``."""
    return _parse_fit_item_document(_read_json(path), FIT_ITEM_SCHEMA)


def read_fit_audit_source(path: Path) -> DdwFitItemRecord:
    """Read the current item or the one old rejected form used by audit."""
    document = _read_json(path)
    schema = document.get("schema_version") if isinstance(document, dict) else None
    if schema == FIT_ITEM_SCHEMA:
        return _parse_fit_item_document(document, FIT_ITEM_SCHEMA)
    from novel_view.preparation.waymo_ddw.legacy.legacy_fit_record import (
        LEGACY_FIT_ITEM_SCHEMA,
        parse_legacy_fit_item_document,
    )

    if schema == LEGACY_FIT_ITEM_SCHEMA:
        legacy_document = parse_legacy_fit_item_document(document)
        return _parse_fit_item_document(legacy_document, LEGACY_FIT_ITEM_SCHEMA)
    raise ValueError("unsupported DDW fit item schema")


def collect_fit_records(
    assignments: tuple[DdwFitAssignment, ...],
    located_records: tuple[tuple[str, DdwFitItemRecord], ...],
) -> DdwFitCollectionRecord:
    """Order explicit current item records against the full assignment axis."""
    expected = {item.global_fit_index: item for item in assignments}
    located: dict[int, DdwFitCollectionEntry] = {}
    for locator, record in located_records:
        assignment = record.decision.assignment
        index = assignment.global_fit_index
        if index in located:
            raise ValueError("collection repeats a fit item index")
        if expected.get(index) != assignment:
            raise ValueError("fit item record disagrees with assignment axis")
        located[index] = DdwFitCollectionEntry(locator, record)

    ordered = tuple(
        located[item.global_fit_index]
        for item in assignments
        if item.global_fit_index in located
    )
    return DdwFitCollectionRecord(
        total=len(assignments),
        accepted=tuple(item for item in ordered if item.item.accepted),
        rejected=tuple(item for item in ordered if not item.item.accepted),
        missing_indices=tuple(
            item.global_fit_index
            for item in assignments
            if item.global_fit_index not in located
        ),
    )


def write_fit_collection_record(
    path: Path,
    record: DdwFitCollectionRecord,
) -> None:
    """Write ordered item links without rebasing their payload locators."""
    _write_json(
        path,
        {
            "schema_version": FIT_COLLECTION_SCHEMA,
            "status": record.status,
            "total": record.total,
            "accepted": [_collection_entry(item) for item in record.accepted],
            "rejected": [_collection_entry(item) for item in record.rejected],
            "missing_indices": list(record.missing_indices),
        },
    )


def write_fit_audit_record(
    path: Path,
    item_record_locator: str,
    result: DdwFitAuditMaterialization,
    depth: DdwDepthTelemetry,
) -> None:
    """Write one v2 visual-audit result with record-relative payload links."""
    decision = result.selection.decision
    assignment = decision.assignment
    candidate = cast("DdwMaskCandidate", decision.candidate)
    _write_json(
        path,
        {
            "schema_version": FIT_AUDIT_SCHEMA,
            "status": "complete",
            "source_item_record": item_record_locator,
            "source_status": "rejected",
            "global_fit_index": assignment.global_fit_index,
            "clip_key": clip_key_to_mapping(assignment.clip_key),
            "variant": _variant_document(assignment.variant),
            "recorded_headroom": asdict(decision.headroom),
            "recorded_observed": asdict(candidate.observed),
            "recorded_reference": asdict(candidate.reference),
            "audit_mask_score": asdict(result.selection.audit_mask_score),
            "render_telemetry": asdict(result.telemetry),
            "depth_telemetry": asdict(depth),
            "payload": {
                "condition_rgb": "condition_rgb.npy",
                "condition_known": "condition_known.npy",
                "preview": "preview.mp4",
            },
        },
    )


def _parse_fit_item_document(value: object, schema: str) -> DdwFitItemRecord:
    document = _mapping(value, _ITEM_FIELDS, "fit item")
    if document["schema_version"] != schema:
        raise ValueError("unsupported DDW fit item schema")

    from novel_view.preparation.waymo_ddw.legacy.assignment import DdwFitAssignment
    from novel_view.preparation.waymo_ddw.legacy.axis import DdwMaskCandidate
    from novel_view.preparation.waymo_ddw.legacy.execution import DdwDepthTelemetry
    from novel_view.preparation.waymo_ddw.legacy.fit import (
        DDW_FIT_MASK_GATE,
        DdwFitDecision,
        DdwFitRenderTelemetry,
    )

    index = document["global_fit_index"]
    if type(index) is not int or index < 0:
        raise ValueError("global_fit_index must be a non-negative integer")
    clip_key = clip_key_from_mapping(document["clip_key"])
    variant = _variant(document["variant"])
    assignment = DdwFitAssignment(index, clip_key, variant)
    headroom = _headroom(document["headroom"])
    observed = _descriptors(document["observed"], "observed")
    reference = (
        None
        if document["reference"] is None
        else _descriptors(document["reference"], "reference")
    )
    candidate = (
        None
        if reference is None
        else DdwMaskCandidate(clip_key, variant, observed, reference)
    )
    mask_score = _mask_score(
        document["mask_score"], observed, reference, DDW_FIT_MASK_GATE
    )
    decision = DdwFitDecision(assignment, headroom, candidate, mask_score)

    status = document["status"]
    if status not in ("accepted", "rejected") or (
        (status == "accepted") != decision.accepted
    ):
        raise ValueError("fit item status disagrees with scientific decision")
    if (
        headroom.passed != (reference is not None)
        or headroom.passed != (mask_score is not None)
    ):
        raise ValueError("fit item reference disagrees with headroom decision")

    measurement = DdwFitRenderTelemetry(
        **_mapping(
            document["measurement_telemetry"],
            _GPU_TELEMETRY_FIELDS,
            "measurement telemetry",
        )
    )
    reference_telemetry = _optional_render_telemetry(
        document["reference_telemetry"]
    )
    recompute_telemetry = _optional_render_telemetry(
        document["recompute_telemetry"]
    )
    if (reference is None) != (reference_telemetry is None):
        raise ValueError("fit item reference telemetry disagrees with evidence")
    if decision.accepted != (recompute_telemetry is not None):
        raise ValueError("fit item recompute telemetry disagrees with decision")

    payload = _payload(document["payload"])
    if decision.accepted != (payload is not None):
        raise ValueError("fit item payload disagrees with decision")
    return DdwFitItemRecord(
        decision=decision,
        observed=observed,
        measurement_telemetry=measurement,
        reference_telemetry=reference_telemetry,
        recompute_telemetry=recompute_telemetry,
        depth_telemetry=DdwDepthTelemetry(
            **_mapping(
                document["depth_telemetry"],
                _DEPTH_TELEMETRY_FIELDS,
                "depth telemetry",
            )
        ),
        payload=payload,
    )


def _item_document(record: DdwFitItemRecord) -> dict[str, object]:
    decision = record.decision
    assignment = decision.assignment
    candidate = decision.candidate
    return {
        "schema_version": FIT_ITEM_SCHEMA,
        "status": "accepted" if decision.accepted else "rejected",
        "global_fit_index": assignment.global_fit_index,
        "clip_key": clip_key_to_mapping(assignment.clip_key),
        "variant": _variant_document(assignment.variant),
        "headroom": asdict(decision.headroom),
        "observed": asdict(record.observed),
        "reference": None if candidate is None else asdict(candidate.reference),
        "mask_score": (
            None if decision.mask_score is None else asdict(decision.mask_score)
        ),
        "measurement_telemetry": asdict(record.measurement_telemetry),
        "reference_telemetry": (
            None
            if record.reference_telemetry is None
            else asdict(record.reference_telemetry)
        ),
        "recompute_telemetry": (
            None
            if record.recompute_telemetry is None
            else asdict(record.recompute_telemetry)
        ),
        "depth_telemetry": asdict(record.depth_telemetry),
        "payload": None if record.payload is None else asdict(record.payload),
    }


def _collection_entry(entry: DdwFitCollectionEntry) -> dict[str, object]:
    assignment = entry.item.decision.assignment
    document = {
        "item_record": entry.item_record,
        "global_fit_index": assignment.global_fit_index,
        "clip_key": clip_key_to_mapping(assignment.clip_key),
        "variant": _variant_document(assignment.variant),
    }
    if entry.item.payload is not None:
        document["payload"] = asdict(entry.item.payload)
    return document


def _variant(value: object) -> DdwVariant:
    from novel_view.preparation.waymo_ddw.legacy.axis import DdwVariant

    document = _mapping(value, {"variant_id", "magnitude_m", "sign"}, "variant")
    magnitude = document["magnitude_m"]
    sign = document["sign"]
    if (
        type(magnitude) is not int
        or magnitude <= 0
        or type(sign) is not int
        or sign not in (-1, 1)
    ):
        raise ValueError("invalid DDW fit variant")
    variant = DdwVariant(magnitude, sign)
    if document["variant_id"] != variant.variant_id:
        raise ValueError("variant id disagrees with magnitude and sign")
    return variant


def _variant_document(variant: DdwVariant) -> dict[str, object]:
    return {
        "variant_id": variant.variant_id,
        "magnitude_m": variant.magnitude_m,
        "sign": variant.sign,
    }


def _headroom(value: object) -> DdwHeadroomScore:
    from novel_view.preparation.waymo_ddw.legacy.headroom import (
        DdwHeadroomMetrics,
        score_ddw_headroom,
    )

    document = _mapping(
        value, {"gate_id", "metrics", "passed", "failures"}, "headroom"
    )
    metrics_document = _mapping(
        document["metrics"],
        {
            "restoration_pixel_count",
            "mean_absolute_corruption",
            "changed_pixel_fraction",
        },
        "headroom metrics",
    )
    restoration_count = metrics_document["restoration_pixel_count"]
    if type(restoration_count) is not int or restoration_count <= 0:
        raise ValueError("invalid headroom restoration count")
    _nonnegative(metrics_document["mean_absolute_corruption"], "headroom mean")
    _fraction(metrics_document["changed_pixel_fraction"], "headroom fraction")
    score = score_ddw_headroom(DdwHeadroomMetrics(**metrics_document))
    if (
        document["gate_id"] != score.gate_id
        or document["passed"] is not score.passed
        or _failures(document["failures"]) != score.failures
    ):
        raise ValueError("headroom decision disagrees with metrics")
    return score


def _descriptors(value: object, name: str) -> DdwMaskDescriptors:
    from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors

    document = _mapping(value, _DESCRIPTOR_FIELDS, name)
    for field in _DESCRIPTOR_FIELDS - {"hole_component_count_p95"}:
        _fraction(document[field], f"{name}.{field}")
    _nonnegative(
        document["hole_component_count_p95"], f"{name}.hole_component_count_p95"
    )
    return DdwMaskDescriptors(**document)


def _mask_score(
    value: object,
    observed: DdwMaskDescriptors,
    reference: DdwMaskDescriptors | None,
    spec: DdwMaskGateSpec,
) -> DdwMaskGateScore | None:
    if value is None:
        return None
    if reference is None:
        raise ValueError("mask score requires reference evidence")
    from novel_view.preparation.waymo_ddw.legacy.gate import score_ddw_masks

    document = _mapping(
        value, {"observed", "reference", "passed", "failures"}, "mask score"
    )
    if (
        _descriptors(document["observed"], "mask score observed") != observed
        or _descriptors(document["reference"], "mask score reference") != reference
    ):
        raise ValueError("mask score descriptors disagree with item evidence")
    score = score_ddw_masks(spec, observed, reference)
    if (
        document["passed"] is not score.passed
        or _failures(document["failures"]) != score.failures
    ):
        raise ValueError("mask score decision disagrees with descriptors")
    return score


def _optional_render_telemetry(value: object) -> DdwFitRenderTelemetry | None:
    if value is None:
        return None
    from novel_view.preparation.waymo_ddw.legacy.fit import DdwFitRenderTelemetry

    return DdwFitRenderTelemetry(
        **_mapping(value, _GPU_TELEMETRY_FIELDS, "render telemetry")
    )


def _payload(value: object) -> DdwFitPayload | None:
    if value is None:
        return None
    document = _mapping(
        value, {"condition_rgb", "condition_known"}, "fit payload"
    )
    if (document["condition_rgb"], document["condition_known"]) != (
        "condition_rgb.npy",
        "condition_known.npy",
    ):
        raise ValueError("v2 fit payload must be relative to its item record")
    return DdwFitPayload(**document)


def _render_telemetry(
    value: DdwLocalMeasurement | DdwReferenceMeasurement,
) -> DdwFitRenderTelemetry:
    from novel_view.preparation.waymo_ddw.legacy.fit import DdwFitRenderTelemetry

    return DdwFitRenderTelemetry(
        value.peak_cuda_allocated_bytes,
        value.peak_cuda_reserved_bytes,
        value.elapsed_seconds,
    )


def _mapping(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has unexpected fields")
    return value


def _failures(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("failures must be a list of text")
    return tuple(value)


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be non-negative")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _fraction(value: object, name: str) -> float:
    number = _nonnegative(value, name)
    if number > 1.0:
        raise ValueError(f"{name} must not exceed one")
    return number


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "FIT_AUDIT_SCHEMA",
    "FIT_COLLECTION_SCHEMA",
    "FIT_ITEM_SCHEMA",
    "DdwFitCollectionEntry",
    "DdwFitCollectionRecord",
    "DdwFitItemRecord",
    "DdwFitPayload",
    "build_fit_item_record",
    "collect_fit_records",
    "read_fit_audit_source",
    "read_fit_item_record",
    "write_fit_audit_record",
    "write_fit_collection_record",
    "write_fit_item_record",
]

"""Rejected-item visual audit for the historical Waymo DDW fit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from novel_view.preparation.waymo_ddw.legacy.axis import DdwMaskCandidate
from novel_view.preparation.waymo_ddw.legacy.fit import (
    DdwFitDecision,
    DdwFitRenderTelemetry,
    encode_waymo_ddw_condition,
    same_ddw_fit_measurement,
)
from novel_view.preparation.waymo_ddw.legacy.gate import (
    DDW_MASK_GATE_ID,
    DdwMaskGateScore,
    DdwMaskGateSpec,
    DdwMaskReferenceTolerance,
    score_ddw_masks,
)
from novel_view.preparation.waymo_ddw.legacy.local import (
    measure_legacy_ddw_rendered,
)
from novel_view.preparation.waymo_ddw.legacy.path import (
    build_legacy_ddw_path,
)
from novel_view.preparation.waymo_ddw.legacy.render import (
    LegacyDdwRenderResult,
    render_legacy_ddw,
)
from novel_view.preparation.waymo_ddw.legacy.source import LegacyDdwSource
from novel_view.preparation.waymo_ddw.warp import WarpResources


AUDIT_MASK_GATE = DdwMaskGateSpec(
    DDW_MASK_GATE_ID,
    0.40,
    0.60,
    40.0,
    0.46,
    0.47,
    0.51,
    DdwMaskReferenceTolerance(0.13, 0.14, 15.75, 0.17, 0.19, 0.23),
    DdwMaskReferenceTolerance(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)


class WaymoDdwAuditError(RuntimeError):
    """The selected item is ineligible or its audit rerender changed."""


@dataclass(frozen=True, slots=True)
class DdwFitAuditSelection:
    """One rejected fit decision admitted by the separate audit gate."""

    decision: DdwFitDecision
    audit_mask_score: DdwMaskGateScore


@dataclass(frozen=True, slots=True, eq=False)
class DdwFitAuditMaterialization:
    """Encoded condition from one matching audit rerender."""

    selection: DdwFitAuditSelection
    telemetry: DdwFitRenderTelemetry
    image_size_hw: tuple[int, int]
    condition_rgb: npt.NDArray[np.uint8]
    condition_known: npt.NDArray[np.uint8]


def select_waymo_ddw_fit_audit(
    decision: DdwFitDecision,
) -> DdwFitAuditSelection:
    """Apply the visual-audit policy without changing the fit decision."""
    candidate = decision.candidate
    if decision.accepted or not decision.headroom.passed or candidate is None:
        raise WaymoDdwAuditError(
            "audit requires a rejected item with complete mask evidence"
        )
    score = score_ddw_masks(
        AUDIT_MASK_GATE,
        candidate.observed,
        candidate.reference,
    )
    if not score.passed:
        raise WaymoDdwAuditError("item does not pass the visual-audit mask gate")
    return DdwFitAuditSelection(decision, score)


def materialize_waymo_ddw_fit_audit(
    source: LegacyDdwSource,
    selection: DdwFitAuditSelection,
    warp_resources: WarpResources,
    outward_log_path: Path,
    return_log_path: Path,
) -> DdwFitAuditMaterialization:
    """Rerender one admitted item and match its recorded fit evidence."""
    decision = selection.decision
    candidate = cast(DdwMaskCandidate, decision.candidate)

    expected_path = build_legacy_ddw_path(source, decision.assignment.variant)
    rendered: LegacyDdwRenderResult | None = None
    try:
        rendered = render_legacy_ddw(
            source,
            expected_path,
            warp_resources,
            outward_log_path,
            return_log_path,
        )
        observed, headroom = measure_legacy_ddw_rendered(source, rendered)
        if not same_ddw_fit_measurement(
            expected_path,
            decision.headroom,
            candidate.observed,
            rendered.path,
            headroom,
            observed,
        ):
            raise WaymoDdwAuditError(
                "audit rerender disagrees with recorded fit measurement"
            )
        rgb, known = encode_waymo_ddw_condition(
            rendered.condition.rgb_minus_one_to_one,
            rendered.condition.known,
        )
        return DdwFitAuditMaterialization(
            selection,
            DdwFitRenderTelemetry(
                rendered.condition.peak_cuda_allocated_bytes,
                rendered.condition.peak_cuda_reserved_bytes,
                rendered.condition.elapsed_seconds,
            ),
            rgb.shape[1:3],
            rgb,
            known,
        )
    finally:
        del rendered


def write_audit_preview(
    path: Path,
    source: LegacyDdwSource,
    result: DdwFitAuditMaterialization,
) -> None:
    """Write the historical source/condition comparison for human review."""
    import cv2

    frames, height, width, _ = result.condition_rgb.shape
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (2 * width, height),
    )
    if not writer.isOpened():
        raise WaymoDdwAuditError("cannot open audit preview")
    try:
        for index in range(frames):
            condition = result.condition_rgb[index].copy()
            known = np.unpackbits(
                result.condition_known[index],
                count=height * width,
                bitorder="little",
            ).reshape(height, width)
            condition[~known.astype(np.bool_)] = (255, 0, 255)
            frame = cv2.cvtColor(
                np.concatenate((source.rgb_thwc[index], condition), axis=1),
                cv2.COLOR_RGB2BGR,
            )
            cv2.putText(
                frame,
                "SOURCE",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "DDW (MAGENTA = UNKNOWN)",
                (width + 24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()


__all__ = [
    "AUDIT_MASK_GATE",
    "DdwFitAuditMaterialization",
    "DdwFitAuditSelection",
    "WaymoDdwAuditError",
    "materialize_waymo_ddw_fit_audit",
    "select_waymo_ddw_fit_audit",
    "write_audit_preview",
]

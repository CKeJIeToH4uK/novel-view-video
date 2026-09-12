"""One preassigned historical DDW fit measurement without queue or retry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from novel_view.models.gen3c.cache4d.backend import Gen3cCache4dResources
from novel_view.preparation.waymo_ddw.legacy._mask_score import (
    DDW_MASK_METRIC_NAMES,
)
from novel_view.preparation.waymo_ddw.legacy.assignment import DdwFitAssignment
from novel_view.preparation.waymo_ddw.legacy.axis import DdwMaskCandidate
from novel_view.preparation.waymo_ddw.legacy.gate import (
    DDW_MASK_GATE_ID,
    DdwMaskGateScore,
    DdwMaskGateSpec,
    DdwMaskReferenceTolerance,
    score_ddw_masks,
)
from novel_view.preparation.waymo_ddw.legacy.headroom import DdwHeadroomScore
from novel_view.preparation.waymo_ddw.legacy.local import (
    DdwLocalMeasurement,
    measure_legacy_ddw_local,
    measure_legacy_ddw_rendered,
)
from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors
from novel_view.preparation.waymo_ddw.legacy.path import (
    LegacyDdwPath,
    build_legacy_ddw_path,
)
from novel_view.preparation.waymo_ddw.legacy.reference import (
    DdwReferenceMeasurement,
    measure_legacy_ddw_reference,
)
from novel_view.preparation.waymo_ddw.legacy.render import (
    LegacyDdwRenderResult,
    render_legacy_ddw,
)
from novel_view.preparation.waymo_ddw.legacy.source import LegacyDdwSource
from novel_view.preparation.waymo_ddw.warp import WarpResources


DDW_FIT_MASK_GATE = DdwMaskGateSpec(
    DDW_MASK_GATE_ID,
    0.56,
    0.36,
    31.0,
    0.46,
    0.47,
    0.51,
    DdwMaskReferenceTolerance(0.13, 0.14, 15.75, 0.17, 0.19, 0.23),
    DdwMaskReferenceTolerance(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
_FRAME_COUNT = 121
_FLOAT_ATOL = 1e-5
_FLOAT_RTOL = 1e-5


class WaymoDdwFitError(RuntimeError):
    """An accepted rerender changed the recorded scientific outcome."""


@dataclass(frozen=True, slots=True)
class DdwFitDecision:
    assignment: DdwFitAssignment
    headroom: DdwHeadroomScore
    candidate: DdwMaskCandidate | None
    mask_score: DdwMaskGateScore | None

    @property
    def accepted(self) -> bool:
        return bool(
            self.headroom.passed
            and self.mask_score is not None
            and self.mask_score.passed
        )


@dataclass(frozen=True, slots=True)
class DdwFitRenderTelemetry:
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True, eq=False)
class DdwAcceptedFitCondition:
    telemetry: DdwFitRenderTelemetry
    image_size_hw: tuple[int, int]
    condition_rgb: npt.NDArray[np.uint8]
    condition_known: npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True, eq=False)
class DdwPreparedFitClip:
    decision: DdwFitDecision
    measurement: DdwLocalMeasurement
    reference: DdwReferenceMeasurement | None
    accepted_condition: DdwAcceptedFitCondition | None

    @property
    def accepted(self) -> bool:
        return self.decision.accepted


def decide_ddw_fit(
    assignment: DdwFitAssignment,
    headroom: DdwHeadroomScore,
    candidate: DdwMaskCandidate | None,
) -> DdwFitDecision:
    """Apply headroom first and the frozen mask gate only when it passes."""
    if not headroom.passed:
        return DdwFitDecision(assignment, headroom, None, None)
    evidence = cast(DdwMaskCandidate, candidate)
    score = score_ddw_masks(
        DDW_FIT_MASK_GATE,
        evidence.observed,
        evidence.reference,
    )
    return DdwFitDecision(assignment, headroom, candidate, score)


def prepare_waymo_ddw_fit_clip(
    source: LegacyDdwSource,
    assignment: DdwFitAssignment,
    warp_resources: WarpResources,
    reference_resources: Gen3cCache4dResources,
    log_directory: Path,
) -> DdwPreparedFitClip:
    """Measure once, then rerender and encode only the accepted variant."""
    measurement = measure_legacy_ddw_local(
        source,
        assignment.variant,
        warp_resources,
        log_directory / "measurement-outward.log",
        log_directory / "measurement-return.log",
    )
    if not measurement.headroom.passed:
        return DdwPreparedFitClip(
            decide_ddw_fit(assignment, measurement.headroom, None),
            measurement,
            None,
            None,
        )

    reference = measure_legacy_ddw_reference(
        source,
        measurement.path,
        reference_resources,
        log_directory / "measurement-cache4d.log",
    )
    decision = decide_ddw_fit(
        assignment,
        measurement.headroom,
        DdwMaskCandidate(
            assignment.clip_key,
            assignment.variant,
            measurement.observed,
            reference.descriptors,
        ),
    )
    if not decision.accepted:
        return DdwPreparedFitClip(decision, measurement, reference, None)

    rendered: LegacyDdwRenderResult | None = None
    try:
        rendered = render_legacy_ddw(
            source,
            build_legacy_ddw_path(source, assignment.variant),
            warp_resources,
            log_directory / "recompute-outward.log",
            log_directory / "recompute-return.log",
        )
        observed, headroom = measure_legacy_ddw_rendered(source, rendered)
        repeated = decide_ddw_fit(
            assignment,
            headroom,
            DdwMaskCandidate(
                assignment.clip_key,
                assignment.variant,
                observed,
                reference.descriptors,
            ),
        )
        if not _same_recompute(
            decision,
            repeated,
            measurement.path,
            rendered.path,
        ):
            raise WaymoDdwFitError(
                "recomputed DDW outcome disagrees with accepted measurement"
            )
        rgb, known = encode_waymo_ddw_condition(
            rendered.condition.rgb_minus_one_to_one,
            rendered.condition.known,
        )
        condition = DdwAcceptedFitCondition(
            DdwFitRenderTelemetry(
                rendered.condition.peak_cuda_allocated_bytes,
                rendered.condition.peak_cuda_reserved_bytes,
                rendered.condition.elapsed_seconds,
            ),
            (rgb.shape[1], rgb.shape[2]),
            rgb,
            known,
        )
        return DdwPreparedFitClip(decision, measurement, reference, condition)
    finally:
        del rendered


def encode_waymo_ddw_condition(
    rgb: npt.NDArray[np.float32],
    known: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Encode exact TCHW RGB and THW known frames for an accepted item."""
    encoded_rgb = np.empty(
        (_FRAME_COUNT, rgb.shape[2], rgb.shape[3], 3),
        dtype=np.uint8,
    )
    encoded_known = np.empty(
        (_FRAME_COUNT, (known.shape[1] * known.shape[2] + 7) // 8),
        dtype=np.uint8,
    )
    for index in range(_FRAME_COUNT):
        encoded_rgb[index] = np.rint(
            np.clip(np.moveaxis(rgb[index], 0, -1) + 1.0, 0.0, 2.0) * 127.5
        ).astype(np.uint8)
        encoded_known[index] = np.packbits(
            known[index].reshape(-1),
            bitorder="little",
        )
    encoded_rgb.setflags(write=False)
    encoded_known.setflags(write=False)
    return encoded_rgb, encoded_known


def same_ddw_fit_measurement(
    expected_path: LegacyDdwPath,
    expected_headroom: DdwHeadroomScore,
    expected_observed: DdwMaskDescriptors,
    actual_path: LegacyDdwPath,
    actual_headroom: DdwHeadroomScore,
    actual_observed: DdwMaskDescriptors,
) -> bool:
    """Compare one rerender with recorded fit evidence at frozen tolerances."""
    return bool(
        _paths_equal(expected_path, actual_path)
        and _headroom_close(expected_headroom, actual_headroom)
        and _descriptors_close(expected_observed, actual_observed)
    )


def _same_recompute(
    measured: DdwFitDecision,
    repeated: DdwFitDecision,
    measured_path: LegacyDdwPath,
    repeated_path: LegacyDdwPath,
) -> bool:
    return bool(
        repeated.accepted
        and measured.candidate is not None
        and repeated.candidate is not None
        and same_ddw_fit_measurement(
            measured_path,
            measured.headroom,
            measured.candidate.observed,
            repeated_path,
            repeated.headroom,
            repeated.candidate.observed,
        )
        and measured.mask_score is not None
        and repeated.mask_score is not None
        and measured.mask_score.failures == repeated.mask_score.failures
    )


def _paths_equal(left: LegacyDdwPath, right: LegacyDdwPath) -> bool:
    return bool(
        left.clip_key == right.clip_key
        and left.variant == right.variant
        and np.array_equal(
            left.target.displacement_m,
            right.target.displacement_m,
        )
        and np.array_equal(
            left.source_world_to_camera_cv,
            right.source_world_to_camera_cv,
        )
        and np.array_equal(
            left.target.virtual_world_to_camera_cv,
            right.target.virtual_world_to_camera_cv,
        )
    )


def _headroom_close(left: DdwHeadroomScore, right: DdwHeadroomScore) -> bool:
    return bool(
        left.passed == right.passed
        and left.failures == right.failures
        and left.metrics.restoration_pixel_count
        == right.metrics.restoration_pixel_count
        and np.isclose(
            left.metrics.mean_absolute_corruption,
            right.metrics.mean_absolute_corruption,
            atol=_FLOAT_ATOL,
            rtol=_FLOAT_RTOL,
        )
        and np.isclose(
            left.metrics.changed_pixel_fraction,
            right.metrics.changed_pixel_fraction,
            atol=_FLOAT_ATOL,
            rtol=_FLOAT_RTOL,
        )
    )


def _descriptors_close(
    left: DdwMaskDescriptors,
    right: DdwMaskDescriptors,
) -> bool:
    return all(
        bool(
            np.isclose(
                getattr(left, name),
                getattr(right, name),
                atol=_FLOAT_ATOL,
                rtol=_FLOAT_RTOL,
            )
        )
        for name in DDW_MASK_METRIC_NAMES
    )


__all__ = [
    "DDW_FIT_MASK_GATE",
    "DdwAcceptedFitCondition",
    "DdwFitDecision",
    "DdwFitRenderTelemetry",
    "DdwPreparedFitClip",
    "WaymoDdwFitError",
    "decide_ddw_fit",
    "encode_waymo_ddw_condition",
    "prepare_waymo_ddw_fit_clip",
    "same_ddw_fit_measurement",
]

"""The separate DDW-v2 canary, v3 probe, and survey orders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from novel_view.models.gen3c.cache4d.backend import Gen3cCache4dResources
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.preparation.waymo_ddw.legacy.axis import (
    DdwMaskCandidate,
    DdwVariant,
)
from novel_view.preparation.waymo_ddw.legacy.headroom import DdwHeadroomScore
from novel_view.preparation.waymo_ddw.legacy.local import measure_legacy_ddw_local
from novel_view.preparation.waymo_ddw.legacy.reference import (
    measure_legacy_ddw_reference,
)
from novel_view.preparation.waymo_ddw.legacy.selection import DDW_VARIANTS
from novel_view.preparation.waymo_ddw.legacy.source import LegacyDdwSource
from novel_view.preparation.waymo_ddw.legacy.v2 import (
    DDW_V2_VARIANTS,
    DdwV2ScientificStop,
)
from novel_view.preparation.waymo_ddw.legacy.v2_calibration import (
    DdwV2MaskCalibration,
    calibrate_ddw_v2_mask_gate,
)
from novel_view.preparation.waymo_ddw.warp import WarpResources


DDW_V3_PROBE_VARIANTS = DDW_V2_VARIANTS[2:]
DDW_SURVEY_VARIANTS = DDW_VARIANTS


@dataclass(frozen=True, slots=True)
class DdwV2VariantEvidence:
    candidate: DdwMaskCandidate
    headroom: DdwHeadroomScore
    local_peak_cuda_allocated_bytes: int
    local_peak_cuda_reserved_bytes: int
    local_elapsed_seconds: float
    reference_peak_cuda_allocated_bytes: int
    reference_peak_cuda_reserved_bytes: int
    reference_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class DdwV2CanaryEvidence:
    clip_key: WaymoClipKey
    variants: tuple[DdwV2VariantEvidence, ...]
    calibration: DdwV2MaskCalibration

    @property
    def candidates(self) -> tuple[DdwMaskCandidate, ...]:
        return tuple(item.candidate for item in self.variants)


@dataclass(frozen=True, slots=True)
class DdwV3ProbeOutcome:
    variant: DdwVariant
    headroom: DdwHeadroomScore
    candidate: DdwMaskCandidate | None
    local_peak_cuda_allocated_bytes: int
    local_peak_cuda_reserved_bytes: int
    local_elapsed_seconds: float
    reference_peak_cuda_allocated_bytes: int | None
    reference_peak_cuda_reserved_bytes: int | None
    reference_elapsed_seconds: float | None


@dataclass(frozen=True, slots=True)
class DdwV3ProbeEvidence:
    clip_key: WaymoClipKey
    outcomes: tuple[DdwV3ProbeOutcome, ...]

    @property
    def passed(self) -> bool:
        return all(item.headroom.passed for item in self.outcomes)


@dataclass(frozen=True, slots=True)
class DdwSurveyEvidence:
    clip_key: WaymoClipKey
    outcomes: tuple[DdwV3ProbeOutcome, ...]

    @property
    def passed(self) -> bool:
        return all(item.headroom.passed for item in self.outcomes)


def measure_waymo_ddw_v2_canary(
    source: LegacyDdwSource,
    axis: tuple[DdwVariant, ...],
    warp_resources: WarpResources,
    reference_resources: Gen3cCache4dResources,
    log_directory: Path,
) -> DdwV2CanaryEvidence:
    """Stop at the first failed v2 headroom, before its reference."""
    variants = _measure_stop_axis(
        source,
        axis,
        warp_resources,
        reference_resources,
        log_directory,
    )
    calibration = calibrate_ddw_v2_mask_gate(
        tuple(item.candidate for item in variants)
    )
    return DdwV2CanaryEvidence(source.clip_key, variants, calibration)


def measure_waymo_ddw_v3_probe(
    source: LegacyDdwSource,
    axis: tuple[DdwVariant, ...],
    warp_resources: WarpResources,
    reference_resources: Gen3cCache4dResources,
    log_directory: Path,
) -> DdwV3ProbeEvidence:
    """Finish all four d3/d4 outcomes before exposing the verdict."""
    return DdwV3ProbeEvidence(
        source.clip_key,
        _measure_complete_axis(
            source,
            axis,
            warp_resources,
            reference_resources,
            log_directory,
        ),
    )


def measure_waymo_ddw_survey(
    source: LegacyDdwSource,
    axis: tuple[DdwVariant, ...],
    warp_resources: WarpResources,
    reference_resources: Gen3cCache4dResources,
    log_directory: Path,
) -> DdwSurveyEvidence:
    """Finish all eight d1..d4 outcomes for one explicit survey clip."""
    return DdwSurveyEvidence(
        source.clip_key,
        _measure_complete_axis(
            source,
            axis,
            warp_resources,
            reference_resources,
            log_directory,
        ),
    )


def _measure_stop_axis(
    source: LegacyDdwSource,
    axis: tuple[DdwVariant, ...],
    warp_resources: WarpResources,
    reference_resources: Gen3cCache4dResources,
    log_directory: Path,
) -> tuple[DdwV2VariantEvidence, ...]:
    evidence: list[DdwV2VariantEvidence] = []
    for variant in axis:
        local = measure_legacy_ddw_local(
            source,
            variant,
            warp_resources,
            log_directory / f"{variant.variant_id}__outward.log",
            log_directory / f"{variant.variant_id}__return.log",
        )
        if not local.headroom.passed:
            raise DdwV2ScientificStop(variant, local.headroom)
        reference = measure_legacy_ddw_reference(
            source,
            local.path,
            reference_resources,
            log_directory / f"{variant.variant_id}__cache4d.log",
        )
        evidence.append(
            DdwV2VariantEvidence(
                DdwMaskCandidate(
                    source.clip_key,
                    variant,
                    local.observed,
                    reference.descriptors,
                ),
                local.headroom,
                local.peak_cuda_allocated_bytes,
                local.peak_cuda_reserved_bytes,
                local.elapsed_seconds,
                reference.peak_cuda_allocated_bytes,
                reference.peak_cuda_reserved_bytes,
                reference.elapsed_seconds,
            )
        )
    return tuple(evidence)


def _measure_complete_axis(
    source: LegacyDdwSource,
    axis: tuple[DdwVariant, ...],
    warp_resources: WarpResources,
    reference_resources: Gen3cCache4dResources,
    log_directory: Path,
) -> tuple[DdwV3ProbeOutcome, ...]:
    outcomes: list[DdwV3ProbeOutcome] = []
    for variant in axis:
        local = measure_legacy_ddw_local(
            source,
            variant,
            warp_resources,
            log_directory / f"{variant.variant_id}__outward.log",
            log_directory / f"{variant.variant_id}__return.log",
        )
        if not local.headroom.passed:
            outcomes.append(
                DdwV3ProbeOutcome(
                    variant,
                    local.headroom,
                    None,
                    local.peak_cuda_allocated_bytes,
                    local.peak_cuda_reserved_bytes,
                    local.elapsed_seconds,
                    None,
                    None,
                    None,
                )
            )
            continue
        reference = measure_legacy_ddw_reference(
            source,
            local.path,
            reference_resources,
            log_directory / f"{variant.variant_id}__cache4d.log",
        )
        outcomes.append(
            DdwV3ProbeOutcome(
                variant,
                local.headroom,
                DdwMaskCandidate(
                    source.clip_key,
                    variant,
                    local.observed,
                    reference.descriptors,
                ),
                local.peak_cuda_allocated_bytes,
                local.peak_cuda_reserved_bytes,
                local.elapsed_seconds,
                reference.peak_cuda_allocated_bytes,
                reference.peak_cuda_reserved_bytes,
                reference.elapsed_seconds,
            )
        )
    return tuple(outcomes)


__all__ = [
    "DDW_SURVEY_VARIANTS",
    "DDW_V3_PROBE_VARIANTS",
    "DdwSurveyEvidence",
    "DdwV2CanaryEvidence",
    "DdwV2VariantEvidence",
    "DdwV3ProbeEvidence",
    "DdwV3ProbeOutcome",
    "measure_waymo_ddw_survey",
    "measure_waymo_ddw_v2_canary",
    "measure_waymo_ddw_v3_probe",
]

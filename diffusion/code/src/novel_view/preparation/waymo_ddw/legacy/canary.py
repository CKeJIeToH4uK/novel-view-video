"""The original eight-variant DDW engineering canary order."""

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
from novel_view.preparation.waymo_ddw.legacy.local import (
    measure_legacy_ddw_local,
)
from novel_view.preparation.waymo_ddw.legacy.reference import (
    measure_legacy_ddw_reference,
)
from novel_view.preparation.waymo_ddw.legacy.source import LegacyDdwSource
from novel_view.preparation.waymo_ddw.warp import WarpResources


DDW_ENGINEERING_CANARY_ID = "waymo-ddw-engineering-canary-v1"


class WaymoDdwCanaryError(RuntimeError):
    """The original canary stopped at one failed scientific headroom gate."""


@dataclass(frozen=True, slots=True)
class DdwCanaryVariantEvidence:
    candidate: DdwMaskCandidate
    headroom: DdwHeadroomScore
    local_peak_cuda_allocated_bytes: int
    local_peak_cuda_reserved_bytes: int
    local_elapsed_seconds: float
    reference_peak_cuda_allocated_bytes: int
    reference_peak_cuda_reserved_bytes: int
    reference_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class DdwEngineeringCanaryEvidence:
    canary_id: str
    clip_key: WaymoClipKey
    variants: tuple[DdwCanaryVariantEvidence, ...]

    @property
    def candidates(self) -> tuple[DdwMaskCandidate, ...]:
        return tuple(item.candidate for item in self.variants)


def measure_waymo_ddw_engineering_canary(
    source: LegacyDdwSource,
    axis: tuple[DdwVariant, ...],
    warp_resources: WarpResources,
    reference_resources: Gen3cCache4dResources,
    log_directory: Path,
) -> DdwEngineeringCanaryEvidence:
    """Run the exact v1 axis, stopping before reference on failed headroom."""
    evidence: list[DdwCanaryVariantEvidence] = []
    for variant in axis:
        local = measure_legacy_ddw_local(
            source,
            variant,
            warp_resources,
            log_directory / f"{variant.variant_id}__outward.log",
            log_directory / f"{variant.variant_id}__return.log",
        )
        if not local.headroom.passed:
            raise WaymoDdwCanaryError(
                f"DDW headroom STOP for {variant.variant_id}: "
                f"{','.join(local.headroom.failures)}"
            )
        reference = measure_legacy_ddw_reference(
            source,
            local.path,
            reference_resources,
            log_directory / f"{variant.variant_id}__cache4d.log",
        )
        evidence.append(
            DdwCanaryVariantEvidence(
                candidate=DdwMaskCandidate(
                    source.clip_key,
                    variant,
                    local.observed,
                    reference.descriptors,
                ),
                headroom=local.headroom,
                local_peak_cuda_allocated_bytes=local.peak_cuda_allocated_bytes,
                local_peak_cuda_reserved_bytes=local.peak_cuda_reserved_bytes,
                local_elapsed_seconds=local.elapsed_seconds,
                reference_peak_cuda_allocated_bytes=(
                    reference.peak_cuda_allocated_bytes
                ),
                reference_peak_cuda_reserved_bytes=(
                    reference.peak_cuda_reserved_bytes
                ),
                reference_elapsed_seconds=reference.elapsed_seconds,
            )
        )
    return DdwEngineeringCanaryEvidence(
        DDW_ENGINEERING_CANARY_ID,
        source.clip_key,
        tuple(evidence),
    )


__all__ = [
    "DDW_ENGINEERING_CANARY_ID",
    "DdwCanaryVariantEvidence",
    "DdwEngineeringCanaryEvidence",
    "WaymoDdwCanaryError",
    "measure_waymo_ddw_engineering_canary",
]

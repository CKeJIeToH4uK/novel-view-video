"""Pure aggregation of frozen Waymo DDW variants across eight clips."""

from __future__ import annotations

from dataclasses import dataclass

from novel_view.preparation.waymo_ddw.legacy.axis import (
    DdwMaskCandidate,
    DdwVariant,
)
from novel_view.preparation.waymo_ddw.legacy.gate import (
    DdwMaskGateScore,
    DdwMaskGateSpec,
    score_ddw_masks,
)
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.inputs.waymo.types import WaymoContractError


DDW_VARIANTS = tuple(
    DdwVariant(magnitude, sign)
    for magnitude in range(1, 5)
    for sign in (-1, 1)
)


@dataclass(frozen=True, slots=True)
class DdwMaskCandidateScore:
    clip_key: WaymoClipKey
    variant: DdwVariant
    gate: DdwMaskGateScore


@dataclass(frozen=True, slots=True)
class DdwMaskSelection:
    accepted_variants: tuple[DdwVariant, ...]
    scores: tuple[DdwMaskCandidateScore, ...]


def select_ddw_variants(
    spec: DdwMaskGateSpec,
    selection_keys: tuple[WaymoClipKey, ...],
    candidates: tuple[DdwMaskCandidate, ...],
) -> DdwMaskSelection:
    """Keep a magnitude only when both signs pass all eight frozen clips."""
    _selection_keys(selection_keys)
    required = tuple(
        (clip_key, variant)
        for clip_key in selection_keys
        for variant in DDW_VARIANTS
    )
    actual = tuple(
        (candidate.clip_key, candidate.variant)
        for candidate in candidates
    )
    if actual != required:
        raise WaymoContractError(
            "candidates must match ordered 8 clips x 8 variants"
        )
    scores = tuple(
        DdwMaskCandidateScore(
            candidate.clip_key,
            candidate.variant,
            score_ddw_masks(spec, candidate.observed, candidate.reference),
        )
        for candidate in candidates
    )
    return DdwMaskSelection(_accepted_variants(scores), scores)


def _accepted_variants(
    scores: tuple[DdwMaskCandidateScore, ...],
) -> tuple[DdwVariant, ...]:
    passed = {
        variant: all(
            score.gate.passed for score in scores if score.variant == variant
        )
        for variant in DDW_VARIANTS
    }
    accepted = tuple(
        variant
        for variant in DDW_VARIANTS
        if passed[variant]
        and passed[DdwVariant(variant.magnitude_m, -variant.sign)]
    )
    if not accepted:
        raise WaymoContractError("no DDW magnitude passes both signs")
    return accepted


def _selection_keys(keys: tuple[WaymoClipKey, ...]) -> None:
    if (
        len(keys) != 8
        or len({(key.official_partition, key.segment_id) for key in keys}) != 8
    ):
        raise WaymoContractError(
            "selection_keys must contain eight distinct segments"
        )


__all__ = [
    "DDW_VARIANTS",
    "DdwMaskCandidate",
    "DdwMaskCandidateScore",
    "DdwMaskSelection",
    "DdwVariant",
    "select_ddw_variants",
]

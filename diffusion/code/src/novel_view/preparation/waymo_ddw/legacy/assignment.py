"""Balanced deterministic variant assignment for the historical DDW fit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from novel_view.preparation.waymo_ddw.legacy.axis import DdwVariant
from novel_view.preparation.waymo_depth.keyset import (
    WaymoClipKey,
    clip_key_to_mapping,
)


DDW_FIT_RECIPE_ID = "waymo-ddw-fit-random-one-v1"
DDW_FIT_VARIANTS = tuple(
    DdwVariant(magnitude, sign)
    for magnitude in range(1, 5)
    for sign in (-1, 1)
)


@dataclass(frozen=True, slots=True)
class DdwFitAssignment:
    """One full-keyset row and its deterministic historical variant."""

    global_fit_index: int
    clip_key: WaymoClipKey
    variant: DdwVariant


@dataclass(frozen=True, slots=True)
class DdwFitAssignmentTable:
    """Assignments in the original full keyset order, before any subset."""

    assignments: tuple[DdwFitAssignment, ...]


def build_ddw_fit_assignment(
    keys: tuple[WaymoClipKey, ...],
) -> DdwFitAssignmentTable:
    """Rank the full keyset once, then deal variants round-robin by score."""
    ranked = sorted(
        enumerate(keys),
        key=lambda item: (_assignment_score(item[1]), item[0]),
    )
    variants: list[DdwVariant | None] = [None] * len(keys)
    for rank, (index, _) in enumerate(ranked):
        variants[index] = DDW_FIT_VARIANTS[rank % len(DDW_FIT_VARIANTS)]
    return DdwFitAssignmentTable(
        tuple(
            DdwFitAssignment(index, key, cast(DdwVariant, variants[index]))
            for index, key in enumerate(keys)
        )
    )


def _assignment_score(key: WaymoClipKey) -> bytes:
    payload = json.dumps(
        clip_key_to_mapping(key),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2s(
        DDW_FIT_RECIPE_ID.encode("utf-8") + b"\0" + payload,
        digest_size=32,
    ).digest()


__all__ = [
    "DDW_FIT_RECIPE_ID",
    "DDW_FIT_VARIANTS",
    "DdwFitAssignment",
    "DdwFitAssignmentTable",
    "build_ddw_fit_assignment",
]

"""Version-neutral value objects shared by Waymo DDW policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from novel_view.preparation.waymo_ddw.legacy.masks import DdwMaskDescriptors
    from novel_view.preparation.waymo_depth.keyset import WaymoClipKey


@dataclass(frozen=True, slots=True, order=True)
class DdwVariant:
    magnitude_m: int
    sign: int

    @property
    def variant_id(self) -> str:
        suffix = "minus" if self.sign == -1 else "plus"
        return f"d{self.magnitude_m}m-sign-{suffix}"


@dataclass(frozen=True, slots=True)
class DdwMaskCandidate:
    clip_key: WaymoClipKey
    variant: DdwVariant
    observed: DdwMaskDescriptors
    reference: DdwMaskDescriptors


__all__ = ["DdwMaskCandidate", "DdwVariant"]

"""Frozen DDW-v2 protocol values before any Waymo or model payload."""

from __future__ import annotations

from novel_view.preparation.waymo_ddw.legacy.axis import DdwVariant
from novel_view.preparation.waymo_ddw.legacy.headroom import DdwHeadroomScore


DDW_V2_RECIPE_ID = "waymo-ddw-v2"
DDW_V2_VARIANTS = tuple(
    DdwVariant(magnitude, sign)
    for magnitude in (2, 3, 4)
    for sign in (-1, 1)
)


class DdwV2ScientificStop(RuntimeError):
    """One small expected STOP caused only by failed DDW headroom."""

    def __init__(self, variant: DdwVariant, headroom: DdwHeadroomScore) -> None:
        self._variant = variant
        self._headroom = headroom
        super().__init__(
            f"DDW-v2 headroom STOP for {variant.variant_id}: "
            f"{','.join(headroom.failures)}",
        )

    @property
    def variant(self) -> DdwVariant:
        return self._variant

    @property
    def headroom(self) -> DdwHeadroomScore:
        return self._headroom


__all__ = [
    "DDW_V2_RECIPE_ID",
    "DDW_V2_VARIANTS",
    "DdwV2ScientificStop",
]

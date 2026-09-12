"""Historical DDW identity around the shared cosine target path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from novel_view.preparation.waymo_ddw.legacy.axis import DdwVariant
from novel_view.preparation.waymo_ddw.legacy.source import LegacyDdwSource
from novel_view.preparation.waymo_ddw.target_path import DdwPath, build_target_path
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey


@dataclass(frozen=True, slots=True, eq=False)
class LegacyDdwPath:
    """Clip/variant identity plus the shared source-neutral target path."""

    clip_key: WaymoClipKey
    variant: DdwVariant
    source_world_to_camera_cv: npt.NDArray[np.float64]
    target: DdwPath


def build_legacy_ddw_path(
    source: LegacyDdwSource,
    variant: DdwVariant,
) -> LegacyDdwPath:
    """Attach the historical identity without duplicating trajectory math."""
    target = build_target_path(
        source.world_to_camera_cv,
        float(variant.magnitude_m),
        variant.sign,
    )
    return LegacyDdwPath(
        clip_key=source.clip_key,
        variant=variant,
        source_world_to_camera_cv=source.world_to_camera_cv,
        target=target,
    )


__all__ = ["LegacyDdwPath", "build_legacy_ddw_path"]

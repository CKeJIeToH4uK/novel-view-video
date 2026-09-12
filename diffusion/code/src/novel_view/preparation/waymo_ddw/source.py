"""Exact physical Waymo source selected for one DDW sample."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from novel_view.inputs.waymo.frame import WaymoFrameBundle
from novel_view.inputs.waymo.reader import WaymoV2FrameReader
from novel_view.inputs.waymo.types import WaymoFrameKey
from novel_view.preparation.waymo_ddw.selection import SelectedDdwSample


_FRAME_COUNT = 121


@dataclass(frozen=True, slots=True, eq=False)
class WaymoFrontSource:
    """One stable sample identity joined to its audited physical frame keys."""

    selected: SelectedDdwSample
    frame_keys: tuple[WaymoFrameKey, ...]
    frames: Iterator[WaymoFrameBundle]


def prepare_front_source(
    waymo_root: str | Path,
    selected: SelectedDdwSample,
) -> WaymoFrontSource:
    """Open the exact selected 121-frame stream without rasterizing it."""
    frames = WaymoV2FrameReader(
        waymo_root,
        selected.partition,
        selected.segment_id,
        selected.start_frame_index,
        _FRAME_COUNT,
    )
    return WaymoFrontSource(
        selected=selected,
        frame_keys=frames.index.window,
        frames=frames,
    )


__all__ = ["WaymoFrontSource", "prepare_front_source"]

"""Join validation IDs and read the exact raw Waymo FRONT target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from novel_view.preparation.waymo_ddw.record import PreparedItem, PreparedRecord
from novel_view.training.gen3c.data import (
    R4cPreparedItem,
    R4cTrainingSplit,
    resolve_r4c_items,
)


@dataclass(frozen=True, slots=True)
class DdwEvaluationSample:
    """One validation identity with prepared and raw-source coordinates."""

    prepared: PreparedItem
    model_input: R4cPreparedItem


@dataclass(frozen=True, slots=True, eq=False)
class RawFrontTarget:
    """Rectified RGB for the exact 121 raw frames recorded by preparation."""

    sample_id: str
    frame_timestamps_micros: tuple[int, ...]
    rgb_thwc: npt.NDArray[np.uint8]


def resolve_ddw_evaluation_samples(
    prepared: PreparedRecord,
    split: R4cTrainingSplit,
) -> tuple[tuple[R4cPreparedItem, ...], tuple[DdwEvaluationSample, ...]]:
    """Preserve split order while retaining evaluation-only record fields."""
    training, validation = resolve_r4c_items(prepared, split)
    by_sample_id = {item.sample_id: item for item in prepared.items}
    return training, tuple(
        DdwEvaluationSample(by_sample_id[item.sample_id], item) for item in validation
    )


def read_raw_front_target(
    waymo_root: Path,
    sample: DdwEvaluationSample,
) -> RawFrontTarget:
    """Read and rectify FRONT once, requiring all recorded timestamps."""
    from itertools import chain

    from novel_view.preparation.waymo_ddw.raster import (
        RASTER_SIZE_HW,
        build_front_raster_plan,
        rectify_front_rgb,
    )
    from novel_view.preparation.waymo_ddw.selection import SelectedDdwSample
    from novel_view.preparation.waymo_ddw.source import prepare_front_source

    item = sample.prepared
    selected = SelectedDdwSample(
        sample_id=item.sample_id,
        partition=item.partition,
        segment_id=item.segment_id,
        start_frame_index=item.start_frame_index,
        magnitude_m=item.magnitude_m,
        sign=item.sign,
    )
    source = prepare_front_source(waymo_root, selected)
    timestamps = tuple(key.frame_timestamp_micros for key in source.frame_keys)
    if timestamps != item.frame_timestamps_micros:
        raise ValueError("raw Waymo timestamps differ from PreparedRecord")

    frames = iter(source.frames)
    first = next(frames)
    plan = build_front_raster_plan(first.camera("FRONT").calibration)
    height, width = RASTER_SIZE_HW
    rgb = np.empty((len(source.frame_keys), height, width, 3), dtype=np.uint8)
    for index, frame in enumerate(chain((first,), frames)):
        if frame.key != source.frame_keys[index]:
            raise ValueError("raw Waymo frame order differs from its audited index")
        rgb[index] = rectify_front_rgb(plan, frame.camera("FRONT"))
    return RawFrontTarget(item.sample_id, timestamps, rgb)


__all__ = [
    "DdwEvaluationSample",
    "RawFrontTarget",
    "read_raw_front_target",
    "resolve_ddw_evaluation_samples",
]

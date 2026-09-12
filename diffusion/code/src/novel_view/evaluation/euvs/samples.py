"""Load exact generated RGB slots and measured EUVS target frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from novel_view.generation.euvs.record import (
    EuvsGen3cRunRecord,
    load_run_record,
)
from novel_view.inputs.euvs.index import FramesIndex
from novel_view.inputs.euvs.pair import EuvsPairInput, load_euvs_pair_input
from novel_view.inputs.euvs.rgb import (
    EuvsRasterizedFrame,
    EuvsRasterizedSequence,
    rasterize_euvs_sequence,
)


@dataclass(frozen=True, slots=True, eq=False)
class EuvsSample:
    """One measured target and its generated RGB in recorded order."""

    target_index: int
    output_index: int
    source_index: int
    target: EuvsRasterizedFrame
    prediction_rgb: npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True, eq=False)
class EuvsSamples:
    """One generation record and all target-order evaluation samples."""

    record: EuvsGen3cRunRecord
    pair: EuvsPairInput
    target: EuvsRasterizedSequence
    frames: tuple[EuvsSample, ...]


def load_euvs_samples(
    record_path: Path,
    frames_index: FramesIndex,
    *,
    allow_legacy_record: bool = False,
) -> EuvsSamples:
    """Open one exact record/output and own only its selected target slots."""
    record = load_run_record(record_path, allow_legacy=allow_legacy_record)
    selection = record.pair
    pair = load_euvs_pair_input(
        name=selection.name,
        location=selection.location,
        channel=selection.channel,
        source_traversal=selection.source.traversal,
        source_image_tokens=selection.source.image_tokens,
        target_traversal=selection.target.traversal,
        target_image_tokens=selection.target.image_tokens,
        frames_index=frames_index,
    )
    target = rasterize_euvs_sequence(pair.target, record.output.shape[1:3])
    output: np.ndarray | None = None
    try:
        output = np.load(
            record_path.with_name(record.output.file),
            mmap_mode="r",
            allow_pickle=False,
        )
        frames = tuple(
            EuvsSample(
                target_index=index,
                output_index=output_index,
                source_index=record.target_source_sequence_index[index],
                target=target.frames[index],
                prediction_rgb=_readonly(output[output_index]),
            )
            for index, output_index in enumerate(record.target_output_index)
        )
    finally:
        if isinstance(output, np.memmap):
            output._mmap.close()
    return EuvsSamples(record, pair, target, frames)


def _readonly(value: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    owned = np.array(value, copy=True, order="C")
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(
        owned.shape
    )


__all__ = ["EuvsSample", "EuvsSamples", "load_euvs_samples"]

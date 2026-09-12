"""Rasterize ordered EUVS sequences without assigning model semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from novel_view.inputs.euvs.pair import (
    EuvsFrameInput,
    EuvsFrameSequence,
    EuvsPairInput,
)
from novel_view.inputs.nuplan.db import RawCameraCalibration
from novel_view.inputs.nuplan.raster import (
    NuPlanRasterPlan,
    RasterizedRgb,
    build_nuplan_raster_plan,
    read_rasterized_rgb,
)


class EuvsRgbError(ValueError):
    """One physical camera key has conflicting calibration."""


_CameraKey: TypeAlias = tuple[Path, str]


@dataclass(frozen=True, slots=True, eq=False)
class EuvsRasterizedFrame:
    input: EuvsFrameInput
    raster: RasterizedRgb

    @property
    def image_token(self) -> str:
        return self.input.image_token


@dataclass(frozen=True, slots=True, eq=False)
class EuvsRasterizedSequence:
    input_sequence: EuvsFrameSequence
    frames: tuple[EuvsRasterizedFrame, ...]

    @property
    def sequence_id(self) -> tuple[str, ...]:
        return self.input_sequence.sequence_id

    @property
    def image_size_hw(self) -> tuple[int, int]:
        return self.frames[0].raster.plan.output_size_hw


def rasterize_euvs_sequence(
    sequence: EuvsFrameSequence,
    output_size_hw: tuple[int, int],
) -> EuvsRasterizedSequence:
    camera_keys: list[_CameraKey] = []
    requests: dict[_CameraKey, tuple[RawCameraCalibration, EuvsFrameInput]] = {}
    for frame in sequence.frames:
        key = (frame.ref.db_path, frame.ref.camera_token)
        request = requests.get(key)
        if request is None:
            requests[key] = (frame.calibration, frame)
        elif frame.calibration != request[0]:
            raise EuvsRgbError(
                f"image_token {frame.image_token!r} reuses camera key "
                f"{key!r} with conflicting calibration"
            )
        camera_keys.append(key)

    plans = {
        key: build_nuplan_raster_plan(frame.calibration, output_size_hw)
        for key, (_, frame) in requests.items()
    }
    frames = tuple(
        EuvsRasterizedFrame(
            input=frame,
            raster=read_rasterized_rgb(frame.ref.image_path, plans[key]),
        )
        for frame, key in zip(sequence.frames, camera_keys, strict=True)
    )
    return EuvsRasterizedSequence(input_sequence=sequence, frames=frames)


def rasterize_euvs_source(
    pair: EuvsPairInput,
    output_size_hw: tuple[int, int],
) -> EuvsRasterizedSequence:
    return rasterize_euvs_sequence(pair.source, output_size_hw)

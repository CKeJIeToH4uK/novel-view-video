"""Persistent source-neutral VGGT-Omega arrays and their existing summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from novel_view.models.vggt.request import VggtOmegaRawPrediction
from novel_view.models.vggt.spec import (
    VggtOmegaInputMode,
    build_vggt_omega_input_plan,
)


PREDICTION_FILES = {
    "depth_model_units": "depth_model_units.npy",
    "depth_confidence": "depth_confidence.npy",
    "pose_encoding": "pose_encoding.npy",
    "model_w2c": "predicted_w2c.npy",
    "model_intrinsics": "predicted_intrinsics.npy",
}


class VggtOmegaRecordError(ValueError):
    """A persistent VGGT-Omega result cannot be decoded."""


@dataclass(frozen=True, slots=True, eq=False)
class SavedVggtOmegaPrediction:
    """Raw arrays plus the source identity actually present in summary.json."""

    prediction: VggtOmegaRawPrediction
    source_tokens: tuple[str, ...] | None
    source_frame_count: int
    first_source_token: str
    last_source_token: str
    frame_order_preserved: bool
    provenance: str


def save_vggt_omega_prediction(
    prediction: VggtOmegaRawPrediction,
    source_tokens: tuple[str, ...],
    source_size_hw: tuple[int, int],
    directory: Path,
) -> None:
    """Write the five established arrays and a complete ordered summary."""
    directory.mkdir()
    arrays = {
        "depth_model_units": prediction.depth_model_units,
        "depth_confidence": prediction.depth_confidence,
        "pose_encoding": prediction.pose_encoding,
        "model_w2c": prediction.model_w2c,
        "model_intrinsics": prediction.model_intrinsics,
    }
    for name, filename in PREDICTION_FILES.items():
        np.save(directory / filename, arrays[name], allow_pickle=False)
    summary = {
        "source_frames": len(source_tokens),
        "source_size_hw": list(source_size_hw),
        "first_image_token": source_tokens[0],
        "last_image_token": source_tokens[-1],
        "frame_order_preserved": True,
        "source_image_tokens": list(source_tokens),
        "mode": prediction.input_plan.mode.value,
        "model_size_hw": list(prediction.input_plan.model_size_hw),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_saved_vggt_omega_prediction(
    directory: Path,
) -> SavedVggtOmegaPrediction:
    """Read current or historical arrays without assigning dataset policy."""
    try:
        summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        source_size_hw = cast(
            tuple[int, int],
            tuple(summary["source_size_hw"]),
        )
        mode = VggtOmegaInputMode(summary["mode"])
        plan = build_vggt_omega_input_plan(source_size_hw, mode)
        if list(plan.model_size_hw) != summary["model_size_hw"]:
            raise VggtOmegaRecordError(
                "saved VGGT model grid disagrees with its mode"
            )
        token_values = summary.get("source_image_tokens")
        source_tokens = (
            None
            if token_values is None
            else tuple(cast(list[str], token_values))
        )
        provenance = (
            "legacy-attested"
            if source_tokens is None
            else "ordered-source-tokens"
        )
        values = {
            name: np.load(
                directory / filename,
                mmap_mode="r",
                allow_pickle=False,
            )
            for name, filename in PREDICTION_FILES.items()
        }
        try:
            prediction = VggtOmegaRawPrediction(
                input_plan=plan,
                depth_model_units=cast(
                    npt.NDArray[np.float32], _owned(values["depth_model_units"])
                ),
                depth_confidence=cast(
                    npt.NDArray[np.float32], _owned(values["depth_confidence"])
                ),
                pose_encoding=cast(
                    npt.NDArray[np.float32], _owned(values["pose_encoding"])
                ),
                model_w2c=cast(
                    npt.NDArray[np.float32], _owned(values["model_w2c"])
                ),
                model_intrinsics=cast(
                    npt.NDArray[np.float32], _owned(values["model_intrinsics"])
                ),
            )
        finally:
            for value in values.values():
                if isinstance(value, np.memmap):
                    value._mmap.close()
        return SavedVggtOmegaPrediction(
            prediction=prediction,
            source_tokens=source_tokens,
            source_frame_count=int(summary["source_frames"]),
            first_source_token=str(summary["first_image_token"]),
            last_source_token=str(summary["last_image_token"]),
            frame_order_preserved=summary["frame_order_preserved"] is True,
            provenance=provenance,
        )
    except VggtOmegaRecordError:
        raise
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        raise VggtOmegaRecordError(
            f"cannot read saved VGGT prediction {directory}: {error}"
        ) from error


def _owned(value: npt.NDArray[np.generic]) -> npt.NDArray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


__all__ = [
    "PREDICTION_FILES",
    "SavedVggtOmegaPrediction",
    "VggtOmegaRecordError",
    "load_saved_vggt_omega_prediction",
    "save_vggt_omega_prediction",
]

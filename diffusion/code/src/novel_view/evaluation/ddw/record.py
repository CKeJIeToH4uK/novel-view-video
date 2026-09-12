"""Versioned scientific records for one DDW heldout-point evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from novel_view.evaluation.ddw.metrics import (
    DdwItemMetrics,
    DepthMetrics,
    FrameDepthEvidence,
    RgbMetrics,
    VARIANTS,
    VariantAggregate,
    VariantItemMetrics,
    aggregate_ddw_items,
)
from novel_view.evaluation.ddw.sampling import DdwVariant
from novel_view.evaluation.ddw.spec import DdwSamplingSpec


EVALUATION_FORMAT = "novel-view/ddw-heldout-point-evaluation/v1"
ITEM_RESULT_FORMAT = "novel-view/ddw-heldout-point-item/v1"
DEPTH_INTERPRETATION = (
    "relative-depth shape after per-frame nonheldout LiDAR calibration"
)


@dataclass(frozen=True, slots=True)
class EvaluationCheckpoint:
    """Exact readable-record and binary locator used for one variant."""

    record: str
    checkpoint: str
    source_attempt: str
    completed_step: int
    objective_name: str
    objective_version: int
    depth_weight: float | None
    observer_lineage: str | None


@dataclass(frozen=True, slots=True)
class DdwItemResult:
    """One worker-produced item result and its persistent video locator."""

    sample_id: str
    video: str
    metrics: DdwItemMetrics


@dataclass(frozen=True, slots=True)
class DdwEvaluationRecord:
    """Portable final scientific evidence; no temporary array locators."""

    prepared_record: str
    split: str
    model_contract: str
    base_checkpoint: str
    tokenizer: str
    depth_checkpoint: str
    comparison_step: int
    sampling: DdwSamplingSpec
    v1_checkpoint: EvaluationCheckpoint
    v2_checkpoint: EvaluationCheckpoint
    items: tuple[DdwItemResult, ...]
    aggregates: tuple[VariantAggregate, ...]


def write_item_result(path: Path, result: DdwItemResult) -> None:
    """Write the strict temporary result handed back by the decode worker."""
    document = {
        "format": ITEM_RESULT_FORMAT,
        "sample_id": result.sample_id,
        "video": result.video,
        "variants": _variant_items_document(result.metrics.variants),
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def read_item_result(path: Path) -> DdwItemResult:
    """Read one process result before assembling the final record."""
    document = _fields(
        json.loads(path.read_text(encoding="utf-8")),
        {"format", "sample_id", "video", "variants"},
        "DDW item result",
    )
    if document["format"] != ITEM_RESULT_FORMAT:
        raise ValueError("unsupported DDW item result format")
    sample_id = _text(document["sample_id"], "sample_id")
    metrics = DdwItemMetrics(sample_id, _parse_variant_items(document["variants"]))
    return DdwItemResult(
        sample_id,
        _text(document["video"], "video"),
        metrics,
    )


def write_evaluation_record(path: Path, record: DdwEvaluationRecord) -> None:
    """Write the final evaluation JSON directly after all items complete."""
    document = {
        "format": EVALUATION_FORMAT,
        "kind": "heldout-point-validation",
        "workflow": {"name": "ddw_evaluation", "version": 1},
        "input": {
            "prepared_record": record.prepared_record,
            "split": record.split,
        },
        "model": {
            "contract": record.model_contract,
            "base_checkpoint": record.base_checkpoint,
            "tokenizer": record.tokenizer,
            "depth_backend": "moge",
            "depth_checkpoint": record.depth_checkpoint,
        },
        "comparison_step": record.comparison_step,
        "sampling": _sampling_document(record.sampling),
        "checkpoints": {
            "v1": _checkpoint_document(record.v1_checkpoint),
            "v2": _checkpoint_document(record.v2_checkpoint),
        },
        "depth_interpretation": DEPTH_INTERPRETATION,
        "items": [
            {
                "sample_id": item.sample_id,
                "video": item.video,
                "variants": _variant_items_document(item.metrics.variants),
            }
            for item in record.items
        ],
        "aggregates": _aggregates_document(record.aggregates),
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def read_evaluation_record(path: Path) -> DdwEvaluationRecord:
    """Strictly read the final heldout-point scientific boundary."""
    document = _fields(
        json.loads(path.read_text(encoding="utf-8")),
        {
            "format",
            "kind",
            "workflow",
            "input",
            "model",
            "comparison_step",
            "sampling",
            "checkpoints",
            "depth_interpretation",
            "items",
            "aggregates",
        },
        "DDW evaluation record",
    )
    if document["format"] != EVALUATION_FORMAT:
        raise ValueError("unsupported DDW evaluation format")
    if document["kind"] != "heldout-point-validation":
        raise ValueError("DDW evaluation must be named heldout-point validation")
    if document["workflow"] != {"name": "ddw_evaluation", "version": 1}:
        raise ValueError("DDW evaluation record has the wrong workflow")
    if document["depth_interpretation"] != DEPTH_INTERPRETATION:
        raise ValueError("DDW evaluation depth interpretation differs")
    input_values = _fields(
        document["input"],
        {"prepared_record", "split"},
        "DDW evaluation input",
    )
    model = _fields(
        document["model"],
        {
            "contract",
            "base_checkpoint",
            "tokenizer",
            "depth_backend",
            "depth_checkpoint",
        },
        "DDW evaluation model",
    )
    if model["depth_backend"] != "moge":
        raise ValueError("DDW evaluation depth backend differs")
    checkpoints = _fields(
        document["checkpoints"],
        {"v1", "v2"},
        "DDW evaluation checkpoints",
    )
    items_value = document["items"]
    if not isinstance(items_value, list) or not items_value:
        raise ValueError("DDW evaluation items must be a nonempty list")
    items = tuple(_parse_final_item(value) for value in items_value)
    if len({item.sample_id for item in items}) != len(items):
        raise ValueError("DDW evaluation contains duplicate sample_id")
    comparison_step = _positive_integer(
        document["comparison_step"],
        "comparison_step",
    )
    v1_checkpoint = _parse_checkpoint(checkpoints["v1"], "v1")
    v2_checkpoint = _parse_checkpoint(checkpoints["v2"], "v2")
    if (
        v1_checkpoint.completed_step,
        v2_checkpoint.completed_step,
    ) != (comparison_step, comparison_step):
        raise ValueError("DDW evaluation checkpoint steps differ")
    aggregates = _parse_aggregates(document["aggregates"])
    expected_aggregates = aggregate_ddw_items(tuple(item.metrics for item in items))
    if aggregates != expected_aggregates:
        raise ValueError("DDW evaluation aggregates differ from item metrics")
    return DdwEvaluationRecord(
        prepared_record=_text(input_values["prepared_record"], "prepared_record"),
        split=_text(input_values["split"], "split"),
        model_contract=_text(model["contract"], "model contract"),
        base_checkpoint=_text(model["base_checkpoint"], "base checkpoint"),
        tokenizer=_text(model["tokenizer"], "tokenizer"),
        depth_checkpoint=_text(model["depth_checkpoint"], "depth checkpoint"),
        comparison_step=comparison_step,
        sampling=_parse_sampling(document["sampling"]),
        v1_checkpoint=v1_checkpoint,
        v2_checkpoint=v2_checkpoint,
        items=items,
        aggregates=aggregates,
    )


def _variant_items_document(
    variants: tuple[VariantItemMetrics, ...],
) -> dict[str, object]:
    return {
        row.variant: {
            "depth": {
                "median_abs_rel": row.depth.median_abs_rel,
                "median_abs_z_m": row.depth.median_abs_z_m,
                "q95_abs_z_m": row.depth.q95_abs_z_m,
                "valid_fraction": row.depth.valid_fraction,
                "temporal_scale_jitter": row.depth.temporal_scale_jitter,
                "frames": [
                    {
                        "frame_index": frame.frame_index,
                        "scale": frame.scale,
                        "nonheldout_fit_count": frame.nonheldout_fit_count,
                        "heldout_count": frame.heldout_count,
                        "valid_count": frame.valid_count,
                    }
                    for frame in row.depth.frames
                ],
            },
            "rgb": {
                "mae": row.rgb.mae,
                "mae_improvement_vs_base": row.rgb.mae_improvement_vs_base,
                "improved_frame_fraction_vs_base": (
                    row.rgb.improved_frame_fraction_vs_base
                ),
                "mean_abs_change_vs_base": row.rgb.mean_abs_change_vs_base,
            },
        }
        for row in variants
    }


def _aggregates_document(
    aggregates: tuple[VariantAggregate, ...],
) -> dict[str, object]:
    return {
        row.variant: {
            "median_abs_rel": row.median_abs_rel,
            "median_abs_z_m": row.median_abs_z_m,
            "q95_abs_z_m": row.q95_abs_z_m,
            "valid_fraction": row.valid_fraction,
            "temporal_scale_jitter": row.temporal_scale_jitter,
            "rgb_mae": row.rgb_mae,
            "mae_improvement_vs_base": row.mae_improvement_vs_base,
            "improved_frame_fraction_vs_base": (row.improved_frame_fraction_vs_base),
            "mean_abs_change_vs_base": row.mean_abs_change_vs_base,
        }
        for row in aggregates
    }


def _checkpoint_document(value: EvaluationCheckpoint) -> dict[str, object]:
    return {
        "record": value.record,
        "checkpoint": value.checkpoint,
        "source_attempt": value.source_attempt,
        "completed_step": value.completed_step,
        "objective": {
            "name": value.objective_name,
            "version": value.objective_version,
            "depth_weight": value.depth_weight,
        },
        "observer_lineage": value.observer_lineage,
    }


def _sampling_document(value: DdwSamplingSpec) -> dict[str, object]:
    return {
        "seed": value.seed,
        "steps": value.steps,
        "guidance": value.guidance,
        "condition_augment_sigma": value.condition_augment_sigma,
    }


def _parse_final_item(value: object) -> DdwItemResult:
    item = _fields(
        value,
        {"sample_id", "video", "variants"},
        "DDW evaluation item",
    )
    sample_id = _text(item["sample_id"], "sample_id")
    return DdwItemResult(
        sample_id,
        _text(item["video"], "video"),
        DdwItemMetrics(sample_id, _parse_variant_items(item["variants"])),
    )


def _parse_variant_items(value: object) -> tuple[VariantItemMetrics, ...]:
    variants = _fields(value, set(VARIANTS), "DDW item variants")
    return tuple(
        _parse_variant_item(variant, variants[variant]) for variant in VARIANTS
    )


def _parse_variant_item(
    variant: DdwVariant,
    value: object,
) -> VariantItemMetrics:
    row = _fields(value, {"depth", "rgb"}, f"{variant} item metrics")
    depth = _fields(
        row["depth"],
        {
            "median_abs_rel",
            "median_abs_z_m",
            "q95_abs_z_m",
            "valid_fraction",
            "temporal_scale_jitter",
            "frames",
        },
        f"{variant} depth metrics",
    )
    frames_value = depth["frames"]
    if not isinstance(frames_value, list) or len(frames_value) != 120:
        raise ValueError("DDW depth metrics require frames 1..120")
    frames = tuple(_parse_frame(value) for value in frames_value)
    if tuple(frame.frame_index for frame in frames) != tuple(range(1, 121)):
        raise ValueError("DDW depth frame order differs")
    rgb = _fields(
        row["rgb"],
        {
            "mae",
            "mae_improvement_vs_base",
            "improved_frame_fraction_vs_base",
            "mean_abs_change_vs_base",
        },
        f"{variant} RGB metrics",
    )
    optional = variant == "base"
    return VariantItemMetrics(
        variant,
        DepthMetrics(
            _nonnegative(depth["median_abs_rel"], "median_abs_rel"),
            _nonnegative(depth["median_abs_z_m"], "median_abs_z_m"),
            _nonnegative(depth["q95_abs_z_m"], "q95_abs_z_m"),
            _fraction(depth["valid_fraction"], "valid_fraction"),
            _nonnegative(
                depth["temporal_scale_jitter"],
                "temporal_scale_jitter",
            ),
            frames,
        ),
        RgbMetrics(
            _nonnegative(rgb["mae"], "RGB MAE"),
            _optional_finite(
                rgb["mae_improvement_vs_base"],
                "MAE improvement",
                optional,
            ),
            _optional_fraction(
                rgb["improved_frame_fraction_vs_base"],
                "improved frame fraction",
                optional,
            ),
            _optional_nonnegative(
                rgb["mean_abs_change_vs_base"],
                "mean absolute change",
                optional,
            ),
        ),
    )


def _parse_frame(value: object) -> FrameDepthEvidence:
    row = _fields(
        value,
        {
            "frame_index",
            "scale",
            "nonheldout_fit_count",
            "heldout_count",
            "valid_count",
        },
        "DDW frame depth evidence",
    )
    fit_count = _positive_integer(
        row["nonheldout_fit_count"],
        "nonheldout_fit_count",
    )
    heldout_count = _nonnegative_integer(row["heldout_count"], "heldout_count")
    valid_count = _nonnegative_integer(row["valid_count"], "valid_count")
    if valid_count > heldout_count:
        raise ValueError("valid heldout count exceeds heldout count")
    scale = _finite(row["scale"], "scale")
    if scale <= 0.0:
        raise ValueError("DDW scale must be positive")
    return FrameDepthEvidence(
        _positive_integer(row["frame_index"], "frame_index"),
        scale,
        fit_count,
        heldout_count,
        valid_count,
    )


def _parse_aggregates(value: object) -> tuple[VariantAggregate, ...]:
    rows = _fields(value, set(VARIANTS), "DDW aggregates")
    result = []
    for variant in VARIANTS:
        row = _fields(
            rows[variant],
            {
                "median_abs_rel",
                "median_abs_z_m",
                "q95_abs_z_m",
                "valid_fraction",
                "temporal_scale_jitter",
                "rgb_mae",
                "mae_improvement_vs_base",
                "improved_frame_fraction_vs_base",
                "mean_abs_change_vs_base",
            },
            f"{variant} aggregate",
        )
        optional = variant == "base"
        result.append(
            VariantAggregate(
                variant,
                _nonnegative(row["median_abs_rel"], "median_abs_rel"),
                _nonnegative(row["median_abs_z_m"], "median_abs_z_m"),
                _nonnegative(row["q95_abs_z_m"], "q95_abs_z_m"),
                _fraction(row["valid_fraction"], "valid_fraction"),
                _nonnegative(
                    row["temporal_scale_jitter"],
                    "temporal_scale_jitter",
                ),
                _nonnegative(row["rgb_mae"], "rgb_mae"),
                _optional_finite(
                    row["mae_improvement_vs_base"],
                    "MAE improvement",
                    optional,
                ),
                _optional_fraction(
                    row["improved_frame_fraction_vs_base"],
                    "improved frame fraction",
                    optional,
                ),
                _optional_nonnegative(
                    row["mean_abs_change_vs_base"],
                    "mean absolute change",
                    optional,
                ),
            )
        )
    return tuple(result)


def _parse_checkpoint(value: object, variant: str) -> EvaluationCheckpoint:
    row = _fields(
        value,
        {
            "record",
            "checkpoint",
            "source_attempt",
            "completed_step",
            "objective",
            "observer_lineage",
        },
        f"{variant} evaluation checkpoint",
    )
    objective = _fields(
        row["objective"],
        {"name", "version", "depth_weight"},
        f"{variant} objective",
    )
    if variant == "v1":
        if (
            objective
            != {
                "name": "gen3c-kendall-edm",
                "version": 1,
                "depth_weight": None,
            }
            or row["observer_lineage"] is not None
        ):
            raise ValueError("v1 evaluation checkpoint objective differs")
        depth_weight = None
        lineage = None
    else:
        if (
            objective["name"] != "gen3c-kendall-edm-lidar-depth"
            or objective["version"] != 1
        ):
            raise ValueError("v2 evaluation checkpoint objective differs")
        depth_weight = _positive(objective["depth_weight"], "depth_weight")
        lineage = _text(row["observer_lineage"], "observer_lineage")
    return EvaluationCheckpoint(
        _text(row["record"], "checkpoint record"),
        _text(row["checkpoint"], "checkpoint"),
        _text(row["source_attempt"], "source_attempt"),
        _positive_integer(row["completed_step"], "completed_step"),
        cast(str, objective["name"]),
        cast(int, objective["version"]),
        depth_weight,
        lineage,
    )


def _parse_sampling(value: object) -> DdwSamplingSpec:
    row = _fields(
        value,
        {"seed", "steps", "guidance", "condition_augment_sigma"},
        "DDW sampling",
    )
    return DdwSamplingSpec(
        _nonnegative_integer(row["seed"], "sampling seed"),
        _positive_integer(row["steps"], "sampling steps"),
        _finite(row["guidance"], "guidance"),
        _nonnegative(row["condition_augment_sigma"], "condition augment sigma"),
    )


def _fields(value: object, expected: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} has unexpected fields")
    return dict(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _finite(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _fraction(value: object, name: str) -> float:
    result = _nonnegative(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must not exceed one")
    return result


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _optional_finite(value: object, name: str, require_none: bool) -> float | None:
    if require_none:
        if value is not None:
            raise ValueError(f"base {name} must be null")
        return None
    return _finite(value, name)


def _optional_fraction(value: object, name: str, require_none: bool) -> float | None:
    if require_none:
        if value is not None:
            raise ValueError(f"base {name} must be null")
        return None
    return _fraction(value, name)


def _optional_nonnegative(
    value: object,
    name: str,
    require_none: bool,
) -> float | None:
    if require_none:
        if value is not None:
            raise ValueError(f"base {name} must be null")
        return None
    return _nonnegative(value, name)


__all__ = [
    "DEPTH_INTERPRETATION",
    "DdwEvaluationRecord",
    "DdwItemResult",
    "EVALUATION_FORMAT",
    "EvaluationCheckpoint",
    "read_evaluation_record",
    "read_item_result",
    "write_evaluation_record",
    "write_item_result",
]

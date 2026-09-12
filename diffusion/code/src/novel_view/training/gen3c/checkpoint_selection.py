"""Explicit v1/v2 checkpoint-record selection without directory discovery."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novel_view.training.gen3c.lora.depth_record import CheckpointRecordV2
from novel_view.training.gen3c.lora.depth_spec import (
    DepthObjectiveV2,
    depth_objective_document,
    parse_depth_objective,
)
from novel_view.training.gen3c.lora.record import CheckpointRecordV1


R4C_CHECKPOINT_SELECTION_FORMAT_V1 = "novel-view/gen3c-checkpoint-selection/v1"
R4C_CHECKPOINT_SELECTION_FORMAT_V2 = "novel-view/gen3c-checkpoint-selection/v2"
_V1_FIELDS = frozenset(
    {
        "format",
        "input_checkpoint_records",
        "selected_checkpoint_record",
        "checkpoint",
        "completed_step",
        "metric",
    }
)
_V2_FIELDS = _V1_FIELDS | {"objective", "observer_lineage"}


@dataclass(frozen=True, slots=True)
class CheckpointSelectionV1:
    """Readable result of one ordered v1 record selection."""

    input_checkpoint_records: tuple[str, ...]
    selected_checkpoint_record: str
    checkpoint: str
    completed_step: int
    metric_name: str
    metric_value: float


@dataclass(frozen=True, slots=True)
class CheckpointSelectionV2:
    """Readable result of one objective/lineage-pure v2 selection."""

    input_checkpoint_records: tuple[str, ...]
    selected_checkpoint_record: str
    checkpoint: str
    completed_step: int
    metric_name: str
    metric_value: float
    objective: DepthObjectiveV2
    observer_lineage: str


def select_checkpoint_v1(
    candidates: tuple[CheckpointRecordV1, ...],
) -> CheckpointRecordV1:
    """Choose the stable minimum loss, then the earliest completed step."""
    return min(
        candidates,
        key=lambda record: (record.validation_loss, record.completed_step),
    )


def build_checkpoint_selection_v1(
    record_paths: tuple[Path, ...],
    candidates: tuple[CheckpointRecordV1, ...],
    selected: CheckpointRecordV1,
) -> CheckpointSelectionV1:
    """Bind the selected record to its exact runs-relative input locator."""
    selected_index = candidates.index(selected)
    selected_record = record_paths[selected_index]
    return CheckpointSelectionV1(
        input_checkpoint_records=tuple(str(path) for path in record_paths),
        selected_checkpoint_record=str(selected_record),
        checkpoint=str(selected_record.parent / selected.checkpoint),
        completed_step=selected.completed_step,
        metric_name="validation_loss",
        metric_value=selected.validation_loss,
    )


def select_checkpoint_v2(
    candidates: tuple[CheckpointRecordV2, ...],
) -> CheckpointRecordV2:
    """Choose one v2 minimum only within an exact objective and lineage."""
    objective = candidates[0].objective
    lineage = candidates[0].observer_lineage
    if any(
        item.objective != objective or item.observer_lineage != lineage
        for item in candidates[1:]
    ):
        raise ValueError("v2 checkpoint selection mixes objective or observer lineage")
    return min(
        candidates,
        key=lambda record: (record.validation_total_loss, record.completed_step),
    )


def build_checkpoint_selection_v2(
    record_paths: tuple[Path, ...],
    candidates: tuple[CheckpointRecordV2, ...],
    selected: CheckpointRecordV2,
) -> CheckpointSelectionV2:
    """Bind the selected v2 record to its ordered exact input locator."""
    selected_record = record_paths[candidates.index(selected)]
    return CheckpointSelectionV2(
        input_checkpoint_records=tuple(str(path) for path in record_paths),
        selected_checkpoint_record=str(selected_record),
        checkpoint=str(selected_record.parent / selected.checkpoint),
        completed_step=selected.completed_step,
        metric_name="validation_total_loss",
        metric_value=selected.validation_total_loss,
        objective=selected.objective,
        observer_lineage=selected.observer_lineage,
    )


def write_checkpoint_selection_v1(
    path: Path,
    selection: CheckpointSelectionV1,
) -> None:
    """Write the one user-readable selection result directly."""
    path.write_text(
        json.dumps(
            {
                "format": R4C_CHECKPOINT_SELECTION_FORMAT_V1,
                "input_checkpoint_records": list(selection.input_checkpoint_records),
                "selected_checkpoint_record": selection.selected_checkpoint_record,
                "checkpoint": selection.checkpoint,
                "completed_step": selection.completed_step,
                "metric": {
                    "name": selection.metric_name,
                    "value": selection.metric_value,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_checkpoint_selection_v2(
    path: Path,
    selection: CheckpointSelectionV2,
) -> None:
    """Write the one user-readable v2 selection result directly."""
    path.write_text(
        json.dumps(
            {
                "format": R4C_CHECKPOINT_SELECTION_FORMAT_V2,
                "input_checkpoint_records": list(selection.input_checkpoint_records),
                "selected_checkpoint_record": selection.selected_checkpoint_record,
                "checkpoint": selection.checkpoint,
                "completed_step": selection.completed_step,
                "metric": {
                    "name": selection.metric_name,
                    "value": selection.metric_value,
                },
                "objective": depth_objective_document(selection.objective),
                "observer_lineage": selection.observer_lineage,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_checkpoint_selection_v1(path: Path) -> CheckpointSelectionV1:
    """Strictly read one portable v1 selection result."""
    document = _selection_document(path, _V1_FIELDS)
    if document["format"] != R4C_CHECKPOINT_SELECTION_FORMAT_V1:
        raise ValueError("unsupported v1 checkpoint selection format")
    records, selected, checkpoint, step, metric = _selection_values(
        document, "validation_loss"
    )
    return CheckpointSelectionV1(
        records, selected, checkpoint, step, "validation_loss", metric
    )


def read_checkpoint_selection_v2(path: Path) -> CheckpointSelectionV2:
    """Strictly read one portable v2 selection result."""
    document = _selection_document(path, _V2_FIELDS)
    if document["format"] != R4C_CHECKPOINT_SELECTION_FORMAT_V2:
        raise ValueError("unsupported v2 checkpoint selection format")
    records, selected, checkpoint, step, metric = _selection_values(
        document, "validation_total_loss"
    )
    return CheckpointSelectionV2(
        records,
        selected,
        checkpoint,
        step,
        "validation_total_loss",
        metric,
        parse_depth_objective(document["objective"]),
        _text(document["observer_lineage"], "observer_lineage"),
    )


def _selection_document(path: Path, fields: frozenset[str]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("checkpoint selection fields differ")
    return dict(value)


def _selection_values(
    document: Mapping[str, Any], metric_name: str
) -> tuple[tuple[str, ...], str, str, int, float]:
    raw_records = document["input_checkpoint_records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("checkpoint selection inputs must be a non-empty list")
    records = tuple(_relative_text(value, "checkpoint record") for value in raw_records)
    selected = _relative_text(document["selected_checkpoint_record"], "selected record")
    checkpoint = _relative_text(document["checkpoint"], "checkpoint")
    if selected not in records or Path(selected).parent != Path(checkpoint).parent:
        raise ValueError("selected checkpoint does not belong to its input record")
    step = document["completed_step"]
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError("completed_step must be positive")
    raw_metric = document["metric"]
    if not isinstance(raw_metric, Mapping) or set(raw_metric) != {"name", "value"}:
        raise ValueError("checkpoint selection metric fields differ")
    if raw_metric["name"] != metric_name:
        raise ValueError("checkpoint selection metric name differs")
    metric = raw_metric["value"]
    if (
        not isinstance(metric, (int, float))
        or isinstance(metric, bool)
        or not math.isfinite(metric)
    ):
        raise ValueError("checkpoint selection metric must be finite")
    return records, selected, checkpoint, step, float(metric)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _relative_text(value: object, name: str) -> str:
    text = _text(value, name)
    if Path(text).is_absolute():
        raise ValueError(f"{name} must be runs-relative")
    return text


__all__ = [
    "CheckpointSelectionV1",
    "CheckpointSelectionV2",
    "R4C_CHECKPOINT_SELECTION_FORMAT_V1",
    "R4C_CHECKPOINT_SELECTION_FORMAT_V2",
    "build_checkpoint_selection_v1",
    "build_checkpoint_selection_v2",
    "read_checkpoint_selection_v1",
    "read_checkpoint_selection_v2",
    "select_checkpoint_v1",
    "select_checkpoint_v2",
    "write_checkpoint_selection_v1",
    "write_checkpoint_selection_v2",
]

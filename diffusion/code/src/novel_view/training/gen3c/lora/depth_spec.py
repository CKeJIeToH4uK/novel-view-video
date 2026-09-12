"""Строгая внешняя конфигурация экспериментального R4c depth recipe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from novel_view.config.load import reject_unknown_fields
from novel_view.training.gen3c.lora.spec import (
    _integer,
    _mapping,
    _nonnegative_integer,
    _positive_integer,
    _require_fields,
    _text,
)


R4C_DEPTH_OBJECTIVE_NAME = "gen3c-kendall-edm-lidar-depth"
R4C_DEPTH_OBJECTIVE_VERSION = 1
DEPTH_REDUCER_NAME = "front-a-lidar-log-camera-z-grid"
DEPTH_REDUCER_VERSION = 1
DEPTH_OBSERVER_NAME = "latent-depth-observer"
DEPTH_OBSERVER_VERSION = 1
DEPTH_OBSERVER_ARCHITECTURE = "conv3d-16x32x1-silu-conv3d-32x1x1"
DEPTH_OBSERVER_PARAMETER_COUNT = 577


@dataclass(frozen=True, slots=True)
class DepthObserverFitSpec:
    """Зарегистрированный порядок подгонки маленького depth observer."""

    init_seed: int
    fit_order_seed: int
    fit_epochs: int
    items_per_step: int
    optimizer_name: str
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    scheduler: str


@dataclass(frozen=True, slots=True)
class DepthObjectiveV2:
    """Exact scientific identity shared by v2 PT, JSON and selection."""

    depth_weight: float


@dataclass(frozen=True, slots=True)
class DepthObserverFitIdentity:
    """Inputs and fixed procedure that produced one frozen observer."""

    training_sample_ids: tuple[str, ...]
    training_segment_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    validation_segment_ids: tuple[str, ...]
    spec: DepthObserverFitSpec


@dataclass(frozen=True, slots=True)
class DepthObserverFitResult:
    """Item-balanced losses retained as the observer signal evidence."""

    training_loss: float
    validation_loss: float
    constant_comparator_loss: float


@dataclass(frozen=True, slots=True)
class R4cLoraDepthSpec:
    """Параметры v2, отделённые от неизменного baseline v1."""

    base_checkpoint: Path
    epochs: int
    training_seed: int
    validation_seed: int
    depth_weight: float
    observer: DepthObserverFitSpec


def parse_r4c_lora_depth_spec(
    values: Mapping[str, object],
) -> R4cLoraDepthSpec:
    """Разобрать одну заранее зарегистрированную v2 recipe."""
    reject_unknown_fields(values, frozenset({"model", "training"}))
    _require_fields(values, frozenset({"model", "training"}), "parameters")
    model = _mapping(values, "model")
    reject_unknown_fields(model, frozenset({"checkpoint"}))
    _require_fields(model, frozenset({"checkpoint"}), "model")

    training = _mapping(values, "training")
    fields = frozenset(
        {"method", "objective", "epochs", "training_seed", "validation_seed"}
    )
    reject_unknown_fields(training, fields)
    _require_fields(training, fields, "training")
    if _text(training, "method") != "lora":
        raise ValueError("R4c depth training requires method lora")

    objective = _mapping(training, "objective")
    objective_fields = frozenset({"name", "version", "depth_weight", "observer"})
    reject_unknown_fields(objective, objective_fields)
    _require_fields(objective, objective_fields, "objective")
    if (_text(objective, "name"), _integer(objective, "version")) != (
        R4C_DEPTH_OBJECTIVE_NAME,
        R4C_DEPTH_OBJECTIVE_VERSION,
    ):
        raise ValueError("R4c depth training requires its v1 objective")

    depth_weight = _finite_number(objective, "depth_weight")
    if depth_weight <= 0:
        raise ValueError("depth_weight must be positive")
    return R4cLoraDepthSpec(
        base_checkpoint=Path(_text(model, "checkpoint")),
        epochs=_positive_integer(training, "epochs"),
        training_seed=_nonnegative_integer(training, "training_seed"),
        validation_seed=_nonnegative_integer(training, "validation_seed"),
        depth_weight=depth_weight,
        observer=_parse_observer(_mapping(objective, "observer")),
    )


def depth_objective_document(value: DepthObjectiveV2) -> dict[str, object]:
    """Encode the one registered v2 objective."""
    return {
        "name": R4C_DEPTH_OBJECTIVE_NAME,
        "version": R4C_DEPTH_OBJECTIVE_VERSION,
        "depth_weight": value.depth_weight,
    }


def parse_depth_objective(value: object) -> DepthObjectiveV2:
    """Strictly decode the objective stored at an external PT/JSON boundary."""
    document = _external_mapping(value, "objective")
    _exact_fields(document, {"name", "version", "depth_weight"}, "objective")
    if (document["name"], document["version"]) != (
        R4C_DEPTH_OBJECTIVE_NAME,
        R4C_DEPTH_OBJECTIVE_VERSION,
    ):
        raise ValueError("R4c depth objective differs")
    weight = _finite_value(document["depth_weight"], "depth_weight")
    if weight <= 0:
        raise ValueError("depth_weight must be positive")
    return DepthObjectiveV2(weight)


def observer_fit_identity_document(
    value: DepthObserverFitIdentity,
) -> dict[str, object]:
    """Encode the complete deterministic observer-fit identity."""
    spec = value.spec
    return {
        "reducer": {"name": DEPTH_REDUCER_NAME, "version": DEPTH_REDUCER_VERSION},
        "observer": {
            "name": DEPTH_OBSERVER_NAME,
            "version": DEPTH_OBSERVER_VERSION,
            "architecture": DEPTH_OBSERVER_ARCHITECTURE,
            "parameter_count": DEPTH_OBSERVER_PARAMETER_COUNT,
        },
        "training_sample_ids": list(value.training_sample_ids),
        "training_segment_ids": list(value.training_segment_ids),
        "validation_sample_ids": list(value.validation_sample_ids),
        "validation_segment_ids": list(value.validation_segment_ids),
        "init_seed": spec.init_seed,
        "fit_order_seed": spec.fit_order_seed,
        "fit_epochs": spec.fit_epochs,
        "items_per_step": spec.items_per_step,
        "optimizer": {
            "name": spec.optimizer_name,
            "learning_rate": spec.learning_rate,
            "betas": list(spec.betas),
            "epsilon": spec.epsilon,
            "weight_decay": spec.weight_decay,
            "scheduler": spec.scheduler,
        },
    }


def parse_observer_fit_identity(value: object) -> DepthObserverFitIdentity:
    """Strictly decode one saved observer-fit identity."""
    document = _external_mapping(value, "observer_fit_identity")
    _exact_fields(
        document,
        {
            "reducer",
            "observer",
            "training_sample_ids",
            "training_segment_ids",
            "validation_sample_ids",
            "validation_segment_ids",
            "init_seed",
            "fit_order_seed",
            "fit_epochs",
            "items_per_step",
            "optimizer",
        },
        "observer_fit_identity",
    )
    if document["reducer"] != {
        "name": DEPTH_REDUCER_NAME,
        "version": DEPTH_REDUCER_VERSION,
    } or document["observer"] != {
        "name": DEPTH_OBSERVER_NAME,
        "version": DEPTH_OBSERVER_VERSION,
        "architecture": DEPTH_OBSERVER_ARCHITECTURE,
        "parameter_count": DEPTH_OBSERVER_PARAMETER_COUNT,
    }:
        raise ValueError("R4c depth observer identity differs")
    optimizer = _external_mapping(document["optimizer"], "observer optimizer")
    _exact_fields(
        optimizer,
        {"name", "learning_rate", "betas", "epsilon", "weight_decay", "scheduler"},
        "observer optimizer",
    )
    training_samples = _text_list(
        document["training_sample_ids"], "training_sample_ids"
    )
    training_segments = _text_list(
        document["training_segment_ids"], "training_segment_ids"
    )
    validation_samples = _text_list(
        document["validation_sample_ids"], "validation_sample_ids"
    )
    validation_segments = _text_list(
        document["validation_segment_ids"], "validation_segment_ids"
    )
    if len(training_samples) != len(training_segments) or len(
        validation_samples
    ) != len(validation_segments):
        raise ValueError("observer fit sample/segment counts differ")
    if set(training_samples) & set(validation_samples) or set(training_segments) & set(
        validation_segments
    ):
        raise ValueError("observer fit split overlaps")
    betas = optimizer["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("observer optimizer betas differ")
    parsed_betas = tuple(_finite_value(item, "betas") for item in betas)
    spec = DepthObserverFitSpec(
        init_seed=_saved_nonnegative(document["init_seed"], "init_seed"),
        fit_order_seed=_saved_nonnegative(document["fit_order_seed"], "fit_order_seed"),
        fit_epochs=_saved_positive(document["fit_epochs"], "fit_epochs"),
        items_per_step=_saved_positive(document["items_per_step"], "items_per_step"),
        optimizer_name=str(optimizer["name"]),
        learning_rate=_finite_value(optimizer["learning_rate"], "learning_rate"),
        betas=(parsed_betas[0], parsed_betas[1]),
        epsilon=_finite_value(optimizer["epsilon"], "epsilon"),
        weight_decay=_finite_value(optimizer["weight_decay"], "weight_decay"),
        scheduler=str(optimizer["scheduler"]),
    )
    if (
        spec.items_per_step != 1
        or spec.optimizer_name != "adamw"
        or spec.scheduler != "constant"
        or spec.learning_rate <= 0
        or spec.epsilon <= 0
        or spec.weight_decay < 0
        or not all(0 <= item < 1 for item in spec.betas)
    ):
        raise ValueError("observer fit procedure differs")
    return DepthObserverFitIdentity(
        training_samples,
        training_segments,
        validation_samples,
        validation_segments,
        spec,
    )


def observer_fit_result_document(value: DepthObserverFitResult) -> dict[str, float]:
    """Encode the three signal-gate measurements."""
    return {
        "training_loss": value.training_loss,
        "validation_loss": value.validation_loss,
        "constant_comparator_loss": value.constant_comparator_loss,
    }


def parse_observer_fit_result(value: object) -> DepthObserverFitResult:
    """Strictly decode finite non-negative signal-gate measurements."""
    document = _external_mapping(value, "observer_fit_result")
    _exact_fields(
        document,
        {"training_loss", "validation_loss", "constant_comparator_loss"},
        "observer_fit_result",
    )
    result = DepthObserverFitResult(
        *(
            _finite_value(document[field], field)
            for field in (
                "training_loss",
                "validation_loss",
                "constant_comparator_loss",
            )
        )
    )
    if (
        min(
            result.training_loss,
            result.validation_loss,
            result.constant_comparator_loss,
        )
        < 0
        or result.validation_loss >= result.constant_comparator_loss
    ):
        raise ValueError("observer fit result has no validation signal")
    return result


def _parse_observer(values: Mapping[str, object]) -> DepthObserverFitSpec:
    fields = frozenset(
        {"init_seed", "fit_order_seed", "fit_epochs", "items_per_step", "optimizer"}
    )
    reject_unknown_fields(values, fields)
    _require_fields(values, fields, "observer")
    optimizer = _mapping(values, "optimizer")
    optimizer_fields = frozenset(
        {"name", "learning_rate", "betas", "epsilon", "weight_decay", "scheduler"}
    )
    reject_unknown_fields(optimizer, optimizer_fields)
    _require_fields(optimizer, optimizer_fields, "optimizer")
    if _text(optimizer, "name") != "adamw":
        raise ValueError("depth observer requires adamw")
    if _text(optimizer, "scheduler") != "constant":
        raise ValueError("depth observer requires constant scheduler")
    betas = optimizer["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("betas must contain two numbers")
    parsed_betas = tuple(_finite_value(value, "betas") for value in betas)
    if not all(0 <= value < 1 for value in parsed_betas):
        raise ValueError("betas must be within [0, 1)")
    learning_rate = _finite_number(optimizer, "learning_rate")
    epsilon = _finite_number(optimizer, "epsilon")
    weight_decay = _finite_number(optimizer, "weight_decay")
    if learning_rate <= 0 or epsilon <= 0 or weight_decay < 0:
        raise ValueError("observer optimizer values are outside their domains")
    items_per_step = _positive_integer(values, "items_per_step")
    if items_per_step != 1:
        raise ValueError("depth observer requires one item per optimizer step")
    return DepthObserverFitSpec(
        init_seed=_nonnegative_integer(values, "init_seed"),
        fit_order_seed=_nonnegative_integer(values, "fit_order_seed"),
        fit_epochs=_positive_integer(values, "fit_epochs"),
        items_per_step=items_per_step,
        optimizer_name="adamw",
        learning_rate=learning_rate,
        betas=(parsed_betas[0], parsed_betas[1]),
        epsilon=epsilon,
        weight_decay=weight_decay,
        scheduler="constant",
    )


def _finite_number(values: Mapping[str, object], field: str) -> float:
    return _finite_value(values[field], field)


def _finite_value(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _external_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _exact_fields(value: Mapping[str, object], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} fields differ")


def _text_list(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty text list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    return result


def _saved_nonnegative(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _saved_positive(value: object, name: str) -> int:
    parsed = _saved_nonnegative(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


__all__ = [
    "DEPTH_OBSERVER_ARCHITECTURE",
    "DEPTH_OBSERVER_NAME",
    "DEPTH_OBSERVER_PARAMETER_COUNT",
    "DEPTH_OBSERVER_VERSION",
    "DEPTH_REDUCER_NAME",
    "DEPTH_REDUCER_VERSION",
    "DepthObjectiveV2",
    "DepthObserverFitIdentity",
    "DepthObserverFitResult",
    "DepthObserverFitSpec",
    "R4C_DEPTH_OBJECTIVE_NAME",
    "R4C_DEPTH_OBJECTIVE_VERSION",
    "R4cLoraDepthSpec",
    "depth_objective_document",
    "observer_fit_identity_document",
    "observer_fit_result_document",
    "parse_depth_objective",
    "parse_observer_fit_identity",
    "parse_observer_fit_result",
    "parse_r4c_lora_depth_spec",
]

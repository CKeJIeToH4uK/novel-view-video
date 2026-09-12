"""Tagged v2 binary state for LoRA plus one frozen depth observer."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novel_view.training.gen3c.lora.checkpoint import (
    R4C_CHECKPOINT_STATE_FIELDS_V1,
    LoadedCheckpointV1,
    ResumeStateV1,
    TrainingStateV1,
    decode_training_state_v1,
    pack_training_state_v1,
    restore_lora_v1,
    restore_training_state_v1,
)
from novel_view.training.gen3c.lora.depth_spec import (
    DepthObjectiveV2,
    DepthObserverFitIdentity,
    DepthObserverFitResult,
    depth_objective_document,
    observer_fit_identity_document,
    observer_fit_result_document,
    parse_depth_objective,
    parse_observer_fit_identity,
    parse_observer_fit_result,
)


R4C_CHECKPOINT_FORMAT_V2 = "novel-view/gen3c-lora-training-checkpoint/v2"
_V2_FIELDS = R4C_CHECKPOINT_STATE_FIELDS_V1 | {
    "format",
    "objective",
    "observer",
    "observer_fit_identity",
    "observer_fit_result",
    "observer_lineage",
}
_OBSERVER_SHAPES = {
    "input.weight": (32, 16, 1, 1, 1),
    "input.bias": (32,),
    "output.weight": (1, 32, 1, 1, 1),
    "output.bias": (1,),
}


@dataclass(frozen=True, slots=True)
class TrainingStateV2:
    """One completed v2 update and the already-fitted observer facts."""

    baseline: TrainingStateV1
    objective: DepthObjectiveV2
    observer: Any
    observer_fit_identity: DepthObserverFitIdentity
    observer_fit_result: DepthObserverFitResult
    observer_lineage: str


@dataclass(frozen=True, slots=True)
class LoadedCheckpointV2:
    """Strictly decoded v2 state; the baseline body keeps v1 restore order."""

    baseline: LoadedCheckpointV1
    objective: DepthObjectiveV2
    observer: Mapping[str, Any]
    observer_fit_identity: DepthObserverFitIdentity
    observer_fit_result: DepthObserverFitResult
    observer_lineage: str


def save_checkpoint_v2(path: Path, state: TrainingStateV2) -> None:
    """Write one tagged v2 PT without observer optimizer or gradients."""
    torch = importlib.import_module("torch")
    observer = {
        name: state.observer.state_dict()[name]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .clone()
        for name in _OBSERVER_SHAPES
    }
    payload = {
        "format": R4C_CHECKPOINT_FORMAT_V2,
        **pack_training_state_v1(state.baseline),
        "objective": depth_objective_document(state.objective),
        "observer": observer,
        "observer_fit_identity": observer_fit_identity_document(
            state.observer_fit_identity
        ),
        "observer_fit_result": observer_fit_result_document(state.observer_fit_result),
        "observer_lineage": state.observer_lineage,
    }
    torch.save(payload, path)


def load_checkpoint_v2(path: Path) -> LoadedCheckpointV2:
    """Restricted-load only the exact tagged v2 format; never adapt untagged PT."""
    torch = importlib.import_module("torch")
    raw = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise ValueError("R4c v2 checkpoint root must be a mapping")
    payload = dict(raw)
    if set(payload) != _V2_FIELDS or payload["format"] != R4C_CHECKPOINT_FORMAT_V2:
        raise ValueError("unsupported R4c v2 checkpoint format")
    baseline = decode_training_state_v1(
        {field: payload[field] for field in R4C_CHECKPOINT_STATE_FIELDS_V1},
        torch,
    )
    observer = _decode_observer(payload["observer"], torch)
    fit_identity = parse_observer_fit_identity(payload["observer_fit_identity"])
    if (
        fit_identity.training_sample_ids
        != baseline.resume.data_cursor.training_sample_ids
    ):
        raise ValueError("R4c v2 observer fit differs from checkpoint training order")
    return LoadedCheckpointV2(
        baseline=baseline,
        objective=parse_depth_objective(payload["objective"]),
        observer=observer,
        observer_fit_identity=fit_identity,
        observer_fit_result=parse_observer_fit_result(payload["observer_fit_result"]),
        observer_lineage=_attempt_reference(payload["observer_lineage"]),
    )


def restore_lora_v2(checkpoint: LoadedCheckpointV2, model: Any) -> None:
    """Restore only raw LoRA weights before CP/recompute/DDP."""
    restore_lora_v1(checkpoint.baseline, model)


def restore_depth_observer_v2(
    checkpoint: LoadedCheckpointV2,
    *,
    device: object,
) -> Any:
    """Build the exact observer from saved state, freeze it and never refit it."""
    from novel_view.training.gen3c.lora.depth import LatentDepthObserver

    torch = importlib.import_module("torch")
    observer = LatentDepthObserver()
    observer.load_state_dict(dict(checkpoint.observer), strict=True)
    observer.to(device=device, dtype=torch.float32)
    observer.eval()
    observer.requires_grad_(False)
    return observer


def restore_training_state_v2(
    checkpoint: LoadedCheckpointV2,
    method: Any,
    *,
    global_rank: int,
) -> ResumeStateV1:
    """Restore baseline optimizer/scheduler/progress/RNG without observer fit."""
    return restore_training_state_v1(
        checkpoint.baseline,
        method,
        global_rank=global_rank,
    )


def _decode_observer(value: object, torch: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_OBSERVER_SHAPES):
        raise ValueError("R4c v2 observer fields differ")
    result = dict(value)
    if any(
        not torch.is_tensor(result[name])
        or result[name].device.type != "cpu"
        or result[name].dtype != torch.float32
        or tuple(result[name].shape) != shape
        or not result[name].is_contiguous()
        for name, shape in _OBSERVER_SHAPES.items()
    ):
        raise ValueError("R4c v2 observer tensor differs")
    return result


def _attempt_reference(value: object) -> str:
    if not isinstance(value, str) or len(value.split("/")) != 3 or "//" in value:
        raise ValueError("observer_lineage must be job/run/attempt")
    return value


__all__ = [
    "LoadedCheckpointV2",
    "R4C_CHECKPOINT_FORMAT_V2",
    "TrainingStateV2",
    "load_checkpoint_v2",
    "restore_depth_observer_v2",
    "restore_lora_v2",
    "restore_training_state_v2",
    "save_checkpoint_v2",
]

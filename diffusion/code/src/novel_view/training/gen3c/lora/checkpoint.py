"""Строгий binary checkpoint baseline R4c LoRA и точный resume state."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novel_view.models.gen3c import lora_weights as model_lora
from novel_view.training.gen3c.data import (
    R4C_ORDER_VERSION,
    R4cDataCursor,
    load_r4c_legacy_index_map,
)
from novel_view.training.gen3c.lora.spec import R4C_LORA_SPEC
from novel_view.training.gen3c.topology import Gen3cTrainingTopology


R4C_CHECKPOINT_FORMAT_V1 = "novel-view/gen3c-lora-training-checkpoint/v1"
R4C_CHECKPOINT_STATE_FIELDS_V1 = frozenset(
    {
        "adapter",
        "optimizer",
        "scheduler",
        "progress",
        "topology",
        "rng_states",
    }
)
_TOP_LEVEL_FIELDS = R4C_CHECKPOINT_STATE_FIELDS_V1 | {"format"}
_LEGACY_TOP_LEVEL_FIELDS = R4C_CHECKPOINT_STATE_FIELDS_V1
_ADAPTER_FIELDS = frozenset({"model_name", "spec", "state_dict"})
_SPEC_FIELDS = frozenset(
    {
        "rank",
        "scale",
        "block_count",
        "self_attention_targets",
        "cross_attention_targets",
    }
)
_OPTIMIZER_FIELDS = frozenset({"group", "adam_w_mode", "state"})
_OPTIMIZER_GROUP_FIELDS = frozenset(
    {"lr", "weight_decay", "bias_correction", "betas", "eps", "initial_lr", "step"}
)
_OPTIMIZER_STATE_FIELDS = frozenset({"master", "exp_avg", "exp_avg_sq"})
_PROGRESS_FIELDS = frozenset({"next_step", "data_cursor"})
_LEGACY_PROGRESS_FIELDS = frozenset({"next_step", "data_cursor", "resolved_request"})
_CURSOR_FIELDS = frozenset(
    {"order_version", "training_sample_ids", "training_seed", "epoch", "offset"}
)
_LEGACY_CURSOR_FIELDS = frozenset(
    {
        "recipe_id",
        "split_id",
        "keyset_id",
        "training_seed",
        "order_version",
        "accepted_global_fit_indices",
        "epoch",
        "offset",
    }
)
_TOPOLOGY_FIELDS = frozenset(
    {"world_size", "context_parallel_size", "gradient_average_size"}
)
_LEGACY_TOPOLOGY_FIELDS = frozenset({"world_size", "context_parallel_size"})
_RNG_FIELDS = frozenset({"global_rank", "torch_cpu", "torch_cuda"})
_LEGACY_RNG_FIELDS = frozenset({"cpu", "cuda"})
_LEGACY_CURSOR_IDENTITY = {
    "recipe_id": "waymo-ddw-fit-random-one-v1",
    "split_id": "waymo-ddw-lora-v1",
    "keyset_id": "fit-central-v1",
    "order_version": "gen3c-waymo-ddw-random-one-order-v1",
}


@dataclass(frozen=True, slots=True)
class RankRngState:
    """CPU/CUDA RNG одного process с явным global-rank identity."""

    global_rank: int
    torch_cpu: Any
    torch_cuda: Any


@dataclass(frozen=True, slots=True)
class ResumeStateV1:
    """Следующий update, canonical cursor и RNG всех ranks."""

    next_step: int
    data_cursor: R4cDataCursor
    rank_rng_states: tuple[RankRngState, ...]


@dataclass(frozen=True, slots=True)
class TrainingStateV1:
    """Живое состояние, снимаемое одним binary writer после update."""

    model: Any
    optimizer: Any
    scheduler: Any
    next_step: int
    data_cursor: R4cDataCursor
    topology: Gen3cTrainingTopology
    rank_rng_states: tuple[RankRngState, ...]


@dataclass(frozen=True, slots=True)
class LoadedCheckpointV1:
    """Разобранный tagged либо узко адаптированный legacy checkpoint."""

    adapter: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    resume: ResumeStateV1
    topology: Mapping[str, int]
    legacy: bool


def save_checkpoint_v1(path: Path, state: TrainingStateV1) -> None:
    """Напрямую записать один completed-step tagged checkpoint."""
    payload = {"format": R4C_CHECKPOINT_FORMAT_V1, **pack_training_state_v1(state)}
    importlib.import_module("torch").save(payload, path)


def pack_training_state_v1(state: TrainingStateV1) -> dict[str, Any]:
    """Pack the shared v1 state body reused by the tagged v2 writer."""
    return {
        "adapter": _pack_adapter(state.model),
        "optimizer": pack_r4c_fused_adam(state.model, state.optimizer),
        "scheduler": _plain_cpu(state.scheduler.state_dict()),
        "progress": {
            "next_step": state.next_step,
            "data_cursor": _cursor_document(state.data_cursor),
        },
        "topology": {
            "world_size": state.topology.world_size,
            "context_parallel_size": state.topology.context_parallel_size,
            "gradient_average_size": state.topology.gradient_average_size,
        },
        "rng_states": [
            {
                "global_rank": item.global_rank,
                "torch_cpu": item.torch_cpu.detach().cpu().clone(),
                "torch_cuda": item.torch_cuda.detach().cpu().clone(),
            }
            for item in sorted(state.rank_rng_states, key=lambda item: item.global_rank)
        ],
    }


def load_checkpoint_v1(
    path: Path,
    *,
    legacy_index_map: Path | None = None,
) -> LoadedCheckpointV1:
    """Restricted-load tagged v1 или ровно один исторический six-field format."""
    torch = importlib.import_module("torch")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("R4c checkpoint root must be a mapping")
    if "format" in payload:
        if payload["format"] != R4C_CHECKPOINT_FORMAT_V1:
            raise ValueError("unsupported R4c checkpoint format")
        _exact_fields(payload, _TOP_LEVEL_FIELDS, "R4c checkpoint")
        return decode_training_state_v1(
            {field: payload[field] for field in R4C_CHECKPOINT_STATE_FIELDS_V1},
            torch,
        )
    if legacy_index_map is None:
        raise ValueError("untagged R4c checkpoint requires the legacy index map")
    return _decode_legacy_checkpoint(
        dict(payload),
        load_r4c_legacy_index_map(legacy_index_map),
        torch,
    )


def restore_lora_v1(checkpoint: LoadedCheckpointV1, model: Any) -> None:
    """Восстановить raw adapter до CP/recompute/DDP."""
    model_lora.replace_gen3c_lora_adapter(
        model,
        checkpoint.adapter["state_dict"],
    )


def restore_training_state_v1(
    checkpoint: LoadedCheckpointV1,
    method: Any,
    *,
    global_rank: int,
) -> ResumeStateV1:
    """Восстановить scheduler, FP32 FusedAdam и RNG выбранного global rank."""
    method.scheduler.load_state_dict(dict(checkpoint.scheduler))
    restore_r4c_fused_adam(method.model.model, method.optimizer, checkpoint.optimizer)
    rng = next(
        item
        for item in checkpoint.resume.rank_rng_states
        if item.global_rank == global_rank
    )
    torch = importlib.import_module("torch")
    torch.set_rng_state(rng.torch_cpu)
    torch.cuda.set_rng_state(rng.torch_cuda, torch.cuda.current_device())
    return checkpoint.resume


def capture_rank_rng_state(global_rank: int) -> RankRngState:
    """Снять CPU/CUDA RNG текущего rank после завершённого update."""
    torch = importlib.import_module("torch")
    return RankRngState(
        global_rank,
        torch.get_rng_state(),
        torch.cuda.get_rng_state(torch.cuda.current_device()),
    )


def gather_rank_rng_states(
    local: RankRngState,
    topology: Gen3cTrainingTopology,
) -> tuple[RankRngState, ...]:
    """Собрать RNG и нормализовать порядок только по global rank."""
    distributed = importlib.import_module("torch.distributed")
    gathered: list[RankRngState | None] = [None] * topology.world_size
    distributed.all_gather_object(
        gathered,
        local,
        group=topology.context_parallel_group,
    )
    return tuple(
        sorted(
            (item for item in gathered if item is not None),
            key=lambda item: item.global_rank,
        )
    )


def pack_r4c_fused_adam(model: Any, optimizer: Any) -> dict[str, Any]:
    """Скопировать FP32 master/moments по стабильному LoRA key."""
    group = _fused_adam_group(optimizer)
    parameters = model_lora.gen3c_lora_state_by_name(model, keep_vars=True)
    names_by_id = {id(parameter): name for name, parameter in parameters.items()}
    masters = optimizer.param_groups_master[0]["params"]
    state: dict[str, Any] = {}
    for parameter, master in zip(group["params"], masters, strict=True):
        name = names_by_id[id(parameter)]
        moments = optimizer.state[parameter]
        state[name] = {
            "master": master.detach().cpu().clone(),
            "exp_avg": moments["exp_avg"].detach().cpu().clone(),
            "exp_avg_sq": moments["exp_avg_sq"].detach().cpu().clone(),
        }
    return {
        "group": _plain_cpu(
            {key: value for key, value in group.items() if key != "params"}
        ),
        "adam_w_mode": optimizer.adam_w_mode,
        "state": state,
    }


def restore_r4c_fused_adam(
    model: Any,
    optimizer: Any,
    payload: Mapping[str, Any],
) -> None:
    """Восстановить fresh FusedAdam без BF16 round-trip master weights."""
    group = _fused_adam_group(optimizer)
    if optimizer.param_groups_master is not None or optimizer.state:
        raise RuntimeError("R4c FusedAdam must be fresh before restore")
    parameters = model_lora.gen3c_lora_state_by_name(model, keep_vars=True)
    names_by_id = {id(parameter): name for name, parameter in parameters.items()}
    torch = importlib.import_module("torch")
    device = group["params"][0].device
    for key, value in payload["group"].items():
        group[key] = (
            value.to(device=device).clone() if torch.is_tensor(value) else value
        )

    masters = []
    for parameter in group["params"]:
        values = payload["state"][names_by_id[id(parameter)]]
        master = values["master"].to(device=device).clone()
        masters.append(master)
        optimizer.state[parameter] = {
            "exp_avg": values["exp_avg"].to(device=device).clone(),
            "exp_avg_sq": values["exp_avg_sq"].to(device=device).clone(),
        }
    optimizer.param_groups_master = [{"params": masters}]


def decode_training_state_v1(
    payload: Mapping[str, Any], torch: Any
) -> LoadedCheckpointV1:
    """Decode the exact shared v1 state body from a tagged checkpoint."""
    payload = dict(payload)
    adapter = _decode_adapter(payload["adapter"], torch)
    optimizer = _decode_optimizer(payload["optimizer"], adapter["state_dict"], torch)
    scheduler = _mapping(payload["scheduler"], "scheduler")
    progress = _mapping(payload["progress"], "progress")
    _exact_fields(progress, _PROGRESS_FIELDS, "progress")
    next_step = _positive_integer(progress["next_step"], "next_step")
    cursor = _decode_cursor(progress["data_cursor"], next_step)
    topology = _decode_topology(payload["topology"], legacy=False)
    rng_states = _decode_rng_states(
        payload["rng_states"], topology["world_size"], torch
    )
    return LoadedCheckpointV1(
        adapter,
        optimizer,
        scheduler,
        ResumeStateV1(next_step, cursor, rng_states),
        topology,
        False,
    )


def _decode_legacy_checkpoint(
    payload: dict[str, Any], index_map: Any, torch: Any
) -> LoadedCheckpointV1:
    _exact_fields(payload, _LEGACY_TOP_LEVEL_FIELDS, "legacy R4c checkpoint")
    adapter = _decode_adapter(payload["adapter"], torch)
    optimizer = _decode_optimizer(payload["optimizer"], adapter["state_dict"], torch)
    scheduler = _mapping(payload["scheduler"], "scheduler")
    progress = _mapping(payload["progress"], "progress")
    _exact_fields(progress, _LEGACY_PROGRESS_FIELDS, "legacy progress")
    next_step = _positive_integer(progress["next_step"], "next_step")
    cursor = _decode_legacy_cursor(progress["data_cursor"], index_map, next_step)
    topology = _decode_topology(payload["topology"], legacy=True)
    rng_states = _decode_legacy_rng_states(
        payload["rng_states"], topology["world_size"], torch
    )
    return LoadedCheckpointV1(
        adapter,
        optimizer,
        scheduler,
        ResumeStateV1(next_step, cursor, rng_states),
        topology,
        True,
    )


def _pack_adapter(model: Any) -> dict[str, Any]:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model_lora.gen3c_lora_state_by_name(model).items()
    }
    return {
        "model_name": model_lora.GEN3C_MODEL_NAME,
        "spec": {
            "rank": R4C_LORA_SPEC.rank,
            "scale": R4C_LORA_SPEC.scale,
            "block_count": R4C_LORA_SPEC.block_count,
            "self_attention_targets": list(R4C_LORA_SPEC.self_attention_targets),
            "cross_attention_targets": list(R4C_LORA_SPEC.cross_attention_targets),
        },
        "state_dict": state,
    }


def _decode_adapter(value: Any, torch: Any) -> dict[str, Any]:
    adapter = _mapping(value, "adapter")
    _exact_fields(adapter, _ADAPTER_FIELDS, "adapter")
    spec = _mapping(adapter["spec"], "adapter spec")
    _exact_fields(spec, _SPEC_FIELDS, "adapter spec")
    if (
        adapter["model_name"] != model_lora.GEN3C_MODEL_NAME
        or model_lora.decode_gen3c_lora_spec(spec) != R4C_LORA_SPEC
    ):
        raise ValueError("R4c checkpoint adapter identity differs")
    state = _mapping(adapter["state_dict"], "adapter state_dict")
    if not state or any(
        not isinstance(name, str)
        or "_lora." not in name
        or not _cpu_tensor(torch, tensor, torch.bfloat16)
        for name, tensor in state.items()
    ):
        raise ValueError("R4c checkpoint adapter state is invalid")
    return {"model_name": adapter["model_name"], "spec": spec, "state_dict": state}


def _decode_optimizer(
    value: Any, adapter: Mapping[str, Any], torch: Any
) -> dict[str, Any]:
    optimizer = _mapping(value, "optimizer")
    _exact_fields(optimizer, _OPTIMIZER_FIELDS, "optimizer")
    group = _mapping(optimizer["group"], "optimizer group")
    _exact_fields(group, _OPTIMIZER_GROUP_FIELDS, "optimizer group")
    state = _mapping(optimizer["state"], "optimizer state")
    if optimizer["adam_w_mode"] != 1 or set(state) != set(adapter):
        raise ValueError("R4c checkpoint optimizer identity differs")
    for name, tensor in adapter.items():
        values = _mapping(state[name], f"optimizer state {name}")
        _exact_fields(values, _OPTIMIZER_STATE_FIELDS, f"optimizer state {name}")
        if any(
            not _cpu_tensor(torch, values[field], torch.float32, tensor.shape)
            for field in _OPTIMIZER_STATE_FIELDS
        ):
            raise ValueError(f"R4c checkpoint optimizer tensor differs: {name}")
    return {"group": group, "adam_w_mode": 1, "state": state}


def _decode_cursor(value: Any, next_step: int) -> R4cDataCursor:
    cursor = _mapping(value, "data cursor")
    _exact_fields(cursor, _CURSOR_FIELDS, "data cursor")
    if cursor["order_version"] != R4C_ORDER_VERSION:
        raise ValueError("unsupported R4c data order")
    sample_ids = _text_sequence(cursor["training_sample_ids"], "training_sample_ids")
    training_seed = _nonnegative_integer(cursor["training_seed"], "training_seed")
    epoch = _nonnegative_integer(cursor["epoch"], "epoch")
    offset = _nonnegative_integer(cursor["offset"], "offset")
    if offset >= len(sample_ids) or epoch * len(sample_ids) + offset != next_step:
        raise ValueError("R4c checkpoint cursor differs from next_step")
    return R4cDataCursor(R4C_ORDER_VERSION, sample_ids, training_seed, epoch, offset)


def _decode_legacy_cursor(value: Any, index_map: Any, next_step: int) -> R4cDataCursor:
    cursor = _mapping(value, "legacy data cursor")
    _exact_fields(cursor, _LEGACY_CURSOR_FIELDS, "legacy data cursor")
    if any(
        cursor[field] != expected for field, expected in _LEGACY_CURSOR_IDENTITY.items()
    ):
        raise ValueError("unsupported legacy R4c data cursor")
    old_indices = cursor["accepted_global_fit_indices"]
    if (
        not isinstance(old_indices, (list, tuple))
        or not old_indices
        or any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in old_indices
        )
    ):
        raise ValueError("legacy R4c indices must be a non-empty sequence")
    by_old_index = {
        item.old_global_fit_index: item.sample_id for item in index_map.items
    }
    try:
        sample_ids = tuple(by_old_index[index] for index in old_indices)
    except (KeyError, TypeError) as error:
        raise ValueError("legacy R4c cursor contains an unknown index") from error
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("legacy R4c cursor contains duplicate indices")
    training_seed = _nonnegative_integer(cursor["training_seed"], "training_seed")
    epoch = _nonnegative_integer(cursor["epoch"], "epoch")
    offset = _nonnegative_integer(cursor["offset"], "offset")
    if offset >= len(sample_ids) or epoch * len(sample_ids) + offset != next_step:
        raise ValueError("legacy R4c cursor differs from next_step")
    return R4cDataCursor(R4C_ORDER_VERSION, sample_ids, training_seed, epoch, offset)


def _decode_topology(value: Any, *, legacy: bool) -> dict[str, int]:
    topology = _mapping(value, "topology")
    fields = _LEGACY_TOPOLOGY_FIELDS if legacy else _TOPOLOGY_FIELDS
    _exact_fields(topology, fields, "topology")
    decoded = {field: _positive_integer(topology[field], field) for field in fields}
    if legacy:
        decoded["gradient_average_size"] = decoded["context_parallel_size"]
    return decoded


def _decode_rng_states(
    value: Any, world_size: int, torch: Any
) -> tuple[RankRngState, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != world_size:
        raise ValueError("R4c checkpoint RNG state count differs")
    states = []
    for item in value:
        row = _mapping(item, "RNG state")
        _exact_fields(row, _RNG_FIELDS, "RNG state")
        rank = _nonnegative_integer(row["global_rank"], "global_rank")
        if not _cpu_tensor(torch, row["torch_cpu"], torch.uint8) or not _cpu_tensor(
            torch, row["torch_cuda"], torch.uint8
        ):
            raise ValueError("R4c checkpoint RNG tensor differs")
        states.append(RankRngState(rank, row["torch_cpu"], row["torch_cuda"]))
    states.sort(key=lambda item: item.global_rank)
    if tuple(item.global_rank for item in states) != tuple(range(world_size)):
        raise ValueError("R4c checkpoint RNG global ranks differ")
    return tuple(states)


def _decode_legacy_rng_states(
    value: Any, world_size: int, torch: Any
) -> tuple[RankRngState, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != world_size:
        raise ValueError("legacy R4c checkpoint RNG state count differs")
    states = []
    for rank, item in enumerate(value):
        row = _mapping(item, "legacy RNG state")
        _exact_fields(row, _LEGACY_RNG_FIELDS, "legacy RNG state")
        if not _cpu_tensor(torch, row["cpu"], torch.uint8) or not _cpu_tensor(
            torch, row["cuda"], torch.uint8
        ):
            raise ValueError("legacy R4c checkpoint RNG tensor differs")
        states.append(RankRngState(rank, row["cpu"], row["cuda"]))
    return tuple(states)


def _cursor_document(cursor: R4cDataCursor) -> dict[str, Any]:
    return {
        "order_version": cursor.order_version,
        "training_sample_ids": list(cursor.training_sample_ids),
        "training_seed": cursor.training_seed,
        "epoch": cursor.epoch,
        "offset": cursor.offset,
    }


def _fused_adam_group(optimizer: Any) -> dict[str, Any]:
    if (
        optimizer.master_weights is not True
        or optimizer.capturable is not True
        or optimizer.adam_w_mode != 1
    ):
        raise RuntimeError("R4c optimizer is not the pinned FP32-master FusedAdam")
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("R4c FusedAdam must have one parameter group")
    return optimizer.param_groups[0]


def _plain_cpu(value: Any) -> Any:
    torch = importlib.import_module("torch")
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _plain_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        values = [_plain_cpu(item) for item in value]
        return tuple(values) if isinstance(value, tuple) else values
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"checkpoint value is not weights-only compatible: {type(value)}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _exact_fields(value: Mapping[str, Any], fields: frozenset[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} fields differ")


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty text sequence")
    values = tuple(value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return values


def _cpu_tensor(torch: Any, value: Any, dtype: Any, shape: Any = None) -> bool:
    return (
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.dtype == dtype
        and (shape is None or value.shape == shape)
    )


__all__ = [
    "LoadedCheckpointV1",
    "R4C_CHECKPOINT_FORMAT_V1",
    "R4C_CHECKPOINT_STATE_FIELDS_V1",
    "RankRngState",
    "ResumeStateV1",
    "TrainingStateV1",
    "capture_rank_rng_state",
    "decode_training_state_v1",
    "gather_rank_rng_states",
    "load_checkpoint_v1",
    "pack_r4c_fused_adam",
    "pack_training_state_v1",
    "restore_lora_v1",
    "restore_r4c_fused_adam",
    "restore_training_state_v1",
    "save_checkpoint_v1",
]

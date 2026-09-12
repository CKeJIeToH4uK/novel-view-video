"""Подготовленные R4c items, deterministic order и старый index handoff."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import numpy as np

from novel_view.config.load import load_yaml_mapping, reject_unknown_fields
from novel_view.models.gen3c.spec import R4C_GEN3C_MODEL_CONTRACT
from novel_view.preparation.waymo_ddw import artifacts as prepared_artifacts
from novel_view.preparation.waymo_ddw.record import PreparedRecord


R4C_LEGACY_INDEX_MAP_FORMAT = "novel-view/r4c-ddw85-legacy-index-map/v1"
R4C_ORDER_VERSION = "r4c-pcg64-sample-id/v1"
_SPLIT_FIELDS = frozenset(
    {"schema_version", "training_sample_ids", "validation_sample_ids"}
)
_LEGACY_MAP_FIELDS = frozenset({"format", "items"})
_LEGACY_ITEM_FIELDS = frozenset({"old_global_fit_index", "sample_id"})
_R4C_DDW85_SAMPLE_COUNT = 85


@dataclass(frozen=True, slots=True)
class R4cTrainingSplit:
    """Ordered stable sample IDs selected for training and validation."""

    training_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R4cPreparedItem:
    """One training identity with unopened Stage 6 artifact locators."""

    sample_id: str
    segment_id: str
    base_latent: Path
    pose_latent: Path
    prompt_embedding: Path
    magnitude_m: float
    sign: int


@dataclass(frozen=True, slots=True)
class R4cDataCursor:
    """Normalized position of the next item in deterministic epoch order."""

    order_version: Literal["r4c-pcg64-sample-id/v1"]
    training_sample_ids: tuple[str, ...]
    training_seed: int
    epoch: int
    offset: int


@dataclass(frozen=True, slots=True, eq=False)
class R4cTrainingBatch:
    """Four raw model inputs materialized for one baseline update."""

    sample_id: str
    clean_latent: Any
    source_latent: Any
    pose_latent: Any
    prompt_embedding: Any
    next_cursor: R4cDataCursor


@dataclass(frozen=True, slots=True)
class LegacyR4cIndexItem:
    """One old DDW85 fit index translated to a stable sample ID."""

    old_global_fit_index: int
    sample_id: str


@dataclass(frozen=True, slots=True)
class LegacyR4cIndexMap:
    """Exact old DDW85 order used only to read an untagged checkpoint."""

    items: tuple[LegacyR4cIndexItem, ...]


def load_r4c_training_split(path: Path) -> R4cTrainingSplit:
    """Read the ordered training/validation split without opening artifacts."""
    values = load_yaml_mapping(path)
    reject_unknown_fields(values, _SPLIT_FIELDS)
    _require_fields(values, _SPLIT_FIELDS, "R4c split")
    if _integer(values, "schema_version") != 1:
        raise ValueError("unsupported R4c split schema_version")

    training = _sample_ids(values, "training_sample_ids")
    validation = _sample_ids(values, "validation_sample_ids")
    if set(training) & set(validation):
        raise ValueError("R4c training and validation sample IDs overlap")
    return R4cTrainingSplit(training, validation)


def resolve_r4c_items(
    prepared: PreparedRecord,
    split: R4cTrainingSplit,
) -> tuple[tuple[R4cPreparedItem, ...], tuple[R4cPreparedItem, ...]]:
    """Join one ordered split to an already validated prepared record."""
    by_sample_id = {item.sample_id: item for item in prepared.items}
    requested = split.training_sample_ids + split.validation_sample_ids
    missing = [sample_id for sample_id in requested if sample_id not in by_sample_id]
    if missing:
        raise ValueError(f"R4c split contains unknown sample IDs: {missing}")

    prompt = Path(prepared.empty_prompt)

    def resolve(sample_ids: tuple[str, ...]) -> tuple[R4cPreparedItem, ...]:
        return tuple(
            R4cPreparedItem(
                sample_id=source.sample_id,
                segment_id=source.segment_id,
                base_latent=Path(source.base_latent),
                pose_latent=Path(source.pose_latent),
                prompt_embedding=prompt,
                magnitude_m=source.magnitude_m,
                sign=source.sign,
            )
            for source in (by_sample_id[sample_id] for sample_id in sample_ids)
        )

    training = resolve(split.training_sample_ids)
    validation = resolve(split.validation_sample_ids)
    training_segments = tuple(item.segment_id for item in training)
    validation_segments = tuple(item.segment_id for item in validation)
    if (
        len(set(training_segments)) != len(training_segments)
        or len(set(validation_segments)) != len(validation_segments)
        or set(training_segments) & set(validation_segments)
    ):
        raise ValueError("R4c training and validation segments must be disjoint")
    return training, validation


def epoch_item_order(
    items: tuple[R4cPreparedItem, ...],
    *,
    training_seed: int,
    epoch: int,
) -> tuple[R4cPreparedItem, ...]:
    """Return the literal local PCG64 permutation for one epoch."""
    permutation = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([training_seed, epoch]))
    ).permutation(len(items))
    return tuple(items[int(index)] for index in permutation)


def r4c_data_cursor(
    items: tuple[R4cPreparedItem, ...],
    *,
    training_seed: int,
    completed_steps: int = 0,
) -> R4cDataCursor:
    """Build the canonical next-item cursor from completed update count."""
    epoch, offset = divmod(completed_steps, len(items))
    return R4cDataCursor(
        order_version=R4C_ORDER_VERSION,
        training_sample_ids=tuple(item.sample_id for item in items),
        training_seed=training_seed,
        epoch=epoch,
        offset=offset,
    )


def iter_r4c_items(
    items: tuple[R4cPreparedItem, ...],
    cursor: R4cDataCursor,
) -> Iterator[tuple[R4cPreparedItem, R4cDataCursor]]:
    """Yield items forever with the cursor already advanced past each item."""
    epoch = cursor.epoch
    offset = cursor.offset
    while True:
        order = epoch_item_order(
            items,
            training_seed=cursor.training_seed,
            epoch=epoch,
        )
        while offset < len(order):
            item = order[offset]
            next_epoch = epoch + 1 if offset + 1 == len(order) else epoch
            next_offset = 0 if next_epoch != epoch else offset + 1
            yield item, R4cDataCursor(
                order_version=R4C_ORDER_VERSION,
                training_sample_ids=cursor.training_sample_ids,
                training_seed=cursor.training_seed,
                epoch=next_epoch,
                offset=next_offset,
            )
            offset += 1
        epoch, offset = epoch + 1, 0


def materialize_r4c_training_batch(
    item: R4cPreparedItem,
    next_cursor: R4cDataCursor,
    *,
    artifact_root: Path,
    device: object,
) -> R4cTrainingBatch:
    """Open one baseline item through the strict Stage 6 artifact readers."""
    base = prepared_artifacts.read_base_latents(
        artifact_root / item.base_latent,
        item.sample_id,
    )
    pose = prepared_artifacts.read_pose_latent(
        artifact_root / item.pose_latent,
        item.sample_id,
        item.magnitude_m,
        item.sign,
    )
    prompt = prepared_artifacts.read_empty_prompt(artifact_root / item.prompt_embedding)
    clean, source, pose_tensor, prompt_tensor = (
        tensor.to(device=device)
        for tensor in (
            base.clean_latent,
            base.source_latent,
            pose.pose_latent,
            prompt.t5_text_embeddings,
        )
    )
    return R4cTrainingBatch(
        sample_id=item.sample_id,
        clean_latent=clean,
        source_latent=source,
        pose_latent=pose_tensor,
        prompt_embedding=prompt_tensor,
        next_cursor=next_cursor,
    )


def iter_r4c_training_batches(
    items: tuple[R4cPreparedItem, ...],
    cursor: R4cDataCursor,
    *,
    artifact_root: Path,
    device: object,
    context_parallel_group: object,
) -> Iterator[R4cTrainingBatch]:
    """Materialize on the source rank and broadcast four raw inputs once."""
    torch = importlib.import_module("torch")
    distributed = importlib.import_module("torch.distributed")
    ranks = distributed.get_process_group_ranks(context_parallel_group)
    source_rank = min(ranks)
    rank = distributed.get_rank()

    for item, next_cursor in iter_r4c_items(items, cursor):
        batch: R4cTrainingBatch | None = None
        local_error: str | None = None
        try:
            if rank == source_rank:
                batch = materialize_r4c_training_batch(
                    item,
                    next_cursor,
                    artifact_root=artifact_root,
                    device=device,
                )
            else:
                batch = _empty_r4c_training_batch(
                    torch,
                    item.sample_id,
                    next_cursor,
                    device,
                )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        raise_r4c_source_errors(
            distributed,
            context_parallel_group,
            ranks,
            local_error,
            operation="data materialization",
        )

        ready = cast(R4cTrainingBatch, batch)
        for tensor in _r4c_batch_tensors(ready):
            distributed.broadcast(
                tensor,
                source_rank,
                group=context_parallel_group,
            )
        yield ready


def _empty_r4c_training_batch(
    torch: Any,
    sample_id: str,
    next_cursor: R4cDataCursor,
    device: object,
) -> R4cTrainingBatch:
    contract = R4C_GEN3C_MODEL_CONTRACT
    clean, source, pose, prompt = (
        torch.empty(shape, dtype=torch.bfloat16, device=device)
        for shape in (
            contract.base_latent_shape,
            contract.base_latent_shape,
            contract.pose_latent_shape,
            contract.prompt_embedding_shape,
        )
    )
    return R4cTrainingBatch(
        sample_id=sample_id,
        clean_latent=clean,
        source_latent=source,
        pose_latent=pose,
        prompt_embedding=prompt,
        next_cursor=next_cursor,
    )


def _r4c_batch_tensors(batch: R4cTrainingBatch) -> tuple[Any, ...]:
    return (
        batch.clean_latent,
        batch.source_latent,
        batch.pose_latent,
        batch.prompt_embedding,
    )


def raise_r4c_source_errors(
    distributed: Any,
    group: object,
    ranks: list[int],
    local_error: str | None,
    *,
    operation: str,
) -> None:
    errors: list[str | None] = [None] * len(ranks)
    distributed.all_gather_object(errors, local_error, group=group)
    failures = [
        f"rank {rank}: {error}"
        for rank, error in zip(ranks, errors, strict=True)
        if error is not None
    ]
    if failures:
        raise RuntimeError(f"R4c {operation} failed; " + "; ".join(failures))


def load_r4c_legacy_index_map(path: Path) -> LegacyR4cIndexMap:
    """Read the closed 85-row map for the exact untagged v1 adapter."""
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("R4c legacy index map root must be a mapping")
    reject_unknown_fields(values, _LEGACY_MAP_FIELDS)
    _require_fields(values, _LEGACY_MAP_FIELDS, "R4c legacy index map")
    if values["format"] != R4C_LEGACY_INDEX_MAP_FORMAT:
        raise ValueError("unsupported R4c legacy index map format")

    rows = values["items"]
    if not isinstance(rows, list) or len(rows) != _R4C_DDW85_SAMPLE_COUNT:
        raise ValueError("R4c legacy index map must contain exactly 85 items")
    items = tuple(_legacy_item(row) for row in rows)
    old_indices = tuple(item.old_global_fit_index for item in items)
    sample_ids = tuple(item.sample_id for item in items)
    if len(set(old_indices)) != len(old_indices):
        raise ValueError("R4c legacy index map contains duplicate old indices")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("R4c legacy index map contains duplicate sample IDs")
    return LegacyR4cIndexMap(items)


def _legacy_item(value: object) -> LegacyR4cIndexItem:
    if not isinstance(value, dict):
        raise ValueError("R4c legacy index map item must be a mapping")
    reject_unknown_fields(value, _LEGACY_ITEM_FIELDS)
    _require_fields(value, _LEGACY_ITEM_FIELDS, "R4c legacy index map item")
    old_index = _integer(value, "old_global_fit_index")
    if old_index < 0:
        raise ValueError("old_global_fit_index must be non-negative")
    return LegacyR4cIndexItem(old_index, _text(value, "sample_id"))


def _sample_ids(values: Mapping[str, object], field: str) -> tuple[str, ...]:
    raw = values[field]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field} must be a non-empty list")
    sample_ids = tuple(_text({field: value}, field) for value in raw)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{field} contains duplicate sample IDs")
    return sample_ids


def _require_fields(
    values: Mapping[str, object],
    fields: frozenset[str],
    name: str,
) -> None:
    missing = fields - values.keys()
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)}")


def _text(values: Mapping[str, object], field: str) -> str:
    value = values[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _integer(values: Mapping[str, object], field: str) -> int:
    value = values[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


__all__ = [
    "LegacyR4cIndexItem",
    "LegacyR4cIndexMap",
    "R4C_LEGACY_INDEX_MAP_FORMAT",
    "R4C_ORDER_VERSION",
    "R4cDataCursor",
    "R4cPreparedItem",
    "R4cTrainingBatch",
    "R4cTrainingSplit",
    "epoch_item_order",
    "iter_r4c_items",
    "iter_r4c_training_batches",
    "load_r4c_legacy_index_map",
    "load_r4c_training_split",
    "materialize_r4c_training_batch",
    "raise_r4c_source_errors",
    "r4c_data_cursor",
    "resolve_r4c_items",
]

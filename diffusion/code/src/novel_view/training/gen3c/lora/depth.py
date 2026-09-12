"""V2-only sparse LiDAR supervision on the Gen3C latent grid."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from novel_view.preparation.waymo_ddw import artifacts as prepared_artifacts
from novel_view.preparation.waymo_ddw.artifacts import (
    FRAME_COUNT,
    LidarDepth,
    read_lidar_depth,
)
from novel_view.preparation.waymo_ddw.record import PreparedRecord
from novel_view.training.gen3c.data import (
    R4cPreparedItem,
    R4cTrainingSplit,
    raise_r4c_source_errors,
    resolve_r4c_items,
)
from novel_view.training.gen3c.lora.depth_spec import (
    DEPTH_OBSERVER_NAME,
    DEPTH_OBSERVER_VERSION,
    DEPTH_REDUCER_NAME,
    DEPTH_REDUCER_VERSION,
    DepthObserverFitResult,
    DepthObserverFitSpec,
)
from novel_view.training.gen3c.topology import Gen3cTrainingTopology


DEPTH_GRID_SHAPE = (1, 1, 16, 88, 160)
_CELL_SIZE_PX = 8.0
_GRID_HEIGHT = DEPTH_GRID_SHAPE[-2]
_GRID_WIDTH = DEPTH_GRID_SHAPE[-1]


@dataclass(frozen=True, slots=True)
class R4cDepthItem:
    """One v2 item with the baseline inputs and its unopened LiDAR locator."""

    prepared: R4cPreparedItem
    lidar_depth: Path


@dataclass(frozen=True, slots=True, eq=False)
class DepthGrid:
    """FP32 log camera-Z values and bool validity on the full latent grid."""

    values: Tensor
    valid: Tensor


class LatentDepthObserver(nn.Module):
    """Small readout fitted before the frozen 7B model is loaded."""

    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.input = nn.Conv3d(16, 32, kernel_size=1)
            self.activation = nn.SiLU()
            self.output = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, latent: Tensor) -> Tensor:
        return self.output(self.activation(self.input(latent)))


def resolve_r4c_depth_items(
    prepared: PreparedRecord,
    split: R4cTrainingSplit,
) -> tuple[tuple[R4cDepthItem, ...], tuple[R4cDepthItem, ...]]:
    """Attach strict Stage 6 LiDAR locators only for the v2 data path."""
    training, validation = resolve_r4c_items(prepared, split)
    source_by_sample = {item.sample_id: item for item in prepared.items}

    def attach(items: tuple[R4cPreparedItem, ...]) -> tuple[R4cDepthItem, ...]:
        return tuple(
            R4cDepthItem(
                prepared=item,
                lidar_depth=Path(source_by_sample[item.sample_id].lidar_depth),
            )
            for item in items
        )

    return attach(training), attach(validation)


def read_r4c_depth_grid(item: R4cDepthItem, *, artifact_root: Path) -> DepthGrid:
    """Open the named sparse artifact directly and reduce it once."""
    artifact = read_lidar_depth(
        artifact_root / item.lidar_depth,
        item.prepared.sample_id,
    )
    return reduce_lidar_depth(artifact)


def reduce_lidar_depth(artifact: LidarDepth) -> DepthGrid:
    """Reduce non-heldout FRONT-A points by exact cell median log camera-Z."""
    offsets = artifact.frame_offsets.numpy()
    frame_indices = np.repeat(
        np.arange(FRAME_COUNT, dtype=np.int64),
        np.diff(offsets),
    )
    xy = artifact.canvas_xy.numpy()
    camera_z = artifact.depth_z_m.numpy()
    heldout = artifact.evaluation_holdout.numpy()
    selected = (frame_indices > 0) & ~heldout

    latent_slots = 1 + (frame_indices[selected] - 1) // 8
    x = np.floor(xy[selected, 0] / _CELL_SIZE_PX).astype(np.int64)
    y = np.floor(xy[selected, 1] / _CELL_SIZE_PX).astype(np.int64)
    cell_keys = (latent_slots * _GRID_HEIGHT + y) * _GRID_WIDTH + x
    log_depth = np.log(camera_z[selected].astype(np.float64))

    if cell_keys.size == 0:
        raise ValueError(
            f"R4c depth sample {artifact.sample_id} has no non-heldout latent cells"
        )

    order = np.lexsort((log_depth, cell_keys))
    sorted_keys = cell_keys[order]
    sorted_depth = log_depth[order]
    unique_keys, starts, counts = np.unique(
        sorted_keys,
        return_index=True,
        return_counts=True,
    )
    lower = starts + (counts - 1) // 2
    upper = starts + counts // 2
    medians = (sorted_depth[lower] + sorted_depth[upper]) * 0.5

    values = np.zeros(np.prod(DEPTH_GRID_SHAPE), dtype=np.float32)
    valid = np.zeros(np.prod(DEPTH_GRID_SHAPE), dtype=np.bool_)
    values[unique_keys] = medians.astype(np.float32)
    valid[unique_keys] = True
    return DepthGrid(
        values=torch.from_numpy(values.reshape(DEPTH_GRID_SHAPE)),
        valid=torch.from_numpy(valid.reshape(DEPTH_GRID_SHAPE)),
    )


def split_depth_grid_cp(depth: DepthGrid, context_parallel_group: object) -> DepthGrid:
    """Split values and mask only through the upstream Gen3C CP primitive."""
    parallel = importlib.import_module("cosmos_predict1.diffusion.module.parallel")
    return DepthGrid(
        values=parallel.split_inputs_cp(
            depth.values,
            seq_dim=2,
            cp_group=context_parallel_group,
        ),
        valid=parallel.split_inputs_cp(
            depth.valid,
            seq_dim=2,
            cp_group=context_parallel_group,
        ),
    )


def fit_depth_observer(
    observer: LatentDepthObserver,
    training_items: tuple[R4cDepthItem, ...],
    validation_items: tuple[R4cDepthItem, ...],
    spec: DepthObserverFitSpec,
    topology: Gen3cTrainingTopology,
    *,
    artifact_root: Path,
    device: object,
) -> DepthObserverFitResult:
    """Fit on the CP source, require validation signal, freeze and broadcast."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(spec.init_seed)
        if cuda_rng is not None:
            torch.cuda.manual_seed_all(spec.init_seed)
        observer.to(device=device, dtype=torch.float32)
        observer.input.reset_parameters()
        observer.output.reset_parameters()

        distributed, ranks, source_rank = _observer_group(topology)
        result: DepthObserverFitResult | None = None
        local_error: str | None = None
        if topology.rank == source_rank:
            try:
                result = _fit_depth_observer_source(
                    observer,
                    training_items,
                    validation_items,
                    spec,
                    artifact_root=artifact_root,
                    device=device,
                )
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"

        if distributed is None:
            if local_error is not None:
                raise RuntimeError(
                    f"R4c depth observer fit failed; rank {source_rank}: {local_error}"
                )
        else:
            raise_r4c_source_errors(
                distributed,
                topology.context_parallel_group,
                ranks,
                local_error,
                operation="depth observer fit",
            )
            measurements = (
                torch.tensor(
                    (
                        result.training_loss,
                        result.validation_loss,
                        result.constant_comparator_loss,
                    ),
                    dtype=torch.float64,
                    device=device,
                )
                if result is not None
                else torch.empty(3, dtype=torch.float64, device=device)
            )
            distributed.broadcast(
                measurements,
                source_rank,
                group=topology.context_parallel_group,
            )
            result = DepthObserverFitResult(*measurements.cpu().tolist())

        observer.eval()
        observer.requires_grad_(False)
        if distributed is not None:
            for parameter in observer.parameters():
                distributed.broadcast(
                    parameter,
                    source_rank,
                    group=topology.context_parallel_group,
                )
        return cast(DepthObserverFitResult, result)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _observer_group(
    topology: Gen3cTrainingTopology,
) -> tuple[Any | None, list[int], int]:
    if topology.context_parallel_group is None:
        return None, [topology.rank], topology.rank
    distributed = torch.distributed
    ranks = distributed.get_process_group_ranks(topology.context_parallel_group)
    return distributed, ranks, min(ranks)


def _fit_depth_observer_source(
    observer: LatentDepthObserver,
    training_items: tuple[R4cDepthItem, ...],
    validation_items: tuple[R4cDepthItem, ...],
    spec: DepthObserverFitSpec,
    *,
    artifact_root: Path,
    device: object,
) -> DepthObserverFitResult:
    observer.train()
    optimizer = torch.optim.AdamW(
        observer.parameters(),
        lr=spec.learning_rate,
        betas=spec.betas,
        eps=spec.epsilon,
        weight_decay=spec.weight_decay,
    )
    training_targets: list[np.ndarray] = []
    training_loss = math.nan
    for epoch in range(spec.fit_epochs):
        permutation = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([spec.fit_order_seed, epoch]))
        ).permutation(len(training_items))
        item_losses: list[float] = []
        for index in permutation:
            clean, target, valid = _read_depth_observer_item(
                training_items[int(index)],
                artifact_root=artifact_root,
                device=device,
            )
            if epoch == 0:
                training_targets.append(
                    target[valid].detach().cpu().numpy().astype(np.float64, copy=False)
                )
            optimizer.zero_grad(set_to_none=True)
            loss = _depth_observer_item_loss(observer(clean), target, valid)
            loss.backward()
            optimizer.step()
            item_losses.append(float(loss.detach().cpu()))
        training_loss = math.fsum(item_losses) / len(item_losses)

    constant = float(np.median(np.concatenate(training_targets)))
    observer.eval()
    validation_losses: list[float] = []
    comparator_losses: list[float] = []
    with torch.no_grad():
        for item in validation_items:
            clean, target, valid = _read_depth_observer_item(
                item,
                artifact_root=artifact_root,
                device=device,
            )
            validation_losses.append(
                float(_depth_observer_item_loss(observer(clean), target, valid).cpu())
            )
            comparator_losses.append(
                float(
                    _depth_observer_item_loss(
                        torch.full_like(target, constant), target, valid
                    ).cpu()
                )
            )
    validation_loss = math.fsum(validation_losses) / len(validation_losses)
    comparator_loss = math.fsum(comparator_losses) / len(comparator_losses)
    if not math.isfinite(validation_loss) or validation_loss >= comparator_loss:
        raise RuntimeError(
            "depth observer has no validation signal: "
            f"observer={validation_loss}, constant={comparator_loss}"
        )
    return DepthObserverFitResult(
        training_loss,
        validation_loss,
        comparator_loss,
    )


def _read_depth_observer_item(
    item: R4cDepthItem,
    *,
    artifact_root: Path,
    device: object,
) -> tuple[Tensor, Tensor, Tensor]:
    base = prepared_artifacts.read_base_latents(
        artifact_root / item.prepared.base_latent,
        item.prepared.sample_id,
    )
    depth = read_r4c_depth_grid(item, artifact_root=artifact_root)
    return (
        base.clean_latent.to(device=device, dtype=torch.float32),
        depth.values.to(device=device),
        depth.valid.to(device=device),
    )


def _depth_observer_item_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
) -> Tensor:
    return F.smooth_l1_loss(
        prediction[valid],
        target[valid],
        beta=0.1,
        reduction="mean",
    )


__all__ = [
    "DEPTH_GRID_SHAPE",
    "DEPTH_OBSERVER_NAME",
    "DEPTH_OBSERVER_VERSION",
    "DEPTH_REDUCER_NAME",
    "DEPTH_REDUCER_VERSION",
    "DepthGrid",
    "DepthObserverFitResult",
    "LatentDepthObserver",
    "R4cDepthItem",
    "fit_depth_observer",
    "read_r4c_depth_grid",
    "reduce_lidar_depth",
    "resolve_r4c_depth_items",
    "split_depth_grid_cp",
]

"""Lightweight inputs used to open one resident Gen3C model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from novel_view.models.gen3c.lora_weights import Gen3cLoraWeights


@dataclass(frozen=True, slots=True)
class Gen3cModelSpec:
    """Official asset tree and the selected network or LoRA checkpoint."""

    shared_checkpoint_root: Path
    network_checkpoint: Path
    lora: Gen3cLoraWeights | None = None


@dataclass(frozen=True, slots=True)
class Gen3cSampling:
    """Sampling values fixed for the lifetime of one resident pipeline."""

    prompt: str
    negative_prompt: str
    guidance: float
    steps: int
    seed: int


@dataclass(frozen=True, slots=True)
class Gen3cTopology:
    """Requested single-node context-parallel size."""

    context_parallel_size: int

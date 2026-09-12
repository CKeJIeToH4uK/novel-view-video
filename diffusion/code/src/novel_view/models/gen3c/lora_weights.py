"""Load model-ready Gen3C LoRA weights from a training checkpoint."""

from __future__ import annotations

import gc
import importlib
import math
import sys
from collections.abc import Mapping
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GEN3C_MODEL_NAME = "GEN3C_Cosmos_7B"
_ALLOWED_ATTENTION_TARGETS = frozenset({"to_q", "to_k", "to_v", "to_out"})
_LORA_SCALE_ATTRIBUTES = (
    "q_lora_scale",
    "k_lora_scale",
    "v_lora_scale",
    "out_lora_scale",
)


@dataclass(frozen=True, slots=True)
class Gen3cLoraWeights:
    """The already selected training checkpoint and inference strength."""

    checkpoint: Path
    strength: float


@dataclass(frozen=True, slots=True)
class Gen3cLoraSpec:
    """LoRA structure shared by Gen3C training and model loading."""

    rank: int = 8
    scale: float = 1.0
    block_count: int = 28
    self_attention_targets: tuple[str, ...] = ("to_q", "to_v")
    cross_attention_targets: tuple[str, ...] = ("to_q", "to_v")

    def __post_init__(self) -> None:
        if not 1 <= self.rank <= 512:
            raise ValueError("LoRA rank must be within [1, 512]")
        if not math.isfinite(self.scale) or not 1e-5 <= self.scale <= 64:
            raise ValueError("LoRA scale must be finite and within [1e-5, 64]")
        if not 1 <= self.block_count <= 28:
            raise ValueError("LoRA block_count must be within [1, 28]")
        targets = self.self_attention_targets + self.cross_attention_targets
        if not targets:
            raise ValueError("LoRA must target at least one attention projection")
        unknown = sorted(set(targets) - _ALLOWED_ATTENTION_TARGETS)
        if unknown:
            raise ValueError(f"unsupported LoRA attention targets: {unknown}")
        if len(set(self.self_attention_targets)) != len(
            self.self_attention_targets
        ) or len(set(self.cross_attention_targets)) != len(
            self.cross_attention_targets
        ):
            raise ValueError("LoRA attention targets must be unique")

    def upstream_control(self) -> dict[str, object]:
        """Return the pinned LayerControlConfigParser input."""
        blocks = r"\b(" + "|".join(map(str, range(self.block_count))) + r")\b"
        edits = []
        for block, targets in (
            ("FA", self.self_attention_targets),
            ("CA", self.cross_attention_targets),
        ):
            if targets:
                edits.append(f"{block}[{', '.join(targets)}]")
        return {
            "enabled": True,
            "customization_type": "LoRA",
            "rank": self.rank,
            "scale": self.scale,
            "edits": [
                {
                    "blocks": blocks,
                    "customization_type": "LoRA",
                    "rank": self.rank,
                    "scale": self.scale,
                    "block_edit": edits,
                }
            ],
        }


def load_gen3c_lora_model(
    *,
    upstream_root: Path | None,
    base_checkpoint: Path,
    weights: Gen3cLoraWeights,
    context_parallel_group: Any | None,
) -> Any:
    """Restricted-load one embedded adapter and return a frozen model."""
    torch = importlib.import_module("torch")
    checkpoint = torch.load(
        weights.checkpoint,
        weights_only=True,
        map_location="cpu",
    )
    return build_gen3c_lora_inference(
        upstream_root=upstream_root,
        base_checkpoint=base_checkpoint,
        adapter=checkpoint["adapter"],
        strength=weights.strength,
        context_parallel_group=context_parallel_group,
    )


def build_gen3c_lora_inference(
    *,
    upstream_root: Path | None,
    base_checkpoint: Path,
    adapter: Mapping[str, Any],
    strength: float,
    context_parallel_group: Any | None,
) -> Any:
    """Build the exact adapter layout and apply model-ready tensors."""
    if adapter["model_name"] != GEN3C_MODEL_NAME:
        raise RuntimeError("LoRA adapter targets a different base model")
    spec = decode_gen3c_lora_spec(adapter["spec"])
    model = build_gen3c_lora(upstream_root, base_checkpoint, spec)
    replace_gen3c_lora_adapter(model.model, adapter["state_dict"])
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    if context_parallel_group is not None:
        model.net.enable_context_parallel(context_parallel_group)
    model.model.eval()
    set_gen3c_lora_strength(model.model, strength)
    return model


def build_gen3c_lora(
    upstream_root: Path | None,
    checkpoint: Path,
    spec: Gen3cLoraSpec,
) -> Any:
    """Build the official Gen3C model, attach LoRA, and load frozen base."""
    model_module = _import_from(
        upstream_root,
        "cosmos_predict1.diffusion.model.model_gen3c",
    )
    inference_utils = _import_from(
        upstream_root,
        "cosmos_predict1.diffusion.inference.inference_utils",
    )
    peft_module = _import_from(
        upstream_root,
        "cosmos_predict1.diffusion.training.utils.peft.peft",
    )
    model = model_module.DiffusionGen3CModel(
        resolve_gen3c_lora_config(upstream_root, spec)
    )
    with inference_utils.skip_init_linear():
        model.set_up_model()
    peft_module.setup_lora_requires_grad(model.model)
    _load_base_checkpoint(model.model, checkpoint)
    return model


def resolve_gen3c_lora_config(
    upstream_root: Path | None,
    spec: Gen3cLoraSpec,
) -> Any:
    """Resolve the official inference config and add official PEFT control."""
    config_module = _import_from(
        upstream_root,
        "cosmos_predict1.diffusion.config.config",
    )
    helper_module = _import_from(
        upstream_root,
        "cosmos_predict1.utils.config_helper",
    )
    omega_module = importlib.import_module("omegaconf")
    config = helper_module.override(
        config_module.make_config(),
        ["--", f"experiment={GEN3C_MODEL_NAME}"],
    )
    config.model.peft_control = omega_module.OmegaConf.create(spec.upstream_control())
    config.validate()
    config.freeze()
    return config.model


def set_gen3c_lora_strength(model: Any, strength: float) -> int:
    """Scale every assembled official attention adapter."""
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("LoRA strength must be finite and non-negative")
    scaled = 0
    for module in model.modules():
        if not getattr(module, "peft_lora_enabled", False):
            continue
        for name in _LORA_SCALE_ATTRIBUTES:
            if hasattr(module, name):
                setattr(module, name, getattr(module, name) * strength)
                scaled += 1
    if scaled == 0:
        raise RuntimeError("assembled Gen3C model has no active LoRA scales")
    return scaled


def replace_gen3c_lora_adapter(
    model: Any,
    adapter: Mapping[str, Any],
) -> None:
    """Replace every stable LoRA tensor in an already assembled model."""
    expected = gen3c_lora_state_by_name(model, keep_vars=True)
    if set(adapter) != set(expected):
        raise RuntimeError("LoRA adapter keys differ from the built model")

    torch = importlib.import_module("torch")
    with torch.no_grad():
        for name, destination in expected.items():
            value = adapter[name]
            if not torch.is_tensor(value):
                raise RuntimeError(f"LoRA adapter value is not a tensor: {name}")
            if value.shape != destination.shape or value.dtype != destination.dtype:
                raise RuntimeError(f"LoRA adapter tensor differs: {name}")
            destination.copy_(value.to(device=destination.device))


@contextmanager
def gen3c_lora_enabled(model: Any, enabled: bool) -> Iterator[None]:
    """Temporarily toggle official LoRA branches and restore their state."""
    modules = [
        module for module in model.modules() if hasattr(module, "peft_lora_enabled")
    ]
    previous = [module.peft_lora_enabled for module in modules]
    for module in modules:
        module.peft_lora_enabled = enabled
    try:
        yield
    finally:
        for module, value in zip(modules, previous, strict=True):
            module.peft_lora_enabled = value


def decode_gen3c_lora_spec(value: Mapping[str, Any]) -> Gen3cLoraSpec:
    """Decode the concrete spec embedded by the training checkpoint."""
    return Gen3cLoraSpec(
        rank=value["rank"],
        scale=value["scale"],
        block_count=value["block_count"],
        self_attention_targets=tuple(value["self_attention_targets"]),
        cross_attention_targets=tuple(value["cross_attention_targets"]),
    )


def gen3c_lora_state_by_name(
    model: Any,
    *,
    keep_vars: bool = False,
) -> dict[str, Any]:
    """Return every raw LoRA tensor under its stable model-state key."""
    state = model.state_dict(keep_vars=keep_vars)
    return {name: state[name] for name in sorted(state) if "_lora." in name}


def _load_base_checkpoint(model: Any, path: Path) -> None:
    torch = importlib.import_module("torch")
    checkpoint = torch.load(
        path,
        weights_only=True,
        mmap=True,
        map_location="cpu",
    )
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("model"), Mapping):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    gc.collect()


def _import_from(upstream_root: Path | None, module_name: str) -> Any:
    if upstream_root is not None and str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    return importlib.import_module(module_name)

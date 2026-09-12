"""Build the shared concrete Gen3C model and sampling request."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from novel_view.generation.gen3c.recipe import (
    BASE_CHECKPOINTS,
    BASE_MODEL_ID,
    Gen3cGenerationRecipe,
)
from novel_view.generation.gen3c.session import (
    Gen3cGenerationModel,
    Gen3cGenerationParameters,
    Gen3cLoraProvenance,
)
from novel_view.generation.gen3c.windows import WINDOW_SEED_AUTOREGRESSIVE
from novel_view.models.gen3c.lora_weights import Gen3cLoraWeights
from novel_view.models.gen3c.request import Gen3cModelSpec


_SHARED_CHECKPOINT_DIRECTORY = Path("gen3c/official/checkpoints")
_BASE_NETWORK_CHECKPOINT = Path("Gen3C-Cosmos-7B/model.pt")


def build_model_request(
    models_root: Path,
    recipe: Gen3cGenerationRecipe,
    *,
    window_seed_policy: str = WINDOW_SEED_AUTOREGRESSIVE,
) -> tuple[Gen3cGenerationModel, Gen3cGenerationParameters]:
    """Resolve one explicit base/full/LoRA request without opening assets."""
    shared = models_root / _SHARED_CHECKPOINT_DIRECTORY
    if recipe.checkpoint in BASE_CHECKPOINTS:
        selected_id = BASE_MODEL_ID
        network = shared / _BASE_NETWORK_CHECKPOINT
    else:
        selected_id = cast(str, recipe.model_id)
        network = Path(recipe.checkpoint)
        if not network.is_absolute():
            network = models_root / network
        if network == shared / _BASE_NETWORK_CHECKPOINT:
            raise ValueError("select the official checkpoint as gen3c/base")

    lora = recipe.lora
    provenance = (
        None
        if lora is None
        else Gen3cLoraProvenance(
            Path(lora.working_manifest),
            Path(lora.evidence_path),
            lora.epochs,
        )
    )
    weights = (
        None
        if lora is None
        else Gen3cLoraWeights(network, lora.strength)
    )
    return (
        Gen3cGenerationModel(
            model_id=selected_id,
            artifact=Gen3cModelSpec(shared, network, weights),
            lora_provenance=provenance,
        ),
        Gen3cGenerationParameters(
            seed=recipe.seed,
            num_steps=recipe.num_steps,
            window_seed_policy=window_seed_policy,
        ),
    )


__all__ = ["build_model_request"]

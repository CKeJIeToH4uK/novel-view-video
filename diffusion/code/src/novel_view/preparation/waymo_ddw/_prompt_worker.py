"""Private one-shot Gen3C empty-prompt worker."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from novel_view.preparation.waymo_ddw.artifacts import (
    PROMPT_EMBEDDING_SHAPE,
    write_empty_prompt,
)


_PROMPT_TOKEN_COUNT = PROMPT_EMBEDDING_SHAPE[1]
_PROMPT_MASK_SHAPE = PROMPT_EMBEDDING_SHAPE[:2]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_encoder(text_encoder_path: Path) -> tuple[Any, Any]:
    """Load the installed image-pinned encoder on the one assigned GPU."""
    torch = importlib.import_module("torch")
    torch.cuda.set_device(0)
    module = importlib.import_module("cosmos_predict1.auxiliary.t5_text_encoder")
    encoder = module.CosmosT5TextEncoder(
        model_name=str(text_encoder_path),
        cache_dir=str(text_encoder_path),
        device="cuda",
    )
    return torch, encoder


def _encode_empty_prompt(torch: Any, encoder: Any) -> Any:
    """Encode one empty string and preserve the proven padding semantics."""
    embedding, mask = encoder.encode_prompts([""], max_length=_PROMPT_TOKEN_COUNT)
    if (
        not torch.is_tensor(embedding)
        or tuple(embedding.shape) != PROMPT_EMBEDDING_SHAPE
        or not torch.is_tensor(mask)
        or tuple(mask.shape) != _PROMPT_MASK_SHAPE
    ):
        raise ValueError("empty prompt has the wrong tensor shapes")
    expected_mask = torch.zeros_like(mask)
    expected_mask[:, 0] = 1
    if not torch.equal(mask, expected_mask):
        raise ValueError("empty prompt has the wrong encoder mask")
    if not bool(torch.all(embedding[:, 1:] == 0)):
        raise ValueError("empty prompt has a nonzero padded tail")
    return embedding.to(device="cpu", dtype=torch.bfloat16).contiguous()


def _run(arguments: argparse.Namespace) -> None:
    torch, encoder = _load_encoder(arguments.text_encoder)
    write_empty_prompt(
        arguments.output,
        _encode_empty_prompt(torch, encoder),
    )


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()

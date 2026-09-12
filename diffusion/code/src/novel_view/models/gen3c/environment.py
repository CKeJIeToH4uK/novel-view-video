"""Concrete environment used by resident Gen3C processes."""

from pathlib import Path

from novel_view.runtime.context import RuntimeContext


def gen3c_environment(
    runtime: RuntimeContext,
    temporary_root: Path,
) -> dict[str, str]:
    """Map fixed container cache roots without probing the machine."""
    cache = runtime.roots.cache
    return {
        "COSMOS_CACHE_DIR": str(cache / "cosmos"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "WARP_CACHE_PATH": str(cache / "warp"),
        "HF_HOME": str(cache / "huggingface"),
        "TORCH_HOME": str(cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch/extensions"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torch/inductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "TMPDIR": str(temporary_root),
        "TMP": str(temporary_root),
        "TEMP": str(temporary_root),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }


__all__ = ["gen3c_environment"]

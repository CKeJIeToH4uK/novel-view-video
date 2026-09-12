"""Checks for the concrete shared Gen3C environment mapping."""

from pathlib import Path
from types import SimpleNamespace

from novel_view.generation.gen3c.resources import build_generation_resources
from novel_view.models.gen3c.environment import gen3c_environment


def test_generation_and_training_share_fixed_cache_mapping() -> None:
    runtime = SimpleNamespace(roots=SimpleNamespace(cache=Path("/cache")))

    environment = gen3c_environment(runtime, Path("/scratch/attempt"))
    generation = build_generation_resources(runtime, 2)

    assert environment["COSMOS_CACHE_DIR"] == "/cache/cosmos"
    assert environment["TMPDIR"] == "/scratch/attempt"
    assert generation.environment_overrides == gen3c_environment(
        runtime,
        Path("/cache"),
    )
    assert generation.context_parallel_size == 2

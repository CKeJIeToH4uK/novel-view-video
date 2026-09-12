"""Build the concrete resident Gen3C process resources for a run."""

from pathlib import Path

from novel_view.generation.gen3c.session import Gen3cGenerationResources
from novel_view.models.gen3c.environment import gen3c_environment
from novel_view.runtime.context import RuntimeContext


def build_generation_resources(
    runtime: RuntimeContext,
    context_parallel_size: int,
    *,
    context_depth_model: Path | None = None,
) -> Gen3cGenerationResources:
    """Use the fixed container cache roots and launcher-owned CP topology."""
    root = runtime.roots.cache
    return Gen3cGenerationResources(
        scratch_root=root,
        environment_overrides=gen3c_environment(runtime, root),
        upstream_root=None,
        context_parallel_size=context_parallel_size,
        unprojection_chunk_size=13,
        context_depth_model=context_depth_model,
    )


__all__ = ["build_generation_resources"]

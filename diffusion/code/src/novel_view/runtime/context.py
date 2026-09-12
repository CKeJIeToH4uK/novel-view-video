"""Фиксированные корни и готовая identity одной container attempt."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ContainerRoots:
    data: Path
    models: Path
    prepared: Path
    runs: Path
    cache: Path
    jobs: Path
    recipes: Path
    selections: Path


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    variant: str
    image_id: str
    source_revision: str
    source_dirty: bool
    lock_revision: str


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    roots: ContainerRoots
    run_id: str
    attempt_id: str
    attempt_kind: str
    attempt_root: Path
    container_name: str
    image: ImageMetadata
    selected_checkpoint: str | None


CONTAINER_ROOTS = ContainerRoots(
    data=Path("/data"),
    models=Path("/models"),
    prepared=Path("/prepared"),
    runs=Path("/runs"),
    cache=Path("/cache"),
    jobs=Path("/project-config/jobs"),
    recipes=Path("/project-config/recipes"),
    selections=Path("/project-config/selections"),
)

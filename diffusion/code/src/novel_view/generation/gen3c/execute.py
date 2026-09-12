"""One-shot wrapper around the resident Gen3C generation session."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from novel_view.generation.gen3c.request import Gen3cConditioningInput
from novel_view.generation.gen3c.session import (
    Gen3cGenerationModel,
    Gen3cGenerationParameters,
    Gen3cGenerationResources,
    Gen3cGenerationResult,
    Gen3cGenerationSession,
)


def generate_gen3c_sequence(
    conditioning_input: Gen3cConditioningInput,
    model: Gen3cGenerationModel,
    parameters: Gen3cGenerationParameters,
    resources: Gen3cGenerationResources,
    output_rgb_path: Path,
) -> Gen3cGenerationResult:
    """Generate one sequence and include model construction in elapsed time."""
    started = time.monotonic()
    with Gen3cGenerationSession(model, parameters, resources) as session:
        result = session.generate(conditioning_input, output_rgb_path)
    return replace(
        result,
        elapsed_seconds=float(time.monotonic() - started),
    )

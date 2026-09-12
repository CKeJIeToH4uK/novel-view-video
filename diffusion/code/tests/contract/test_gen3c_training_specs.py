"""Strict, cold planning contracts for the two R4c training versions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from novel_view.config.job import load_resolved_job
from novel_view.training.gen3c.lora.depth_spec import R4cLoraDepthSpec
from novel_view.training.gen3c.lora.spec import R4cLoraEdmSpec
from novel_view.workflows.gen3c_training import resolve_training_job
from novel_view.workflows.runner import select_job


ROOT = Path(__file__).resolve().parents[4]


def _load(name: str):
    return load_resolved_job(
        ROOT / "jobs" / "waymo" / name / "run.yaml",
        ROOT,
    )


def test_tracked_v1_and_v2_jobs_resolve_to_cp4_core_workflows() -> None:
    v1_job = _load("r4c-lora-edm")
    v2_job = _load("r4c-lora-lidar-depth")
    v1 = resolve_training_job(v1_job)
    v2 = resolve_training_job(v2_job)
    assert isinstance(v1.recipe, R4cLoraEdmSpec)
    assert isinstance(v2.recipe, R4cLoraDepthSpec)
    assert v1_job.execution.preset == v2_job.execution.preset == "training_cp4"
    assert select_job(v1_job).image_variant == select_job(v2_job).image_variant == "core"
    assert v1.input.split == Path("selections/waymo/example-r4c-split.yaml")
    assert v1.input.legacy_index_map is None
    assert v2.recipe.depth_weight == 0.1
    assert v2.recipe.observer.betas == (0.9, 0.999)
    assert v2.recipe.observer.items_per_step == 1

def test_v2_rejects_v1_only_legacy_map() -> None:
    job = _load("r4c-lora-lidar-depth")
    job.input["legacy_index_map"] = "selections/local/legacy-map.json"
    with pytest.raises(ValueError, match="unknown fields"):
        resolve_training_job(job)


def test_v1_planning_does_not_import_depth_or_model_libraries() -> None:
    code = f"""
import sys
from pathlib import Path
from novel_view.config.job import load_resolved_job
from novel_view.workflows.gen3c_training import resolve_training_job
root = Path({str(ROOT)!r})
job = load_resolved_job(root / 'jobs/waymo/r4c-lora-edm/run.yaml', root)
resolve_training_job(job)
assert 'novel_view.training.gen3c.lora.depth_spec' not in sys.modules
assert 'torch' not in sys.modules
assert not any(name.startswith('cosmos_predict1') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)

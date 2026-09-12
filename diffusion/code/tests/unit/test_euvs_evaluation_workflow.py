from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from novel_view.config.job import ResolvedJob
from novel_view.evaluation.euvs.execute import (
    EuvsMetricResources,
    evaluate_pair,
)
from novel_view.evaluation.euvs.record import (
    encode_euvs_metric_result,
    load_euvs_metric_result,
)
from novel_view.evaluation.euvs.spec import parse_euvs_evaluation_job
from novel_view.workflows.euvs_evaluation import plan_euvs_evaluation_records
from novel_view.workflows.runner import select_job
from tests.support.stage5 import make_job


def _job(generation: dict[str, object]) -> ResolvedJob:
    return make_job(
        "euvs_evaluation",
        1,
        {
            "reader": "euvs_nuplan",
            "dataset": "euvs",
            "selection": "selections/euvs/one.yaml",
            "generation": generation,
        },
        {"evaluation": {"mask_recipe": "grounded-sam2-euvs-v1"}},
        preset="inference_cp1",
        recipe="recipes/evaluation/euvs.yaml",
    )


def test_exact_generation_references_and_runner_route() -> None:
    selection = SimpleNamespace(pairs=(SimpleNamespace(name="pair/a"),))
    attempt = _job({"source": "attempt", "path": "generation/attempt"})
    records = _job(
        {"source": "records", "records": ["legacy/pair-a.run.json"]}
    )

    assert select_job(attempt).image_variant == "core"
    for job, locator in (
        (attempt, "generation/attempt/pairs/pair%2Fa/run.json"),
        (records, "legacy/pair-a.run.json"),
    ):
        planned = plan_euvs_evaluation_records(
            parse_euvs_evaluation_job(job),
            selection,
            Path("/runs"),
            Path("/runs/evaluation"),
        )[0]
        assert planned.record_path == Path("/runs") / locator
        assert planned.result_path == Path("/runs/evaluation/metrics/pair%2Fa.json")


def test_pair_workers_get_distinct_attempt_logs(tmp_path):
    from novel_view.workflows.euvs_evaluation import _resources

    runtime = SimpleNamespace(roots=SimpleNamespace(cache=tmp_path / "cache", models=Path("/models")))
    logs = []
    for pair in ("pair-a", "pair-b"):
        pair_logs = tmp_path / "attempt" / "logs" / pair
        resources = _resources(runtime, pair_logs)
        assert tuple(resource.log_path.name for resource in resources) == (
            "source-masks.log", "target-masks.log", "metrics.log",
        )
        for resource in resources:
            assert resource.scratch_root == runtime.roots.cache
            logs.append(resource.log_path)
    assert len(set(logs)) == 6


def test_evaluator_preserves_historical_metric_json(tmp_path, monkeypatch):
    rgb = np.zeros((11, 11, 3), np.uint8)
    full = np.ones((11, 11), bool)
    raster = SimpleNamespace(rgb=rgb, plan=SimpleNamespace(valid_mask=full))
    sample = SimpleNamespace(prediction_rgb=rgb, target=SimpleNamespace(raster=raster))
    views = SimpleNamespace(target_static=full, source_aware=np.zeros_like(full))
    support = SimpleNamespace(mask_recipe_id="grounded-sam2-euvs-v1",
        support_protocol="target-source-dynamic/v1", projection_protocol="four-neighbour-zbuffer/v1",
        frames=(SimpleNamespace(sample=sample, views=views),))

    def worker(arguments, *_):
        np.save(arguments[arguments.index("--output-perceptual") + 1],
                [[[.25, .75, 1.], [np.nan, np.nan, 0.]]])

    monkeypatch.setattr("novel_view.evaluation.euvs.execute.run_process", worker)
    resources = EuvsMetricResources(tmp_path, tmp_path / "metrics.log", tmp_path, tmp_path, tmp_path)
    result = evaluate_pair(support, resources, "generation/attempt/pairs/pair-a/run.json")
    fixture = Path(__file__).parents[1] / "contract/fixtures/euvs_metric_result_v1.json"
    assert encode_euvs_metric_result(result) == fixture.read_bytes()
    assert load_euvs_metric_result(fixture) == result

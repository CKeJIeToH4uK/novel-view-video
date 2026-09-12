"""Public training and checkpoint-selection routes use their explicit version."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from novel_view.config.job import (
    ExecutionReference,
    ResolvedJob,
    WorkflowReference,
    load_resolved_job,
)
from novel_view.training.gen3c.lora.record import (
    read_checkpoint_record_v1,
    write_checkpoint_record_v1,
)
from novel_view.training.gen3c.lora.depth_record import (
    read_checkpoint_record_v2,
    write_checkpoint_record_v2,
)
from novel_view.training.gen3c.lora.depth_spec import DepthObjectiveV2
from novel_view.training.gen3c.checkpoint_selection import (
    read_checkpoint_selection_v1,
    read_checkpoint_selection_v2,
    select_checkpoint_v2,
)
from novel_view.workflows import gen3c_training
from novel_view.workflows.runner import execute_job, select_job
from tests.support.gen3c_training_records import checkpoint_record
from tests.support.stage5 import make_runtime


@pytest.mark.parametrize("version", [1, 2])
def test_selection_keeps_early_step_and_input_order_on_full_tie(tmp_path, version):
    runtime = make_runtime(tmp_path, attempt="selection")
    references = [
        Path(f"train/run/{name}/checkpoints/step-{step}.json")
        for name, step in [("late", 154), ("early-first", 77), ("early-second", 77)]
    ]
    writer = (
        write_checkpoint_record_v1 if version == 1 else write_checkpoint_record_v2
    )
    reader = read_checkpoint_record_v1 if version == 1 else read_checkpoint_record_v2
    first_document = None
    for reference, loss in zip(references, [-0.1, -0.2, -0.2]):
        step = 154 if loss == -0.1 else 77
        record = checkpoint_record(step, loss, str(reference.parent.parent), version)
        path = runtime.roots.runs / reference
        path.parent.mkdir(parents=True)
        writer(path, record)
        assert reader(path) == record
        if first_document is None:
            first_document = json.loads(path.read_text())
    assert (
        first_document["format"]
        == f"novel-view/gen3c-lora-checkpoint-record/v{version}"
    )
    invalid = runtime.roots.runs / references[0]
    invalid.write_text(json.dumps(first_document | {"unexpected": True}))
    with pytest.raises(ValueError):
        reader(invalid)
    invalid.write_text(json.dumps(first_document))
    job = ResolvedJob(
        1,
        "select",
        WorkflowReference("gen3c_checkpoint_selection", version),
        {"checkpoint_records": [str(path) for path in references]},
        None,
        {},
        ExecutionReference("cpu_test"),
    )
    assert (select_job(job).image_variant, execute_job(job, runtime)) == ("core", 0)
    output = runtime.attempt_root / "selection.json"
    result = json.loads(output.read_text())
    parsed = (read_checkpoint_selection_v1 if version == 1 else read_checkpoint_selection_v2)(
        output
    )
    assert result["format"] == f"novel-view/gen3c-checkpoint-selection/v{version}"
    assert result["input_checkpoint_records"] == [str(path) for path in references]
    assert (parsed.selected_checkpoint_record, parsed.completed_step) == (str(references[1]), 77)
    assert result["checkpoint"] == str(references[1].parent / "step-000000077.pt")
    assert result["metric"] == {
        "name": "validation_loss" if version == 1 else "validation_total_loss",
        "value": -0.2,
    }
    assert (
        json.loads((runtime.attempt_root / "attempt.json").read_text())["recorded_state"]
        == "succeeded"
    )
    if version == 2:
        assert result["objective"]["depth_weight"] == 0.1
        assert result["observer_lineage"] == "train/run/fresh"


@pytest.mark.parametrize("version", [1, 2])
def test_workflow_selects_cp4_worker_and_explicit_resume(tmp_path, monkeypatch, version):
    root = Path(__file__).resolve().parents[4]
    name = "r4c-lora-edm" if version == 1 else "r4c-lora-lidar-depth"
    job = load_resolved_job(root / f"jobs/waymo/{name}/run.yaml", root)
    calls = []
    monkeypatch.setattr(
        gen3c_training,
        "start_torchrun",
        lambda *args, **kwargs: calls.append((args, kwargs)) or NS(wait=lambda: None),
    )
    fresh = make_runtime(tmp_path / "fresh", attempt="fresh")
    resumed = replace(
        make_runtime(tmp_path / "resume", attempt="resumed"),
        attempt_kind="resume",
        selected_checkpoint="/runs/train/source/checkpoints/step-000000077.pt",
    )
    assert execute_job(job, fresh) == execute_job(job, resumed) == 0
    assert len(calls) == 2
    for args, kwargs in calls:
        assert Path(args[1]).name == f"_worker_v{version}.py" and kwargs["process_count"] == 4
        assert kwargs["log_path"].name == f"training-v{version}.log"
        assert "--prepared-record" in args[2] and "--source-attempt" in args[2]
        assert ("--depth-weight" in args[2]) == (version == 2)
        assert ("--observer-fit-epochs" in args[2]) == (version == 2)
    assert "--resume-checkpoint" not in calls[0][0][2]
    args = calls[1][0][2]
    assert args[args.index("--resume-checkpoint") + 1] == resumed.selected_checkpoint


def test_v2_selection_does_not_mix_depth_objective_or_observer_lineage():
    first = checkpoint_record(77, -0.2, "train/run/a", 2)
    for other in [
        replace(first, objective=DepthObjectiveV2(0.2)),
        replace(first, observer_lineage="train/run/other"),
    ]:
        with pytest.raises(ValueError):
            select_checkpoint_v2((first, other))

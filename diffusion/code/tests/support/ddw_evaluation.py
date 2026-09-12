"""Small scientific values shared by DDW evaluation tests."""

from novel_view.evaluation.ddw.spec import (
    DdwEvaluationInput,
    DdwEvaluationRecipe,
    DdwEvaluationSpec,
    DdwSamplingSpec,
)
from novel_view.preparation.waymo_ddw.record import PreparedItem
from tests.support.gen3c_training import training_item


def evaluation_spec() -> DdwEvaluationSpec:
    return DdwEvaluationSpec(
        DdwEvaluationInput("prepared.json", "split.yaml", "v1.json", "v2.json"),
        DdwEvaluationRecipe(
            "gen3c-cosmos-7b-121x704x1280-fps10-v1",
            77,
            DdwSamplingSpec(0, 35, 1.0, 0.001),
            "gen3c/official/model.pt",
            "gen3c/tokenizer",
            "moge",
            "moge/model.pt",
        ),
    )


def matched_items():
    return (
        (training_item("train-a", "segment-a"), training_item("train-b", "segment-b")),
        (training_item("validation-a", "segment-v"),),
    )


def prepared_item(sample_id: str, partition: str, segment: str) -> PreparedItem:
    return PreparedItem(
        sample_id,
        partition,
        segment,
        0,
        tuple(range(121)),
        1.0,
        1,
        f"items/{sample_id}/base.pt",
        f"items/{sample_id}/pose.pt",
        f"items/{sample_id}/lidar-depth.pt",
    )

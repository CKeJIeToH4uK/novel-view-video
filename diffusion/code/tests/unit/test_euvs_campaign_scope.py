"""Сохранённые EUVS выборки остаются раздельными и упорядоченными."""

from itertools import combinations
from pathlib import Path

import pytest

from novel_view.inputs.euvs.spec import load_euvs_selection
from novel_view.workflows.euvs_generation import plan_euvs_generation_outputs


SELECTIONS = Path(__file__).parents[4] / "selections/euvs"


@pytest.mark.parametrize(
    "name",
    (
        "euvs-setting1-primary35",
        "euvs-setting1-extended16",
        "euvs-setting1-raw29",
        "euvs-location-2-tr2-to-tr6",
    ),
)
def test_generation_outputs_keep_real_selection_order(name: str) -> None:
    selection = load_euvs_selection(SELECTIONS / f"{name}.yaml")
    outputs = plan_euvs_generation_outputs(
        selection, Path("/missing/source-attempt"), Path("/missing/output"),
    )
    assert tuple(output.selection for output in outputs) == selection.pairs


def test_primary_extended_and_raw_keep_separate_pair_scopes() -> None:
    scopes = [
        {
            (pair.location, pair.source.traversal, pair.target.traversal)
            for pair in load_euvs_selection(
                SELECTIONS / f"euvs-setting1-{scope}.yaml"
            ).pairs
        }
        for scope in ("primary35", "extended16", "raw29")
    ]
    assert all(left.isdisjoint(right) for left, right in combinations(scopes, 2))

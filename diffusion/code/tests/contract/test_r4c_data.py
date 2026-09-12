"""Declared split, lazy artifact joins and independent PCG64 cursor examples."""

import json
from pathlib import Path

import numpy as np
import pytest

from novel_view.training.gen3c import data
from tests.support.gen3c_training import (
    legacy_index_document,
    training_item,
    prepared_record,
    prepared_record_item,
)


def test_split_and_legacy_index_external_documents(tmp_path):
    root = Path(__file__).resolve().parents[4]
    split = data.load_r4c_training_split(root / "selections/waymo/example-r4c-split.yaml")
    assert (split.training_sample_ids, split.validation_sample_ids) == (
        ("example-front-0002",),
        ("example-front-0001",),
    )
    source = tmp_path / "split.yaml"
    source.write_text(
        "schema_version: 1\ntraining_sample_ids: [train]\n"
        "validation_sample_ids: [validation]\nhidden: true\n"
    )
    with pytest.raises(ValueError):
        data.load_r4c_training_split(source)
    source.write_text(json.dumps(legacy_index_document(stride=3, digits=4)))
    mapping = data.load_r4c_legacy_index_map(source)
    assert len(mapping.items) == 85
    assert (mapping.items[17].old_global_fit_index, mapping.items[17].sample_id) == (
        51,
        "sample-0017",
    )


def test_prepared_join_keeps_split_order_lazy_paths_and_separate_segments():
    prepared = prepared_record(
        tuple(
            prepared_record_item(name, segment)
            for name, segment in [
                ("validation", "segment-v"),
                ("train-b", "segment-b"),
                ("train-a", "segment-a"),
            ]
        )
    )
    training, validation = data.resolve_r4c_items(
        prepared, data.R4cTrainingSplit(("train-a", "train-b"), ("validation",))
    )
    assert tuple(item.sample_id for item in training) == ("train-a", "train-b")
    assert validation[0].sample_id == "validation"
    assert training[0].base_latent == Path("items/train-a/base.pt")
    assert training[0].prompt_embedding == Path("empty-prompt.pt")
    overlapping = prepared_record(
        tuple(prepared_record_item(name, "same") for name in ["train", "validation"])
    )
    for name in ["missing", "validation"]:
        with pytest.raises(ValueError):
            data.resolve_r4c_items(overlapping, data.R4cTrainingSplit(("train",), (name,)))


def test_pcg64_epoch_order_resume_cursor_and_global_rng():
    items = tuple(
        training_item(f"sample-{index}", f"segment-{index}", magnitude_m=3.0, sign=-1)
        for index in range(5)
    )
    np.random.seed(123)
    before = np.random.get_state()
    iterator = data.iter_r4c_items(items, data.r4c_data_cursor(items, training_seed=17))
    outputs = [next(iterator) for _ in range(7)]
    assert [item.sample_id for item, _ in outputs[:5]] == [f"sample-{i}" for i in [4, 0, 2, 1, 3]]
    assert (outputs[4][1].epoch, outputs[4][1].offset) == (1, 0)
    assert next(data.iter_r4c_items(items, outputs[1][1])) == outputs[2]
    assert [item.sample_id for item in data.epoch_item_order(items, training_seed=17, epoch=1)] == (
        [item.sample_id for item, _ in outputs[5:]] + ["sample-1", "sample-3", "sample-0"]
    )
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]

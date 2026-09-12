"""One numeric Cache4D exchange, with and without retained context."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from novel_view.generation.gen3c.protocol import write_generation_request
from novel_view.generation.gen3c.request import Gen3cGenerationRequest
from novel_view.models.gen3c.cache4d.protocol import read_cache4d_conditioning
from novel_view.models.gen3c.cache4d.request import (
    Gen3cConditioning,
    Gen3cContextConditioning,
    Gen3cSourceFrame,
)


@dataclass
class Rows:
    offset: int
    count: int = 3
    reads: list = field(default_factory=list)

    def __len__(self):
        return self.count

    def read(self, index):
        self.reads.append(index)
        return Gen3cSourceFrame(
            np.full((2, 3, 3), self.offset + index, np.uint8),
            np.full((2, 3), self.offset + index + 0.5, np.float32),
            np.full((2, 3), index % 2 == 0, np.bool_),
        )


@pytest.mark.parametrize("with_context", [False, True])
def test_streaming_arrays_indices_and_context(tmp_path, with_context):
    rows, context_rows = Rows(10), Rows(60, 2)
    source_k = np.tile(np.diag([3.0, 4.0, 1.0]), (3, 1, 1))
    source_w2c = np.tile(np.eye(4), (3, 1, 1))
    source_w2c[:, 0, 3] = [2, 3, 4]
    query_w2c = np.tile(np.eye(4), (4, 1, 1))
    query_w2c[:, 1, 3] = [8, 9, 10, 11]
    query_k = np.tile(np.diag([5.0, 6.0, 1.0]), (4, 1, 1))
    context = (
        Gen3cContextConditioning(
            context_rows,
            source_w2c[:2],
            source_k[:2],
            np.array([0, 0, 1, 1]),
        )
        if with_context
        else None
    )
    conditioning = Gen3cConditioning(
        rows,
        source_k,
        source_w2c,
        np.array([0, 1, 2, 2]),
        query_w2c,
        query_k,
        np.array([0, 3]),
        context,
    )
    output_slots = np.array([1, 2]) if with_context else None
    write_generation_request(
        tmp_path,
        Gen3cGenerationRequest(
            conditioning,
            tmp_path / "generated.npy",
            output_slots,
        ),
    )
    arrays = read_cache4d_conditioning(tmp_path)
    expected = {
        "source_intrinsics": source_k,
        "source_w2c": source_w2c,
        "source_index": np.array([0, 1, 2, 2]),
        "query_w2c": query_w2c,
        "query_intrinsics": query_k,
        "selected_slots": np.array([0, 3]),
    }
    for prefix, offset, count in [("source", 10, 3)] + (
        [("context", 60, 2)] if with_context else []
    ):
        expected[prefix + "_rgb"] = np.broadcast_to(
            np.arange(offset, offset + count, dtype=np.uint8)[:, None, None, None],
            (count, 2, 3, 3),
        )
        expected[prefix + "_depth"] = expected[prefix + "_rgb"][..., 0].astype(np.float32) + 0.5
        expected[prefix + "_valid"] = np.broadcast_to(
            (np.arange(count) % 2 == 0)[:, None, None],
            (count, 2, 3),
        )
    if with_context:
        expected.update(
            context_w2c=source_w2c[:2],
            context_intrinsics=source_k[:2],
            context_index=np.array([0, 0, 1, 1]),
        )
        np.testing.assert_array_equal(np.load(tmp_path / "context_depth_output_slots.npy"), [1, 2])
    try:
        assert arrays.keys() == expected.keys()
        for name, value in expected.items():
            np.testing.assert_array_equal(arrays[name], value)
            assert arrays[name].dtype == value.dtype
        assert rows.reads == [0, 1, 2]
        assert context_rows.reads == ([0, 1] if with_context else [])
    finally:
        for array in arrays.values():
            array._mmap.close()

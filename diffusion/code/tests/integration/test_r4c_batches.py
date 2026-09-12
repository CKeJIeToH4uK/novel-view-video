"""One source rank supplies the four distinct training tensors."""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from novel_view.training.gen3c import data
from tests.support.gen3c_training import training_item


def test_source_reads_lazy_artifacts_and_receiver_stops_before_broadcast(monkeypatch):
    opened, broadcasts = [], []
    rank, source_error = 0, None

    def read(path, **values):
        opened.append(path)
        return NS(
            **{name: torch.tensor([value], dtype=torch.bfloat16) for name, value in values.items()}
        )

    monkeypatch.setattr(
        data.prepared_artifacts,
        "read_base_latents",
        lambda path, *_: read(path, clean_latent=1, source_latent=2),
    )
    monkeypatch.setattr(
        data.prepared_artifacts, "read_pose_latent", lambda path, *_: read(path, pose_latent=3)
    )
    monkeypatch.setattr(
        data.prepared_artifacts,
        "read_empty_prompt",
        lambda path: read(path, t5_text_embeddings=4),
    )
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _group: [0, 1])
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, error, **_: output.__setitem__(slice(None), [source_error or error, None]),
    )
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda value, source, **_: broadcasts.append(value.clone()),
    )
    items = (training_item("sample-1", "segment-1", magnitude_m=3.0, sign=-1),)
    options = dict(artifact_root=Path("/prepared/run"), device="cpu", context_parallel_group="cp")
    batch = next(
        data.iter_r4c_training_batches(
            items, data.r4c_data_cursor(items, training_seed=17), **options
        )
    )
    assert opened == [
        Path("/prepared/run") / name
        for name in ["items/sample-1/base.pt", "items/sample-1/pose.pt", "empty-prompt.pt"]
    ]
    assert [tensor.item() for tensor in broadcasts] == [1, 2, 3, 4]
    assert all(
        tensor.dtype == torch.bfloat16 and tensor.device.type == "cpu" for tensor in broadcasts
    )
    assert [
        value.item()
        for value in (
            batch.clean_latent,
            batch.source_latent,
            batch.pose_latent,
            batch.prompt_embedding,
        )
    ] == [1, 2, 3, 4]
    assert batch.sample_id == "sample-1"
    assert batch.next_cursor == data.r4c_data_cursor(items, training_seed=17, completed_steps=1)
    rank, source_error = 1, "OSError: broken base artifact"
    with pytest.raises(RuntimeError):
        next(
            data.iter_r4c_training_batches(
                items, data.r4c_data_cursor(items, training_seed=17), **options
            )
        )
    assert len(broadcasts) == 4 and len(opened) == 3

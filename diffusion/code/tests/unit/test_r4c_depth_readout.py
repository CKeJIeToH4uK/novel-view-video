"""LiDAR time/grid reduction, the real 577-parameter fit and frozen readout."""

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from novel_view.training.gen3c.data import R4cTrainingSplit
from novel_view.training.gen3c.lora import depth
from novel_view.training.gen3c.topology import Gen3cTrainingTopology
from tests.support.gen3c_training import (
    install_observer_data,
    lidar_depth_artifact,
    local_topology,
    observer_fit_spec,
    observer_item,
    prepared_record_item,
    prepared_record,
)


def test_lidar_log_median_slots_holdout_and_declared_item_locator(tmp_path, monkeypatch):
    artifact = lidar_depth_artifact(
        (
            (0, 8.0, 16.0, 25.0, False),
            (1, 8.0, 16.0, 1.0, False),
            (8, 15.9, 23.9, 9.0, False),
            (8, 8.0, 16.0, 81.0, True),
            (9, 24.0, 32.0, 4.0, False),
            (120, 32.0, 40.0, 16.0, False),
        )
    )
    grid = depth.reduce_lidar_depth(artifact)
    assert (depth.DEPTH_REDUCER_NAME, depth.DEPTH_REDUCER_VERSION) == (
        "front-a-lidar-log-camera-z-grid",
        1,
    )
    assert grid.values.shape == depth.DEPTH_GRID_SHAPE and grid.values.dtype == torch.float32
    assert (
        grid.valid.dtype == torch.bool and not grid.valid[:, :, 0].any() and grid.valid.sum() == 3
    )
    assert grid.values[0, 0, 3, 0, 0].item() == 0.0
    observed = [grid.values[0, 0, t, y, x].item() for t, y, x in [(1, 2, 1), (2, 4, 3), (15, 5, 4)]]
    np.testing.assert_allclose(observed, np.log([3.0, 4.0, 16.0]), rtol=1e-7)
    prepared = prepared_record(
        (
            prepared_record_item("validation", "segment-v"),
            prepared_record_item("sample-a", "segment-a"),
        )
    )
    training, validation = depth.resolve_r4c_depth_items(
        prepared, R4cTrainingSplit(("sample-a",), ("validation",))
    )
    opened = []
    monkeypatch.setattr(
        depth,
        "read_lidar_depth",
        lambda path, sample_id: opened.append((path, sample_id)) or artifact,
    )
    loaded = depth.read_r4c_depth_grid(training[0], artifact_root=tmp_path)
    assert training[0].lidar_depth == Path("items/sample-a/lidar-depth.pt")
    assert validation[0].lidar_depth == Path("items/validation/lidar-depth.pt")
    assert opened == [(tmp_path / training[0].lidar_depth, "sample-a")]
    torch.testing.assert_close(loaded.values, grid.values)
    torch.testing.assert_close(loaded.valid, grid.valid)


def test_values_and_mask_take_the_same_upstream_cp_slice(monkeypatch):
    calls = []

    def split(value, *, seq_dim, cp_group):
        calls.append((seq_dim, cp_group))
        return value[:, :, 4:8]

    monkeypatch.setattr(depth.importlib, "import_module", lambda _: NS(split_inputs_cp=split))
    full = depth.DepthGrid(
        torch.arange(16.0).reshape(1, 1, 16, 1, 1),
        torch.tensor([i % 2 == 0 for i in range(16)]).reshape(1, 1, 16, 1, 1),
    )
    local = depth.split_depth_grid_cp(full, "cp4")
    assert calls == [(2, "cp4")] * 2
    assert local.values.flatten().tolist() == [4.0, 5.0, 6.0, 7.0]
    assert local.valid.flatten().tolist() == [True, False, True, False]


def test_real_observer_fit_is_rng_neutral_deterministic_and_broadcast_frozen(monkeypatch):
    install_observer_data(
        depth,
        monkeypatch,
        {
            "t0": (-1.0, -1.6),
            "t1": (-0.5, -0.6),
            "t2": (0.5, 1.4),
            "t3": (1.0, 2.4),
            "v0": (-0.75, -1.1),
            "v1": (0.75, 1.9),
        },
    )
    training = tuple(observer_item(f"t{i}") for i in range(4))
    validation = tuple(observer_item(f"v{i}") for i in range(2))
    torch.manual_seed(97)
    before = torch.get_rng_state().clone()
    first = depth.LatentDepthObserver()
    result = depth.fit_depth_observer(
        first,
        training,
        validation,
        observer_fit_spec(),
        local_topology(),
        artifact_root=Path("."),
        device="cpu",
    )
    assert torch.equal(torch.get_rng_state(), before)
    broadcasts = []
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [5, 3])
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda rows, _value, **_: rows.__setitem__(slice(None), [None, None]),
    )
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, source, **_: broadcasts.append((tensor, source)),
    )
    torch.manual_seed(12345)
    second = depth.LatentDepthObserver()
    repeated = depth.fit_depth_observer(
        second,
        training,
        validation,
        observer_fit_spec(),
        Gen3cTrainingTopology(2, 2, 3, 1, "cp", "cp", 2),
        artifact_root=Path("."),
        device="cpu",
    )
    assert sum(parameter.numel() for parameter in first.parameters()) == 577
    assert result == repeated and result.validation_loss < result.constant_comparator_loss
    assert not first.training and not second.training
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert left.dtype == torch.float32 and not left.requires_grad and not right.requires_grad
    assert broadcasts[0][0].shape == (3,) and len(broadcasts) == 5
    assert {source for _, source in broadcasts} == {3}
    for (tensor, _), parameter in zip(broadcasts[1:], second.parameters()):
        torch.testing.assert_close(tensor, parameter, rtol=0, atol=0)
    install_observer_data(depth, monkeypatch, {"t0": (-1.0, 0.0), "v0": (1.0, 0.0)})
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        depth.fit_depth_observer(
            depth.LatentDepthObserver(),
            (observer_item("t0"),),
            (observer_item("v0"),),
            observer_fit_spec(),
            local_topology(),
            artifact_root=Path("."),
            device="cpu",
        )
    assert torch.equal(torch.get_rng_state(), before)

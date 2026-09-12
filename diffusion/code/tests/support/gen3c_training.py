"""Small CPU doubles shared by Gen3C training-state tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from novel_view.preparation.waymo_ddw.artifacts import LidarDepth
from novel_view.preparation.waymo_ddw.record import PreparedItem
from novel_view.training.gen3c import data
from novel_view.training.gen3c.data import (
    R4C_ORDER_VERSION,
    R4cDataCursor,
    R4cTrainingBatch,
)
from novel_view.training.gen3c.lora.depth import DepthGrid, R4cDepthItem
from novel_view.training.gen3c.lora.depth_method import (
    TrainingStepRecordV2,
    ValidationRecordV2,
)
from novel_view.training.gen3c.lora.checkpoint import RankRngState, TrainingStateV1
from novel_view.training.gen3c.lora.depth_spec import (
    DepthObserverFitSpec,
)
from novel_view.training.gen3c.lora.step import (
    TrainingStepRecord,
    ValidationRecordV1,
)
from novel_view.training.gen3c.topology import Gen3cTrainingTopology


class ObjectiveCondition:
    def __init__(self, source: torch.Tensor, indicator: torch.Tensor) -> None:
        self.gt_latent = source
        self.condition_video_indicator = indicator
        self.video_cond_bool = False

    def to_dict(self) -> dict[str, object]:
        return {}


class ObjectiveScheduler:
    @staticmethod
    def precondition_inputs(sample: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return sample / torch.sqrt(sigma.square() + 1).view(-1, 1, 1, 1, 1)

    @staticmethod
    def precondition_noise(sigma: torch.Tensor) -> torch.Tensor:
        return sigma.log() / 4

    @staticmethod
    def precondition_outputs(sample, model_output, sigma):
        return model_output


class ObjectiveModel:
    def __init__(self, *, context_parallel: bool = False) -> None:
        self.net = SimpleNamespace(
            is_context_parallel_enabled=context_parallel,
            cp_group="cp",
        )
        self.model = SimpleNamespace(logvar=lambda timestep: torch.zeros_like(timestep))
        self.scheduler = ObjectiveScheduler()
        self.tensor_kwargs = {"device": torch.device("cpu"), "dtype": torch.float32}
        self.sigma_data = 1.0

    @staticmethod
    def _reverse_precondition_input(sample, sigma):
        return sample * torch.sqrt(sigma.square() + 1).view(-1, 1, 1, 1, 1)


class TinyLoraNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(7.0), requires_grad=False)
        self.q_lora = torch.nn.Linear(1, 1, bias=False)
        self.context_parallel_group = None

    def enable_context_parallel(self, group: object) -> None:
        self.context_parallel_group = group


def method_batch(sample_id: str = "sample-1") -> R4cTrainingBatch:
    cursor = R4cDataCursor(R4C_ORDER_VERSION, (sample_id,), 0, 0, 1)
    latent = torch.ones(1, 1, 2, 1, 1)
    return R4cTrainingBatch(
        sample_id,
        latent,
        torch.zeros_like(latent),
        torch.zeros(1, 4, 2, 1, 1),
        torch.zeros(1, 1, 1),
        cursor,
    )


def method_topology(group: object = None) -> Gen3cTrainingTopology:
    return Gen3cTrainingTopology(4, 4, 0, 0, group, "grad", 4)


def legacy_index_document(*, stride: int = 1, digits: int = 3) -> dict[str, object]:
    return {
        "format": data.R4C_LEGACY_INDEX_MAP_FORMAT,
        "items": [
            {"old_global_fit_index": index * stride, "sample_id": f"sample-{index:0{digits}d}"}
            for index in range(85)
        ],
    }


def prepared_record_item(sample_id: str, segment_id: str) -> PreparedItem:
    return PreparedItem(
        sample_id,
        "training",
        segment_id,
        0,
        (),
        3.0,
        -1,
        f"items/{sample_id}/base.pt",
        f"items/{sample_id}/pose.pt",
        f"items/{sample_id}/lidar-depth.pt",
    )


def prepared_record(items: tuple[PreparedItem, ...]) -> object:
    from novel_view.preparation.waymo_ddw.record import PreparedRecord

    return PreparedRecord(
        "waymo_v2",
        "waymo",
        "contract",
        "moge",
        "checkpoint",
        "tokenizer",
        "encoder",
        "empty-prompt.pt",
        items,
    )


class CheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = torch.nn.Module()
        self.attention.to_q_lora = torch.nn.Linear(
            2,
            1,
            bias=False,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            self.attention.to_q_lora.weight.copy_(torch.tensor([[0.2, -0.1]], dtype=torch.bfloat16))

    def forward(self, value):
        return torch.nn.functional.linear(
            value,
            self.attention.to_q_lora.weight.float(),
        )


class FakeFusedAdam:
    master_weights = True
    capturable = True
    adam_w_mode = 1

    def __init__(self, model: CheckpointModel) -> None:
        parameters = list(model.parameters())
        self.param_groups = [
            {
                "params": parameters,
                "lr": torch.tensor(0.03, dtype=torch.float32),
                "betas": (0.8, 0.9),
                "eps": 1e-6,
                "weight_decay": 0.0,
                "bias_correction": True,
            }
        ]
        self.param_groups_master = None
        self.state: dict[torch.Tensor, dict[str, torch.Tensor]] = {}

    def zero_grad(self) -> None:
        for parameter in self.param_groups[0]["params"]:
            parameter.grad = None

    def step(self) -> None:
        group = self.param_groups[0]
        if self.param_groups_master is None:
            self.param_groups_master = [
                {"params": [parameter.detach().float().clone() for parameter in group["params"]]}
            ]
        group["step"] = (
            group.get(
                "step",
                torch.zeros(1, dtype=torch.int32),
            )
            + 1
        )
        beta1, beta2 = group["betas"]
        for parameter, master in zip(
            group["params"],
            self.param_groups_master[0]["params"],
            strict=True,
        ):
            gradient = parameter.grad.float()
            state = self.state.setdefault(
                parameter,
                {
                    "exp_avg": torch.zeros_like(master),
                    "exp_avg_sq": torch.zeros_like(master),
                },
            )
            state["exp_avg"].mul_(beta1).add_(gradient, alpha=1 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(
                gradient,
                gradient,
                value=1 - beta2,
            )
            denominator = state["exp_avg_sq"].sqrt().add_(group["eps"])
            master.addcdiv_(
                state["exp_avg"],
                denominator,
                value=-float(group["lr"]),
            )
            with torch.no_grad():
                parameter.copy_(master.to(parameter.dtype))


class Scheduler:
    def __init__(self, optimizer: FakeFusedAdam) -> None:
        self.optimizer = optimizer
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        group = self.optimizer.param_groups[0]
        group["lr"] = group["lr"] * 0.8

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.steps = state["steps"]


def optimization_step(model, optimizer, scheduler, value):
    optimizer.zero_grad()
    loss = model(value).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return loss.detach()


def fake_method(model: CheckpointModel) -> SimpleNamespace:
    optimizer = FakeFusedAdam(model)
    optimizer.param_groups[0]["initial_lr"] = optimizer.param_groups[0]["lr"].clone()
    return SimpleNamespace(
        model=SimpleNamespace(model=model),
        optimizer=optimizer,
        scheduler=Scheduler(optimizer),
    )


def fake_topology() -> Gen3cTrainingTopology:
    return Gen3cTrainingTopology(2, 2, 0, 0, "cp", "gradient", 2)


def fake_cursor(*, epoch: int = 0, offset: int = 1) -> R4cDataCursor:
    return R4cDataCursor(
        R4C_ORDER_VERSION,
        ("sample-train-a", "sample-train-b"),
        0,
        epoch,
        offset,
    )


def checkpoint_training_state(model, method, rank_zero_rng) -> TrainingStateV1:
    return TrainingStateV1(
        model,
        method.optimizer,
        method.scheduler,
        1,
        fake_cursor(),
        fake_topology(),
        (
            RankRngState(
                1,
                torch.Generator().manual_seed(99).get_state(),
                torch.tensor([19], dtype=torch.uint8),
            ),
            RankRngState(0, rank_zero_rng, torch.tensor([7], dtype=torch.uint8)),
        ),
    )


def legacy_checkpoint_payload(payload: dict, *, step: int = 77) -> dict:
    payload = dict(payload)
    payload.pop("format")
    payload["topology"].pop("gradient_average_size")
    payload["progress"] = {
        "next_step": step,
        "data_cursor": {
            "recipe_id": "waymo-ddw-fit-random-one-v1",
            "split_id": "waymo-ddw-lora-v1",
            "keyset_id": "fit-central-v1",
            "training_seed": 0,
            "order_version": "gen3c-waymo-ddw-random-one-order-v1",
            "accepted_global_fit_indices": list(range(77)),
            "epoch": step // 77,
            "offset": 0,
        },
        "resolved_request": {"historical": True},
    }
    payload["rng_states"] = [
        {"cpu": row["torch_cpu"], "cuda": row["torch_cuda"]} for row in payload["rng_states"]
    ]
    return payload


def training_item(
    name: str, segment: str, *, magnitude_m: float = 1.0, sign: int = 1
) -> data.R4cPreparedItem:
    return data.R4cPreparedItem(
        name,
        segment,
        Path(f"items/{name}/base.pt"),
        Path(f"items/{name}/pose.pt"),
        Path("empty-prompt.pt"),
        magnitude_m,
        sign,
    )


def loop_batch(name: str = "sample") -> data.R4cTrainingBatch:
    return data.R4cTrainingBatch(name, None, None, None, None, object())


def fake_training_batches(items, cursor, **_kwargs):
    for item, next_cursor in data.iter_r4c_items(items, cursor):
        yield data.R4cTrainingBatch(
            item.sample_id,
            None,
            None,
            None,
            None,
            next_cursor,
        )


def install_training_loop_fakes(module, monkeypatch, *, version: int):
    """Install shared model/process doubles; scenario assertions stay in tests."""
    model = SimpleNamespace(model=torch.nn.Linear(1, 1), net=object())
    living = SimpleNamespace(
        model=model,
        network=object(),
        optimizer=object(),
        scheduler=object(),
        trainable_parameters=(),
    )
    result = SimpleNamespace(
        model=model,
        living=living,
        events=[],
        states=[],
        records=[],
    )
    monkeypatch.setattr(
        module,
        "build_r4c_lora_model",
        lambda **_kwargs: result.events.append("model") or model,
    )
    monkeypatch.setattr(module, "build_r4c_lora_method", lambda *_args: living)
    monkeypatch.setattr(module, "iter_r4c_training_batches", fake_training_batches)
    monkeypatch.setattr(module, "capture_rank_rng_state", lambda _rank: object())
    monkeypatch.setattr(module, "gather_rank_rng_states", lambda *_args: ())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda _device: torch.tensor([7], dtype=torch.uint8),
    )
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda _seed: None)
    if version == 1:
        monkeypatch.setattr(
            module,
            "run_training_step_v1",
            lambda **values: TrainingStepRecord(
                values["batch"].sample_id,
                values["batch"].next_cursor,
                -0.1,
                0.5,
                1e-4,
                (1.0,),
                (0.1,),
            ),
        )
        monkeypatch.setattr(
            module,
            "run_validation_v1",
            lambda **values: ValidationRecordV1(-0.25, len(tuple(values["batches"]))),
        )
        save_name, writer_name = "save_checkpoint_v1", "write_checkpoint_record_v1"
    else:
        monkeypatch.setattr(
            module,
            "read_r4c_depth_grid",
            lambda *_args, **_kwargs: DepthGrid(torch.zeros(1), torch.ones(1, dtype=torch.bool)),
        )
        monkeypatch.setattr(
            module,
            "run_training_step_v2",
            lambda **values: TrainingStepRecordV2(
                values["batch"].sample_id,
                values["batch"].next_cursor,
                -0.1,
                0.2,
                -0.08,
                1,
                0.5,
                1e-4,
                (1.0,),
                (0.1,),
            ),
        )
        monkeypatch.setattr(
            module,
            "_run_validation_epoch_v2",
            lambda **_kwargs: ValidationRecordV2(-0.2, 0.3, -0.17, 1),
        )
        save_name, writer_name = "save_checkpoint_v2", "write_checkpoint_record_v2"
    monkeypatch.setattr(module, save_name, lambda _path, state: result.states.append(state))
    monkeypatch.setattr(module, writer_name, lambda _path, record: result.records.append(record))
    return result


def observer_item(sample_id: str) -> R4cDepthItem:
    path = Path(sample_id)
    prepared = data.R4cPreparedItem(sample_id, sample_id, path, path, path, 3.0, -1)
    return R4cDepthItem(prepared, path)


def observer_fit_spec() -> DepthObserverFitSpec:
    return DepthObserverFitSpec(0, 0, 10, 1, "adamw", 0.03, (0.9, 0.999), 1e-8, 0.0, "constant")


def install_observer_data(module, monkeypatch, targets) -> None:
    def read_base(_path: Path, sample_id: str):
        latent = torch.zeros((1, 16, 1, 1, 1), dtype=torch.bfloat16)
        latent[:, 0] = targets[sample_id][0]
        return SimpleNamespace(clean_latent=latent)

    def read_depth(item: R4cDepthItem, *, artifact_root: Path) -> DepthGrid:
        value = torch.full((1, 1, 1, 1, 1), targets[item.prepared.sample_id][1])
        return DepthGrid(value, torch.ones_like(value, dtype=torch.bool))

    monkeypatch.setattr(module.prepared_artifacts, "read_base_latents", read_base)
    monkeypatch.setattr(module, "read_r4c_depth_grid", read_depth)


def local_topology() -> Gen3cTrainingTopology:
    return Gen3cTrainingTopology(1, 1, 0, 0, None, None, 1)


def lidar_depth_artifact(rows) -> LidarDepth:
    ordered = sorted(rows)
    counts = np.bincount([row[0] for row in ordered], minlength=121)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return LidarDepth(
        "sample-a",
        "FRONT",
        (704, 1280),
        torch.zeros((121, 3, 3), dtype=torch.float64),
        torch.from_numpy(offsets),
        torch.tensor([(row[1], row[2]) for row in ordered], dtype=torch.float32),
        torch.tensor([row[3] for row in ordered], dtype=torch.float32),
        torch.tensor([row[4] for row in ordered], dtype=torch.bool),
    )


def assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_nested_equal(first, second)
    else:
        assert left == right

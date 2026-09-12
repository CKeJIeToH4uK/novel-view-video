"""Concrete CP4/LoRA policy and a numerical clipped update, no model weights."""

import re
from types import SimpleNamespace as NS

import pytest
import torch

from novel_view.training.gen3c import topology
from novel_view.training.gen3c.lora import method, step
from tests.support.gen3c_training import TinyLoraNetwork, method_batch, method_topology


def test_cp4_uses_real_cp_and_gradient_groups(monkeypatch):
    options = {}
    parallel = NS(
        initialize_model_parallel=lambda **kw: options.update(kw),
        get_context_parallel_group=lambda: "cp",
        get_data_parallel_group=lambda **kw: options.update(kw) or "gradient",
    )
    modules = {
        "cosmos_predict1.utils.distributed": NS(init=lambda: None),
        "megatron.core.parallel_state": parallel,
        "torch": NS(
            distributed=NS(get_world_size=lambda group=None: 4, get_rank=lambda group=None: 2)
        ),
    }
    monkeypatch.setattr(topology.importlib, "import_module", modules.__getitem__)
    actual = topology.initialize_r4c_cp4()
    assert options == dict(
        tensor_model_parallel_size=1, context_parallel_size=4, with_context_parallel=True
    )
    assert actual == topology.Gen3cTrainingTopology(4, 4, 2, 2, "cp", "gradient", 4)


def test_lora_layout_frozen_base_and_optimizer_parameters(monkeypatch):
    control = method.R4C_LORA_SPEC.upstream_control()
    assert (control["rank"], control["scale"]) == (8, 1.0)
    edit = control["edits"][0]
    assert edit["block_edit"] == ["FA[to_q, to_v]", "CA[to_q, to_v]"]
    assert [i for i in range(30) if re.fullmatch(edit["blocks"], str(i))] == list(range(28))
    net, options = TinyLoraNetwork(), {}
    optimizer = NS(param_groups=[{"lr": 1e-4}])
    scheduler = NS(get_last_lr=lambda: [1e-4])
    modules = {
        "cosmos_predict1.utils.distributed": NS(parallel_model_wrapper=lambda config, value: value),
        "cosmos_predict1.utils.config": NS(DDPConfig=lambda: None),
        "cosmos_predict1.diffusion.training.utils.optim_instantiate": NS(
            get_base_optimizer=lambda model, **kw: options.update(kw) or optimizer
        ),
        "torch.optim.lr_scheduler": NS(LambdaLR=lambda *_args, **_kwargs: scheduler),
    }
    original = method.importlib.import_module
    monkeypatch.setattr(
        method.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original(name),
    )
    monkeypatch.setattr(method, "_apply_activation_recompute", lambda _: None)
    built = method.build_r4c_lora_method(NS(net=net, model=net), method_topology("cp"))
    assert built.trainable_parameters == (net.q_lora.weight,)
    assert not net.base.requires_grad and net.training and net.context_parallel_group == "cp"
    assert options == dict(
        lr=1e-4,
        weight_decay=0.1,
        betas=[0.9, 0.99],
        eps=1e-10,
        optim_type="fusedadam",
        sharding=False,
        master_weights=True,
        capturable=True,
    )
    assert built.scheduler.get_last_lr() == [1e-4]
    net.base.requires_grad_(True)
    with pytest.raises(RuntimeError):
        method.build_r4c_lora_method(NS(net=net, model=net), method_topology("cp"))


def test_clipped_update_validation_and_remote_failure(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    base = torch.nn.Parameter(torch.tensor(9.0), requires_grad=False)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    monkeypatch.setattr(
        step, "sample_r4c_noise", lambda *_: NS(sigma=torch.ones(1), condition_sigma=torch.ones(1))
    )
    monkeypatch.setattr(step, "prepare_r4c_condition", lambda *_: None)
    monkeypatch.setattr(step, "forward_r4c_edm", lambda *_: NS(loss=(parameter - 1).square()))
    arguments = dict(model=None, network=None, topology=method_topology())
    record = step.run_training_step_v1(
        **arguments,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_parameters=(parameter,),
        batch=method_batch()
    )
    assert (parameter.item(), record.loss, record.gradient_norm) == pytest.approx((0.1, 1.0, 2.0))
    assert record.sample_id == "sample-1" and parameter.grad is None
    assert base.item() == 9.0 and base.grad is None
    validation = step.run_validation_v1(**arguments, batches=[method_batch(), method_batch()])
    assert (validation.loss, validation.sample_count) == pytest.approx((0.81, 2))
    assert parameter.item() == pytest.approx(0.1) and parameter.grad is None
    assert scheduler.last_epoch == 1
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, **_: value.zero_())
    with pytest.raises(FloatingPointError):
        step.apply_r4c_lora_update(
            loss=(parameter - 1).square(),
            optimizer=optimizer,
            scheduler=scheduler,
            trainable_parameters=(parameter,),
            topology=method_topology("cp"),
        )
    assert parameter.grad is None and parameter.item() == pytest.approx(0.1)

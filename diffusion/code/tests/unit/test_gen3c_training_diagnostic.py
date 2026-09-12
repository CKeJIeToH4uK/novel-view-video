"""CPU-проверки конкретной A100/CP4 training-диагностики."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from novel_view.diagnostics import training


def _rows(name: str = "NVIDIA A100-SXM4-80GB") -> list[dict[str, object]]:
    total = 80 * 1024**3
    return [
        {
            "name": name,
            "compute_capability": "8.0",
            "total_memory_bytes": total,
            "free_memory_bytes": int(total * 0.95),
            "mig_mode": "Disabled",
        }
        for _ in range(4)
    ]


def test_profile_and_torchrun_environment_are_exact(monkeypatch) -> None:
    profile = training._resolve_profile("a100-cp4")
    assert (profile.name, profile.gpu_count) == ("training_cp4", 4)
    with pytest.raises(RuntimeError, match="unsupported"):
        training._resolve_profile("h200-cp4")

    values = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "RANK": "2",
        "LOCAL_RANK": "2",
        "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "4",
    }
    monkeypatch.setattr(os, "environ", values)
    assert training._visible_devices(profile.gpu_count) == ("0", "1", "2", "3")
    assert training._distributed_environment(profile) == (2, 2)


def test_hardware_contract_is_only_in_explicit_diagnostic() -> None:
    training._validate_hardware(_rows(), 4)
    with pytest.raises(RuntimeError, match="A100"):
        training._validate_hardware(_rows("NVIDIA H100 80GB HBM3"), 4)


def test_distributed_failure_always_destroys_group(monkeypatch) -> None:
    events: list[str] = []
    state = {"initialized": False}
    total = 80 * 1024**3

    def initialize(**_kwargs):
        state["initialized"] = True
        events.append("initialize")

    def gather_hardware(outputs, local):
        outputs[:] = [dict(local) for _ in range(4)]

    def fail_reduce(_value):
        raise RuntimeError("synthetic collective failure")

    distributed = SimpleNamespace(
        init_process_group=initialize,
        all_gather_object=gather_hardware,
        all_reduce=fail_reduce,
        is_initialized=lambda: state["initialized"],
        destroy_process_group=lambda: events.append("destroy"),
    )
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 4,
        set_device=lambda _rank: None,
        get_device_properties=lambda _rank: SimpleNamespace(
            name="NVIDIA A100-SXM4-80GB",
            major=8,
            minor=0,
            total_memory=total,
            uuid="00000007-0000-0000-0000-000000000007",
        ),
        mem_get_info=lambda _rank: (int(total * 0.95), total),
    )
    torch = SimpleNamespace(
        cuda=cuda,
        distributed=distributed,
        bfloat16="bfloat16",
        tensor=lambda *_args, **_kwargs: object(),
    )
    checked_devices = []
    monkeypatch.setattr(
        training,
        "_require_mig_disabled",
        lambda device: checked_devices.append(device) or "Disabled",
    )

    with pytest.raises(RuntimeError, match="synthetic collective failure"):
        training._run_distributed_preflight(
            torch,
            training._resolve_profile("a100-cp4"),
            rank=0,
            local_rank=0,
            fused_adam="apex.optimizers.FusedAdam",
            amp_extension="amp_C",
        )
    assert events == ["initialize", "destroy"]
    assert checked_devices == ["GPU-00000007-0000-0000-0000-000000000007"]


def test_job_doctor_stops_without_visible_cuda(monkeypatch) -> None:
    from novel_view.diagnostics import _job_worker

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    with pytest.raises(RuntimeError, match="visible CUDA"):
        _job_worker.main(
            ["--backend", "generation", "--gpu-count", "2", "--t5-root", "/absent/t5"]
        )

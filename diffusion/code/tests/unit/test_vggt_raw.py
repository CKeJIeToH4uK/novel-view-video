"""Raw VGGT grids and numeric array transport, without camera alignment."""

from pathlib import Path

import numpy as np
import pytest

from novel_view.models.vggt import backend
from novel_view.models.vggt.request import VggtOmegaRequest, VggtOmegaTelemetry
from novel_view.models.vggt.spec import (
    VggtOmegaError,
    VggtOmegaInputMode,
    build_vggt_omega_input_plan,
)


def test_proven_grid_modes_and_pixel_scale():
    for mode, size in {
        VggtOmegaInputMode.BALANCED_512: (384, 688),
        VggtOmegaInputMode.BALANCED_896: (672, 1216),
        VggtOmegaInputMode.DIRECT_704_1280: (704, 1280),
    }.items():
        plan = build_vggt_omega_input_plan((704, 1280), mode)
        assert plan.model_size_hw == size
        np.testing.assert_array_equal(
            plan.source_to_model_pixels, np.diag([size[1] / 1280, size[0] / 704, 1])
        )
    with pytest.raises(VggtOmegaError):
        build_vggt_omega_input_plan((100, 1000), VggtOmegaInputMode.BALANCED_512)


def test_ordered_rgb_and_all_raw_outputs_cross_fixed_prefix(tmp_path, monkeypatch):
    frames = [np.full((2, 3, 3), value, np.uint8) for value in [19, 2, 31]]
    plan = build_vggt_omega_input_plan((2, 3), VggtOmegaInputMode.BALANCED_512)
    values = {
        "depth_model_units": np.arange(6, dtype=np.float32).reshape(3, 1, 2),
        "depth_confidence": np.full((3, 1, 2), 2, np.float32),
        "pose_encoding": np.arange(27, dtype=np.float32).reshape(3, 9),
        "model_w2c": np.arange(36, dtype=np.float32).reshape(3, 3, 4),
        "model_intrinsics": np.arange(27, dtype=np.float32).reshape(3, 3, 3),
    }

    def run(command, *_args):
        assert command[0] == str(backend.VGGT_PYTHON)
        assert command[1].endswith("models/vggt/_worker.py")
        exchange = Path(command[command.index("--exchange") + 1])
        np.testing.assert_array_equal(np.load(exchange / "rgb.npy"), frames)
        for name, value in values.items():
            np.save(exchange / (name + ".npy"), value)
        np.save(exchange / "peak_ram_mib.npy", 456.0)
        np.save(exchange / "peak_vram_mib.npy", 123.0)

    monkeypatch.setattr(backend, "run_process", run)
    result = backend.run_vggt_omega(
        VggtOmegaRequest(iter(frames), 3, plan),
        backend.VggtOmegaResources(tmp_path, Path("model.pt")),
    )
    for name, expected in values.items():
        np.testing.assert_array_equal(getattr(result.prediction, name), expected)
    assert result.telemetry == VggtOmegaTelemetry(456.0, 123.0)

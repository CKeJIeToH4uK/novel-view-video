"""Raw union masks and permanent logs across the disposable worker exchange."""

from pathlib import Path

import numpy as np
import pytest

from novel_view.models.grounded_sam2 import backend
from novel_view.models.grounded_sam2._worker import combine_object_masks
from novel_view.models.grounded_sam2.request import GroundedSam2Request
from novel_view.runtime.process import ProcessError


def test_empty_and_overlapping_object_masks():
    empty = combine_object_masks(np.empty((0, 4, 5), np.bool_), (4, 5))
    assert empty.dtype == np.bool_ and not empty.any()
    masks = np.zeros((3, 1, 4, 5), np.float32)
    masks[:2, 0, 0, 0] = 1
    masks[1, 0, 1, 2] = masks[2, 0, 3, 4] = 1
    expected = np.zeros((4, 5), np.bool_)
    expected[0, 0] = expected[1, 2] = expected[3, 4] = True
    np.testing.assert_array_equal(combine_object_masks(masks, (4, 5)), expected)


def test_logs_survive_success_and_failed_scratch(tmp_path, monkeypatch):
    resources = backend.GroundedSam2Resources(tmp_path, Path("grounding"), Path("sam2.pt"))
    request = GroundedSam2Request(
        (np.zeros((2, 3, 3), np.uint8),),
        1,
        (2, 3),
        "sam2.yaml",
        "car.",
        0.4,
        0.3,
    )
    exchanges = []

    def worker(command, log_path, *_args):
        assert command[0] == str(backend.GEN3C_PYTHON)
        exchange = Path(command[command.index("--exchange") + 1])
        exchanges.append(exchange)
        log_path.write_text("full SAM2 log\n")
        if log_path.name == "failure.log":
            raise ProcessError("SAM2 exited with code 23: original traceback")
        np.save(exchange / "dynamic_mask.npy", np.ones((1, 2, 3), np.bool_))

    monkeypatch.setattr(backend, "run_process", worker)
    result = backend.run_grounded_sam2(request, resources, tmp_path / "success.log")
    with pytest.raises(RuntimeError, match="code 23: original traceback"):
        backend.run_grounded_sam2(request, resources, tmp_path / "failure.log")
    assert result.dynamic_mask.all()
    assert all(not exchange.exists() for exchange in exchanges)
    for name in ["success.log", "failure.log"]:
        assert (tmp_path / name).read_text() == "full SAM2 log\n"

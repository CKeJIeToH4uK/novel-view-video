"""Real EUVS records, selected RGB slots, measured cameras and two supports."""

import csv
import math
import pickle
import sqlite3

import numpy as np
from PIL import Image
import pytest

from novel_view.evaluation.euvs.execute import EuvsMetricResources, evaluate_pair
from novel_view.evaluation.euvs.masks import load_or_predict_masks
from novel_view.evaluation.euvs.record import encode_euvs_metric_result
from novel_view.evaluation.euvs.samples import load_euvs_samples
from novel_view.evaluation.euvs.support_views import build_euvs_support
from novel_view.generation.euvs.record import (
    EuvsGen3cRunRecord, Gen3cRunExecution, Gen3cRunGeometry, Gen3cRunModel,
    Gen3cRunOutput, Gen3cRunSampling, encode_run_record,
)
from novel_view.inputs.euvs.index import FramesIndex
from novel_view.inputs.euvs.spec import EuvsFrameSelection, EuvsPairSelection
from novel_view.metrics.image import MetricStatus
from novel_view.models.vggt.record import save_vggt_omega_prediction
from novel_view.models.vggt.request import VggtOmegaRawPrediction
from novel_view.models.vggt.spec import VggtOmegaInputMode, build_vggt_omega_input_plan
from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import ProcessError


def _inputs(root):
    size = (704, 1280)
    tokens = tuple(f"{i:016x}" for i in range(1, 6))
    centers = [(0, 0), (1, 0), (0, 1), (0, 1), (0, 0)]
    rows = [dict(
        image_path=f"{token}.png", db_path="frames.db", location="place",
        logical_locations="place", traversal="source" if i < 3 else "target",
        timestamp_us=i + 1, channel="CAM_F0", image_token=token,
        ego_pose_token=token, camera_token="00000000000000aa",
    ) for i, token in enumerate(tokens)]
    with (root / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with sqlite3.connect(root / "frames.db") as db:
        db.executescript("""
            CREATE TABLE ego_pose (token BLOB, timestamp INTEGER, x REAL, y REAL,
                z REAL, qw REAL, qx REAL, qy REAL, qz REAL, epsg INTEGER);
            CREATE TABLE camera (token BLOB, channel TEXT, model TEXT,
                translation BLOB, rotation BLOB, intrinsic BLOB, distortion BLOB,
                width INTEGER, height INTEGER);
        """)
        db.executemany("INSERT INTO ego_pose VALUES (?,?,?,?,?,?,?,?,?,?)", [
            (bytes.fromhex(token), i + 1, x, y, 0, 1, 0, 0, 0, 32611)
            for i, (token, (x, y)) in enumerate(zip(tokens, centers))
        ])
        db.execute("INSERT INTO camera VALUES (?,?,?,?,?,?,?,?,?)", (
            bytes.fromhex("00000000000000aa"), "CAM_F0", "pinhole",
            pickle.dumps([0., 0., 0.]), pickle.dumps([1., 0., 0., 0.]),
            pickle.dumps(np.eye(3).tolist()), pickle.dumps([0.] * 5), 1280, 704,
        ))
    pair = EuvsPairSelection("pair/a", (), "place", "CAM_F0",
        EuvsFrameSelection("source", tokens[:3]), EuvsFrameSelection("target", tokens[3:]))
    record = EuvsGen3cRunRecord("native", "generation", pair,
        "direct-projected-nearest-single-source/v1", (1, 7), (2, 0),
        Gen3cRunGeometry("vggt-omega", "geometry", "ordered-source-tokens"),
        Gen3cRunModel("base", "gen3c/base.pt"), Gen3cRunSampling(7, "", "", 1., 35),
        Gen3cRunExecution(1), Gen3cRunOutput("rgb.npy", (121, *size, 3)))
    (root / "run.json").write_bytes(encode_run_record(record))
    output = np.lib.format.open_memmap(root / "rgb.npy", mode="w+",
                                      dtype=np.uint8, shape=record.output.shape)
    output[1], output[7] = 10, 20
    output.flush()
    del output
    # Source PNGs do not exist until after load_euvs_samples: target-only IO is real.
    for token in tokens[3:]:
        Image.fromarray(np.zeros((*size, 3), np.uint8)).save(root / f"{token}.png")
    return tokens, size


def test_record_slots_target_only_rgb_two_support_and_metric_failure(tmp_path, monkeypatch):
    tokens, size = _inputs(tmp_path)
    samples = load_euvs_samples(tmp_path / "run.json", FramesIndex.load(tmp_path))
    assert [(f.output_index, f.source_index) for f in samples.frames] == [(1, 2), (7, 0)]
    assert [f.target.input.ref.image_token for f in samples.frames] == list(tokens[3:])
    for frame, value in zip(samples.frames, (10, 20)):
        np.testing.assert_array_equal(frame.prediction_rgb, np.full((*size, 3), value, np.uint8))
        assert not frame.prediction_rgb.flags.writeable
        assert not frame.target.raster.rgb.any()
    for token in tokens[:3]:
        Image.fromarray(np.zeros((*size, 3), np.uint8)).save(tmp_path / f"{token}.png")
    plan = build_vggt_omega_input_plan(size, VggtOmegaInputMode.BALANCED_512)
    w2c = np.tile(np.eye(4, dtype=np.float32)[:3], (3, 1, 1))
    w2c[1, 0, 3], w2c[2, 1, 3] = -1, -1
    prediction = VggtOmegaRawPrediction(plan,
        np.ones((3, *plan.model_size_hw), np.float32),
        np.ones((3, *plan.model_size_hw), np.float32), np.zeros((3, 9), np.float32),
        w2c, np.tile(plan.source_to_model_pixels.astype(np.float32), (3, 1, 1)))
    save_vggt_omega_prediction(prediction, tokens[:3], size, tmp_path / "geometry")
    native = tmp_path / "masks" / "native"
    native.mkdir(parents=True)
    for i, token in enumerate(tokens):
        mask = np.full(size, 255 if i == 2 else 0, np.uint8)
        if i >= 3:
            mask[:, 0] = 255
        Image.fromarray(mask).save(native / f"{token}.png")
    source_masks = load_or_predict_masks(samples.pair.source, native.parent)
    target_masks = load_or_predict_masks(samples.pair.target, native.parent)
    support = build_euvs_support(samples, source_masks, target_masks, tmp_path)
    target_static = np.ones(size, bool)
    target_static[:, 0] = False
    for frame in support.frames:
        np.testing.assert_array_equal(frame.views.target_static, target_static)
    assert not support.frames[0].views.source_aware.any()
    np.testing.assert_array_equal(support.frames[1].views.source_aware, target_static)

    def perceptual(arguments, log_path, *_):
        assert arguments[0] == str(GEN3C_PYTHON)
        rgb = np.load(arguments[arguments.index("--input-rgb") + 1])
        masks = np.load(arguments[arguments.index("--input-support") + 1])
        np.testing.assert_array_equal(rgb[:, 0, 0, 0, 0], [10, 20])
        assert not rgb[:, 1].any()
        np.testing.assert_array_equal(masks[:, 0], np.stack([target_static] * 2))
        assert not masks[0, 1].any()
        np.testing.assert_array_equal(masks[1, 1], target_static)
        np.save(arguments[arguments.index("--output-perceptual") + 1],
                [[[.25, .75, 1.], [np.nan, np.nan, 0.]], [[.5, .6, 2.], [.5, .6, 2.]]])
        log_path.write_text("original perceptual output\n")

    monkeypatch.setattr("novel_view.evaluation.euvs.execute.run_process", perceptual)
    resources = EuvsMetricResources(tmp_path, tmp_path / "metrics.log",
                                    tmp_path / "dino", tmp_path / "dino.pt", tmp_path / "torch")
    result = evaluate_pair(support, resources, "generation/run.json")
    for frame, value in zip(result.frames, (10, 20)):
        assert frame.target_static.psnr.value == pytest.approx(20 * math.log10(255 / value))
        assert frame.target_static.support_pixel_count == 704 * 1279
        assert frame.target_static.lpips_weight_sum == 704 * 1279
    assert result.frames[0].source_aware_static.psnr.status is MetricStatus.UNDEFINED
    assert result.frames[0].target_static.lpips_alex.value == .25
    assert result.frames[1].source_aware_static.dinov2_cosine.value == .6
    assert b'"run_record": "generation/run.json"' in encode_euvs_metric_result(result)
    assert resources.log_path.read_text() == "original perceptual output\n"
    failure = ProcessError("perceptual worker exited 23")

    def fail(arguments, log_path, *_):
        log_path.write_text("original traceback\n")
        raise failure

    monkeypatch.setattr("novel_view.evaluation.euvs.execute.run_process", fail)
    with pytest.raises(ProcessError) as caught:
        evaluate_pair(support, resources, "generation/run.json")
    assert caught.value is failure and resources.log_path.read_text() == "original traceback\n"
    assert not list(tmp_path.glob("euvs-evaluation-*"))

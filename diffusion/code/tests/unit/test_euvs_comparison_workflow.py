"""Real saved evaluation comparison with equal camera/pair/location means."""

from dataclasses import replace
from urllib.parse import quote

from novel_view.evaluation.euvs.benchmark import compare_evaluations, write_euvs_comparison
from novel_view.evaluation.euvs.record import (
    EuvsMetricFrameRecord, EuvsMetricResultRecord, EuvsMetricTrackResult, encode_euvs_metric_result,
)
from novel_view.generation.euvs.record import (
    EuvsGen3cRunRecord, Gen3cRunExecution, Gen3cRunGeometry, Gen3cRunModel,
    Gen3cRunOutput, Gen3cRunSampling, encode_run_record, load_run_record,
)
from novel_view.inputs.euvs.spec import EuvsFrameSelection, EuvsPairSelection, EuvsSelection
from novel_view.metrics.image import MetricValue
from novel_view.workflows.runner import select_job
from tests.support.stage5 import make_job


def _pair(name: str, location: str) -> EuvsPairSelection:
    return EuvsPairSelection(
        name,
        (),
        location,
        "CAM_F0",
        EuvsFrameSelection("source", (f"s-{name}",)),
        EuvsFrameSelection("target", (f"t-{name}",)),
    )


def _metrics(locator: str, value: float) -> EuvsMetricResultRecord:
    metric = MetricValue.finite(value)
    track = EuvsMetricTrackResult(
        4, 4, 1.0, 1, 4.0, 1.0, metric, metric, metric, metric
    )
    return EuvsMetricResultRecord(
        locator,
        "audited-static/v1",
        "target-source-dynamic/v1",
        "four-neighbour-zbuffer/v1",
        "mask-v1",
        (EuvsMetricFrameRecord(track, track),),
    )


def test_real_records_macro_camera_pair_location_and_incomplete_rows(tmp_path):
    pairs = (_pair("a/b", "one"), _pair("b", "one"), _pair("c", "two"))
    pairs = (replace(pairs[0], target=replace(pairs[0].target, image_tokens=("t-a0", "t-a1"))), *pairs[1:])
    for pair, values in zip(pairs, ((0., 2.), (3.,), (10.,))):
        for variant in ("base", "tuned"):
            locator = f"records/{quote(pair.name, safe='')}-{variant}.json"
            output = tmp_path / locator
            output.parent.mkdir(exist_ok=True)
            record = EuvsGen3cRunRecord("native", "generation", pair,
                "direct-projected-nearest-single-source/v1",
                tuple(range(1, len(values) + 1)), (0,) * len(values),
                Gen3cRunGeometry("vggt", "geometry/common", "ordered-source-tokens"),
                Gen3cRunModel(variant, f"models/{variant}.pt"),
                Gen3cRunSampling(7, "", "", 1., 35), Gen3cRunExecution(1),
                Gen3cRunOutput("rgb.npy", (121, 704, 1280, 3)))
            output.write_bytes(encode_run_record(record))
            metric = replace(_metrics(locator, 0.), frames=tuple(
                _metrics(locator, value if variant == "tuned" else 0.).frames[0]
                for value in values))
            path = tmp_path / variant / "metrics" / f"{quote(pair.name, safe='')}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encode_euvs_metric_result(metric))
    tables = compare_evaluations("comparison", EuvsSelection(pairs), "base", "tuned", tmp_path)
    global_values = {row[5]: row[-1] for row in tables.summary
                     if row[1:5] == ("global", None, None, "target_static")}
    assert global_values["psnr"] == 6. and global_values["lpips_alex"] == -6.
    assert {row[-16:] for row in tables.pairs if row[1] == "tuned" and row[4] == "a/b"} == {(1., 2, 0, 0) * 4}
    assert next(row[-1] for row in tables.summary
                if row[1:3] == ("pair", "a/b") and row[4:6] == ("target_static", "psnr")) == 1.
    # One truly missing metric and one scientific geometry mismatch remain explicit.
    (tmp_path / "tuned/metrics/c.json").unlink()
    path = tmp_path / "records/b-tuned.json"
    record = load_run_record(path)
    path.write_bytes(encode_run_record(replace(record, geometry=replace(record.geometry, backend="other"))))
    partial = compare_evaluations("partial", EuvsSelection(pairs), "base", "tuned", tmp_path)
    states = {row[6] for row in partial.summary if row[1] == "pair"}
    assert states == {"completed", "incompatible", "missing"}
    (tmp_path / "report").mkdir()
    write_euvs_comparison(tmp_path / "report", partial)
    assert {path.name for path in (tmp_path / "report").iterdir()} == {"frames.csv", "pairs.csv", "summary.csv"}

    job = make_job("euvs_comparison", 1, {
        "reader": "euvs_nuplan", "dataset": "euvs", "selection": "s.yaml",
        "base_evaluation": "base", "tuned_evaluation": "tuned",
    }, {}, preset="cpu_test")
    selected = select_job(job)
    assert (selected.image_variant, selected.preset.gpu_count) == ("core", 0)

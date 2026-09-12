"""Compare two ready EUVS evaluation attempts without running models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import quote

from novel_view.evaluation.euvs.record import (
    EuvsMetricResultRecord,
    EuvsMetricTrackResult,
    load_euvs_metric_result,
)
from novel_view.generation.euvs.record import (
    EuvsGen3cRunRecord,
    load_run_record,
)
from novel_view.inputs.euvs.spec import EuvsPairSelection, EuvsSelection
from novel_view.metrics.aggregate import FrameMacroMetric, mean_columns, summarize_cameras
from novel_view.metrics.image import MetricValue

_TRACKS = ("target_static", "source_aware_static")
_METRICS = ("psnr", "ssim", "lpips_alex", "dinov2_cosine")
_LOWER_IS_BETTER = {"lpips_alex"}
_FloatQuad = tuple[float, float, float, float]

_FRAME_COLUMNS = (
    "experiment",
    "variant",
    "model_id",
    "seed",
    "pair",
    "location",
    "tags",
    "model_checkpoint",
    "record_provenance",
    "context_parallel_size",
    "geometry_backend",
    "geometry_provenance",
    "metric_protocol",
    "support_protocol",
    "projection_protocol",
    "mask_recipe_id",
    "run_record",
    "metric_result",
    "target_index",
    "target_image_token",
    "output_index",
    "source_sequence_index",
    "source_image_token",
    "track",
    "support_pixel_count",
    "raster_valid_pixel_count",
    "support_fraction",
    "ssim_window_count",
    "lpips_weight_sum",
    "dinov2_patch_weight_sum",
    *(
        field
        for metric in _METRICS
        for field in (f"{metric}_status", f"{metric}_value", f"{metric}_reason")
    ),
)
_PAIR_COLUMNS = (
    "experiment",
    "variant",
    "model_id",
    "seed",
    "pair",
    "location",
    "tags",
    "model_checkpoint",
    "record_provenance",
    "context_parallel_size",
    "geometry_backend",
    "geometry_provenance",
    "metric_protocol",
    "support_protocol",
    "projection_protocol",
    "mask_recipe_id",
    "track",
    "status",
    "reason_code",
    "error",
    "run_record",
    "metric_result",
    "target_frame_count",
    *(
        field
        for metric in _METRICS
        for field in (
            f"{metric}_finite_mean",
            f"{metric}_finite_count",
            f"{metric}_positive_infinity_count",
            f"{metric}_undefined_count",
        )
    ),
)
_SUMMARY_COLUMNS = (
    "experiment",
    "level",
    "pair",
    "location",
    "track",
    "metric",
    "status",
    "reason_code",
    "reason",
    "seed",
    "base_model_id",
    "base_checkpoint",
    "base_record_provenance",
    "base_context_parallel_size",
    "base_geometry_provenance",
    "base_metric_protocol",
    "base_support_protocol",
    "base_projection_protocol",
    "base_mask_recipe_id",
    "tuned_model_id",
    "tuned_checkpoint",
    "tuned_record_provenance",
    "tuned_context_parallel_size",
    "tuned_geometry_provenance",
    "tuned_metric_protocol",
    "tuned_support_protocol",
    "tuned_projection_protocol",
    "tuned_mask_recipe_id",
    "expected_pair_count",
    "completed_pair_count",
    "scalar_eligible_pair_count",
    "variant_expected_count",
    "variant_completed_count",
    "variant_failed_count",
    "variant_missing_count",
    "paired_expected_count",
    "paired_completed_count",
    "paired_missing_count",
    "paired_failed_count",
    "paired_incompatible_count",
    "scalar_exceptional_pair_count",
    "expected_location_count",
    "complete_location_count",
    "base_mean",
    "tuned_mean",
    "delta_tuned_minus_base",
    "improvement",
)


class EuvsComparisonError(ValueError):
    """Saved inputs cannot form an honest paired comparison."""


@dataclass(frozen=True, slots=True)
class EuvsComparisonCell:
    """One exact attempt/pair cell; only an absent metric file is missing."""

    variant: str
    pair: EuvsPairSelection
    metric_result: str
    record: EuvsGen3cRunRecord | None = None
    metrics: EuvsMetricResultRecord | None = None

    @property
    def status(self) -> str:
        return "completed" if self.metrics is not None else "missing"


@dataclass(frozen=True, slots=True)
class EuvsComparisonTables:
    """Rows for the three direct comparison CSV outputs."""

    frames: tuple[tuple[object, ...], ...]
    pairs: tuple[tuple[object, ...], ...]
    summary: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class _PairMetricComparison:
    pair: EuvsPairSelection
    track: str
    metric: str
    status: str
    reason_code: str | None
    reason: str | None
    base_mean: float | None
    tuned_mean: float | None
    delta: float | None
    improvement: float | None


def compare_evaluations(
    experiment: str,
    selection: EuvsSelection,
    base_evaluation: str,
    tuned_evaluation: str,
    runs_root: Path,
) -> EuvsComparisonTables:
    """Read the exact matrix and derive equal-camera/pair/location rows."""
    cells = tuple(
        _load_cell(variant, pair, attempt, runs_root)
        for pair in selection.pairs
        for variant, attempt in (
            ("base", base_evaluation),
            ("tuned", tuned_evaluation),
        )
    )
    matrix = {(cell.pair.name, cell.variant): cell for cell in cells}
    comparisons = tuple(
        item
        for pair in selection.pairs
        for item in _compare_pair(
            pair,
            matrix[(pair.name, "base")],
            matrix[(pair.name, "tuned")],
        )
    )
    return EuvsComparisonTables(
        _frame_rows(experiment, cells),
        _pair_rows(experiment, cells),
        _summary_rows(experiment, selection, cells, comparisons),
    )


def write_euvs_comparison(
    attempt_root: Path,
    tables: EuvsComparisonTables,
) -> None:
    """Write all three tables directly to the current attempt root."""
    for name, columns, rows in (
        ("frames.csv", _FRAME_COLUMNS, tables.frames),
        ("pairs.csv", _PAIR_COLUMNS, tables.pairs),
        ("summary.csv", _SUMMARY_COLUMNS, tables.summary),
    ):
        with (attempt_root / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)


def _load_cell(
    variant: str,
    pair: EuvsPairSelection,
    evaluation: str,
    runs_root: Path,
) -> EuvsComparisonCell:
    metric_locator = (
        Path(evaluation) / "metrics" / f"{quote(pair.name, safe='')}.json"
    ).as_posix()
    try:
        metrics = load_euvs_metric_result(runs_root / metric_locator)
    except FileNotFoundError:
        return EuvsComparisonCell(variant, pair, metric_locator)
    record = load_run_record(runs_root / metrics.run_record, allow_legacy=True)
    _require_selected_pair(pair, record, metrics)
    return EuvsComparisonCell(variant, pair, metric_locator, record, metrics)


def _require_selected_pair(
    selected: EuvsPairSelection,
    record: EuvsGen3cRunRecord,
    metrics: EuvsMetricResultRecord,
) -> None:
    actual = record.pair
    if (
        actual.name,
        actual.location,
        actual.channel,
        actual.source.traversal,
        actual.source.image_tokens,
        actual.target.traversal,
        actual.target.image_tokens,
    ) != (
        selected.name,
        selected.location,
        selected.channel,
        selected.source.traversal,
        selected.source.image_tokens,
        selected.target.traversal,
        selected.target.image_tokens,
    ):
        raise EuvsComparisonError(
            f"generation record disagrees with selected pair {selected.name!r}"
        )
    if len(metrics.frames) != len(selected.target.image_tokens):
        raise EuvsComparisonError(
            f"metric frame count disagrees with selected pair {selected.name!r}"
        )


def _frame_rows(
    experiment: str,
    cells: tuple[EuvsComparisonCell, ...],
) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for cell in cells:
        if cell.status == "missing":
            continue
        record, result = _record(cell), _metrics(cell)
        for target_index, frame in enumerate(result.frames):
            source_index = record.target_source_sequence_index[target_index]
            common = (
                experiment,
                cell.variant,
                record.model.model_id,
                record.sampling.seed,
                cell.pair.name,
                cell.pair.location,
                "|".join(cell.pair.tags),
                record.model.models_relative_checkpoint,
                record.record_provenance,
                _execution_size(record),
                record.geometry.backend,
                record.geometry.provenance,
                result.metric_protocol,
                result.support_protocol,
                result.projection_protocol,
                result.mask_recipe_id,
                result.run_record,
                cell.metric_result,
                target_index,
                cell.pair.target.image_tokens[target_index],
                record.target_output_index[target_index],
                source_index,
                cell.pair.source.image_tokens[source_index],
            )
            for track in _TRACKS:
                rows.append(common + _track_row(track, getattr(frame, track)))
    return tuple(rows)


def _track_row(
    track: str,
    value: EuvsMetricTrackResult,
) -> tuple[object, ...]:
    return (
        track,
        value.support_pixel_count,
        value.raster_valid_pixel_count,
        value.support_fraction,
        value.ssim_window_count,
        value.lpips_weight_sum,
        value.dinov2_patch_weight_sum,
        *(
            field
            for metric in _METRICS
            for field in _metric_fields(getattr(value, metric))
        ),
    )


def _pair_rows(
    experiment: str,
    cells: tuple[EuvsComparisonCell, ...],
) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for cell in cells:
        record, result = cell.record, cell.metrics
        for track in _TRACKS:
            common = (
                experiment,
                cell.variant,
                record.model.model_id if record else None,
                record.sampling.seed if record else None,
                cell.pair.name,
                cell.pair.location,
                "|".join(cell.pair.tags),
                record.model.models_relative_checkpoint if record else None,
                record.record_provenance if record else None,
                _execution_size(record) if record else None,
                record.geometry.backend if record else None,
                record.geometry.provenance if record else None,
                result.metric_protocol if result else None,
                result.support_protocol if result else None,
                result.projection_protocol if result else None,
                result.mask_recipe_id if result else None,
                track,
                cell.status,
                "run_missing" if cell.status == "missing" else None,
                None,
                result.run_record if result else None,
                cell.metric_result,
                len(cell.pair.target.image_tokens),
            )
            values = (
                tuple(
                    field
                    for metric in _METRICS
                    for field in _macro_fields(
                        summarize_cameras(_metric_values(result, track, metric))
                    )
                )
                if result
                else (None,) * (len(_METRICS) * 4)
            )
            rows.append(common + values)
    return tuple(rows)


def _summary_rows(
    experiment: str,
    selection: EuvsSelection,
    cells: tuple[EuvsComparisonCell, ...],
    comparisons: tuple[_PairMetricComparison, ...],
) -> tuple[tuple[object, ...], ...]:
    matrix = {(cell.pair.name, cell.variant): cell for cell in cells}
    locations = tuple(dict.fromkeys(pair.location for pair in selection.pairs))
    variant_completed = sum(cell.status == "completed" for cell in cells)
    variant_missing = len(cells) - variant_completed
    groups = {
        pair.name: tuple(item for item in comparisons if item.pair == pair)
        for pair in selection.pairs
    }
    paired_completed = sum(
        all(item.status not in {"missing", "incompatible"} for item in group)
        for group in groups.values()
    )
    paired_missing = sum(
        any(item.status == "missing" for item in group) for group in groups.values()
    )
    paired_incompatible = sum(
        any(item.status == "incompatible" for item in group)
        for group in groups.values()
    )
    common_counts = (
        len(cells),
        variant_completed,
        0,
        variant_missing,
        len(selection.pairs),
        paired_completed,
        paired_missing,
        0,
        paired_incompatible,
    )
    pair_rows: list[tuple[object, ...]] = []
    location_rows: list[tuple[object, ...]] = []
    global_rows: list[tuple[object, ...]] = []
    for item in comparisons:
        pair_cells = (
            matrix[(item.pair.name, "base")],
            matrix[(item.pair.name, "tuned")],
        )
        pair_rows.append(
            _summary_row(
                experiment,
                "pair",
                item.pair.name,
                item.pair.location,
                item.track,
                item.metric,
                item.status,
                item.reason_code,
                item.reason,
                _uniform_seed(pair_cells),
                _identity(pair_cells),
                1,
                int(item.status not in {"missing", "incompatible"}),
                int(item.status == "completed"),
                common_counts,
                int(item.status == "exceptional"),
                len(locations),
                _complete_location_count(
                    selection, comparisons, item.track, item.metric
                ),
                item.base_mean,
                item.tuned_mean,
                item.delta,
                item.improvement,
            )
        )
    for track in _TRACKS:
        for metric in _METRICS:
            complete_locations: list[_FloatQuad] = []
            for location in locations:
                selected = tuple(
                    item
                    for item in comparisons
                    if item.pair.location == location
                    and item.track == track
                    and item.metric == metric
                )
                eligible = tuple(
                    item for item in selected if item.status == "completed"
                )
                complete = len(eligible) == len(selected)
                values: tuple[object, ...]
                if complete:
                    values = _aggregate(eligible)
                    complete_locations.append(cast(_FloatQuad, values))
                else:
                    values = (None,) * 4
                location_cells = tuple(
                    cell for cell in cells if cell.pair.location == location
                )
                location_rows.append(
                    _summary_row(
                        experiment,
                        "location",
                        None,
                        location,
                        track,
                        metric,
                        "completed" if complete else "partial",
                        None if complete else "incomplete_pair_coverage",
                        (
                            None
                            if complete
                            else "not every expected pair is scalar-eligible"
                        ),
                        _uniform_seed(location_cells),
                        _identity(location_cells),
                        len(selected),
                        sum(
                            item.status not in {"missing", "incompatible"}
                            for item in selected
                        ),
                        len(eligible),
                        common_counts,
                        sum(item.status == "exceptional" for item in selected),
                        len(locations),
                        _complete_location_count(selection, comparisons, track, metric),
                        *values,
                    )
                )
            complete = len(complete_locations) == len(locations)
            values = mean_columns(complete_locations) if complete else (None,) * 4
            selected = tuple(
                item
                for item in comparisons
                if item.track == track and item.metric == metric
            )
            global_rows.append(
                _summary_row(
                    experiment,
                    "global",
                    None,
                    None,
                    track,
                    metric,
                    "completed" if complete else "partial",
                    None if complete else "incomplete_location_coverage",
                    None if complete else "not every expected location is complete",
                    _uniform_seed(cells),
                    _identity(cells),
                    len(selection.pairs),
                    paired_completed,
                    sum(item.status == "completed" for item in selected),
                    common_counts,
                    sum(item.status == "exceptional" for item in selected),
                    len(locations),
                    len(complete_locations),
                    *values,
                )
            )
    return tuple(pair_rows + location_rows + global_rows)


def _compare_pair(
    pair: EuvsPairSelection,
    base: EuvsComparisonCell,
    tuned: EuvsComparisonCell,
) -> tuple[_PairMetricComparison, ...]:
    compatibility = None
    if base.status == "completed" and tuned.status == "completed":
        compatibility = _compatibility_error(base, tuned)
    rows: list[_PairMetricComparison] = []
    for track in _TRACKS:
        for metric in _METRICS:
            status = "missing" if "missing" in {base.status, tuned.status} else None
            reason_code = "run_missing" if status else None
            reason = "base or tuned evaluation result is missing" if status else None
            base_mean = tuned_mean = delta = improvement = None
            if status is None and compatibility is not None:
                status, reason_code, reason = (
                    "incompatible",
                    "paired_identity_mismatch",
                    compatibility,
                )
            elif status is None:
                base_values = _metric_values(_metrics(base), track, metric)
                tuned_values = _metric_values(_metrics(tuned), track, metric)
                if tuple(
                    (value.status, value.undefined_reason) for value in base_values
                ) != tuple(
                    (value.status, value.undefined_reason) for value in tuned_values
                ):
                    status, reason_code, reason = (
                        "exceptional",
                        "metric_state_mismatch",
                        "per-frame metric states differ",
                    )
                else:
                    left = summarize_cameras(base_values)
                    right = summarize_cameras(tuned_values)
                    if left.finite_mean is None or right.finite_mean is None:
                        status, reason_code, reason = (
                            "exceptional",
                            "no_finite_values",
                            "no finite paired frame values",
                        )
                    else:
                        status = "completed"
                        base_mean, tuned_mean = left.finite_mean, right.finite_mean
                        delta = tuned_mean - base_mean
                        improvement = -delta if metric in _LOWER_IS_BETTER else delta
            rows.append(
                _PairMetricComparison(
                    pair,
                    track,
                    metric,
                    status or "incompatible",
                    reason_code,
                    reason,
                    base_mean,
                    tuned_mean,
                    delta,
                    improvement,
                )
            )
    return tuple(rows)


def _compatibility_error(
    base: EuvsComparisonCell,
    tuned: EuvsComparisonCell,
) -> str | None:
    left, right = _record(base), _record(tuned)
    if left.execution is None or right.execution is None:
        return "base or tuned execution identity is unknown"
    checks = (
        ("pair", left.pair, right.pair),
        (
            "conditioning protocol",
            left.conditioning_protocol,
            right.conditioning_protocol,
        ),
        ("target output slots", left.target_output_index, right.target_output_index),
        (
            "source mapping",
            left.target_source_sequence_index,
            right.target_source_sequence_index,
        ),
        ("geometry", left.geometry, right.geometry),
        ("sampling", left.sampling, right.sampling),
        ("execution", left.execution, right.execution),
        ("output grid", left.output.shape, right.output.shape),
        ("output dtype", left.output.dtype, right.output.dtype),
        ("output color space", left.output.color_space, right.output.color_space),
    )
    for name, lhs, rhs in checks:
        if lhs != rhs:
            return f"base and tuned differ in {name}"
    base_metrics, tuned_metrics = _metrics(base), _metrics(tuned)
    for name in (
        "metric_protocol",
        "support_protocol",
        "projection_protocol",
        "mask_recipe_id",
    ):
        if getattr(base_metrics, name) != getattr(tuned_metrics, name):
            return f"base and tuned differ in {name}"
    for index, (lhs, rhs) in enumerate(
        zip(base_metrics.frames, tuned_metrics.frames, strict=True)
    ):
        for track in _TRACKS:
            left_track, right_track = getattr(lhs, track), getattr(rhs, track)
            if any(
                getattr(left_track, name) != getattr(right_track, name)
                for name in (
                    "support_pixel_count",
                    "raster_valid_pixel_count",
                    "support_fraction",
                    "ssim_window_count",
                    "lpips_weight_sum",
                    "dinov2_patch_weight_sum",
                )
            ):
                return f"base and tuned differ in frame {index} {track} support masses"
    return None


def _complete_location_count(
    selection: EuvsSelection,
    comparisons: tuple[_PairMetricComparison, ...],
    track: str,
    metric: str,
) -> int:
    count = 0
    for location in dict.fromkeys(pair.location for pair in selection.pairs):
        selected = tuple(
            item
            for item in comparisons
            if item.pair.location == location
            and item.track == track
            and item.metric == metric
        )
        count += bool(selected) and all(
            item.status == "completed" for item in selected
        )
    return count


def _aggregate(
    comparisons: tuple[_PairMetricComparison, ...],
) -> _FloatQuad:
    return cast(
        _FloatQuad,
        mean_columns(
            tuple(
                cast(
                    _FloatQuad,
                    (item.base_mean, item.tuned_mean, item.delta, item.improvement),
                )
                for item in comparisons
            )
        ),
    )


def _identity(cells: tuple[EuvsComparisonCell, ...]) -> tuple[object, ...]:
    return tuple(
        field
        for variant in ("base", "tuned")
        for field in _variant_identity(
            tuple(cell for cell in cells if cell.variant == variant)
        )
    )


def _variant_identity(cells: tuple[EuvsComparisonCell, ...]) -> tuple[object, ...]:
    records = tuple(cell.record for cell in cells if cell.record is not None)
    results = tuple(cell.metrics for cell in cells if cell.metrics is not None)
    return (
        _uniform(tuple(record.model.model_id for record in records)),
        _uniform(tuple(record.model.models_relative_checkpoint for record in records)),
        _uniform(tuple(record.record_provenance for record in records)),
        _uniform(tuple(_execution_size(record) for record in records)),
        _uniform(tuple(record.geometry.provenance for record in records)),
        _uniform(tuple(result.metric_protocol for result in results)),
        _uniform(tuple(result.support_protocol for result in results)),
        _uniform(tuple(result.projection_protocol for result in results)),
        _uniform(tuple(result.mask_recipe_id for result in results)),
    )


def _summary_row(
    experiment: str,
    level: str,
    pair: str | None,
    location: str | None,
    track: str,
    metric: str,
    status: str,
    reason_code: str | None,
    reason: str | None,
    seed: object,
    identity: tuple[object, ...],
    expected_pairs: int,
    completed_pairs: int,
    eligible_pairs: int,
    common_counts: tuple[int, ...],
    exceptional_pairs: int,
    expected_locations: int,
    complete_locations: int,
    base_mean: float | None,
    tuned_mean: float | None,
    delta: float | None,
    improvement: float | None,
) -> tuple[object, ...]:
    return (
        experiment,
        level,
        pair,
        location,
        track,
        metric,
        status,
        reason_code,
        reason,
        seed,
        *identity,
        expected_pairs,
        completed_pairs,
        eligible_pairs,
        *common_counts,
        exceptional_pairs,
        expected_locations,
        complete_locations,
        base_mean,
        tuned_mean,
        delta,
        improvement,
    )


def _uniform_seed(cells: tuple[EuvsComparisonCell, ...]) -> object:
    return _uniform(
        tuple(cell.record.sampling.seed for cell in cells if cell.record is not None)
    )


def _uniform(values: tuple[object, ...]) -> object:
    unique = tuple(dict.fromkeys(values))
    return None if not unique else unique[0] if len(unique) == 1 else "mixed"


def _execution_size(record: EuvsGen3cRunRecord) -> int | None:
    return record.execution.context_parallel_size if record.execution else None


def _metric_values(
    result: EuvsMetricResultRecord,
    track: str,
    metric: str,
) -> tuple[MetricValue, ...]:
    return tuple(getattr(getattr(frame, track), metric) for frame in result.frames)


def _metric_fields(value: MetricValue) -> tuple[object, ...]:
    return (
        value.status.value,
        value.value,
        value.undefined_reason.value if value.undefined_reason else None,
    )


def _macro_fields(value: FrameMacroMetric) -> tuple[object, ...]:
    return (
        value.finite_mean,
        value.finite_frame_count,
        value.positive_infinity_frame_count,
        value.undefined_frame_count,
    )


def _record(cell: EuvsComparisonCell) -> EuvsGen3cRunRecord:
    return cast(EuvsGen3cRunRecord, cell.record)


def _metrics(cell: EuvsComparisonCell) -> EuvsMetricResultRecord:
    return cast(EuvsMetricResultRecord, cell.metrics)


__all__ = [
    "EuvsComparisonError",
    "EuvsComparisonTables",
    "compare_evaluations",
    "write_euvs_comparison",
]

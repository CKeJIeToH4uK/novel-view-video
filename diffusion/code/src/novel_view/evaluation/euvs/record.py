"""Strict metric-result v1 JSON for one evaluated EUVS pair."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from novel_view.evaluation.euvs.support_views import (
    EUVS_SUPPORT_PROTOCOL,
)
from novel_view.evaluation.euvs.support_projection import (
    SOURCE_DYNAMIC_PROJECTION_PROTOCOL,
)
from novel_view.metrics.image import (
    IMAGE_METRIC_PROTOCOL,
    MetricStatus,
    MetricValue,
    UndefinedReason,
)

EUVS_METRIC_RESULT_SCHEMA = "novel-view/euvs-metric-result/v1"


class EuvsMetricResultError(ValueError):
    """One external metric result violates the scientific v1 contract."""


@dataclass(frozen=True, slots=True)
class EuvsMetricTrackResult:
    """Four metrics and their effective support for one frame and track."""

    support_pixel_count: int
    raster_valid_pixel_count: int
    support_fraction: float
    ssim_window_count: int
    lpips_weight_sum: float
    dinov2_patch_weight_sum: float
    psnr: MetricValue
    ssim: MetricValue
    lpips_alex: MetricValue
    dinov2_cosine: MetricValue

    def __post_init__(self) -> None:
        counts = (
            self.support_pixel_count,
            self.raster_valid_pixel_count,
            self.ssim_window_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise EuvsMetricResultError("metric counts must be non-negative")
        if self.support_pixel_count > self.raster_valid_pixel_count:
            raise EuvsMetricResultError("support exceeds raster validity")
        expected_fraction = (
            self.support_pixel_count / self.raster_valid_pixel_count
            if self.raster_valid_pixel_count
            else 0.0
        )
        if not math.isclose(
            self.support_fraction,
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise EuvsMetricResultError("support fraction disagrees with counts")
        if self.lpips_weight_sum != float(self.support_pixel_count):
            raise EuvsMetricResultError("LPIPS mass must equal support pixels")
        if (
            not math.isfinite(self.dinov2_patch_weight_sum)
            or self.dinov2_patch_weight_sum < 0.0
        ):
            raise EuvsMetricResultError(
                "DINOv2 mass must be finite and non-negative"
            )
        _metric_mass(
            self.psnr,
            float(self.support_pixel_count),
            UndefinedReason.EMPTY_SUPPORT,
            allow_infinity=True,
        )
        _metric_mass(
            self.ssim,
            float(self.ssim_window_count),
            UndefinedReason.NO_FULL_WINDOWS,
        )
        _metric_mass(
            self.lpips_alex,
            self.lpips_weight_sum,
            UndefinedReason.EMPTY_SUPPORT,
        )
        _metric_mass(
            self.dinov2_cosine,
            self.dinov2_patch_weight_sum,
            UndefinedReason.EMPTY_SUPPORT,
        )


@dataclass(frozen=True, slots=True)
class EuvsMetricFrameRecord:
    """Both support tracks for one target in recorded order."""

    target_static: EuvsMetricTrackResult
    source_aware_static: EuvsMetricTrackResult

    def __post_init__(self) -> None:
        target = self.target_static
        source = self.source_aware_static
        if target.raster_valid_pixel_count != source.raster_valid_pixel_count:
            raise EuvsMetricResultError("tracks disagree on raster population")
        if (
            source.support_pixel_count > target.support_pixel_count
            or source.ssim_window_count > target.ssim_window_count
        ):
            raise EuvsMetricResultError("source-aware support exceeds target-static")
        if source.dinov2_patch_weight_sum > target.dinov2_patch_weight_sum and (
            not math.isclose(
                source.dinov2_patch_weight_sum,
                target.dinov2_patch_weight_sum,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise EuvsMetricResultError(
                "source-aware DINOv2 mass exceeds target-static"
            )


@dataclass(frozen=True, slots=True)
class EuvsMetricResultRecord:
    """Versioned metrics linked to one runs-relative generation record."""

    run_record: str
    metric_protocol: str
    support_protocol: str
    projection_protocol: str
    mask_recipe_id: str
    frames: tuple[EuvsMetricFrameRecord, ...]

    schema = EUVS_METRIC_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if not self.run_record or not self.mask_recipe_id or not self.frames:
            raise EuvsMetricResultError(
                "record locator, recipe and frames are required"
            )
        if (
            self.metric_protocol != IMAGE_METRIC_PROTOCOL
            or self.support_protocol != EUVS_SUPPORT_PROTOCOL
            or self.projection_protocol != SOURCE_DYNAMIC_PROJECTION_PROTOCOL
        ):
            raise EuvsMetricResultError("metric/support/projection protocol mismatch")


def encode_euvs_metric_result(record: EuvsMetricResultRecord) -> bytes:
    """Encode deterministic standards-compliant metric JSON."""
    mapping = asdict(record)
    mapping["schema"] = EUVS_METRIC_RESULT_SCHEMA
    return (
        json.dumps(
            mapping,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_euvs_metric_result(
    path: Path,
    record: EuvsMetricResultRecord,
) -> Path:
    """Write directly to the caller-selected result path."""
    path.write_bytes(encode_euvs_metric_result(record))
    return path


def load_euvs_metric_result(path: Path) -> EuvsMetricResultRecord:
    """Read one exact v1 result without resolving neighbouring artifacts."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return _record(_object(value, "record"))
    except EuvsMetricResultError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EuvsMetricResultError(f"cannot decode metric result: {error}") from error


def _record(value: Mapping[str, object]) -> EuvsMetricResultRecord:
    data = _fields(
        value,
        "record",
        (
            "schema",
            "run_record",
            "metric_protocol",
            "support_protocol",
            "projection_protocol",
            "mask_recipe_id",
            "frames",
        ),
    )
    if data["schema"] != EUVS_METRIC_RESULT_SCHEMA:
        raise EuvsMetricResultError("unsupported metric-result schema")
    return EuvsMetricResultRecord(
        run_record=_text(data["run_record"], "run_record"),
        metric_protocol=_text(data["metric_protocol"], "metric_protocol"),
        support_protocol=_text(data["support_protocol"], "support_protocol"),
        projection_protocol=_text(
            data["projection_protocol"], "projection_protocol"
        ),
        mask_recipe_id=_text(data["mask_recipe_id"], "mask_recipe_id"),
        frames=tuple(
            _frame(_object(item, f"frames[{index}]"))
            for index, item in enumerate(_array(data["frames"], "frames"))
        ),
    )


def _frame(value: Mapping[str, object]) -> EuvsMetricFrameRecord:
    data = _fields(value, "frame", ("target_static", "source_aware_static"))
    return EuvsMetricFrameRecord(
        _track(data["target_static"], "target_static"),
        _track(data["source_aware_static"], "source_aware_static"),
    )


def _track(value: object, name: str) -> EuvsMetricTrackResult:
    data = _fields(
        _object(value, name),
        name,
        (
            "support_pixel_count",
            "raster_valid_pixel_count",
            "support_fraction",
            "ssim_window_count",
            "lpips_weight_sum",
            "dinov2_patch_weight_sum",
            "psnr",
            "ssim",
            "lpips_alex",
            "dinov2_cosine",
        ),
    )
    return EuvsMetricTrackResult(
        support_pixel_count=_integer(data["support_pixel_count"], name),
        raster_valid_pixel_count=_integer(
            data["raster_valid_pixel_count"], name
        ),
        support_fraction=_number(data["support_fraction"], name),
        ssim_window_count=_integer(data["ssim_window_count"], name),
        lpips_weight_sum=_number(data["lpips_weight_sum"], name),
        dinov2_patch_weight_sum=_number(
            data["dinov2_patch_weight_sum"], name
        ),
        psnr=_metric(data["psnr"], f"{name}.psnr"),
        ssim=_metric(data["ssim"], f"{name}.ssim"),
        lpips_alex=_metric(data["lpips_alex"], f"{name}.lpips_alex"),
        dinov2_cosine=_metric(
            data["dinov2_cosine"], f"{name}.dinov2_cosine"
        ),
    )


def _metric(value: object, name: str) -> MetricValue:
    data = _fields(
        _object(value, name),
        name,
        ("status", "value", "undefined_reason"),
    )
    try:
        status = MetricStatus(data["status"])
        reason = (
            None
            if data["undefined_reason"] is None
            else UndefinedReason(data["undefined_reason"])
        )
        number = (
            None if data["value"] is None else _number(data["value"], name)
        )
        return MetricValue(status, number, reason)
    except (TypeError, ValueError) as error:
        raise EuvsMetricResultError(f"invalid {name} state") from error


def _metric_mass(
    value: MetricValue,
    mass: float,
    empty_reason: UndefinedReason,
    *,
    allow_infinity: bool = False,
) -> None:
    if mass == 0.0:
        if not (
            value.status is MetricStatus.UNDEFINED
            and value.undefined_reason is empty_reason
        ):
            raise EuvsMetricResultError("zero-mass metric has a defined value")
        return
    allowed = {MetricStatus.FINITE}
    if allow_infinity:
        allowed.add(MetricStatus.POSITIVE_INFINITY)
    if value.status not in allowed:
        raise EuvsMetricResultError("positive-mass metric is undefined")


def _fields(
    value: Mapping[str, object],
    name: str,
    expected: Sequence[str],
) -> Mapping[str, object]:
    if set(value) != set(expected):
        raise EuvsMetricResultError(f"{name} has unexpected fields")
    return value


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise EuvsMetricResultError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise EuvsMetricResultError(f"{name} must be an array")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EuvsMetricResultError(f"{name} must be non-empty text")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EuvsMetricResultError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EuvsMetricResultError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise EuvsMetricResultError(f"{name} must be finite")
    return number


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EuvsMetricResultError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise EuvsMetricResultError(f"non-standard JSON number: {value}")


__all__ = [
    "EUVS_METRIC_RESULT_SCHEMA",
    "EuvsMetricFrameRecord",
    "EuvsMetricResultError",
    "EuvsMetricResultRecord",
    "EuvsMetricTrackResult",
    "encode_euvs_metric_result",
    "load_euvs_metric_result",
    "write_euvs_metric_result",
]

"""Evaluate one exact EUVS support set through CPU and perceptual metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
import tempfile

import numpy as np
import numpy.typing as npt

from novel_view.evaluation.euvs.record import (
    EuvsMetricFrameRecord,
    EuvsMetricResultRecord,
    EuvsMetricTrackResult,
)
from novel_view.evaluation.euvs.support_views import EuvsSupport
from novel_view.metrics.image import (
    IMAGE_METRIC_PROTOCOL,
    MetricValue,
    UndefinedReason,
    masked_psnr,
    reduce_ssim_map,
    ssim_map,
)
from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import run_process


class EuvsEvaluationError(RuntimeError):
    """One exact pair cannot complete its scientific evaluation."""


@dataclass(frozen=True, slots=True)
class EuvsMetricResources:
    """Literal process and model paths used by the perceptual worker."""

    scratch_root: Path
    log_path: Path
    dinov2_repository: Path
    dinov2_checkpoint: Path
    torch_home: Path
    environment_overrides: Mapping[str, str] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class _PixelTrack:
    support_count: int
    raster_count: int
    support_fraction: float
    ssim_window_count: int
    psnr: MetricValue
    ssim: MetricValue


def evaluate_pair(
    support: EuvsSupport,
    resources: EuvsMetricResources,
    run_record_locator: str,
) -> EuvsMetricResultRecord:
    """Compute two tracks and return one detached metric-result v1 record."""
    perceptual: np.ndarray | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="euvs-evaluation-",
            dir=resources.scratch_root,
        ) as temporary_directory:
            exchange = Path(temporary_directory)
            cpu_tracks = _write_exchange(support, exchange)
            _run_worker(exchange, resources)
            perceptual = np.load(
                exchange / "perceptual.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            frames = _metric_frames(support, cpu_tracks, perceptual)
    finally:
        if isinstance(perceptual, np.memmap):
            perceptual._mmap.close()
    return EuvsMetricResultRecord(
        run_record=run_record_locator,
        metric_protocol=IMAGE_METRIC_PROTOCOL,
        support_protocol=support.support_protocol,
        projection_protocol=support.projection_protocol,
        mask_recipe_id=support.mask_recipe_id,
        frames=frames,
    )


def _write_exchange(
    support: EuvsSupport,
    exchange: Path,
) -> tuple[tuple[_PixelTrack, _PixelTrack], ...]:
    first = support.frames[0].sample.prediction_rgb
    height, width = first.shape[:2]
    rgb = np.lib.format.open_memmap(
        exchange / "rgb.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(support.frames), 2, height, width, 3),
    )
    masks = np.lib.format.open_memmap(
        exchange / "support.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(support.frames), 2, height, width),
    )
    tracks: list[tuple[_PixelTrack, _PixelTrack]] = []
    try:
        for index, frame in enumerate(support.frames):
            prediction = frame.sample.prediction_rgb
            target = frame.sample.target.raster.rgb
            track_masks = (frame.views.target_static, frame.views.source_aware)
            rgb[index, 0] = prediction
            rgb[index, 1] = target
            masks[index] = track_masks
            raster_count = int(
                np.count_nonzero(frame.sample.target.raster.plan.valid_mask)
            )
            image_ssim = ssim_map(prediction, target)
            tracks.append(
                tuple(
                    _pixel_track(
                        prediction,
                        target,
                        mask,
                        raster_count,
                        image_ssim,
                    )
                    for mask in track_masks
                )
            )
        rgb.flush()
        masks.flush()
    finally:
        rgb._mmap.close()
        masks._mmap.close()
    return tuple(tracks)


def _pixel_track(
    prediction: npt.NDArray[np.uint8],
    target: npt.NDArray[np.uint8],
    support: npt.NDArray[np.bool_],
    raster_count: int,
    image_ssim: npt.NDArray[np.float64],
) -> _PixelTrack:
    support_count = int(np.count_nonzero(support))
    ssim, window_count = reduce_ssim_map(image_ssim, support)
    return _PixelTrack(
        support_count,
        raster_count,
        support_count / raster_count if raster_count else 0.0,
        window_count,
        masked_psnr(prediction, target, support),
        ssim,
    )


def _run_worker(exchange: Path, resources: EuvsMetricResources) -> None:
    run_process(
        [
            str(GEN3C_PYTHON),
            str(Path(__file__).with_name("_worker.py")),
            "--input-rgb",
            str(exchange / "rgb.npy"),
            "--input-support",
            str(exchange / "support.npy"),
            "--output-perceptual",
            str(exchange / "perceptual.npy"),
            "--dinov2-repository",
            str(resources.dinov2_repository),
            "--dinov2-checkpoint",
            str(resources.dinov2_checkpoint),
            "--torch-home",
            str(resources.torch_home),
        ],
        resources.log_path,
        "LPIPS AlexNet and DINOv2 image metrics",
        resources.environment_overrides,
    )


def _metric_frames(
    support: EuvsSupport,
    cpu_tracks: tuple[tuple[_PixelTrack, _PixelTrack], ...],
    perceptual: npt.NDArray[np.generic],
) -> tuple[EuvsMetricFrameRecord, ...]:
    frames: list[EuvsMetricFrameRecord] = []
    for frame_index, tracks in enumerate(cpu_tracks):
        completed: list[EuvsMetricTrackResult] = []
        for track_index, track in enumerate(tracks):
            lpips, dinov2, dinov2_mass = (
                float(value) for value in perceptual[frame_index, track_index]
            )
            completed.append(
                EuvsMetricTrackResult(
                    support_pixel_count=track.support_count,
                    raster_valid_pixel_count=track.raster_count,
                    support_fraction=track.support_fraction,
                    ssim_window_count=track.ssim_window_count,
                    lpips_weight_sum=float(track.support_count),
                    dinov2_patch_weight_sum=dinov2_mass,
                    psnr=track.psnr,
                    ssim=track.ssim,
                    lpips_alex=_perceptual_metric(lpips, track.support_count),
                    dinov2_cosine=_perceptual_metric(dinov2, dinov2_mass),
                )
            )
        frames.append(EuvsMetricFrameRecord(completed[0], completed[1]))
    return tuple(frames)


def _perceptual_metric(value: float, mass: float) -> MetricValue:
    if mass == 0.0:
        if math.isnan(value):
            return MetricValue.undefined(UndefinedReason.EMPTY_SUPPORT)
        raise EuvsEvaluationError("worker defined a zero-support metric")
    if not math.isfinite(value):
        raise EuvsEvaluationError("worker returned a non-finite metric")
    return MetricValue.finite(value)


__all__ = [
    "EuvsEvaluationError",
    "EuvsMetricResources",
    "evaluate_pair",
]

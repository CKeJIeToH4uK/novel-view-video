"""Run the ordered MoGe then VGGT historical depth comparison."""

from __future__ import annotations

import os
import threading
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from novel_view.models.moge.backend import MogeResources
from novel_view.models.vggt.backend import VggtOmegaResources
from novel_view.preparation.waymo_depth.metrics import measure_metric_depth
from novel_view.preparation.waymo_depth.moge import (
    MogeDepthUnavailable,
    run_moge_metric_depth,
)
from novel_view.preparation.waymo_depth.candidate_record import write_depth_outcome
from novel_view.preparation.waymo_depth.selection import (
    CANDIDATE_EXECUTION_FAILURE,
    DepthCandidateFailure,
    DepthCandidateOutcome,
    DepthCandidateReport,
    DepthClipMetrics,
)
from novel_view.preparation.waymo_depth.vggt import (
    VggtDepthUnavailable,
    run_vggt_metric_depth,
)

if TYPE_CHECKING:
    from novel_view.preparation.waymo_depth.clip import (
        DepthComparisonCamera,
        DepthComparisonClip,
    )


@dataclass(frozen=True, slots=True)
class DepthComparisonResources:
    moge: MogeResources
    vggt: VggtOmegaResources


class _PeakResidentMemory:
    """Measure one method without carrying the process lifetime high-water mark."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.peak_mib = 0.0
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._error: OSError | ValueError | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _PeakResidentMemory:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, error_type, _error, _traceback) -> bool:
        self._stop.set()
        self._thread.join()
        try:
            self._sample()
        except (OSError, ValueError) as sampling_error:
            self._error = sampling_error
        if error_type is None and self._error is not None:
            raise self._error
        return False

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._sample()
            except (OSError, ValueError) as error:
                self._error = error
                return

    def _sample(self) -> None:
        self.peak_mib = max(self.peak_mib, _read_current_rss_mib())


def compare_depth_candidates(
    clip: DepthComparisonClip,
    resources: DepthComparisonResources,
    output_directory: Path,
    log_directory: Path,
) -> tuple[DepthCandidateOutcome, DepthCandidateOutcome]:
    """Persist MoGe before starting VGGT, then return both separate outcomes."""
    moge = measure_moge(clip, resources.moge, log_directory)
    write_depth_outcome(output_directory / "moge-v1-lidar-scale.json", moge)
    vggt = measure_vggt(clip, resources.vggt, log_directory)
    write_depth_outcome(output_directory / "vggt-omega-sim3.json", vggt)
    return moge, vggt


def measure_moge(
    clip: DepthComparisonClip,
    resources: MogeResources,
    log_directory: Path,
) -> DepthCandidateOutcome:
    """Run MoGe across the three cameras or return its scientific refusal."""
    entries: list[DepthClipMetrics] = []
    try:
        with _PeakResidentMemory() as operator_memory:
            for camera in clip.cameras:
                from novel_view.preparation.waymo_depth.clip import (
                    iter_depth_comparison_rgb,
                )

                with closing(iter_depth_comparison_rgb(camera)) as rgb_frames:
                    result = run_moge_metric_depth(
                        rgb_frames,
                        camera.evidence,
                        camera.raster_plan.rectification_known,
                        camera.K_canvas,
                        camera.world_to_camera_cv,
                        resources,
                        _log_path(log_directory, "moge-v1-lidar-scale", camera),
                    )
                entries.append(
                    measure_metric_depth(
                        result.metric_depth.clip,
                        camera.evidence,
                        result.metric_depth.scale_by_frame,
                        elapsed_seconds=result.elapsed_seconds,
                        peak_ram_mib=result.worker_peak_ram_mib,
                        peak_vram_mib=result.worker_peak_vram_mib,
                    )
                )
                del result
    except MogeDepthUnavailable as error:
        return DepthCandidateFailure(
            "moge-v1-lidar-scale",
            (f"{CANDIDATE_EXECUTION_FAILURE}{error}",),
        )
    return DepthCandidateReport(
        "moge-v1-lidar-scale",
        _add_operator_memory(entries, operator_memory.peak_mib),
    )


def measure_vggt(
    clip: DepthComparisonClip,
    resources: VggtOmegaResources,
    log_directory: Path,
) -> DepthCandidateOutcome:
    """Run one clip-wide VGGT forward per camera or return its scientific refusal."""
    entries: list[DepthClipMetrics] = []
    try:
        with _PeakResidentMemory() as operator_memory:
            for camera in clip.cameras:
                from novel_view.preparation.waymo_depth.clip import (
                    iter_depth_comparison_rgb,
                )

                with closing(iter_depth_comparison_rgb(camera)) as rgb_frames:
                    result = run_vggt_metric_depth(
                        rgb_frames,
                        camera.evidence,
                        camera.raster_plan.rectification_known,
                        camera.K_canvas,
                        camera.world_to_camera_cv,
                        resources,
                        _log_path(log_directory, "vggt-omega-sim3", camera),
                    )
                scales = np.full(
                    len(camera.evidence.clip_key.frame_timestamps_micros),
                    result.metric_depth.alignment.scale_m_per_model_unit,
                    dtype=np.float64,
                )
                entries.append(
                    measure_metric_depth(
                        result.metric_depth.clip,
                        camera.evidence,
                        scales,
                        elapsed_seconds=result.elapsed_seconds,
                        peak_ram_mib=result.worker_peak_ram_mib,
                        peak_vram_mib=result.worker_peak_vram_mib,
                    )
                )
                del result, scales
    except VggtDepthUnavailable as error:
        return DepthCandidateFailure(
            "vggt-omega-sim3",
            (f"{CANDIDATE_EXECUTION_FAILURE}{error}",),
        )
    return DepthCandidateReport(
        "vggt-omega-sim3",
        _add_operator_memory(entries, operator_memory.peak_mib),
    )


def _log_path(
    directory: Path,
    backend: str,
    camera: DepthComparisonCamera,
) -> Path:
    return directory / f"{backend}__{camera.evidence.camera_name.lower()}.log"


def _add_operator_memory(
    entries: list[DepthClipMetrics],
    operator_peak_mib: float,
) -> tuple[DepthClipMetrics, ...]:
    return tuple(
        replace(entry, peak_ram_mib=entry.peak_ram_mib + operator_peak_mib)
        for entry in entries
    )


def _read_current_rss_mib() -> float:
    resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


__all__ = [
    "DepthComparisonResources",
    "compare_depth_candidates",
    "measure_moge",
    "measure_vggt",
]

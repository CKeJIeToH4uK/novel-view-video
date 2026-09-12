"""Open explicit FRONT/MoGe inputs for the historical DDW orders."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from novel_view.inputs.waymo.reader import WaymoV2FrameReader
from novel_view.models.gen3c.cache4d.backend import Gen3cCache4dResources
from novel_view.models.gen3c.environment import gen3c_environment
from novel_view.models.moge.backend import MogeResources
from novel_view.preparation.waymo_ddw.legacy.source import (
    LegacyDdwSource,
    open_legacy_ddw_source,
)
from novel_view.preparation.waymo_ddw.legacy.spec import LegacyDdwSpec
from novel_view.preparation.waymo_ddw.warp import WarpResources
from novel_view.preparation.waymo_depth.clip import (
    build_depth_comparison_clip,
    iter_depth_comparison_rgb,
)
from novel_view.preparation.waymo_depth.moge import run_moge_metric_depth
from novel_view.preparation.waymo_depth.selection import DepthGateDecision
from novel_view.preparation.waymo_depth.selection_record import read_selection_record
from novel_view.preparation.waymo_depth.keyset import WaymoClipKey
from novel_view.preparation.waymo_depth.spec import load_depth_clip_selection
from novel_view.runtime.context import RuntimeContext


@dataclass(frozen=True, slots=True)
class DdwDepthTelemetry:
    elapsed_seconds: float
    worker_peak_ram_mib: float
    worker_peak_vram_mib: float


@dataclass(frozen=True, slots=True)
class LegacyDdwMeasurement:
    source: LegacyDdwSource
    depth: DdwDepthTelemetry
    warp: WarpResources
    reference: Gen3cCache4dResources
    logs: Path


@contextmanager
def open_legacy_ddw_measurement(
    spec: LegacyDdwSpec,
    runtime: RuntimeContext,
) -> Iterator[LegacyDdwMeasurement]:
    """Resolve the selected method before reading the clip or starting MoGe."""
    selection = read_selection_record(runtime.roots.runs / spec.input.depth_selection)
    if not isinstance(selection.outcome, DepthGateDecision) or (
        selection.outcome.selected_backend != "moge-v1-lidar-scale"
    ):
        raise ValueError("legacy DDW requires a selected MoGe depth result")
    clip_key = load_depth_clip_selection(
        runtime.roots.selections.parent / spec.input.selection
    )
    with open_legacy_ddw_clip(
        clip_key, spec.recipe.moge_checkpoint, runtime, runtime.attempt_root / "logs"
    ) as measurement:
        yield measurement


@contextmanager
def open_legacy_ddw_clip(
    clip_key: WaymoClipKey,
    moge_checkpoint: str,
    runtime: RuntimeContext,
    logs: Path,
) -> Iterator[LegacyDdwMeasurement]:
    """Build one selected clip, keeping its borrowed RGB mapping alive."""
    reader = WaymoV2FrameReader(
        runtime.roots.data / "waymo",
        clip_key.official_partition,
        clip_key.segment_id,
        clip_key.start_frame_index,
        len(clip_key.frame_timestamps_micros),
    )
    logs.mkdir(parents=True, exist_ok=True)
    environment = gen3c_environment(runtime, runtime.roots.cache)
    moge = MogeResources(
        runtime.roots.cache,
        runtime.roots.models / moge_checkpoint,
        environment,
    )
    warp = WarpResources(runtime.roots.cache, environment)
    reference = Gen3cCache4dResources(
        runtime.roots.cache, Path("/opt/upstream/gen3c"), environment
    )
    with TemporaryDirectory(prefix="legacy-ddw-", dir=runtime.roots.cache) as temporary:
        clip = build_depth_comparison_clip(clip_key, reader, Path(temporary))
        camera = clip.camera("FRONT")
        depth = run_moge_metric_depth(
            iter_depth_comparison_rgb(camera),
            camera.evidence,
            camera.raster_plan.rectification_known,
            camera.K_canvas,
            camera.world_to_camera_cv,
            moge,
            logs / "moge.log",
        )
        telemetry = DdwDepthTelemetry(
            depth.elapsed_seconds, depth.worker_peak_ram_mib, depth.worker_peak_vram_mib
        )
        with open_legacy_ddw_source(camera, depth.metric_depth.clip) as source:
            yield LegacyDdwMeasurement(source, telemetry, warp, reference, logs)

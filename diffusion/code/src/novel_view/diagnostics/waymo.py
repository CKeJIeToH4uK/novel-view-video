"""Явные Waymo v2 ingress и raster diagnostics на frozen окне."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import yaml

from novel_view.preparation.waymo_depth.raster import (
    GEN3C_CANVAS_SIZE_HW,
    build_waymo_gen3c_raster_plan,
    rectify_waymo_projection_xy,
    rectify_waymo_rgb,
)
from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.inputs.waymo.reader import WaymoV2FrameReader


_EXPOSED_DEBUG = "10203656353524179475_7625_000_7645_000"
_EXPECTED_TIMELINE = 198
_START_FRAME_INDEX = 51
_FRAME_COUNT = 121
_MAX_RSS_MIB = 1024.0
_EXPECTED_CAMERAS = 5
_EXPECTED_LIDARS = 5
_BATCH_SIZES = {
    "vehicle_pose": 1,
    "camera_image": 5,
    "lidar": 5,
    "lidar_camera_projection": 5,
    "lidar_pose": 1,
}


def run_v2_ingress_main(argv: Sequence[str] | None = None) -> int:
    args = _parse_v2_ingress_args(argv)
    _require_frozen_canary(
        args.start_frame_index,
        args.count,
        args.max_rss_mib,
    )
    started = time.monotonic()
    _require_exposed_debug(args.split_config)
    reader = WaymoV2FrameReader(
        args.waymo_root,
        "validation",
        _EXPOSED_DEBUG,
        args.start_frame_index,
        args.count,
    )
    if len(reader.index.timeline) != _EXPECTED_TIMELINE:
        raise WaymoContractError(
            f"expected {_EXPECTED_TIMELINE} exposed-debug frames, "
            f"got {len(reader.index.timeline)}"
        )

    context_identity: int | None = None
    previous_timestamp: int | None = None
    frame_deltas_ms: list[float] = []
    camera_pose_offsets_ms: list[float] = []
    trigger_intervals_ms: list[float] = []
    observed = 0
    for bundle in reader:
        observed += 1
        if len(bundle.cameras) != _EXPECTED_CAMERAS:
            raise WaymoContractError("canary frame does not contain five cameras")
        if len(bundle.lidars) != _EXPECTED_LIDARS:
            raise WaymoContractError("canary frame does not contain five lidars")
        if context_identity is None:
            context_identity = id(bundle.context)
        elif id(bundle.context) != context_identity:
            raise WaymoContractError("canary reader copied its calibration context")
        timestamp = bundle.key.frame_timestamp_micros
        if previous_timestamp is not None:
            frame_deltas_ms.append((timestamp - previous_timestamp) / 1000.0)
        previous_timestamp = timestamp
        for camera in bundle.cameras:
            camera_pose_offsets_ms.append(
                (
                    camera.pose_timestamp_seconds
                    - bundle.key.frame_timestamp_micros / 1_000_000.0
                )
                * 1000.0
            )
            trigger_intervals_ms.append(
                camera.trigger_to_readout_done_interval_seconds * 1000.0
            )
        del bundle

    if observed != args.count:
        raise WaymoContractError(
            f"expected {args.count} streamed frames, got {observed}"
        )
    max_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    within_memory_limit = max_rss_mib <= args.max_rss_mib
    summary = {
        "status": "passed" if within_memory_limit else "failed_memory",
        "segment_id": _EXPOSED_DEBUG,
        "official_partition": "validation",
        "start_frame_index": args.start_frame_index,
        "count": observed,
        "timeline_count": len(reader.index.timeline),
        "batch_sizes": _BATCH_SIZES,
        "selected_row_groups": {
            component: list(reader.index.row_groups(component))
            for component in _BATCH_SIZES
        },
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_mib": max_rss_mib,
        "max_rss_limit_mib": args.max_rss_mib,
        "frame_delta_ms": _summary(frame_deltas_ms),
        "camera_pose_offset_ms": _summary(camera_pose_offsets_ms),
        "trigger_to_readout_done_ms": _summary(trigger_intervals_ms),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not within_memory_limit:
        raise WaymoContractError(
            f"R4a peak RSS {max_rss_mib:.1f} MiB exceeds frozen "
            f"limit {args.max_rss_mib:.1f} MiB"
        )
    return 0


def _parse_v2_ingress_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--start-frame-index", type=int, default=51)
    parser.add_argument("--count", type=int, default=121)
    parser.add_argument("--max-rss-mib", type=float, required=True)
    return parser.parse_args(argv)


def _require_exposed_debug(path: Path) -> None:
    if not path.is_file():
        raise WaymoContractError("split config must be an existing file")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise WaymoContractError("split config must contain a mapping")
    if document.get("schema_version") != "waymo-segment-split/v1" or (
        document.get("split_id") != "waymo-ddw-lora-v1"
    ):
        raise WaymoContractError("canary requires the frozen Waymo split contract")
    splits = document.get("splits")
    if not isinstance(splits, dict):
        raise WaymoContractError("split config lacks splits")
    memberships: list[str] = []
    for role in ("fit", "dev", "debug_subset", "exposed_debug", "transfer_test"):
        entry = splits.get(role)
        if not isinstance(entry, dict):
            raise WaymoContractError(f"split config lacks role {role!r}")
        segment_ids = entry.get("segment_ids")
        if not isinstance(segment_ids, list) or any(
            type(value) is not str for value in segment_ids
        ):
            raise WaymoContractError(f"split role {role!r} has invalid segment IDs")
        memberships.extend(role for value in segment_ids if value == _EXPOSED_DEBUG)
    exposed = splits["exposed_debug"]
    if memberships != ["exposed_debug"] or (
        exposed.get("source_partition") != "validation"
    ):
        raise WaymoContractError(
            "canary segment must occur exactly once in validation exposed_debug"
        )


def _require_frozen_canary(
    start_frame_index: int,
    count: int,
    max_rss_mib: float,
) -> None:
    if (start_frame_index, count, max_rss_mib) != (
        _START_FRAME_INDEX,
        _FRAME_COUNT,
        _MAX_RSS_MIB,
    ):
        raise WaymoContractError(
            "canary requires frozen start=51, count=121 and max RSS=1024 MiB"
        )


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


_EXPECTED_CAMERA_NAMES = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")
_CAMERA_IDS = dict(zip(_EXPECTED_CAMERA_NAMES, (1, 2, 3), strict=True))
_MIN_ROI_RETAINED = 0.75
_MIN_KNOWN = 0.995
_MIN_PROJECTION_RETAINED = 0.70


def run_raster_main(argv: Sequence[str] | None = None) -> int:
    args = _parse_raster_args(argv)
    _require_exposed_debug(args.split_config)
    summary = _check_clip(args.waymo_root)
    if tuple(summary) != _EXPECTED_CAMERA_NAMES:
        raise WaymoContractError("raster canary must report all three cameras")
    result = {
        "status": "passed",
        "segment_id": _EXPOSED_DEBUG,
        "start_frame_index": _START_FRAME_INDEX,
        "count": _FRAME_COUNT,
        "thresholds": {
            "roi_retained_fraction": _MIN_ROI_RETAINED,
            "known_fraction": _MIN_KNOWN,
            "projection_retained_fraction": _MIN_PROJECTION_RETAINED,
        },
        "cameras": summary,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _check_clip(waymo_root: Path) -> dict[str, object]:
    reader = WaymoV2FrameReader(
        waymo_root,
        "validation",
        _EXPOSED_DEBUG,
        _START_FRAME_INDEX,
        _FRAME_COUNT,
    )

    plans = None
    raw_counts = {name: 0 for name in _CAMERA_IDS}
    valid_counts = {name: 0 for name in _CAMERA_IDS}
    observed = 0
    for bundle in reader:
        observed += 1
        cameras = {camera.name: camera for camera in bundle.cameras}
        if plans is None:
            plans = {
                name: build_waymo_gen3c_raster_plan(cameras[name].calibration)
                for name in _CAMERA_IDS
            }
            for name, plan in plans.items():
                rgb = rectify_waymo_rgb(plan, cameras[name])
                if rgb.shape != (*GEN3C_CANVAS_SIZE_HW, 3):
                    raise WaymoContractError("rectified RGB has unexpected shape")
                del rgb
        for lidar in bundle.lidars:
            for lidar_return in (lidar.return1, lidar.return2):
                projection = lidar_return.camera_projection
                for name, camera_id in _CAMERA_IDS.items():
                    for offset in (0, 3):
                        selected = projection[..., offset] == camera_id
                        count = int(np.count_nonzero(selected))
                        if count == 0:
                            continue
                        xy = np.array(
                            projection[..., offset + 1 : offset + 3][selected],
                            dtype=np.float32,
                            order="C",
                            copy=True,
                        )
                        xy.setflags(write=False)
                        _, valid = rectify_waymo_projection_xy(plans[name], xy)
                        raw_counts[name] += count
                        valid_counts[name] += int(np.count_nonzero(valid))
        del bundle, cameras

    if observed != _FRAME_COUNT or plans is None:
        raise WaymoContractError(
            f"raster canary did not consume exactly {_FRAME_COUNT} frames"
        )
    camera_summaries: dict[str, dict[str, float | int | list[int]]] = {}
    for name, plan in plans.items():
        camera_summaries[name] = _camera_summary(
            plan,
            raw_counts[name],
            valid_counts[name],
        )
    return camera_summaries


def _camera_summary(plan, raw_count: int, valid_count: int):
    retained = valid_count / raw_count if raw_count else 0.0
    summary = {
        "valid_roi_xywh": list(plan.valid_roi_xywh),
        "crop_xywh": list(plan.crop_xywh),
        "roi_retained_fraction": plan.roi_retained_fraction,
        "known_fraction": plan.known_fraction,
        "projection_count": raw_count,
        "projection_retained_fraction": retained,
    }
    if (
        plan.roi_retained_fraction < _MIN_ROI_RETAINED
        or plan.known_fraction < _MIN_KNOWN
        or retained < _MIN_PROJECTION_RETAINED
    ):
        raise WaymoContractError(
            f"camera fails the frozen no-padding raster thresholds: {summary}"
        )
    return summary


def _parse_raster_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    return parser.parse_args(argv)

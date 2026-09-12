"""Build target-static and source-aware EUVS support views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from novel_view.evaluation.euvs.masks import (
    EuvsDynamicMaskSequence,
    rasterize_dynamic_mask,
)
from novel_view.evaluation.euvs.samples import EuvsSample, EuvsSamples
from novel_view.evaluation.euvs.support_projection import (
    SOURCE_DYNAMIC_PROJECTION_PROTOCOL,
    SourceDynamicProjection,
    project_source_dynamic,
)
from novel_view.generation.euvs.source_views import (
    build_source_geometry_input,
    resolve_euvs_vggt_geometry,
)
from novel_view.inputs.euvs.rgb import rasterize_euvs_source
from novel_view.inputs.nuplan.camera import build_reference_to_camera_transforms
from novel_view.models.vggt.spec import VGGT_OMEGA_BACKEND

EUVS_SUPPORT_PROTOCOL = "target-source-dynamic/v1"


@dataclass(frozen=True, slots=True, eq=False)
class EuvsSupportViews:
    """Two independent support tracks on one target raster."""

    target_static: npt.NDArray[np.bool_]
    source_aware: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True, eq=False)
class EuvsSupportFrame:
    """Two support tracks and the recorded source row for one sample."""

    sample: EuvsSample
    source_projection: SourceDynamicProjection
    views: EuvsSupportViews


@dataclass(frozen=True, slots=True, eq=False)
class EuvsSupport:
    """Ordered support tied to one exact generation sample set."""

    samples: EuvsSamples
    mask_recipe_id: str
    frames: tuple[EuvsSupportFrame, ...]

    support_protocol = EUVS_SUPPORT_PROTOCOL
    projection_protocol = SOURCE_DYNAMIC_PROJECTION_PROTOCOL


def build_support_views(
    raster_valid: npt.NDArray[np.bool_],
    target_dynamic: npt.NDArray[np.bool_],
    projected_source_dynamic: npt.NDArray[np.bool_],
) -> EuvsSupportViews:
    """Exclude target dynamics, then visible source dynamics, not disocclusion."""
    target_static = raster_valid & ~target_dynamic
    source_aware = target_static & ~projected_source_dynamic
    return EuvsSupportViews(_readonly(target_static), _readonly(source_aware))


def build_euvs_support(
    samples: EuvsSamples,
    source_masks: EuvsDynamicMaskSequence,
    target_masks: EuvsDynamicMaskSequence,
    runs_root: Path,
    *,
    allow_legacy_geometry: bool = False,
) -> EuvsSupport:
    """Resolve recorded source geometry and build both target-order tracks."""
    record = samples.record
    if record.geometry.backend != VGGT_OMEGA_BACKEND:
        raise ValueError(
            f"unsupported geometry backend: {record.geometry.backend}"
        )
    if record.geometry.provenance == "legacy-attested" and not (
        allow_legacy_geometry
    ):
        raise ValueError(
            "legacy-attested geometry requires an explicit legacy record source"
        )
    output_size_hw = samples.target.frames[0].raster.plan.output_size_hw
    source = rasterize_euvs_source(samples.pair, output_size_hw)
    resolved = resolve_euvs_vggt_geometry(
        build_source_geometry_input(source),
        runs_root / record.geometry.runs_relative_directory,
        expected_provenance=record.geometry.provenance,
    )
    target_w2c = build_reference_to_camera_transforms(
        samples.pair.source.frames[0].geometry,
        tuple(frame.geometry for frame in samples.pair.target.frames),
    )
    source_cache: dict[int, npt.NDArray[np.bool_]] = {}
    frames: list[EuvsSupportFrame] = []
    for sample in samples.frames:
        source_index = sample.source_index
        source_dynamic = source_cache.get(source_index)
        if source_dynamic is None:
            source_dynamic = rasterize_dynamic_mask(
                source_masks.frames[source_index],
                source.frames[source_index].raster.plan,
            ).dynamic_mask
            source_cache[source_index] = source_dynamic
        target_plan = sample.target.raster.plan
        target_dynamic = rasterize_dynamic_mask(
            target_masks.frames[sample.target_index],
            target_plan,
        ).dynamic_mask
        projection = project_source_dynamic(
            resolved.posed_depth.depth_z_m[source_index],
            resolved.posed_depth.geometry_valid_mask[source_index],
            source_dynamic,
            resolved.posed_depth.intrinsics[source_index],
            resolved.posed_depth.reference_to_camera[source_index],
            target_plan.output_intrinsics,
            target_w2c[sample.target_index],
            target_plan.output_size_hw,
        )
        frames.append(
            EuvsSupportFrame(
                sample,
                projection,
                build_support_views(
                    target_plan.valid_mask,
                    target_dynamic,
                    projection.visible_label_mask,
                ),
            )
        )
    return EuvsSupport(samples, source_masks.recipe_id, tuple(frames))


def _readonly(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    return np.frombuffer(mask.tobytes(order="C"), dtype=np.bool_).reshape(mask.shape)


__all__ = [
    "EUVS_SUPPORT_PROTOCOL",
    "EuvsSupport",
    "EuvsSupportFrame",
    "EuvsSupportViews",
    "build_euvs_support",
    "build_support_views",
]

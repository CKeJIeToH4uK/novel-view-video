"""Publish exact dense Gen3C outputs as native PNG handoffs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal, cast

import numpy as np
from PIL import Image

from novel_view.generation.gaussian.dense import (
    StandardCameraTable,
    load_standard_camera_table,
)
from novel_view.generation.gaussian.record import (
    GaussianRunV2,
    GaussianRunV3,
    read_gaussian_run_record,
)
from novel_view.geometry.raster import CoverCropTransform


DenseMethod = Literal["independent", "overlap21"]


@dataclass(frozen=True, slots=True)
class DenseHandoffRow:
    """One requested dense camera joined to its saved RGB output slot."""

    method: DenseMethod
    dense_camera_id: int
    chunk_index: int
    target_output_index: int
    run_record: Path
    generated_rgb: Path


@dataclass(frozen=True, slots=True, eq=False)
class DenseHandoff:
    """One method's exact records, camera table and ordered output rows."""

    method: DenseMethod
    camera_table: StandardCameraTable
    dense_transforms: Path
    dense_intrinsics: Path
    model_size_hw: tuple[int, int]
    rows: tuple[DenseHandoffRow, ...]


def select_dense_outputs(
    independent_attempt: Path,
    overlap_attempt: Path,
    dense_camera_ids: tuple[int, ...],
    *,
    scene_id: int,
    bake_dense_stride: int,
) -> tuple[DenseHandoff, DenseHandoff]:
    """Join ordered IDs to exact v2/v3 records from two named attempts."""
    _check_selection_stride(dense_camera_ids, bake_dense_stride)
    independent = _select_method(
        independent_attempt,
        "independent",
        dense_camera_ids,
        scene_id,
    )
    overlap = _select_method(
        overlap_attempt,
        "overlap21",
        dense_camera_ids,
        scene_id,
    )
    if independent.model_size_hw != overlap.model_size_hw:
        raise ValueError("dense attempts use different generated RGB rasters")
    _match_camera_tables(independent.camera_table, overlap.camera_table)
    return independent, overlap


def materialize_dense_png(
    handoff: DenseHandoff,
    output_directory: Path,
    *,
    bake_dense_stride: int,
    dense_transforms: Path,
    dense_intrinsics: Path,
) -> Path:
    """Write ordered native PNGs and their direct prepare-v1 join."""
    output_directory.mkdir(parents=True)
    transform = CoverCropTransform(
        handoff.camera_table.image_size_hw,
        handoff.model_size_hw,
    )
    entries: list[dict[str, object]] = []
    opened: dict[Path, np.memmap] = {}
    try:
        for row in handoff.rows:
            generated = opened.get(row.generated_rgb)
            if generated is None:
                generated = np.load(
                    row.generated_rgb,
                    mmap_mode="r",
                    allow_pickle=False,
                )
                opened[row.generated_rgb] = generated
            name = f"frame_{row.dense_camera_id:05d}.png"
            _to_native(
                np.asarray(generated[row.target_output_index]),
                transform,
            ).save(output_directory / name)
            entries.append(
                {
                    "image": name,
                    "dense_camera_id": row.dense_camera_id,
                    "chunk_index": row.chunk_index,
                    "run_record": os.path.relpath(row.run_record, output_directory),
                    "target_output_index": row.target_output_index,
                }
            )
    finally:
        for generated in opened.values():
            generated._mmap.close()

    record = {
        "format": "dense_gen3c_prepare",
        "format_version": 1,
        "method": handoff.method,
        "camera": handoff.camera_table.camera_name,
        "shift": 0.0,
        "image_pattern": "frame_{dense_camera_id:05d}.png",
        "frame_count": len(entries),
        "bake_dense_stride": bake_dense_stride,
        "dense_transforms": os.path.relpath(dense_transforms, output_directory),
        "dense_intrinsics": os.path.relpath(dense_intrinsics, output_directory),
        "native_size_hw": list(handoff.camera_table.image_size_hw),
        "frames": entries,
    }
    output = output_directory / "prepare.json"
    output.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _select_method(
    attempt: Path,
    method: DenseMethod,
    dense_camera_ids: tuple[int, ...],
    scene_id: int,
) -> DenseHandoff:
    first_path = attempt / "chunks/chunk-000/run.json"
    first = read_gaussian_run_record(first_path)
    expected_type = GaussianRunV2 if method == "independent" else GaussianRunV3
    if not isinstance(first, expected_type):
        raise ValueError(f"{method} attempt has the wrong Gaussian record version")
    chunk_count = cast(int, first.chunk["count"])
    identity = _record_identity(first)
    first_rgb = cast(dict[str, object], first.outputs["generated_rgb"])
    first_shape = cast(list[int], first_rgb["shape"])
    model_size_hw = (first_shape[1], first_shape[2])
    if identity[0] != scene_id:
        raise ValueError(f"{method} attempt disagrees with the selected scene")

    owners: dict[int, DenseHandoffRow] = {}
    for chunk_index in range(chunk_count):
        path = attempt / "chunks" / f"chunk-{chunk_index:03d}" / "run.json"
        record = first if chunk_index == 0 else read_gaussian_run_record(path)
        if not isinstance(record, expected_type):
            raise ValueError(f"{method} attempt mixes Gaussian record versions")
        if (
            cast(int, record.chunk["index"]) != chunk_index
            or cast(int, record.chunk["count"]) != chunk_count
            or _record_identity(record) != identity
        ):
            raise ValueError(f"{method} chunk records disagree on their identity")
        production_count = (
            len(cast(list[object], record.input["dense_camera_ids"]))
            if isinstance(record, GaussianRunV2)
            else cast(int, record.chunk["production_target_count"])
        )
        dense_ids = cast(list[int], record.input["dense_camera_ids"])
        output_slots = cast(list[int], record.input["target_output_index"])
        rgb = cast(dict[str, object], record.outputs["generated_rgb"])
        generated = path.parent / Path(cast(str, rgb["path"]))
        for dense_id, output_slot in zip(
            dense_ids[:production_count],
            output_slots[:production_count],
            strict=True,
        ):
            if dense_id in owners:
                raise ValueError(
                    f"{method} dense camera {dense_id} has two production owners"
                )
            owners[dense_id] = DenseHandoffRow(
                method,
                dense_id,
                chunk_index,
                output_slot,
                path,
                generated,
            )

    rows = []
    for dense_id in dense_camera_ids:
        try:
            rows.append(owners[dense_id])
        except KeyError as error:
            raise ValueError(
                f"{method} attempt has no production owner for dense camera {dense_id}"
            ) from error
    transforms = first_path.parent / Path(cast(str, first.input["dense_transforms"]))
    intrinsics = first_path.parent / Path(cast(str, first.input["dense_intrinsics"]))
    table = load_standard_camera_table(
        transforms,
        intrinsics,
        cast(str, first.input["dense_camera"]),
    )
    if any(row.dense_camera_id >= table.target_timestamp_ns.size for row in rows):
        raise ValueError(f"{method} dense camera ID is absent from its camera table")
    return DenseHandoff(
        method,
        table,
        transforms,
        intrinsics,
        model_size_hw,
        tuple(rows),
    )


def _record_identity(record: GaussianRunV2 | GaussianRunV3) -> tuple[object, ...]:
    fields = (
        "scene_id",
        "manifest",
        "splats_variant",
        "source_camera",
        "dense_camera",
        "dense_transforms",
        "dense_intrinsics",
    )
    return tuple(record.input[field] for field in fields)


def _match_camera_tables(
    independent: StandardCameraTable,
    overlap: StandardCameraTable,
) -> None:
    scalar_identity = (
        "scene_origin_frame",
        "source_frame_id",
        "camera_name",
        "source_timestamp_ns",
        "image_size_hw",
    )
    if any(
        getattr(independent, field) != getattr(overlap, field)
        for field in scalar_identity
    ):
        raise ValueError("dense attempts describe different camera tables")
    if (
        not np.array_equal(independent.target_timestamp_ns, overlap.target_timestamp_ns)
        or not np.array_equal(independent.intrinsics, overlap.intrinsics)
        or not np.allclose(independent.w2c, overlap.w2c, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("dense attempts describe different camera trajectories")


def _check_selection_stride(
    dense_camera_ids: tuple[int, ...],
    bake_dense_stride: int,
) -> None:
    if bake_dense_stride <= 0:
        raise ValueError("bake_dense_stride must be positive")
    if len(dense_camera_ids) > 1 and any(
        right - left != bake_dense_stride
        for left, right in zip(
            dense_camera_ids[:-1], dense_camera_ids[1:], strict=True
        )
    ):
        raise ValueError("dense handoff IDs do not match bake_dense_stride")


def _to_native(frame: np.ndarray, transform: CoverCropTransform) -> Image.Image:
    """Restore the uncropped raster and resize it to the native image size."""
    resized_height, resized_width = transform.resized_size_hw
    output_height, output_width = transform.output_size_hw
    padded = np.pad(
        frame,
        (
            (
                transform.crop_top,
                resized_height - output_height - transform.crop_top,
            ),
            (
                transform.crop_left,
                resized_width - output_width - transform.crop_left,
            ),
            (0, 0),
        ),
        mode="edge",
    )
    native_height, native_width = transform.source_size_hw
    return Image.fromarray(padded).resize(
        (native_width, native_height),
        Image.Resampling.BICUBIC,
    )


__all__ = [
    "DenseHandoff",
    "DenseHandoffRow",
    "materialize_dense_png",
    "select_dense_outputs",
]

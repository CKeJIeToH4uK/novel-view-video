"""Dense Gaussian camera tables, independent chunks, and overlap21 seams."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from novel_view.generation.gen3c.request import gen3c_diagnostic_slots
from novel_view.geometry.raster import CoverCropTransform
from novel_view.geometry.trajectory import (
    interpolate_w2c_shortest,
    translation_quaternion_xyzw_to_w2c,
    w2c_to_translation_quaternion_xyzw,
)
from novel_view.inputs.gaussian.reader import (
    GaussianExport,
    GaussianFrame,
    read_gaussian_raster,
)
from novel_view.inputs.gaussian.spec import GaussianPoseRange
from novel_view.models.gen3c.cache4d.request import (
    Gen3cConditioning,
    Gen3cContextConditioning,
    Gen3cSourceFrame,
)
from novel_view.models.gen3c.spec import GEN3C_IMAGE_SIZE_HW

if TYPE_CHECKING:
    from novel_view.generation.gen3c.session import Gen3cGenerationResult


@dataclass(frozen=True, slots=True, eq=False)
class DenseTrajectory:
    """One shortest-SO(3) trajectory with stable zero-based camera IDs."""

    densification_factor: int
    source_w2c: npt.NDArray[np.float64]
    target_w2c: npt.NDArray[np.float64]
    target_timestamp_ns: npt.NDArray[np.int64]
    source_pose_index: npt.NDArray[np.int64]

    @property
    def dense_camera_count(self) -> int:
        return int(self.target_w2c.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class StandardCameraTable:
    """One track in the reconstruction transforms/intrinsics JSON schema."""

    scene_origin_frame: str
    source_frame_id: str
    camera_name: str
    source_timestamp_ns: int
    target_timestamp_ns: npt.NDArray[np.int64]
    w2c: npt.NDArray[np.float64]
    intrinsics: npt.NDArray[np.float64]
    image_size_hw: tuple[int, int]


@dataclass(frozen=True, slots=True, eq=False)
class DenseIndependentChunk:
    """One fresh source seed and at most ``target_capacity`` owned targets."""

    chunk_index: int
    anchor_source_pose_index: int
    dense_camera_ids: npt.NDArray[np.int64]
    target_output_index: npt.NDArray[np.int64]
    target_source_pose_indices: npt.NDArray[np.int64]
    model_dense_camera_ids: npt.NDArray[np.int64]
    model_source_pose_indices: npt.NDArray[np.int64]
    source_pose_indices: npt.NDArray[np.int64]
    padding_frame_count: int

    @property
    def production_frame_count(self) -> int:
        return int(self.dense_camera_ids.size)


@dataclass(frozen=True, slots=True, eq=False)
class DenseIndependentSequence:
    """Physical source rows and one independent dense-camera schedule."""

    export: GaussianExport
    trajectory: DenseTrajectory
    chunk: DenseIndependentChunk
    raster_transform: CoverCropTransform
    output_intrinsics: npt.NDArray[np.float64]
    source_frames: tuple[GaussianFrame, ...]

    @property
    def image_size_hw(self) -> tuple[int, int]:
        return GEN3C_IMAGE_SIZE_HW

    @property
    def model_frame_count(self) -> int:
        return 1 + int(self.chunk.model_dense_camera_ids.size)

    @property
    def unpadded_frame_count(self) -> int:
        return 1 + self.chunk.production_frame_count

    @property
    def target_output_index(self) -> npt.NDArray[np.int64]:
        return self.chunk.target_output_index

    @property
    def context_depth_output_slots(self) -> None:
        return None

    @property
    def target_source_pose_indices(self) -> tuple[int, ...]:
        return tuple(
            self.export.frames[int(index)].pose_index
            for index in self.chunk.target_source_pose_indices
        )

    def __len__(self) -> int:
        return len(self.source_frames)

    def read(self, index: int) -> Gen3cSourceFrame:
        raster = read_gaussian_raster(
            self.source_frames[index],
            self.export.source_size_hw,
            self.raster_transform,
        )
        return Gen3cSourceFrame(raster.rgb, raster.depth_z_m, raster.valid)

    def build_conditioning(self) -> Gen3cConditioning:
        """Rebase this chunk to its fresh physical anchor and hold its tail."""
        return _build_dense_conditioning(
            source_rows=self,
            source_frames=self.source_frames,
            output_intrinsics=self.output_intrinsics,
            trajectory=self.trajectory,
            anchor_source_pose_index=self.chunk.anchor_source_pose_index,
            model_dense_camera_ids=self.chunk.model_dense_camera_ids,
            model_source_pose_indices=self.chunk.model_source_pose_indices,
            source_pose_indices=self.chunk.source_pose_indices,
            unpadded_frame_count=self.unpadded_frame_count,
        )


@dataclass(frozen=True, slots=True, eq=False)
class DenseOverlapChunk:
    """One fresh source seed, owned targets, and trailing context targets."""

    chunk_index: int
    anchor_source_pose_index: int
    production_dense_camera_ids: npt.NDArray[np.int64]
    production_target_output_index: npt.NDArray[np.int64]
    context_dense_camera_ids: npt.NDArray[np.int64]
    context_target_output_index: npt.NDArray[np.int64]
    model_dense_camera_ids: npt.NDArray[np.int64]
    model_source_pose_indices: npt.NDArray[np.int64]
    source_pose_indices: npt.NDArray[np.int64]
    padding_frame_count: int

    @property
    def production_frame_count(self) -> int:
        return int(self.production_dense_camera_ids.size)

    @property
    def context_frame_count(self) -> int:
        return int(self.context_dense_camera_ids.size)


@dataclass(frozen=True, slots=True, eq=False)
class DenseOverlapContext:
    """Generated RGB and metric depth handed to exactly the next chunk."""

    dense_camera_ids: npt.NDArray[np.int64]
    rgb: npt.NDArray[np.uint8]
    depth_z_m: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]

    def __len__(self) -> int:
        return int(self.dense_camera_ids.size)

    def read(self, index: int) -> Gen3cSourceFrame:
        return Gen3cSourceFrame(
            self.rgb[index],
            self.depth_z_m[index],
            self.valid[index],
        )


@dataclass(frozen=True, slots=True, eq=False)
class DenseOverlapSequence:
    """Fresh physical evidence plus the previous chunk's explicit seam."""

    export: GaussianExport
    trajectory: DenseTrajectory
    chunk: DenseOverlapChunk
    raster_transform: CoverCropTransform
    output_intrinsics: npt.NDArray[np.float64]
    source_frames: tuple[GaussianFrame, ...]
    previous_context: DenseOverlapContext | None

    @property
    def image_size_hw(self) -> tuple[int, int]:
        return GEN3C_IMAGE_SIZE_HW

    @property
    def model_frame_count(self) -> int:
        return 1 + int(self.chunk.model_dense_camera_ids.size)

    @property
    def unpadded_frame_count(self) -> int:
        return 1 + self.chunk.production_frame_count + self.chunk.context_frame_count

    @property
    def target_output_index(self) -> npt.NDArray[np.int64]:
        return self.chunk.production_target_output_index

    @property
    def context_depth_output_slots(self) -> npt.NDArray[np.int64] | None:
        if self.chunk.context_frame_count:
            return self.chunk.context_target_output_index
        return None

    @property
    def target_source_pose_indices(self) -> tuple[int, ...]:
        dense_ids = np.concatenate(
            (
                self.chunk.production_dense_camera_ids,
                self.chunk.context_dense_camera_ids,
            )
        )
        return tuple(
            self.export.frames[int(index)].pose_index
            for index in self.trajectory.source_pose_index[dense_ids]
        )

    def __len__(self) -> int:
        return len(self.source_frames)

    def read(self, index: int) -> Gen3cSourceFrame:
        raster = read_gaussian_raster(
            self.source_frames[index],
            self.export.source_size_hw,
            self.raster_transform,
        )
        return Gen3cSourceFrame(raster.rgb, raster.depth_z_m, raster.valid)

    def build_conditioning(self) -> Gen3cConditioning:
        """Add prior generated context to this chunk's fresh source plan."""
        conditioning = _build_dense_conditioning(
            source_rows=self,
            source_frames=self.source_frames,
            output_intrinsics=self.output_intrinsics,
            trajectory=self.trajectory,
            anchor_source_pose_index=self.chunk.anchor_source_pose_index,
            model_dense_camera_ids=self.chunk.model_dense_camera_ids,
            model_source_pose_indices=self.chunk.model_source_pose_indices,
            source_pose_indices=self.chunk.source_pose_indices,
            unpadded_frame_count=self.unpadded_frame_count,
        )
        context = self.previous_context
        if context is None:
            return conditioning
        expected = self.chunk.production_dense_camera_ids[: len(context)]
        if not np.array_equal(context.dense_camera_ids, expected):
            raise ValueError("overlap context does not match the current seam")
        anchor_inverse = np.linalg.inv(
            self.source_frames[
                int(np.flatnonzero(
                    self.chunk.source_pose_indices
                    == self.chunk.anchor_source_pose_index
                )[0])
            ].source_w2c
        )
        return Gen3cConditioning(
            source_rows=conditioning.source_rows,
            source_intrinsics=conditioning.source_intrinsics,
            anchor_to_source_camera=conditioning.anchor_to_source_camera,
            source_sequence_index=conditioning.source_sequence_index,
            anchor_to_query_camera=conditioning.anchor_to_query_camera,
            query_intrinsics=conditioning.query_intrinsics,
            selected_global_slots=conditioning.selected_global_slots,
            context=Gen3cContextConditioning(
                rows=context,
                anchor_to_context_camera=(
                    self.trajectory.target_w2c[context.dense_camera_ids]
                    @ anchor_inverse
                ),
                context_intrinsics=np.repeat(
                    self.output_intrinsics[None], len(context), axis=0
                ),
                context_sequence_index=np.clip(
                    np.arange(self.model_frame_count, dtype=np.int64) - 1,
                    0,
                    len(context) - 1,
                ),
            ),
        )


def select_dense_export(
    export: GaussianExport,
    pose_range: GaussianPoseRange,
) -> GaussianExport:
    """Select the exact inclusive physical-pose interval from the export."""
    frames = tuple(
        frame
        for frame in export.frames
        if pose_range.start <= frame.pose_index <= pose_range.stop
    )
    if (
        not frames
        or frames[0].pose_index != pose_range.start
        or frames[-1].pose_index != pose_range.stop
    ):
        raise ValueError(
            "Gaussian export does not contain the selected dense pose range"
        )
    return replace(export, frames=frames)


def build_dense_trajectory(
    source_w2c: npt.ArrayLike,
    target_w2c: npt.ArrayLike,
    target_timestamp_ns: npt.ArrayLike,
    *,
    shift_camera_xyz_m: tuple[float, float, float],
    densification_factor: int = 10,
) -> DenseTrajectory:
    """Interpolate adjacent cameras along their unique shortest SO(3) arcs."""
    source = np.asarray(source_w2c, dtype=np.float64)
    target = np.asarray(target_w2c, dtype=np.float64)
    timestamps = np.asarray(target_timestamp_ns, dtype=np.int64)
    if source.shape != target.shape or source.shape[0] < 2:
        raise ValueError("dense trajectory needs matching source/target cameras")
    shift = np.eye(4, dtype=np.float64)
    shift[:3, 3] = -np.asarray(shift_camera_xyz_m, dtype=np.float64)
    if not np.allclose(target, shift[None] @ source, rtol=0.0, atol=1e-9):
        raise ValueError("Gaussian target cameras disagree with their shift")

    dense_count = (source.shape[0] - 1) * densification_factor + 1
    dense_source = np.empty((dense_count, 4, 4), dtype=np.float64)
    dense_target = np.empty_like(dense_source)
    dense_timestamps = np.empty(dense_count, dtype=np.int64)
    for left in range(source.shape[0] - 1):
        start = left * densification_factor
        timestamp_delta = int(timestamps[left + 1] - timestamps[left])
        for offset in range(densification_factor):
            index = start + offset
            if offset == 0:
                dense_source[index] = source[left]
            else:
                dense_source[index] = interpolate_w2c_shortest(
                    source[left],
                    source[left + 1],
                    offset / densification_factor,
                )
            dense_target[index] = shift @ dense_source[index]
            dense_timestamps[index] = (
                int(timestamps[left])
                + _round_half_up(timestamp_delta * offset, densification_factor)
            )
    dense_source[-1] = source[-1]
    dense_target[-1] = target[-1]
    dense_timestamps[-1] = timestamps[-1]

    dense_ids = np.arange(dense_count, dtype=np.int64)
    source_indices = _round_half_up_array(dense_ids, densification_factor)
    np.minimum(source_indices, source.shape[0] - 1, out=source_indices)
    return DenseTrajectory(
        densification_factor,
        _readonly(dense_source),
        _readonly(dense_target),
        _readonly(dense_timestamps),
        _readonly(source_indices),
    )


def load_standard_camera_table(
    transforms_path: Path,
    intrinsics_path: Path,
    camera_name: str,
) -> StandardCameraTable:
    """Read one named external reconstruction camera track."""
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    rows = tuple(
        row
        for row in transforms["transforms"]
        if row["target_frame_id"] == camera_name
    )
    if not rows:
        raise ValueError(f"camera {camera_name!r} has no transform rows")
    source_frames = {str(row["source_frame_id"]) for row in rows}
    source_timestamps = {int(row["source_ts"]) for row in rows}
    if len(source_frames) != 1 or len(source_timestamps) != 1:
        raise ValueError("camera rows disagree on their source identity")
    target_timestamps = np.asarray(
        [int(row["target_ts"]) for row in rows], dtype=np.int64
    )
    if np.any(np.diff(target_timestamps) <= 0):
        raise ValueError("camera target timestamps are not strictly increasing")
    calibration = next(
        row
        for row in json.loads(intrinsics_path.read_text(encoding="utf-8"))
        if row["name"] == camera_name
    )
    intrinsics = np.asarray(
        (
            (calibration["fx"], 0.0, calibration["cx"]),
            (0.0, calibration["fy"], calibration["cy"]),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return StandardCameraTable(
        scene_origin_frame=str(transforms["scene_origin_frame"]),
        source_frame_id=source_frames.pop(),
        camera_name=camera_name,
        source_timestamp_ns=source_timestamps.pop(),
        target_timestamp_ns=_readonly(target_timestamps),
        w2c=_readonly(
            np.stack(
                [
                    translation_quaternion_xyzw_to_w2c(
                        row["transform"]["translation"],
                        row["transform"]["quaternion_xyzw"],
                    )
                    for row in rows
                ]
            )
        ),
        intrinsics=_readonly(intrinsics),
        image_size_hw=(int(calibration["height"]), int(calibration["width"])),
    )


def build_dense_camera_table(
    export: GaussianExport,
    source_transforms_path: Path,
    source_intrinsics_path: Path,
    *,
    densification_factor: int = 10,
) -> tuple[DenseTrajectory, StandardCameraTable]:
    """Bind export cameras to the external table and densify them once."""
    source_table = load_standard_camera_table(
        source_transforms_path,
        source_intrinsics_path,
        export.camera_name,
    )
    export_timestamps = np.asarray(
        [frame.timestamp_ns for frame in export.frames], dtype=np.int64
    )
    rows = np.searchsorted(source_table.target_timestamp_ns, export_timestamps)
    if (
        np.any(rows >= source_table.target_timestamp_ns.size)
        or not np.array_equal(source_table.target_timestamp_ns[rows], export_timestamps)
    ):
        raise ValueError("selected Gaussian timestamps are absent from transforms")
    source_w2c = np.stack([frame.source_w2c for frame in export.frames])
    if not np.allclose(source_table.w2c[rows], source_w2c, rtol=0.0, atol=1e-9):
        raise ValueError("Gaussian export and transforms describe different cameras")
    if source_table.image_size_hw != export.source_size_hw or not np.array_equal(
        source_table.intrinsics,
        export.intrinsics,
    ):
        raise ValueError("Gaussian export and source intrinsics disagree")
    trajectory = build_dense_trajectory(
        source_w2c,
        np.stack([frame.target_w2c for frame in export.frames]),
        export_timestamps,
        shift_camera_xyz_m=export.target_shift_camera_xyz_m,
        densification_factor=densification_factor,
    )
    return trajectory, StandardCameraTable(
        source_table.scene_origin_frame,
        source_table.source_frame_id,
        f"{export.camera_name}/{export.target_name}",
        source_table.source_timestamp_ns,
        trajectory.target_timestamp_ns,
        trajectory.target_w2c,
        source_table.intrinsics,
        source_table.image_size_hw,
    )


def write_standard_camera_table(
    table: StandardCameraTable,
    output_directory: Path,
) -> None:
    """Write the dense track directly in the existing reconstruction schema."""
    transform_rows = []
    for timestamp, w2c in zip(table.target_timestamp_ns, table.w2c, strict=True):
        translation, quaternion = w2c_to_translation_quaternion_xyzw(w2c)
        transform_rows.append(
            {
                "source_frame_id": table.source_frame_id,
                "target_frame_id": table.camera_name,
                "source_ts": table.source_timestamp_ns,
                "target_ts": int(timestamp),
                "transform": {
                    "translation": translation.tolist(),
                    "quaternion_xyzw": quaternion.tolist(),
                },
            }
        )
    height, width = table.image_size_hw
    output_directory.mkdir(parents=True)
    (output_directory / "transforms.json").write_text(
        json.dumps(
            {
                "scene_origin_frame": table.scene_origin_frame,
                "transforms": transform_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "intrinsics.json").write_text(
        json.dumps(
            [
                {
                    "name": table.camera_name,
                    "fx": float(table.intrinsics[0, 0]),
                    "fy": float(table.intrinsics[1, 1]),
                    "cx": float(table.intrinsics[0, 2]),
                    "cy": float(table.intrinsics[1, 2]),
                    "width": width,
                    "height": height,
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def plan_independent_chunks(
    trajectory: DenseTrajectory,
    *,
    target_capacity: int = 120,
) -> tuple[DenseIndependentChunk, ...]:
    """Cover every dense camera once and hold the final target for padding."""
    chunks = []
    for chunk_index, start in enumerate(
        range(0, trajectory.dense_camera_count, target_capacity)
    ):
        stop = min(start + target_capacity, trajectory.dense_camera_count)
        dense_ids = np.arange(start, stop, dtype=np.int64)
        source_indices = trajectory.source_pose_index[dense_ids].copy()
        padding = target_capacity - dense_ids.size
        model_dense_ids = np.pad(
            dense_ids,
            (0, padding),
            constant_values=int(dense_ids[-1]),
        )
        model_source_indices = np.pad(
            source_indices,
            (0, padding),
            constant_values=int(source_indices[-1]),
        )
        anchor = int(source_indices[0])
        evidence_indices = np.unique(
            np.concatenate(([anchor], model_source_indices))
        )
        chunks.append(
            DenseIndependentChunk(
                chunk_index,
                anchor,
                _readonly(dense_ids),
                _readonly(np.arange(1, dense_ids.size + 1, dtype=np.int64)),
                _readonly(source_indices),
                _readonly(model_dense_ids),
                _readonly(model_source_indices),
                _readonly(evidence_indices),
                padding,
            )
        )
    return tuple(chunks)


def plan_overlap_chunks(
    trajectory: DenseTrajectory,
    *,
    production_capacity: int = 100,
    target_capacity: int = 120,
) -> tuple[DenseOverlapChunk, ...]:
    """Cover dense targets once and expose up to one trailing context seam."""
    if target_capacity <= production_capacity:
        raise ValueError("target capacity must exceed production capacity")
    if production_capacity % trajectory.densification_factor:
        raise ValueError("overlap chunk starts must remain on source poses")
    chunks = []
    for chunk_index, start in enumerate(
        range(0, trajectory.dense_camera_count, production_capacity)
    ):
        production_stop = min(
            start + production_capacity,
            trajectory.dense_camera_count,
        )
        model_stop = min(start + target_capacity, trajectory.dense_camera_count)
        production_ids = np.arange(start, production_stop, dtype=np.int64)
        context_ids = np.arange(production_stop, model_stop, dtype=np.int64)
        real_model_ids = np.arange(start, model_stop, dtype=np.int64)
        padding = target_capacity - real_model_ids.size
        model_ids = np.pad(
            real_model_ids,
            (0, padding),
            constant_values=int(real_model_ids[-1]),
        )
        model_source_indices = trajectory.source_pose_index[model_ids].copy()
        anchor = int(trajectory.source_pose_index[start])
        evidence_indices = np.unique(
            np.concatenate(([anchor], model_source_indices))
        )
        chunks.append(
            DenseOverlapChunk(
                chunk_index,
                anchor,
                _readonly(production_ids),
                _readonly(
                    np.arange(1, production_ids.size + 1, dtype=np.int64)
                ),
                _readonly(context_ids),
                _readonly(
                    np.arange(
                        production_ids.size + 1,
                        production_ids.size + context_ids.size + 1,
                        dtype=np.int64,
                    )
                ),
                _readonly(model_ids),
                _readonly(model_source_indices),
                _readonly(evidence_indices),
                padding,
            )
        )
    return tuple(chunks)


def build_dense_independent_sequence(
    export: GaussianExport,
    trajectory: DenseTrajectory,
    chunk: DenseIndependentChunk,
) -> DenseIndependentSequence:
    """Bind one chunk to only the physical rows that its schedule references."""
    transform = CoverCropTransform(export.source_size_hw, GEN3C_IMAGE_SIZE_HW)
    return DenseIndependentSequence(
        export,
        trajectory,
        chunk,
        transform,
        transform.transform_intrinsics(export.intrinsics),
        tuple(export.frames[int(index)] for index in chunk.source_pose_indices),
    )


def build_dense_overlap_sequence(
    export: GaussianExport,
    trajectory: DenseTrajectory,
    chunk: DenseOverlapChunk,
    previous_context: DenseOverlapContext | None,
) -> DenseOverlapSequence:
    """Bind one overlap chunk to fresh physical rows and its prior seam."""
    if (chunk.chunk_index == 0) != (previous_context is None):
        raise ValueError("only the first overlap chunk may omit previous context")
    transform = CoverCropTransform(export.source_size_hw, GEN3C_IMAGE_SIZE_HW)
    return DenseOverlapSequence(
        export,
        trajectory,
        chunk,
        transform,
        transform.transform_intrinsics(export.intrinsics),
        tuple(export.frames[int(index)] for index in chunk.source_pose_indices),
        previous_context,
    )


def load_dense_overlap_context(
    sequence: DenseOverlapSequence,
    result: Gen3cGenerationResult,
) -> DenseOverlapContext:
    """Load the exact trailing RGB/depth/valid seam produced by one request."""
    generated = np.load(
        result.generated_rgb_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    try:
        rgb = np.asarray(
            generated[sequence.chunk.context_target_output_index]
        ).copy()
    finally:
        generated._mmap.close()
    depth = np.load(cast(Path, result.context_depth_path), allow_pickle=False)
    valid = np.load(cast(Path, result.context_valid_path), allow_pickle=False)
    return DenseOverlapContext(
        sequence.chunk.context_dense_camera_ids,
        _readonly(rgb),
        _readonly(depth),
        _readonly(valid),
    )


def _build_dense_conditioning(
    *,
    source_rows: DenseIndependentSequence | DenseOverlapSequence,
    source_frames: tuple[GaussianFrame, ...],
    output_intrinsics: npt.NDArray[np.float64],
    trajectory: DenseTrajectory,
    anchor_source_pose_index: int,
    model_dense_camera_ids: npt.NDArray[np.int64],
    model_source_pose_indices: npt.NDArray[np.int64],
    source_pose_indices: npt.NDArray[np.int64],
    unpadded_frame_count: int,
) -> Gen3cConditioning:
    """Build the common fresh physical layer of both dense methods."""
    source_w2c = np.stack([frame.source_w2c for frame in source_frames])
    local_index = {
        int(source_index): local
        for local, source_index in enumerate(source_pose_indices)
    }
    frame_count = 1 + int(model_dense_camera_ids.size)
    source_sequence_index = np.empty(frame_count, dtype=np.int64)
    source_sequence_index[0] = local_index[anchor_source_pose_index]
    source_sequence_index[1:] = tuple(
        local_index[int(index)] for index in model_source_pose_indices
    )
    anchor_inverse = np.linalg.inv(source_w2c[source_sequence_index[0]])
    rebased_source_w2c = source_w2c @ anchor_inverse
    query_w2c = np.empty((frame_count, 4, 4), dtype=np.float64)
    query_w2c[0] = rebased_source_w2c[source_sequence_index[0]]
    query_w2c[1:] = trajectory.target_w2c[model_dense_camera_ids] @ anchor_inverse
    return Gen3cConditioning(
        source_rows=source_rows,
        source_intrinsics=np.repeat(
            output_intrinsics[None], len(source_frames), axis=0
        ),
        anchor_to_source_camera=rebased_source_w2c,
        source_sequence_index=source_sequence_index,
        anchor_to_query_camera=query_w2c,
        query_intrinsics=np.repeat(output_intrinsics[None], frame_count, axis=0),
        selected_global_slots=gen3c_diagnostic_slots(
            frame_count,
            unpadded_frame_count,
        ),
    )


def _round_half_up(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def _round_half_up_array(
    numerator: npt.NDArray[np.int64],
    denominator: int,
) -> npt.NDArray[np.int64]:
    return (2 * numerator + denominator) // (2 * denominator)


def _readonly(values: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    values.setflags(write=False)
    return values


__all__ = [
    "DenseIndependentChunk",
    "DenseIndependentSequence",
    "DenseOverlapChunk",
    "DenseOverlapContext",
    "DenseOverlapSequence",
    "DenseTrajectory",
    "StandardCameraTable",
    "build_dense_camera_table",
    "build_dense_independent_sequence",
    "build_dense_overlap_sequence",
    "build_dense_trajectory",
    "load_standard_camera_table",
    "load_dense_overlap_context",
    "plan_independent_chunks",
    "plan_overlap_chunks",
    "select_dense_export",
    "write_standard_camera_table",
]

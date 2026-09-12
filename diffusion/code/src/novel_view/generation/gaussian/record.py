"""Read and write the existing Gaussian Gen3C run records v1-v3."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, TypeAlias, cast

import numpy as np

if TYPE_CHECKING:
    from novel_view.generation.gaussian.clips import GaussianClip
    from novel_view.generation.gaussian.dense import (
        DenseIndependentSequence,
        DenseOverlapContext,
        DenseOverlapSequence,
        StandardCameraTable,
    )
    from novel_view.generation.gaussian.full import GaussianFullSequence
    from novel_view.generation.gen3c.session import (
        Gen3cGenerationModel,
        Gen3cGenerationParameters,
        Gen3cGenerationResult,
    )

from novel_view.metrics.image import reduce_rgb_mse_psnr


_FORMAT = "gaussian_scene_gen3c_run"
_IMAGE_SIZE_HW = (704, 1280)
_RGB_FIELDS = frozenset({"path", "shape", "frame_count", "dtype"})
_VIDEO_FIELDS = frozenset({"path", "frame_count", "fps"})
_GENERATION_FIELDS = frozenset(
    {
        "model_id",
        "network_checkpoint",
        "seed",
        "prompt",
        "negative_prompt",
        "guidance",
        "num_steps",
        "context_parallel_size",
        "model_frame_count",
        "unpadded_frame_count",
        "elapsed_seconds",
        "per_rank_peak_cuda_allocated_bytes",
        "per_rank_peak_cuda_reserved_bytes",
    }
)


class GaussianRunRecordError(ValueError):
    """A saved Gaussian run record violates its existing schema."""


@dataclass(frozen=True, slots=True)
class GaussianRunV1:
    """Ordinary full sequence or one independent native clip."""

    input: Mapping[str, object]
    raster: Mapping[str, object]
    generation: Mapping[str, object]
    outputs: Mapping[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "format": _FORMAT,
            "format_version": 1,
            "input": dict(self.input),
            "raster": dict(self.raster),
            "generation": dict(self.generation),
            "outputs": dict(self.outputs),
        }


@dataclass(frozen=True, slots=True)
class GaussianRunV2:
    """One independent dense-camera chunk."""

    input: Mapping[str, object]
    chunk: Mapping[str, object]
    generation: Mapping[str, object]
    outputs: Mapping[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "format": _FORMAT,
            "format_version": 2,
            "method": "dense-independent/v1",
            "input": dict(self.input),
            "chunk": dict(self.chunk),
            "generation": dict(self.generation),
            "outputs": dict(self.outputs),
        }


@dataclass(frozen=True, slots=True)
class GaussianRunV3:
    """One dense overlap21 chunk with optional context and seam evidence."""

    input: Mapping[str, object]
    chunk: Mapping[str, object]
    generation: Mapping[str, object]
    context_depth: Mapping[str, object]
    outputs: Mapping[str, object]
    seam: Mapping[str, object] | None = None

    def to_mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
            "format": _FORMAT,
            "format_version": 3,
            "method": "dense-overlap21/v1",
            "input": dict(self.input),
            "chunk": dict(self.chunk),
            "generation": dict(self.generation),
            "context_depth": dict(self.context_depth),
            "outputs": dict(self.outputs),
        }
        if self.seam is not None:
            value["seam"] = dict(self.seam)
        return value


GaussianRunRecord: TypeAlias = GaussianRunV1 | GaussianRunV2 | GaussianRunV3


def read_gaussian_run_record(path: Path) -> GaussianRunRecord:
    """Decode one exact saved v1-v3 JSON record."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GaussianRunRecordError(f"invalid Gaussian run JSON: {path}") from error
    if not isinstance(value, dict):
        raise GaussianRunRecordError("Gaussian run record must be an object")
    return parse_gaussian_run_record(value)


def parse_gaussian_run_record(value: Mapping[str, object]) -> GaussianRunRecord:
    """Decode an already parsed exact v1-v3 record mapping."""
    if value.get("format") != _FORMAT:
        raise GaussianRunRecordError("unsupported Gaussian run format")
    version = _integer(value, "format_version", "record")
    if version == 1:
        return _read_v1(value)
    if version == 2:
        return _read_v2(value)
    if version == 3:
        return _read_v3(value)
    raise GaussianRunRecordError(f"unsupported Gaussian run version: {version}")


def write_gaussian_run_record(path: Path, record: GaussianRunRecord) -> None:
    """Write the selected existing schema directly to its requested path."""
    path.write_text(
        json.dumps(record.to_mapping(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_gaussian_run_v1(
    sequence: GaussianFullSequence,
    model: Gen3cGenerationModel,
    parameters: Gen3cGenerationParameters,
    result: Gen3cGenerationResult,
    target_video: Path,
    context_parallel_size: int,
    *,
    independent_clip: GaussianClip | None = None,
) -> GaussianRunV1:
    """Build the unchanged full-sequence or independent-clip v1 identity."""
    if model.lora_provenance is not None:
        raise GaussianRunRecordError("Gaussian run v1 cannot record LoRA identity")
    transform = sequence.raster_transform
    input_value: dict[str, object] = {
        "manifest": str(sequence.export.info_file),
        "scene_id": sequence.export.scene_id,
        "splats_variant": sequence.export.splats_variant,
        "camera": sequence.export.camera_name,
        "target": sequence.export.target_name,
        "selected_pose_indices": list(sequence.selected_pose_indices),
        "target_output_index": sequence.target_output_index.tolist(),
    }
    if independent_clip is not None:
        input_value["independent_clip"] = {
            "index": independent_clip.index,
            "count": independent_clip.count,
            "target_start_index": independent_clip.target_start_index,
            "target_stop_index": independent_clip.target_stop_index,
            "full_target_count": independent_clip.full_target_count,
        }
    return GaussianRunV1(
        input=input_value,
        raster={
            "source_size_hw": list(sequence.export.source_size_hw),
            "output_size_hw": list(sequence.image_size_hw),
            "resized_size_hw": list(transform.resized_size_hw),
            "crop_top": transform.crop_top,
            "crop_left": transform.crop_left,
        },
        generation=_generation_mapping(
            sequence, model, parameters, result, context_parallel_size
        ),
        outputs=_output_mapping(
            sequence, result, target_video,
            target_count=len(sequence), frame_rate=sequence.frame_rate,
        ),
    )


def build_gaussian_run_v2(
    sequence: DenseIndependentSequence,
    camera_table: StandardCameraTable,
    chunk_count: int,
    model: Gen3cGenerationModel,
    parameters: Gen3cGenerationParameters,
    result: Gen3cGenerationResult,
    target_video: Path,
    context_parallel_size: int,
) -> GaussianRunV2:
    """Build the existing dense-independent v2 record for one owned chunk."""
    chunk = sequence.chunk
    return GaussianRunV2(
        input=_dense_input_mapping(
            sequence, camera_table, chunk.dense_camera_ids, chunk.target_output_index
        ),
        chunk={
            "index": chunk.chunk_index,
            "count": chunk_count,
            "padding_frame_count": chunk.padding_frame_count,
            "padding_policy": "hold-last/v1",
        },
        generation=_generation_mapping(
            sequence, model, parameters, result, context_parallel_size
        ),
        outputs=_output_mapping(
            sequence, result, target_video, target_count=chunk.production_frame_count
        ),
    )


def build_gaussian_run_v3(
    sequence: DenseOverlapSequence,
    camera_table: StandardCameraTable,
    chunk_count: int,
    model: Gen3cGenerationModel,
    parameters: Gen3cGenerationParameters,
    result: Gen3cGenerationResult,
    target_video: Path,
    context_parallel_size: int,
    context_depth_model: Path,
    previous_context: DenseOverlapContext | None,
) -> GaussianRunV3:
    """Build the existing overlap21 v3 record for one sequential chunk."""
    chunk = sequence.chunk
    dense_ids = np.concatenate(
        (chunk.production_dense_camera_ids, chunk.context_dense_camera_ids)
    )
    output_slots = np.concatenate(
        (
            chunk.production_target_output_index,
            chunk.context_target_output_index,
        )
    )
    outputs = _output_mapping(
        sequence, result, target_video, target_count=chunk.production_frame_count
    )
    if chunk.context_frame_count:
        outputs["context_depth"] = {
            "depth_path": cast(Path, result.context_depth_path).name,
            "valid_path": cast(Path, result.context_valid_path).name,
            "shape": [chunk.context_frame_count, *sequence.image_size_hw],
            "alignment_scale": list(result.context_alignment_scale),
            "alignment_bias": list(result.context_alignment_bias),
            "alignment_mae_m": list(result.context_alignment_mae_m),
            "valid_fraction": list(result.context_valid_fraction),
        }
    return GaussianRunV3(
        input=_dense_input_mapping(sequence, camera_table, dense_ids, output_slots),
        chunk={
            "index": chunk.chunk_index,
            "count": chunk_count,
            "production_target_count": chunk.production_frame_count,
            "context_target_count": chunk.context_frame_count,
            "padding_frame_count": chunk.padding_frame_count,
            "padding_policy": "hold-last/v1",
            "previous_record": (
                None
                if chunk.chunk_index == 0
                else f"../chunk-{chunk.chunk_index - 1:03d}/run.json"
            ),
        },
        generation=_generation_mapping(
            sequence, model, parameters, result, context_parallel_size
        ),
        context_depth={
            "backend": "Ruicheng/moge-vitl",
            "model": str(context_depth_model),
            "alignment": "official-rigid-inverse-depth/v1",
            "max_depth_m": 100.0,
            "local_filter": {"window_size": 5, "ratio_threshold": 0.05},
            "neighbour_threshold": {"absolute_m": 0.10, "relative": 0.05},
        },
        outputs=outputs,
        seam=(
            None
            if previous_context is None
            else _overlap_seam(
                previous_context,
                result.generated_rgb_path,
                chunk.production_target_output_index[: len(previous_context)],
            )
        ),
    )


def _generation_mapping(
    sequence: GaussianFullSequence | DenseIndependentSequence | DenseOverlapSequence,
    model: Gen3cGenerationModel,
    parameters: Gen3cGenerationParameters,
    result: Gen3cGenerationResult,
    context_parallel_size: int,
) -> dict[str, object]:
    """Project the shared realized model and sampling fields."""
    generation: dict[str, object] = {
        "model_id": model.model_id,
        "network_checkpoint": str(model.network_checkpoint),
        "seed": parameters.seed,
        "prompt": parameters.prompt,
        "negative_prompt": parameters.negative_prompt,
        "guidance": parameters.guidance,
        "num_steps": parameters.num_steps,
        "context_parallel_size": context_parallel_size,
        "model_frame_count": sequence.model_frame_count,
        "unpadded_frame_count": sequence.unpadded_frame_count,
        "elapsed_seconds": result.elapsed_seconds,
        "per_rank_peak_cuda_allocated_bytes": list(
            result.per_rank_peak_cuda_allocated_bytes
        ),
        "per_rank_peak_cuda_reserved_bytes": list(
            result.per_rank_peak_cuda_reserved_bytes
        ),
    }
    if model.lora_provenance is not None:
        generation["lora"] = {
            "working_manifest": str(model.lora_provenance.working_manifest),
            "evidence_path": str(model.lora_provenance.evidence_path),
            "epochs": model.lora_provenance.epochs,
            "strength": model.lora_weights.strength,
        }
    return generation


def _output_mapping(
    sequence: GaussianFullSequence | DenseIndependentSequence | DenseOverlapSequence,
    result: Gen3cGenerationResult,
    target_video: Path,
    *,
    target_count: int,
    frame_rate: int = 24,
) -> dict[str, object]:
    """Describe the model RGB array and target-only video."""
    return {
        "generated_rgb": {
            "path": result.generated_rgb_path.name,
            "shape": [sequence.model_frame_count, *sequence.image_size_hw, 3],
            "frame_count": sequence.model_frame_count,
            "dtype": "uint8",
        },
        "target_video": {
            "path": target_video.name,
            "frame_count": target_count,
            "fps": frame_rate,
        },
    }


def _dense_input_mapping(
    sequence: DenseIndependentSequence | DenseOverlapSequence,
    camera_table: StandardCameraTable,
    dense_ids: np.ndarray,
    output_slots: np.ndarray,
) -> dict[str, object]:
    """Describe dense cameras in the caller's production/context order."""
    return {
        "manifest": str(sequence.export.info_file),
        "scene_id": sequence.export.scene_id,
        "splats_variant": sequence.export.splats_variant,
        "source_camera": sequence.export.camera_name,
        "dense_camera": camera_table.camera_name,
        "dense_transforms": "../../dense-camera-table/transforms.json",
        "dense_intrinsics": "../../dense-camera-table/intrinsics.json",
        "dense_camera_ids": dense_ids.tolist(),
        "target_output_index": output_slots.tolist(),
        "target_source_pose_indices": list(sequence.target_source_pose_indices),
    }


def _read_v1(value: Mapping[str, object]) -> GaussianRunV1:
    _fields(
        value,
        {"format", "format_version", "input", "raster", "generation", "outputs"},
        set(),
        "record v1",
    )
    input_value = _section(
        value,
        "input",
        {
            "manifest",
            "scene_id",
            "splats_variant",
            "camera",
            "target",
            "selected_pose_indices",
            "target_output_index",
        },
        {"independent_clip"},
    )
    clip = None
    if "independent_clip" in input_value:
        clip = _section(
            input_value,
            "independent_clip",
            {
                "index",
                "count",
                "target_start_index",
                "target_stop_index",
                "full_target_count",
            },
            set(),
        )
    raster = _section(
        value,
        "raster",
        {
            "source_size_hw",
            "output_size_hw",
            "resized_size_hw",
            "crop_top",
            "crop_left",
        },
        set(),
    )
    generation = _section(value, "generation", set(_GENERATION_FIELDS), set())
    outputs = _outputs(value, allow_context=False)
    slots = _common_identity(input_value, generation, outputs)
    poses = _integers(input_value, "selected_pose_indices", "input")
    if len(poses) != len(slots):
        raise GaussianRunRecordError("selected poses and target slots differ")
    if clip is not None:
        index = _integer(clip, "index", "independent_clip")
        count = _integer(clip, "count", "independent_clip")
        start = _integer(clip, "target_start_index", "independent_clip")
        stop = _integer(clip, "target_stop_index", "independent_clip")
        total = _integer(clip, "full_target_count", "independent_clip")
        if not (
            1 <= index <= count
            and 0 <= start < stop <= total
            and stop - start == len(slots)
        ):
            raise GaussianRunRecordError("independent clip identity is inconsistent")
    if tuple(_integers(raster, "output_size_hw", "raster")) != _IMAGE_SIZE_HW:
        raise GaussianRunRecordError("Gaussian raster output must be 704x1280")
    video_count = _integer(
        _mapping(outputs, "target_video"), "frame_count", "video"
    )
    if video_count != len(slots):
        raise GaussianRunRecordError("target video count differs from target slots")
    return GaussianRunV1(input_value, raster, generation, outputs)


def _read_v2(value: Mapping[str, object]) -> GaussianRunV2:
    _fields(
        value,
        {
            "format",
            "format_version",
            "method",
            "input",
            "chunk",
            "generation",
            "outputs",
        },
        set(),
        "record v2",
    )
    if value["method"] != "dense-independent/v1":
        raise GaussianRunRecordError("unsupported Gaussian v2 method")
    input_value = _dense_input(value)
    chunk = _section(
        value,
        "chunk",
        {"index", "count", "padding_frame_count", "padding_policy"},
        set(),
    )
    generation = _generation(value)
    outputs = _outputs(value, allow_context=False)
    slots = _common_identity(input_value, generation, outputs)
    _dense_identity(input_value, slots)
    video_count = _integer(
        _mapping(outputs, "target_video"), "frame_count", "video"
    )
    if video_count != len(slots):
        raise GaussianRunRecordError("dense v2 video count differs from target slots")
    return GaussianRunV2(input_value, chunk, generation, outputs)


def _read_v3(value: Mapping[str, object]) -> GaussianRunV3:
    _fields(
        value,
        {
            "format",
            "format_version",
            "method",
            "input",
            "chunk",
            "generation",
            "context_depth",
            "outputs",
        },
        {"seam"},
        "record v3",
    )
    if value["method"] != "dense-overlap21/v1":
        raise GaussianRunRecordError("unsupported Gaussian v3 method")
    input_value = _dense_input(value)
    chunk = _section(
        value,
        "chunk",
        {
            "index",
            "count",
            "production_target_count",
            "context_target_count",
            "padding_frame_count",
            "padding_policy",
            "previous_record",
        },
        set(),
    )
    generation = _generation(value)
    context = _context_depth(value)
    outputs = _outputs(value, allow_context=True)
    slots = _common_identity(input_value, generation, outputs)
    _dense_identity(input_value, slots)
    production = _integer(chunk, "production_target_count", "chunk")
    context_count = _integer(chunk, "context_target_count", "chunk")
    if production + context_count != len(slots):
        raise GaussianRunRecordError("dense v3 chunk counts differ from target slots")
    video_count = _integer(
        _mapping(outputs, "target_video"), "frame_count", "video"
    )
    if video_count != production:
        raise GaussianRunRecordError("dense v3 video count differs from production")
    seam = None if "seam" not in value else _seam(value)
    return GaussianRunV3(input_value, chunk, generation, context, outputs, seam)


def _dense_input(value: Mapping[str, object]) -> dict[str, object]:
    return _section(
        value,
        "input",
        {
            "manifest",
            "scene_id",
            "splats_variant",
            "source_camera",
            "dense_camera",
            "dense_transforms",
            "dense_intrinsics",
            "dense_camera_ids",
            "target_output_index",
            "target_source_pose_indices",
        },
        set(),
    )


def _generation(value: Mapping[str, object]) -> dict[str, object]:
    generation = _section(value, "generation", set(_GENERATION_FIELDS), {"lora"})
    if "lora" in generation:
        _section(
            generation,
            "lora",
            {"working_manifest", "evidence_path", "epochs", "strength"},
            set(),
        )
    return generation


def _outputs(value: Mapping[str, object], *, allow_context: bool) -> dict[str, object]:
    outputs = _section(
        value,
        "outputs",
        {"generated_rgb", "target_video"},
        {"context_depth"} if allow_context else set(),
    )
    _section(outputs, "generated_rgb", set(_RGB_FIELDS), set())
    _section(outputs, "target_video", set(_VIDEO_FIELDS), set())
    if "context_depth" in outputs:
        _section(
            outputs,
            "context_depth",
            {
                "depth_path",
                "valid_path",
                "shape",
                "alignment_scale",
                "alignment_bias",
                "alignment_mae_m",
                "valid_fraction",
            },
            set(),
        )
    return outputs


def _context_depth(value: Mapping[str, object]) -> dict[str, object]:
    context = _section(
        value,
        "context_depth",
        {
            "backend",
            "model",
            "alignment",
            "max_depth_m",
            "local_filter",
            "neighbour_threshold",
        },
        set(),
    )
    _section(context, "local_filter", {"window_size", "ratio_threshold"}, set())
    _section(
        context,
        "neighbour_threshold",
        {"absolute_m", "relative"},
        set(),
    )
    return context


def _seam(value: Mapping[str, object]) -> dict[str, object]:
    seam = _section(
        value,
        "seam",
        {"previous_context_vs_current_production"},
        set(),
    )
    _section(
        seam,
        "previous_context_vs_current_production",
        {
            "dense_camera_ids",
            "frame_rgb_mse",
            "frame_rgb_psnr",
            "pooled_rgb_mse",
            "pooled_rgb_psnr",
            "null_psnr_means_positive_infinity",
        },
        set(),
    )
    return seam


def _overlap_seam(
    previous: DenseOverlapContext,
    generated_path: Path,
    current_slots: np.ndarray,
) -> dict[str, object]:
    """Compare the inherited seam with the same newly generated cameras."""
    generated = np.load(generated_path, mmap_mode="r", allow_pickle=False)
    try:
        current = np.asarray(generated[current_slots]).copy()
    finally:
        generated._mmap.close()
    frames = tuple(
        reduce_rgb_mse_psnr(current[index], previous.rgb[index])
        for index in range(len(previous))
    )
    pooled_mse, pooled_psnr = reduce_rgb_mse_psnr(current, previous.rgb)
    return {
        "previous_context_vs_current_production": {
            "dense_camera_ids": previous.dense_camera_ids.tolist(),
            "frame_rgb_mse": [value[0] for value in frames],
            "frame_rgb_psnr": [value[1] for value in frames],
            "pooled_rgb_mse": pooled_mse,
            "pooled_rgb_psnr": pooled_psnr,
            "null_psnr_means_positive_infinity": True,
        }
    }


def _common_identity(
    input_value: Mapping[str, object],
    generation: Mapping[str, object],
    outputs: Mapping[str, object],
) -> tuple[int, ...]:
    slots = _integers(input_value, "target_output_index", "input")
    model_count = _integer(generation, "model_frame_count", "generation")
    unpadded = _integer(generation, "unpadded_frame_count", "generation")
    if (
        not slots
        or slots != tuple(range(1, len(slots) + 1))
        or unpadded != len(slots) + 1
        or model_count < unpadded
    ):
        raise GaussianRunRecordError("invalid Gaussian target/model slot identity")
    rgb = _mapping(outputs, "generated_rgb")
    shape = _integers(rgb, "shape", "generated_rgb")
    if (
        shape != (model_count, *_IMAGE_SIZE_HW, 3)
        or _integer(rgb, "frame_count", "generated_rgb") != model_count
        or rgb.get("dtype") != "uint8"
    ):
        raise GaussianRunRecordError("generated RGB descriptor is inconsistent")
    video = _mapping(outputs, "target_video")
    if _integer(video, "fps", "target_video") != 24:
        raise GaussianRunRecordError("Gaussian target video must be 24 fps")
    cp = _integer(generation, "context_parallel_size", "generation")
    for field in (
        "per_rank_peak_cuda_allocated_bytes",
        "per_rank_peak_cuda_reserved_bytes",
    ):
        if len(_integers(generation, field, "generation")) != cp:
            raise GaussianRunRecordError(f"{field} differs from CP size")
    elapsed = _number(generation, "elapsed_seconds", "generation")
    if not math.isfinite(elapsed) or elapsed < 0:
        raise GaussianRunRecordError("elapsed_seconds must be finite and non-negative")
    return slots


def _dense_identity(input_value: Mapping[str, object], slots: tuple[int, ...]) -> None:
    dense_ids = _integers(input_value, "dense_camera_ids", "input")
    source_poses = _integers(input_value, "target_source_pose_indices", "input")
    if len(dense_ids) != len(slots) or len(source_poses) != len(slots):
        raise GaussianRunRecordError("dense IDs, source poses and slots differ")


def _section(
    values: Mapping[str, object],
    field: str,
    required: set[str],
    optional: set[str],
) -> dict[str, object]:
    result = _mapping(values, field)
    _fields(result, required, optional, field)
    return result


def _fields(
    values: Mapping[str, object],
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = required - values.keys()
    unknown = values.keys() - required - optional
    if missing or unknown:
        raise GaussianRunRecordError(
            f"{context} fields differ: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _mapping(values: Mapping[str, object], field: str) -> dict[str, object]:
    value = values[field]
    if not isinstance(value, dict):
        raise GaussianRunRecordError(f"{field} must be an object")
    return dict(value)


def _integer(values: Mapping[str, object], field: str, context: str) -> int:
    value = values[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise GaussianRunRecordError(f"{context}.{field} must be an integer")
    return value


def _integers(
    values: Mapping[str, object],
    field: str,
    context: str,
) -> tuple[int, ...]:
    value = values[field]
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise GaussianRunRecordError(f"{context}.{field} must be integer list")
    return tuple(value)


def _number(values: Mapping[str, object], field: str, context: str) -> float:
    value = values[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GaussianRunRecordError(f"{context}.{field} must be numeric")
    return float(value)


__all__ = [
    "GaussianRunRecord",
    "GaussianRunRecordError",
    "GaussianRunV1",
    "GaussianRunV2",
    "GaussianRunV3",
    "build_gaussian_run_v1",
    "build_gaussian_run_v2",
    "build_gaussian_run_v3",
    "parse_gaussian_run_record",
    "read_gaussian_run_record",
    "write_gaussian_run_record",
]

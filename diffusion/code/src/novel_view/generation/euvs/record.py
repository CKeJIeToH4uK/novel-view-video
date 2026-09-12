"""Read and write the strict EUVS generation record schemas v1-v5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import cast

from novel_view.inputs.euvs.spec import EuvsFrameSelection, EuvsPairSelection
from novel_view.generation.gen3c.windows import (
    WINDOW_SEED_AUTOREGRESSIVE,
    WINDOW_SEED_POLICIES,
)

LEGACY_RUN_RECORD_SCHEMA = "novel-view/euvs-gen3c-run/v1"
RUN_RECORD_SCHEMA = "novel-view/euvs-gen3c-run/v2"
WINDOW_SEED_RUN_RECORD_SCHEMA = "novel-view/euvs-gen3c-run/v3"
LORA_RUN_RECORD_SCHEMA = "novel-view/euvs-gen3c-run/v4"
LORA_STRENGTH_RUN_RECORD_SCHEMA = "novel-view/euvs-gen3c-run/v5"
GEN3C_CONDITIONING_PROTOCOL = "direct-projected-nearest-single-source/v1"

_GEN3C_HEIGHT = 704
_GEN3C_WIDTH = 1280
_GEN3C_WINDOW_STEP = 120
_RECORD_PROVENANCE = {"native", "legacy-attested"}
_GEOMETRY_PROVENANCE = {"ordered-source-tokens", "legacy-attested"}


class Gen3cRunRecordError(ValueError):
    """An EUVS generation record violates its saved schema."""


@dataclass(frozen=True, slots=True)
class Gen3cRunGeometry:
    """Portable saved-geometry identity.

    ``provenance`` describes how ordered source tokens were attested, not a
    backend-specific geometry format. New backends may therefore use the same
    native provenance without changing the run-record schema.
    """

    backend: str
    runs_relative_directory: str
    provenance: str

    def __post_init__(self) -> None:
        _identifier(self.backend, "geometry.backend")
        _relative_locator(
            self.runs_relative_directory,
            "geometry.runs_relative_directory",
        )
        provenance = _text(self.provenance, "geometry.provenance")
        if provenance not in _GEOMETRY_PROVENANCE:
            raise Gen3cRunRecordError(
                f"unsupported geometry provenance: {self.provenance!r}"
            )


@dataclass(frozen=True, slots=True)
class Gen3cRunModel:
    """Named Gen3C network below ``models_root``."""

    model_id: str
    models_relative_checkpoint: str
    lora_working_manifest: str | None = None
    lora_evidence_path: str | None = None
    lora_epochs: int | None = None
    lora_strength: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.model_id, "model.model_id")
        _relative_locator(
            self.models_relative_checkpoint,
            "model.models_relative_checkpoint",
        )
        lora_identity = (
            self.lora_working_manifest,
            self.lora_evidence_path,
            self.lora_epochs,
        )
        if any(value is not None for value in lora_identity) and not all(
            value is not None for value in lora_identity
        ):
            raise Gen3cRunRecordError(
                "model LoRA manifest, evidence and epochs must be supplied together"
            )
        if self.lora_working_manifest is not None:
            _text(self.lora_working_manifest, "model.lora.working_manifest")
            _text(self.lora_evidence_path, "model.lora.evidence_path")
            epochs = _integer(self.lora_epochs, "model.lora.epochs")
            if epochs <= 0:
                raise Gen3cRunRecordError("model.lora.epochs must be positive")
        if self.lora_strength is not None:
            if self.lora_working_manifest is None:
                raise Gen3cRunRecordError(
                    "model.lora.strength requires LoRA identity"
                )
            strength = _number(self.lora_strength, "model.lora.strength")
            if not math.isfinite(strength) or strength < 0:
                raise Gen3cRunRecordError(
                    "model.lora.strength must be finite and non-negative"
                )
            object.__setattr__(self, "lora_strength", strength)


@dataclass(frozen=True, slots=True)
class Gen3cRunSampling:
    """Realized sampling parameters independent of the model runner."""

    seed: int
    prompt: str
    negative_prompt: str
    guidance: float
    num_steps: int
    window_seed_policy: str = WINDOW_SEED_AUTOREGRESSIVE

    def __post_init__(self) -> None:
        seed = _integer(self.seed, "sampling.seed")
        steps = _integer(self.num_steps, "sampling.num_steps")
        guidance = _number(self.guidance, "sampling.guidance")
        _text(self.prompt, "sampling.prompt", empty=True)
        _text(self.negative_prompt, "sampling.negative_prompt", empty=True)
        if (
            seed < 0
            or seed >= 2**63
            or steps <= 0
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise Gen3cRunRecordError(
                "sampling requires a non-negative seed/guidance and "
                "positive steps"
            )
        if (
            not isinstance(self.window_seed_policy, str)
            or self.window_seed_policy not in WINDOW_SEED_POLICIES
        ):
            raise Gen3cRunRecordError(
                "sampling.window_seed_policy must be one of "
                f"{sorted(WINDOW_SEED_POLICIES)}"
            )
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "guidance", guidance)
        object.__setattr__(self, "num_steps", steps)


@dataclass(frozen=True, slots=True)
class Gen3cRunExecution:
    """Output-affecting distributed topology of one realized run."""

    context_parallel_size: int

    def __post_init__(self) -> None:
        size = _integer(
            self.context_parallel_size,
            "execution.context_parallel_size",
        )
        if size not in (1, 2):
            raise Gen3cRunRecordError(
                "execution.context_parallel_size must be 1 or 2"
            )
        object.__setattr__(self, "context_parallel_size", size)


@dataclass(frozen=True, slots=True)
class Gen3cRunOutput:
    """Lossless sibling RGB array descriptor."""

    file: str
    shape: tuple[int, int, int, int]
    dtype: str = "uint8"
    color_space: str = "RGB"

    def __post_init__(self) -> None:
        _text(self.file, "output.file")
        name = PurePosixPath(self.file)
        if (
            name.name != self.file
            or name.suffix != ".npy"
            or "\\" in self.file
        ):
            raise Gen3cRunRecordError(
                "output.file must be one sibling .npy basename"
            )
        shape = _integers(self.shape, "output.shape")
        if len(shape) != 4 or any(value <= 0 for value in shape):
            raise Gen3cRunRecordError(
                "output.shape must contain four positive integers"
            )
        if shape[1:] != (_GEN3C_HEIGHT, _GEN3C_WIDTH, 3):
            raise Gen3cRunRecordError(
                f"output.shape must be (*, {_GEN3C_HEIGHT}, {_GEN3C_WIDTH}, 3)"
            )
        dtype = _text(self.dtype, "output.dtype")
        color_space = _text(self.color_space, "output.color_space")
        if dtype != "uint8" or color_space != "RGB":
            raise Gen3cRunRecordError("output must be uint8 RGB")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True, slots=True)
class EuvsGen3cRunRecord:
    """Strict portable identity of one published EUVS Gen3C result.

    ``execution=None`` represents a loaded v1 record whose context-parallel
    topology was never recorded. New native records always carry execution
    identity. Autoregressive records encode as v2; records with an explicit
    non-default window seed policy encode as v3. Historical LoRA records
    without inference strength remain v4; new LoRA records encode as v5.
    """

    record_provenance: str
    experiment_name: str
    pair: EuvsPairSelection
    conditioning_protocol: str
    target_output_index: tuple[int, ...]
    target_source_sequence_index: tuple[int, ...]
    geometry: Gen3cRunGeometry
    model: Gen3cRunModel
    sampling: Gen3cRunSampling
    execution: Gen3cRunExecution | None
    output: Gen3cRunOutput

    @property
    def schema(self) -> str:
        """Return the schema implied by execution and window seed identity."""
        if self.execution is None:
            return LEGACY_RUN_RECORD_SCHEMA
        if self.model.lora_working_manifest is not None:
            return (
                LORA_STRENGTH_RUN_RECORD_SCHEMA
                if self.model.lora_strength is not None
                else LORA_RUN_RECORD_SCHEMA
            )
        if self.sampling.window_seed_policy != WINDOW_SEED_AUTOREGRESSIVE:
            return WINDOW_SEED_RUN_RECORD_SCHEMA
        return RUN_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.execution is None and (
            self.sampling.window_seed_policy != WINDOW_SEED_AUTOREGRESSIVE
        ):
            raise Gen3cRunRecordError(
                "legacy records cannot declare a window seed policy"
            )
        record_provenance = _text(
            self.record_provenance,
            "record_provenance",
        )
        if record_provenance not in _RECORD_PROVENANCE:
            raise Gen3cRunRecordError(
                f"unsupported record provenance: {self.record_provenance!r}"
            )
        _identifier(self.experiment_name, "experiment")
        pair = _canonical_pair(self.pair)
        conditioning_protocol = _text(
            self.conditioning_protocol,
            "conditioning.protocol",
        )
        if conditioning_protocol != GEN3C_CONDITIONING_PROTOCOL:
            raise Gen3cRunRecordError(
                f"unsupported conditioning protocol: "
                f"{self.conditioning_protocol!r}"
            )

        slots = _integers(self.target_output_index, "target_output_index")
        sources = _integers(
            self.target_source_sequence_index,
            "target_source_sequence_index",
        )
        target_count = len(pair.target.image_tokens)
        if len(slots) != target_count or len(sources) != target_count:
            raise Gen3cRunRecordError(
                "conditioning arrays must have one row per target token"
            )
        if slots[0] != 1 or any(
            right <= left for left, right in zip(slots, slots[1:])
        ):
            raise Gen3cRunRecordError(
                "target output indices must start at 1 and increase strictly"
            )
        expected_frames = (
            ((slots[-1] + _GEN3C_WINDOW_STEP - 1) // _GEN3C_WINDOW_STEP)
            * _GEN3C_WINDOW_STEP
            + 1
        )
        if self.output.shape[0] != expected_frames:
            raise Gen3cRunRecordError(
                "output frame count must be the minimal compatible "
                "120*k+1 length"
            )
        source_count = len(pair.source.image_tokens)
        if any(index < 0 or index >= source_count for index in sources):
            raise Gen3cRunRecordError(
                "target source index lies outside ordered source frames"
            )
        object.__setattr__(self, "pair", pair)
        object.__setattr__(self, "target_output_index", slots)
        object.__setattr__(self, "target_source_sequence_index", sources)


def build_euvs_gen3c_run_record(
    *,
    experiment_name: str,
    pair: EuvsPairSelection,
    target_output_index: Sequence[int],
    target_source_sequence_index: Sequence[int],
    geometry_backend: str,
    geometry_directory: Path,
    geometry_provenance: str,
    model_id: str,
    network_checkpoint: Path,
    lora_working_manifest: Path | None = None,
    lora_evidence_path: Path | None = None,
    lora_epochs: int | None = None,
    lora_strength: float | None = None,
    sampling: Gen3cRunSampling,
    context_parallel_size: int,
    output_rgb_path: Path,
    output_frame_count: int,
    runs_root: Path,
    models_root: Path,
) -> EuvsGen3cRunRecord:
    """Build one native record from explicit workflow-resolved fields."""
    geometry = _relative_locator(
        os.path.relpath(geometry_directory, runs_root),
        "geometry directory",
    )
    checkpoint = _relative_locator(
        os.path.relpath(network_checkpoint, models_root),
        "model checkpoint",
    )

    return EuvsGen3cRunRecord(
        record_provenance="native",
        experiment_name=experiment_name,
        pair=replace(pair, tags=()),
        conditioning_protocol=GEN3C_CONDITIONING_PROTOCOL,
        target_output_index=_integers(
            target_output_index,
            "target_output_index",
        ),
        target_source_sequence_index=_integers(
            target_source_sequence_index,
            "target_source_sequence_index",
        ),
        geometry=Gen3cRunGeometry(
            geometry_backend,
            geometry,
            geometry_provenance,
        ),
        model=Gen3cRunModel(
            model_id,
            checkpoint,
            (
                None
                if lora_working_manifest is None
                else str(lora_working_manifest)
            ),
            None if lora_evidence_path is None else str(lora_evidence_path),
            lora_epochs,
            lora_strength,
        ),
        sampling=sampling,
        execution=Gen3cRunExecution(context_parallel_size),
        output=Gen3cRunOutput(
            output_rgb_path.name,
            (output_frame_count, _GEN3C_HEIGHT, _GEN3C_WIDTH, 3),
        ),
    )


def run_record_path(output_rgb_path: Path) -> Path:
    """Return the sibling ``<output-stem>.run.json`` path."""
    return output_rgb_path.with_suffix(".run.json")


def encode_run_record(record: EuvsGen3cRunRecord) -> bytes:
    """Encode one record as deterministic standards-compliant UTF-8 JSON."""
    text = json.dumps(
        _to_mapping(record),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{text}\n".encode("utf-8")


def load_run_record(
    path: Path,
    *,
    allow_legacy: bool = False,
) -> EuvsGen3cRunRecord:
    """Load one strict record; schema v1 requires explicit legacy opt-in."""
    content = path.read_text(encoding="utf-8")
    try:
        data = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return _from_mapping(
            _object(data, "record"),
            allow_legacy=allow_legacy,
        )
    except Gen3cRunRecordError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Gen3cRunRecordError(
            f"cannot decode run record {path}: {error}"
        ) from error


def write_run_record(
    path: Path,
    record: EuvsGen3cRunRecord,
) -> Path:
    """Write one record directly to the caller-selected path."""
    path.write_bytes(encode_run_record(record))
    return path


def _to_mapping(record: EuvsGen3cRunRecord) -> dict[str, object]:
    pair = record.pair
    result: dict[str, object] = {
        "schema": record.schema,
        "record_provenance": record.record_provenance,
        "experiment": record.experiment_name,
        "pair": {
            "name": pair.name,
            "location": pair.location,
            "direction": "source_to_target",
            "channel": pair.channel,
            "source": _selection_to_mapping(pair.source),
            "target": _selection_to_mapping(pair.target),
        },
        "conditioning": {
            "protocol": record.conditioning_protocol,
            "target_output_index": list(record.target_output_index),
            "target_source_sequence_index": list(
                record.target_source_sequence_index
            ),
        },
        "geometry": {
            "backend": record.geometry.backend,
            "runs_relative_directory": (
                record.geometry.runs_relative_directory
            ),
            "provenance": record.geometry.provenance,
        },
        "model": {
            "model_id": record.model.model_id,
            "models_relative_checkpoint": (
                record.model.models_relative_checkpoint
            ),
        },
        "sampling": {
            "seed": record.sampling.seed,
            "prompt": record.sampling.prompt,
            "negative_prompt": record.sampling.negative_prompt,
            "guidance": record.sampling.guidance,
            "num_steps": record.sampling.num_steps,
        },
        "output": {
            "file": record.output.file,
            "shape": list(record.output.shape),
            "dtype": record.output.dtype,
            "color_space": record.output.color_space,
        },
    }
    if record.schema in (
        LORA_RUN_RECORD_SCHEMA,
        LORA_STRENGTH_RUN_RECORD_SCHEMA,
    ):
        model = result["model"]
        if isinstance(model, dict):
            model["artifact_type"] = "lora-training-checkpoint"
            lora: dict[str, object] = {
                "working_manifest": record.model.lora_working_manifest,
                "evidence_path": record.model.lora_evidence_path,
                "epochs": record.model.lora_epochs,
            }
            if record.schema == LORA_STRENGTH_RUN_RECORD_SCHEMA:
                lora["strength"] = record.model.lora_strength
            model["lora"] = lora
    if record.schema in (
        WINDOW_SEED_RUN_RECORD_SCHEMA,
        LORA_RUN_RECORD_SCHEMA,
        LORA_STRENGTH_RUN_RECORD_SCHEMA,
    ):
        sampling = result["sampling"]
        if isinstance(sampling, dict):
            sampling["window_seed_policy"] = record.sampling.window_seed_policy
    if record.execution is not None:
        result["execution"] = {
            "context_parallel_size": (
                record.execution.context_parallel_size
            ),
        }
    return result


def _from_mapping(
    root: Mapping[str, object],
    *,
    allow_legacy: bool,
) -> EuvsGen3cRunRecord:
    """Read versioned fields; existing record classes validate their values."""
    root = _object(root, "record")
    if "schema" not in root:
        raise Gen3cRunRecordError(
            "record fields differ from schema; missing=['schema']"
        )
    schema = _text(root["schema"], "schema")
    if schema == LEGACY_RUN_RECORD_SCHEMA:
        if not allow_legacy:
            raise Gen3cRunRecordError(
                "v1 run records require allow_legacy=True"
            )
        fields = (
            "schema record_provenance experiment pair conditioning geometry "
            "model sampling output"
        )
        execution = None
        sampling_fields = "seed prompt negative_prompt guidance num_steps"
    elif schema == RUN_RECORD_SCHEMA:
        fields = (
            "schema record_provenance experiment pair conditioning geometry "
            "model sampling execution output"
        )
        execution_data = _fields(
            root.get("execution"),
            "execution",
            "context_parallel_size",
        )
        execution = Gen3cRunExecution(
            cast(int, execution_data["context_parallel_size"])
        )
        sampling_fields = "seed prompt negative_prompt guidance num_steps"
    elif schema in (
        WINDOW_SEED_RUN_RECORD_SCHEMA,
        LORA_RUN_RECORD_SCHEMA,
        LORA_STRENGTH_RUN_RECORD_SCHEMA,
    ):
        fields = (
            "schema record_provenance experiment pair conditioning geometry "
            "model sampling execution output"
        )
        execution_data = _fields(
            root.get("execution"),
            "execution",
            "context_parallel_size",
        )
        execution = Gen3cRunExecution(
            cast(int, execution_data["context_parallel_size"])
        )
        sampling_fields = (
            "seed prompt negative_prompt guidance num_steps "
            "window_seed_policy"
        )
    else:
        raise Gen3cRunRecordError(f"unsupported run record schema: {schema!r}")
    root = _fields(root, "record", fields)

    pair_data = _fields(
        root["pair"],
        "pair",
        "name location direction channel source target",
    )
    direction = _text(pair_data["direction"], "pair.direction")
    if direction != "source_to_target":
        raise Gen3cRunRecordError(
            f"unsupported pair direction: {direction!r}"
        )
    pair = EuvsPairSelection(
        name=cast(str, pair_data["name"]),
        tags=(),
        location=cast(str, pair_data["location"]),
        channel=cast(str, pair_data["channel"]),
        source=_selection(pair_data["source"], "pair.source"),
        target=_selection(pair_data["target"], "pair.target"),
    )
    conditioning = _fields(
        root["conditioning"],
        "conditioning",
        "protocol target_output_index target_source_sequence_index",
    )
    geometry = _fields(
        root["geometry"],
        "geometry",
        "backend runs_relative_directory provenance",
    )
    model_fields = "model_id models_relative_checkpoint"
    if schema in (LORA_RUN_RECORD_SCHEMA, LORA_STRENGTH_RUN_RECORD_SCHEMA):
        model_fields += " artifact_type lora"
    model = _fields(root["model"], "model", model_fields)
    sampling = _fields(
        root["sampling"],
        "sampling",
        sampling_fields,
    )
    output = _fields(
        root["output"],
        "output",
        "file shape dtype color_space",
    )
    parameters = Gen3cRunSampling(
        seed=cast(int, sampling["seed"]),
        prompt=cast(str, sampling["prompt"]),
        negative_prompt=cast(str, sampling["negative_prompt"]),
        guidance=cast(float, sampling["guidance"]),
        num_steps=cast(int, sampling["num_steps"]),
        window_seed_policy=(
            cast(str, sampling["window_seed_policy"])
            if schema in (
                WINDOW_SEED_RUN_RECORD_SCHEMA,
                LORA_RUN_RECORD_SCHEMA,
                LORA_STRENGTH_RUN_RECORD_SCHEMA,
            )
            else WINDOW_SEED_AUTOREGRESSIVE
        ),
    )
    if (
        schema in (
            WINDOW_SEED_RUN_RECORD_SCHEMA,
            LORA_RUN_RECORD_SCHEMA,
            LORA_STRENGTH_RUN_RECORD_SCHEMA,
        )
        and parameters.window_seed_policy == WINDOW_SEED_AUTOREGRESSIVE
        and schema == WINDOW_SEED_RUN_RECORD_SCHEMA
    ):
        raise Gen3cRunRecordError(
            "v3 run records must declare a non-default window seed policy"
        )

    return EuvsGen3cRunRecord(
        record_provenance=cast(str, root["record_provenance"]),
        experiment_name=cast(str, root["experiment"]),
        pair=pair,
        conditioning_protocol=cast(str, conditioning["protocol"]),
        target_output_index=cast(tuple[int, ...], conditioning["target_output_index"]),
        target_source_sequence_index=cast(
            tuple[int, ...], conditioning["target_source_sequence_index"]
        ),
        geometry=Gen3cRunGeometry(
            cast(str, geometry["backend"]),
            cast(str, geometry["runs_relative_directory"]),
            cast(str, geometry["provenance"]),
        ),
        model=Gen3cRunModel(
            cast(str, model["model_id"]),
            cast(str, model["models_relative_checkpoint"]),
            *(
                _lora_identity_fields(model, schema)
                if schema
                in (LORA_RUN_RECORD_SCHEMA, LORA_STRENGTH_RUN_RECORD_SCHEMA)
                else (None, None, None, None)
            ),
        ),
        sampling=parameters,
        execution=execution,
        output=Gen3cRunOutput(
            cast(str, output["file"]),
            cast(tuple[int, int, int, int], output["shape"]),
            cast(str, output["dtype"]),
            cast(str, output["color_space"]),
        ),
    )


def _lora_identity_fields(
    model: Mapping[str, object],
    schema: str,
) -> tuple[str, str, int, float | None]:
    if model["artifact_type"] != "lora-training-checkpoint":
        raise Gen3cRunRecordError(
            "v4 model.artifact_type must be 'lora-training-checkpoint'"
        )
    lora = _fields(
        model["lora"],
        "model.lora",
        (
            "working_manifest evidence_path epochs strength"
            if schema == LORA_STRENGTH_RUN_RECORD_SCHEMA
            else "working_manifest evidence_path epochs"
        ),
    )
    return (
        _text(lora["working_manifest"], "model.lora.working_manifest"),
        cast(str, lora["evidence_path"]),
        cast(int, lora["epochs"]),
        (
            _number(lora["strength"], "model.lora.strength")
            if schema == LORA_STRENGTH_RUN_RECORD_SCHEMA
            else None
        ),
    )


def _canonical_pair(value: EuvsPairSelection) -> EuvsPairSelection:
    if value.tags:
        raise Gen3cRunRecordError("run-record pair tags must be empty")
    for name, text in (
        ("pair.name", value.name),
        ("pair.location", value.location),
        ("pair.source.traversal", value.source.traversal),
        ("pair.target.traversal", value.target.traversal),
    ):
        _identifier(text, name)
    if value.channel != "CAM_F0":
        raise Gen3cRunRecordError(
            "run-record pair must use source_to_target CAM_F0"
        )
    if value.source.traversal == value.target.traversal:
        raise Gen3cRunRecordError("source and target traversals must differ")
    source = _tokens(value.source.image_tokens, "pair.source.image_tokens")
    target = _tokens(value.target.image_tokens, "pair.target.image_tokens")
    if set(source) & set(target):
        raise Gen3cRunRecordError("source and target tokens must not overlap")
    return EuvsPairSelection(
        name=value.name,
        tags=(),
        location=value.location,
        channel=value.channel,
        source=EuvsFrameSelection(value.source.traversal, source),
        target=EuvsFrameSelection(value.target.traversal, target),
    )


def _selection_to_mapping(value: EuvsFrameSelection) -> dict[str, object]:
    return {
        "traversal": value.traversal,
        "image_tokens": list(value.image_tokens),
    }


def _relative_locator(value: object, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if (
        text == "."
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in text
        or path.parts[0].endswith(":")
        or path.as_posix() != text
    ):
        raise Gen3cRunRecordError(
            f"{name} must be a safe relative POSIX path"
        )
    return text


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Gen3cRunRecordError(f"duplicate run-record field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise Gen3cRunRecordError(f"non-finite JSON number {value!r}")


def _fields(value: object, name: str, fields: str) -> Mapping[str, object]:
    data = _object(value, name)
    expected = set(fields.split())
    actual = set(data)
    if actual != expected:
        raise Gen3cRunRecordError(
            f"{name} fields differ from schema; "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return data


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise Gen3cRunRecordError(f"{name} must be a JSON object")
    return cast(Mapping[str, object], value)


def _selection(value: object, name: str) -> EuvsFrameSelection:
    data = _fields(value, name, "traversal image_tokens")
    return EuvsFrameSelection(
        cast(str, data["traversal"]),
        cast(tuple[str, ...], data["image_tokens"]),
    )


def _tokens(value: object, name: str) -> tuple[str, ...]:
    if not _sequence(value):
        raise Gen3cRunRecordError(f"{name} must be a JSON array")
    result = tuple(
        _text(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not result or len(set(result)) != len(result):
        raise Gen3cRunRecordError(f"{name} must be non-empty and unique")
    return result


def _integers(value: object, name: str) -> tuple[int, ...]:
    if not _sequence(value):
        raise Gen3cRunRecordError(f"{name} must be a JSON array")
    result = tuple(
        _integer(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise Gen3cRunRecordError(f"{name} must not be empty")
    return result


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Gen3cRunRecordError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Gen3cRunRecordError(f"{name} must be a number")
    return float(value)


def _text(value: object, name: str, *, empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or "\x00" in value
        or (not empty and not value)
    ):
        requirement = "string" if empty else "non-empty string"
        raise Gen3cRunRecordError(
            f"{name} must be a {requirement} without NUL"
        )
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise Gen3cRunRecordError(f"{name} must be valid UTF-8") from error
    return value


def _identifier(value: object, name: str) -> str:
    text = _text(value, name)
    if text != text.strip():
        raise Gen3cRunRecordError(
            f"{name} must not have leading or trailing whitespace"
        )
    return text

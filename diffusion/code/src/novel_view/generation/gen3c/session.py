"""Parent lifecycle for one resident Gen3C composition process."""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from novel_view.generation.gen3c.protocol import (
    CONTEXT_DIAGNOSTICS_FILENAME,
    REQUEST_DONE_FILENAME,
    REQUEST_READY_FILENAME,
    STOP_FILENAME,
    WORKER_READY_FILENAME,
    context_output_paths,
    request_root,
    write_generation_request,
)
from novel_view.generation.gen3c.request import (
    Gen3cConditioningInput,
    Gen3cGenerationRequest,
)
from novel_view.generation.gen3c.windows import (
    WINDOW_SEED_AUTOREGRESSIVE,
)
from novel_view.models.gen3c.request import (
    Gen3cModelSpec,
    Gen3cSampling,
)
from novel_view.models.gen3c.lora_weights import Gen3cLoraWeights
from novel_view.runtime.distributed import start_torchrun
from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import ProcessError, RunningProcess

_STARTUP_CUDA_BYTES = "startup_peak_cuda_bytes"
_POLL_INTERVAL_SECONDS = 0.1
_STOP_GRACE_SECONDS = 30.0
_OFFICIAL_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static "
    "with no motion, motion blur, over-saturation, shaky footage, low "
    "resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out "
    "colors, choppy sequences, jerky movements, low frame rate, artifacting, "
    "color banding, unnatural transitions, outdated special effects, fake "
    "elements, unconvincing visuals, poorly edited content, jump cuts, visual "
    "noise, and flickering. Overall, the video is of poor quality."
)


class Gen3cGenerationError(RuntimeError):
    """One resident Gen3C composition process cannot continue."""


@dataclass(frozen=True, slots=True)
class Gen3cLoraProvenance:
    """Training identity retained only for generation records."""

    working_manifest: Path
    evidence_path: Path
    epochs: int


@dataclass(frozen=True, slots=True)
class Gen3cGenerationModel:
    """Record identity around one model-owned artifact."""

    model_id: str
    artifact: Gen3cModelSpec
    lora_provenance: Gen3cLoraProvenance | None = None

    @property
    def shared_checkpoint_root(self) -> Path:
        return self.artifact.shared_checkpoint_root

    @property
    def network_checkpoint(self) -> Path:
        return self.artifact.network_checkpoint

    @property
    def lora_weights(self) -> Gen3cLoraWeights | None:
        return self.artifact.lora


@dataclass(frozen=True, slots=True, init=False)
class Gen3cGenerationParameters:
    """Model sampling plus the generation-owned window seed policy."""

    sampling: Gen3cSampling
    window_seed_policy: str

    def __init__(
        self,
        seed: int,
        prompt: str = "",
        negative_prompt: str = _OFFICIAL_NEGATIVE_PROMPT,
        guidance: float = 1.0,
        num_steps: int = 35,
        window_seed_policy: str = WINDOW_SEED_AUTOREGRESSIVE,
    ) -> None:
        object.__setattr__(
            self,
            "sampling",
            Gen3cSampling(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance=float(guidance),
                steps=num_steps,
                seed=seed,
            ),
        )
        object.__setattr__(self, "window_seed_policy", window_seed_policy)

    @property
    def seed(self) -> int:
        return self.sampling.seed

    @property
    def prompt(self) -> str:
        return self.sampling.prompt

    @property
    def negative_prompt(self) -> str:
        return self.sampling.negative_prompt

    @property
    def guidance(self) -> float:
        return self.sampling.guidance

    @property
    def num_steps(self) -> int:
        return self.sampling.steps


@dataclass(frozen=True, slots=True)
class Gen3cGenerationResources:
    """Process paths and topology selected by the generation caller."""

    scratch_root: Path
    environment_overrides: Mapping[str, str]
    upstream_root: Path | None
    context_parallel_size: int = 2
    unprojection_chunk_size: int = 13
    context_depth_model: Path | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Gen3cGenerationResult:
    """One direct RGB result and the telemetry used by existing records."""

    conditioning_input: Gen3cConditioningInput
    model_id: str
    parameters: Gen3cGenerationParameters
    generated_rgb_path: Path
    elapsed_seconds: float
    per_rank_cuda_allocated_before_bytes: tuple[int, ...]
    per_rank_cuda_reserved_before_bytes: tuple[int, ...]
    per_rank_peak_cuda_allocated_bytes: tuple[int, ...]
    per_rank_peak_cuda_reserved_bytes: tuple[int, ...]
    per_rank_request_peak_cuda_allocated_bytes: tuple[int, ...]
    per_rank_request_peak_cuda_reserved_bytes: tuple[int, ...]
    per_rank_cuda_allocated_after_bytes: tuple[int, ...]
    per_rank_cuda_reserved_after_bytes: tuple[int, ...]
    per_rank_elapsed_seconds: tuple[float, ...]
    context_depth_path: Path | None = None
    context_valid_path: Path | None = None
    context_alignment_scale: tuple[float, ...] = ()
    context_alignment_bias: tuple[float, ...] = ()
    context_alignment_mae_m: tuple[float, ...] = ()
    context_valid_fraction: tuple[float, ...] = ()

    __hash__: ClassVar[None] = None


class Gen3cGenerationSession:
    """Keep one composition worker group alive for sequential requests."""

    def __init__(
        self,
        model: Gen3cGenerationModel,
        parameters: Gen3cGenerationParameters,
        resources: Gen3cGenerationResources,
    ) -> None:
        self._model = model
        self._parameters = parameters
        self._resources = resources
        self._request_lock = threading.Lock()
        self._state = "new"
        self._request_index = 0
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._process: RunningProcess | None = None
        try:
            self._temporary = tempfile.TemporaryDirectory(
                prefix="gen3c-session-",
                dir=resources.scratch_root,
                ignore_cleanup_errors=True,
            )
            self._root = Path(self._temporary.name)
            self._process = _start_worker(
                self._root,
                model,
                parameters,
                resources,
            )
            self._wait_for(self._root / WORKER_READY_FILENAME, "startup")
            (
                self._startup_peak_allocated,
                self._startup_peak_reserved,
            ) = _read_startup_peaks(
                self._root,
                resources.context_parallel_size,
            )
            self._state = "running"
        except Exception as error:
            self._break()
            self._cleanup_exchange()
            if isinstance(error, Gen3cGenerationError):
                raise
            raise Gen3cGenerationError(str(error)) from error

    @property
    def model(self) -> Gen3cGenerationModel:
        return self._model

    @property
    def parameters(self) -> Gen3cGenerationParameters:
        return self._parameters

    @property
    def resources(self) -> Gen3cGenerationResources:
        return self._resources

    def generate(
        self,
        conditioning_input: Gen3cConditioningInput,
        output_rgb_path: Path,
    ) -> Gen3cGenerationResult:
        """Run one direct-output request through the resident group."""
        if not self._request_lock.acquire(blocking=False):
            raise Gen3cGenerationError(
                "Gen3C session already has an active request"
            )
        try:
            return self._generate_request(conditioning_input, output_rgb_path)
        finally:
            self._request_lock.release()

    def _generate_request(
        self,
        conditioning_input: Gen3cConditioningInput,
        output_rgb_path: Path,
    ) -> Gen3cGenerationResult:
        if self._state != "running":
            raise Gen3cGenerationError(
                f"Gen3C session is {self._state}, not running"
            )
        request = request_root(self._root, self._request_index)
        started = time.monotonic()
        try:
            request.mkdir()
            write_generation_request(
                request,
                Gen3cGenerationRequest(
                    conditioning=conditioning_input.build_conditioning(),
                    output_rgb_path=output_rgb_path,
                    context_depth_output_slots=(
                        conditioning_input.context_depth_output_slots
                    ),
                ),
            )
            (request / REQUEST_READY_FILENAME).touch()
            self._wait_for(request / REQUEST_DONE_FILENAME, "request")
            telemetry = _read_rank_telemetry(
                request,
                self.resources.context_parallel_size,
            )
            context = _read_context_result(
                request,
                output_rgb_path,
                conditioning_input.context_depth_output_slots is not None,
            )
            total_peak_allocated = tuple(
                max(startup, current)
                for startup, current in zip(
                    self._startup_peak_allocated,
                    telemetry[2],
                    strict=True,
                )
            )
            total_peak_reserved = tuple(
                max(startup, current)
                for startup, current in zip(
                    self._startup_peak_reserved,
                    telemetry[3],
                    strict=True,
                )
            )
            result = Gen3cGenerationResult(
                conditioning_input=conditioning_input,
                model_id=self.model.model_id,
                parameters=self.parameters,
                generated_rgb_path=output_rgb_path,
                elapsed_seconds=float(time.monotonic() - started),
                per_rank_cuda_allocated_before_bytes=telemetry[0],
                per_rank_cuda_reserved_before_bytes=telemetry[1],
                per_rank_peak_cuda_allocated_bytes=total_peak_allocated,
                per_rank_peak_cuda_reserved_bytes=total_peak_reserved,
                per_rank_request_peak_cuda_allocated_bytes=telemetry[2],
                per_rank_request_peak_cuda_reserved_bytes=telemetry[3],
                per_rank_cuda_allocated_after_bytes=telemetry[4],
                per_rank_cuda_reserved_after_bytes=telemetry[5],
                per_rank_elapsed_seconds=telemetry[6],
                context_depth_path=context[0],
                context_valid_path=context[1],
                context_alignment_scale=context[2],
                context_alignment_bias=context[3],
                context_alignment_mae_m=context[4],
                context_valid_fraction=context[5],
            )
            self._request_index += 1
            return result
        except Exception as error:
            self._break()
            if isinstance(error, Gen3cGenerationError):
                raise
            raise Gen3cGenerationError(str(error)) from error
        finally:
            with suppress(OSError):
                shutil.rmtree(request)

    def _wait_for(self, marker: Path, stage: str) -> None:
        process = self._process
        if process is None:
            raise Gen3cGenerationError("Gen3C worker was not started")
        while not marker.is_file():
            if process.poll() is not None:
                try:
                    process.wait()
                except ProcessError as error:
                    raise Gen3cGenerationError(str(error)) from error
                raise Gen3cGenerationError(
                    f"Gen3C worker exited before {stage} completed"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)

    def _break(self) -> None:
        self._state = "broken"
        if self._process is not None:
            with suppress(Exception):
                self._process.terminate()

    def close(self) -> None:
        """Wait for an active request, then stop the full process group."""
        with self._request_lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        if self._state == "closed":
            return
        failure: Exception | None = None
        process = self._process
        if self._state == "running" and process is not None:
            try:
                (self._root / STOP_FILENAME).touch()
                deadline = time.monotonic() + _STOP_GRACE_SECONDS
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(_POLL_INTERVAL_SECONDS)
                if process.poll() is None:
                    process.terminate()
                else:
                    process.wait()
            except Exception as error:
                with suppress(Exception):
                    process.terminate()
                failure = error
        elif process is not None:
            try:
                process.terminate()
            except Exception as error:
                failure = error
        self._state = "closed"
        self._cleanup_exchange()
        if failure is not None:
            raise Gen3cGenerationError(str(failure)) from failure

    def _cleanup_exchange(self) -> None:
        if self._temporary is not None:
            with suppress(OSError):
                self._temporary.cleanup()

    def __enter__(self) -> Gen3cGenerationSession:
        if self._state != "running":
            raise Gen3cGenerationError(
                f"Gen3C session is {self._state}, not running"
            )
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        if exception_type is None:
            self.close()
        else:
            with suppress(Exception):
                self.close()


def _start_worker(
    session_root: Path,
    model: Gen3cGenerationModel,
    parameters: Gen3cGenerationParameters,
    resources: Gen3cGenerationResources,
) -> RunningProcess:
    arguments = [
        "--session-root",
        str(session_root),
        "--shared-checkpoint-root",
        str(model.shared_checkpoint_root),
        "--network-checkpoint",
        str(model.network_checkpoint),
        "--context-parallel-size",
        str(resources.context_parallel_size),
        "--unprojection-chunk-size",
        str(resources.unprojection_chunk_size),
        "--seed",
        str(parameters.seed),
        "--guidance",
        repr(parameters.guidance),
        "--num-steps",
        str(parameters.num_steps),
        "--window-seed-policy",
        parameters.window_seed_policy,
        "--prompt",
        parameters.prompt,
        "--negative-prompt",
        parameters.negative_prompt,
    ]
    if resources.upstream_root is not None:
        arguments.extend(("--upstream-root", str(resources.upstream_root)))
    if model.lora_weights is not None:
        arguments.extend(
            (
                "--lora-checkpoint",
                str(model.lora_weights.checkpoint),
                "--lora-strength",
                repr(model.lora_weights.strength),
            )
        )
    if resources.context_depth_model is not None:
        arguments.extend(
            (
                "--context-depth-model",
                str(resources.context_depth_model),
            )
        )
    return start_torchrun(
        GEN3C_PYTHON,
        Path(__file__).with_name("_worker.py"),
        arguments,
        process_count=resources.context_parallel_size,
        log_path=session_root / "worker.log",
        description="resident pinned Gen3C composition worker",
        environment_overrides=resources.environment_overrides,
    )


def _read_startup_peaks(
    session_root: Path,
    rank_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rows = tuple(
        np.load(
            session_root / f"{_STARTUP_CUDA_BYTES}_{rank}.npy",
            allow_pickle=False,
        )
        for rank in range(rank_count)
    )
    return (
        tuple(int(row[0]) for row in rows),
        tuple(int(row[1]) for row in rows),
    )


def _read_rank_telemetry(
    request: Path,
    rank_count: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[float, ...],
]:
    memory = tuple(
        np.load(
            request / f"cuda_bytes_{rank}.npy",
            allow_pickle=False,
        )
        for rank in range(rank_count)
    )
    elapsed = tuple(
        np.load(
            request / f"elapsed_seconds_{rank}.npy",
            allow_pickle=False,
        )
        for rank in range(rank_count)
    )
    return (
        *(tuple(int(row[column]) for row in memory) for column in range(6)),
        tuple(float(row[0]) for row in elapsed),
    )


def _read_context_result(
    request: Path,
    output_rgb_path: Path,
    requested: bool,
) -> tuple[
    Path | None,
    Path | None,
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    if not requested:
        return None, None, (), (), (), ()
    diagnostics = np.load(
        request / CONTEXT_DIAGNOSTICS_FILENAME,
        allow_pickle=False,
    )
    depth, valid = context_output_paths(output_rgb_path)
    metrics = tuple(
        tuple(float(value) for value in diagnostics[:, column])
        for column in range(4)
    )
    return depth, valid, *metrics

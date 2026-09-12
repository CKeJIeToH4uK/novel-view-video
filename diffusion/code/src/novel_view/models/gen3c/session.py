"""Resident official Gen3C pipeline inside the composition worker."""

from __future__ import annotations

import gc
import importlib
import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np

from novel_view.models.gen3c.lora_weights import load_gen3c_lora_model
from novel_view.models.gen3c.request import (
    Gen3cModelSpec,
    Gen3cSampling,
    Gen3cTopology,
)
from novel_view.models.gen3c.spec import (
    GEN3C_CHECKPOINT_NAME,
    GEN3C_FPS,
    GEN3C_IMAGE_SIZE_HW,
    GEN3C_WINDOW_SIZE,
)

_PIPELINE_MODULE = "cosmos_predict1.diffusion.inference.gen3c_pipeline"
_INFERENCE_UTILS_MODULE = (
    "cosmos_predict1.diffusion.inference.inference_utils"
)


class Gen3cModelSession:
    """One loaded pipeline, its CP group, device, and request RNG boundary."""

    def __init__(
        self,
        *,
        torch: Any,
        pipeline: Any,
        parallel_state: Any,
        model_parallel_initialized: bool,
        rank: int,
        local_rank: int,
        device: Any,
        sampling: Gen3cSampling,
    ) -> None:
        self.torch = torch
        self.rank = rank
        self.local_rank = local_rank
        self.device = device
        self._sampling = sampling
        self._pipeline = pipeline
        self._parallel_state = parallel_state
        self._model_parallel_initialized = model_parallel_initialized
        self._request_rng_state = _capture_rng_state(torch, device)

    @classmethod
    def open(
        cls,
        spec: Gen3cModelSpec,
        topology: Gen3cTopology,
        sampling: Gen3cSampling,
    ) -> Gen3cModelSession:
        """Initialize CP when requested and construct one official pipeline."""
        torch = importlib.import_module("torch")
        rank, local_rank = _rank_environment(
            torch,
            topology.context_parallel_size,
        )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        distributed = importlib.import_module("cosmos_predict1.utils.distributed")
        misc = importlib.import_module("cosmos_predict1.utils.misc")
        parallel_state = importlib.import_module("megatron.core.parallel_state")
        initialized = False
        try:
            context_parallel_group = None
            if topology.context_parallel_size > 1:
                distributed.init()
                parallel_state.initialize_model_parallel(
                    context_parallel_size=topology.context_parallel_size
                )
                initialized = True
                context_parallel_group = (
                    parallel_state.get_context_parallel_group()
                )
            torch.enable_grad(False)
            misc.set_random_seed(sampling.seed)
            torch.cuda.reset_peak_memory_stats(device)
            pipeline = _build_pipeline(
                spec,
                sampling,
                torch,
                context_parallel_group,
            )
            return cls(
                torch=torch,
                pipeline=pipeline,
                parallel_state=parallel_state,
                model_parallel_initialized=initialized,
                rank=rank,
                local_rank=local_rank,
                device=device,
                sampling=sampling,
            )
        except BaseException:
            if initialized:
                parallel_state.destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            raise

    def restore_request_rng(self) -> None:
        """Return every request to the same post-construction RNG boundary."""
        _restore_rng_state(
            self._request_rng_state,
            self.torch,
            self.device,
        )

    def encode_seed(self, rgb: np.ndarray) -> Any:
        """Convert one uint8 HWC image to the official normalized seed."""
        pixels = self.torch.from_numpy(np.asarray(rgb)).to(
            device=self.device,
            dtype=self.torch.float32,
        )
        return (
            pixels.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
            * (2.0 / 255.0)
            - 1.0
        )

    def generate_window(
        self,
        rendered_rgb: Any,
        rendered_mask: Any,
        seed: Any,
    ) -> np.ndarray:
        """Run one fixed-size official Gen3C window."""
        with self.torch.inference_mode():
            generated = self._pipeline.generate(
                prompt=self._sampling.prompt,
                image_path=seed,
                rendered_warp_images=rendered_rgb,
                rendered_warp_masks=rendered_mask,
                negative_prompt=self._sampling.negative_prompt,
            )
        video = generated[0]
        height, width = GEN3C_IMAGE_SIZE_HW
        if (
            not isinstance(video, np.ndarray)
            or video.dtype != np.uint8
            or video.shape != (GEN3C_WINDOW_SIZE, height, width, 3)
        ):
            raise RuntimeError(
                "Gen3C pipeline must return uint8 [121,704,1280,3]"
            )
        return video

    def barrier(self) -> None:
        """Synchronize the local CP group when distributed is initialized."""
        if self.torch.distributed.is_initialized():
            self.torch.distributed.barrier(device_ids=[self.local_rank])

    def cuda_bytes(self) -> np.ndarray:
        """Return current allocated and reserved CUDA bytes."""
        return np.asarray(
            (
                self.torch.cuda.memory_allocated(self.device),
                self.torch.cuda.memory_reserved(self.device),
            ),
            dtype=np.int64,
        )

    def peak_cuda_bytes(self) -> np.ndarray:
        """Return peak allocated and reserved CUDA bytes."""
        return np.asarray(
            (
                self.torch.cuda.max_memory_allocated(self.device),
                self.torch.cuda.max_memory_reserved(self.device),
            ),
            dtype=np.int64,
        )

    def reset_peak_cuda_bytes(self) -> None:
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def close(self) -> None:
        """Release the pipeline and distributed model/process groups."""
        self._pipeline = None
        gc.collect()
        self.torch.cuda.empty_cache()
        if self._model_parallel_initialized:
            self._parallel_state.destroy_model_parallel()
            self._model_parallel_initialized = False
        if self.torch.distributed.is_initialized():
            self.torch.distributed.destroy_process_group()


def _rank_environment(torch: Any, expected_size: int) -> tuple[int, int]:
    """Resolve the supported single-node CP1/CP2 torchrun topology."""
    if expected_size not in (1, 2):
        raise RuntimeError("context parallel size must be 1 or 2")
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    try:
        rank, local_rank, world_size, local_world_size = (
            int(os.environ[name]) for name in required
        )
    except (KeyError, ValueError) as error:
        raise RuntimeError("worker must be launched by torchrun") from error
    if world_size != expected_size or local_world_size != expected_size:
        raise RuntimeError(
            "WORLD_SIZE and LOCAL_WORLD_SIZE must equal context_parallel_size"
        )
    if rank != local_rank or not 0 <= rank < world_size:
        raise RuntimeError("Gen3C requires one local rank per global rank")
    if not torch.cuda.is_available():
        raise RuntimeError("Gen3C model session requires CUDA")
    if torch.cuda.device_count() != expected_size:
        raise RuntimeError(
            "visible CUDA device count must equal context_parallel_size"
        )
    return rank, local_rank


def _build_pipeline(
    spec: Gen3cModelSpec,
    sampling: Gen3cSampling,
    torch: Any,
    context_parallel_group: Any | None,
) -> Any:
    pipeline_module = importlib.import_module(_PIPELINE_MODULE)
    inference_utils = importlib.import_module(_INFERENCE_UTILS_MODULE)
    pipeline_class = _pipeline_class(
        pipeline_module,
        inference_utils,
        torch,
        spec,
        context_parallel_group,
    )
    height, width = GEN3C_IMAGE_SIZE_HW
    pipeline = pipeline_class(
        inference_type="video2world",
        checkpoint_dir=str(spec.shared_checkpoint_root),
        checkpoint_name=GEN3C_CHECKPOINT_NAME,
        prompt_upsampler_dir=None,
        enable_prompt_upsampler=False,
        has_text_input=True,
        offload_network=False,
        offload_tokenizer=False,
        offload_text_encoder_model=False,
        offload_prompt_upsampler=False,
        offload_guardrail_models=False,
        disable_guardrail=True,
        disable_prompt_encoder=False,
        guidance=sampling.guidance,
        num_steps=sampling.steps,
        height=height,
        width=width,
        fps=GEN3C_FPS,
        num_video_frames=GEN3C_WINDOW_SIZE,
        seed=sampling.seed,
    )
    if spec.lora is None and context_parallel_group is not None:
        pipeline.model.net.enable_context_parallel(context_parallel_group)
    if getattr(pipeline.model, "chunk_size", None) != GEN3C_WINDOW_SIZE:
        raise RuntimeError("pinned Gen3C model chunk size must remain 121")
    return pipeline


def _pipeline_class(
    pipeline_module: Any,
    inference_utils: Any,
    torch: Any,
    spec: Gen3cModelSpec,
    context_parallel_group: Any | None,
) -> type:
    base = pipeline_module.Gen3cPipeline

    class CheckpointGen3cPipeline(base):
        def _load_network(self) -> None:
            if spec.lora is not None:
                self.model = load_gen3c_lora_model(
                    upstream_root=None,
                    base_checkpoint=(
                        spec.shared_checkpoint_root
                        / GEN3C_CHECKPOINT_NAME
                        / "model.pt"
                    ),
                    weights=spec.lora,
                    context_parallel_group=context_parallel_group,
                )
                self.model.cuda()
                return
            with inference_utils.skip_init_linear():
                self.model.set_up_model()
            self.model.model.load_state_dict(
                _checkpoint_state(torch, spec.network_checkpoint),
                strict=False,
            )
            self.model.cuda()

    return CheckpointGen3cPipeline


def _checkpoint_state(torch: Any, path: Any) -> Any:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, Mapping) and isinstance(
        loaded.get("model"), Mapping
    ):
        return loaded["model"]
    return loaded


def _capture_rng_state(torch: Any, device: Any) -> tuple[Any, ...]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        torch.cuda.get_rng_state(device).clone(),
    )


def _restore_rng_state(
    state: tuple[Any, ...],
    torch: Any,
    device: Any,
) -> None:
    python_state, numpy_state, cpu_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, device)

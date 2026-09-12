"""Fixed public facts of the pinned Gen3C model capability."""

from __future__ import annotations

from dataclasses import dataclass


GEN3C_BACKEND = "gen3c-cosmos-7b"
GEN3C_CHECKPOINT_NAME = "Gen3C-Cosmos-7B"
GEN3C_FPS = 24
GEN3C_IMAGE_SIZE_HW = (704, 1280)
GEN3C_WINDOW_SIZE = 121


@dataclass(frozen=True, slots=True)
class Gen3cModelContract:
    """Model-owned R4c raster, latent and conditioner geometry."""

    contract_id: str
    batch_size: int
    video_frame_count: int
    raster_height: int
    raster_width: int
    fps: int
    rgb_channels: int
    condition_frame_count: int
    base_latent_channels: int
    latent_frame_count: int
    latent_height: int
    latent_width: int
    pose_latent_channels: int
    prompt_token_count: int
    prompt_embedding_width: int

    @property
    def raster_size_hw(self) -> tuple[int, int]:
        return self.raster_height, self.raster_width

    @property
    def rgb_thwc_shape(self) -> tuple[int, int, int, int]:
        return (
            self.video_frame_count,
            self.raster_height,
            self.raster_width,
            self.rgb_channels,
        )

    @property
    def known_thw_shape(self) -> tuple[int, int, int]:
        return self.video_frame_count, self.raster_height, self.raster_width

    @property
    def packed_known_shape(self) -> tuple[int, int]:
        pixels = self.raster_height * self.raster_width
        return self.video_frame_count, (pixels + 7) // 8

    @property
    def target_bcthw_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.batch_size,
            self.rgb_channels,
            self.video_frame_count,
            self.raster_height,
            self.raster_width,
        )

    @property
    def source_bcthw_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.batch_size,
            self.rgb_channels,
            self.condition_frame_count,
            self.raster_height,
            self.raster_width,
        )

    @property
    def condition_btnchw_shape(self) -> tuple[int, ...]:
        return (
            self.batch_size,
            self.video_frame_count,
            self.condition_frame_count,
            self.rgb_channels,
            self.raster_height,
            self.raster_width,
        )

    @property
    def condition_known_btn1hw_shape(self) -> tuple[int, ...]:
        return (
            self.batch_size,
            self.video_frame_count,
            self.condition_frame_count,
            1,
            self.raster_height,
            self.raster_width,
        )

    @property
    def base_latent_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.batch_size,
            self.base_latent_channels,
            self.latent_frame_count,
            self.latent_height,
            self.latent_width,
        )

    @property
    def pose_latent_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.batch_size,
            self.pose_latent_channels,
            self.latent_frame_count,
            self.latent_height,
            self.latent_width,
        )

    @property
    def prompt_embedding_shape(self) -> tuple[int, int, int]:
        return (
            self.batch_size,
            self.prompt_token_count,
            self.prompt_embedding_width,
        )

    @property
    def prompt_mask_shape(self) -> tuple[int, int]:
        return self.batch_size, self.prompt_token_count

    @property
    def padding_mask_shape(self) -> tuple[int, int, int, int]:
        return self.batch_size, 1, self.raster_height, self.raster_width

    @property
    def image_size_values(self) -> tuple[int, int, int, int]:
        return (
            self.raster_height,
            self.raster_width,
            self.raster_height,
            self.raster_width,
        )

    @property
    def conditioner_tensor_shapes(
        self,
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return (
            ("t5_text_embeddings", self.prompt_embedding_shape),
            ("t5_text_mask", self.prompt_mask_shape),
            ("fps", (self.batch_size,)),
            ("num_frames", (self.batch_size,)),
            ("image_size", (self.batch_size, len(self.image_size_values))),
            ("padding_mask", self.padding_mask_shape),
        )


R4C_GEN3C_MODEL_CONTRACT = Gen3cModelContract(
    contract_id="gen3c-cosmos-7b-121x704x1280-fps10-v1",
    batch_size=1,
    video_frame_count=121,
    raster_height=704,
    raster_width=1280,
    fps=10,
    rgb_channels=3,
    condition_frame_count=1,
    base_latent_channels=16,
    latent_frame_count=16,
    latent_height=88,
    latent_width=160,
    pose_latent_channels=64,
    prompt_token_count=512,
    prompt_embedding_width=1024,
)


__all__ = [
    "GEN3C_BACKEND",
    "GEN3C_CHECKPOINT_NAME",
    "GEN3C_FPS",
    "GEN3C_IMAGE_SIZE_HW",
    "GEN3C_WINDOW_SIZE",
    "Gen3cModelContract",
    "R4C_GEN3C_MODEL_CONTRACT",
]

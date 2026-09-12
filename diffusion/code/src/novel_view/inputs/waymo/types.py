"""Scalar identity and errors for physical Waymo inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


class WaymoContractError(ValueError):
    """A physical Waymo input violates its concrete data contract."""


WaymoOfficialPartition: TypeAlias = Literal["training", "validation"]


@dataclass(frozen=True, slots=True, order=True)
class WaymoFrameKey:
    """One frame key at the beginning of the first TOP LiDAR scan."""

    segment_id: str
    frame_timestamp_micros: int

    @property
    def frame_timestamp_seconds(self) -> float:
        """Convert the integer microsecond timestamp to seconds."""
        return self.frame_timestamp_micros / 1_000_000.0


__all__ = [
    "WaymoContractError",
    "WaymoFrameKey",
    "WaymoOfficialPartition",
]

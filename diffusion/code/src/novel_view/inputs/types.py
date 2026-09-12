"""Small values shared by concrete input readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Role-neutral reference to one materialized camera frame."""

    image_path: Path
    db_path: Path
    timestamp_us: int
    channel: str
    image_token: str
    ego_pose_token: str
    camera_token: str

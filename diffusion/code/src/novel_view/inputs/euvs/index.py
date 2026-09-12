"""Resolve EUVS frame tokens through the materialized ``frames.csv`` index."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from novel_view.inputs.types import FrameRef

_REQUIRED_COLUMNS = (
    "image_path",
    "db_path",
    "location",
    "logical_locations",
    "traversal",
    "timestamp_us",
    "channel",
    "image_token",
    "ego_pose_token",
    "camera_token",
)


class FramesIndexError(ValueError):
    """The EUVS index cannot identify the requested physical stream."""


@dataclass(frozen=True, slots=True)
class _FrameRow:
    image_path: Path
    db_path: Path
    location: str
    logical_locations: tuple[str, ...]
    traversal: str
    timestamp_us: int
    channel: str
    image_token: str
    ego_pose_token: str
    camera_token: str


class FramesIndex:
    """In-memory token index preserving the order requested by a caller."""

    def __init__(self, dataset_root: Path, rows: dict[str, _FrameRow]) -> None:
        self.dataset_root = dataset_root
        self._rows = rows

    @classmethod
    def load(cls, dataset_root: str | Path) -> FramesIndex:
        root = Path(dataset_root).expanduser()
        rows: dict[str, _FrameRow] = {}
        with (root / "frames.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            _check_header(reader.fieldnames)
            for values in reader:
                row = _parse_row(values, reader.line_num)
                if row.image_token in rows:
                    raise FramesIndexError(
                        f"frames.csv line {reader.line_num}: duplicate "
                        f"image_token {row.image_token!r}"
                    )
                rows[row.image_token] = row
        return cls(root, rows)

    def resolve(
        self,
        image_tokens: Sequence[str],
        *,
        location: str,
        traversal: str,
        channel: str,
    ) -> tuple[FrameRef, ...]:
        frames: list[FrameRef] = []
        for token in image_tokens:
            row = self._rows.get(token)
            if row is None:
                raise FramesIndexError(f"unknown image_token {token!r}")
            if location not in row.logical_locations:
                raise FramesIndexError(
                    f"image_token {token!r}: expected logical location "
                    f"{location!r}, found physical location {row.location!r} "
                    f"with logical_locations {row.logical_locations!r}"
                )
            if row.traversal != traversal:
                raise FramesIndexError(
                    f"image_token {token!r}: expected traversal "
                    f"{traversal!r}, found {row.traversal!r}"
                )
            if row.channel != channel:
                raise FramesIndexError(
                    f"image_token {token!r}: expected channel "
                    f"{channel!r}, found {row.channel!r}"
                )
            frames.append(
                FrameRef(
                    image_path=self.dataset_root / row.image_path,
                    db_path=self.dataset_root / row.db_path,
                    timestamp_us=row.timestamp_us,
                    channel=row.channel,
                    image_token=row.image_token,
                    ego_pose_token=row.ego_pose_token,
                    camera_token=row.camera_token,
                )
            )
        return tuple(frames)


def _check_header(fieldnames: Sequence[str] | None) -> None:
    if fieldnames is None:
        raise FramesIndexError("frames.csv has no header")
    if len(set(fieldnames)) != len(fieldnames):
        raise FramesIndexError("frames.csv header contains duplicate columns")
    missing = sorted(set(_REQUIRED_COLUMNS) - set(fieldnames))
    if missing:
        raise FramesIndexError(
            f"frames.csv is missing required columns: {', '.join(missing)}"
        )


def _parse_row(values: dict[str | None, str | None], line_number: int) -> _FrameRow:
    if None in values:
        raise FramesIndexError(
            f"frames.csv line {line_number}: more values than header columns"
        )
    fields: dict[str, str] = {}
    for name in _REQUIRED_COLUMNS:
        value = values.get(name)
        if value is None or not value.strip():
            raise FramesIndexError(
                f"frames.csv line {line_number}: {name} must not be empty"
            )
        fields[name] = value.strip()
    return _FrameRow(
        image_path=Path(fields["image_path"]),
        db_path=Path(fields["db_path"]),
        location=fields["location"],
        logical_locations=_logical_locations(
            fields["logical_locations"], fields["location"], line_number
        ),
        traversal=fields["traversal"],
        timestamp_us=int(fields["timestamp_us"]),
        channel=fields["channel"],
        image_token=fields["image_token"],
        ego_pose_token=fields["ego_pose_token"],
        camera_token=fields["camera_token"],
    )


def _logical_locations(
    value: str,
    physical_location: str,
    line_number: int,
) -> tuple[str, ...]:
    locations = tuple(value.split("|"))
    if any(not location or location != location.strip() for location in locations):
        raise FramesIndexError(
            f"frames.csv line {line_number}: logical_locations must contain "
            "non-empty values without whitespace around '|'"
        )
    if len(set(locations)) != len(locations):
        raise FramesIndexError(
            f"frames.csv line {line_number}: logical_locations contains duplicates"
        )
    if physical_location not in locations:
        raise FramesIndexError(
            f"frames.csv line {line_number}: physical location "
            f"{physical_location!r} must belong to logical_locations"
        )
    return locations

"""Read role-neutral frame metadata from nuPlan SQLite databases."""

from __future__ import annotations

import io
import math
import pickle
import pickletools
import re
import sqlite3
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, cast

from novel_view.inputs.types import FrameRef

_TOKEN = re.compile(r"^[0-9a-f]{16}$")
_QUERY = "SELECT timestamp,x,y,z,qw,qx,qy,qz,epsg FROM ego_pose WHERE token = ?"
_CAMERA_QUERY = """
SELECT
    CASE WHEN typeof(channel) = 'text'
        THEN substr(channel,1,:text_fetch_chars) END,
    CASE WHEN typeof(model) = 'text'
        THEN substr(model,1,:text_fetch_chars) END,
    CASE WHEN typeof(translation) = 'blob'
        THEN substr(translation,1,:blob_fetch_bytes) END,
    CASE WHEN typeof(rotation) = 'blob'
        THEN substr(rotation,1,:blob_fetch_bytes) END,
    CASE WHEN typeof(intrinsic) = 'blob'
        THEN substr(intrinsic,1,:blob_fetch_bytes) END,
    CASE WHEN typeof(distortion) = 'blob'
        THEN substr(distortion,1,:blob_fetch_bytes) END,
    width,
    height
FROM camera
WHERE token = :token
"""
_FLOAT_FIELDS = ("x", "y", "z", "qw", "qx", "qy", "qz")
_MAX_CAMERA_BLOB_BYTES = 16_384
_MAX_CAMERA_TEXT_CHARS = 256
_NUPLAN_TYPES_MODULE = "nuplan.database.common.data_types"
_NUPLAN_LIST_TYPES = frozenset({"CameraIntrinsic", "Rotation", "Translation"})
_FORBIDDEN_PICKLE_OPCODES = frozenset(
    {"BINPUT", "EXT1", "EXT2", "EXT4", "LONG_BINPUT", "PUT"}
)


class NuPlanDataError(ValueError):
    """Selected nuPlan database data cannot be read safely."""


@dataclass(frozen=True, slots=True)
class RawEgoPose:
    """Uninterpreted numeric fields from one nuPlan ``ego_pose`` row.

    Attributes:
        timestamp_us: Ego-pose timestamp in microseconds. This may differ
            from the linked image timestamp.
        x: Global X coordinate in the row's EPSG system.
        y: Global Y coordinate in the row's EPSG system.
        z: Global Z coordinate in the row's EPSG system.
        qw: Quaternion scalar component stored by nuPlan.
        qx: Quaternion X component stored by nuPlan.
        qy: Quaternion Y component stored by nuPlan.
        qz: Quaternion Z component stored by nuPlan.
        epsg: Coordinate reference system code for the global coordinates.

    The values are not normalized, interpolated, or converted to a matrix.
    """

    timestamp_us: int
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float
    epsg: int


@dataclass(frozen=True, slots=True)
class RawCameraCalibration:
    """Raw nuPlan camera calibration without geometric interpretation.

    Attributes:
        model: Non-empty camera model label from the database.
        translation_xyz: Camera origin expressed in the ego frame, in metres.
        rotation_wxyz: Camera-to-ego quaternion in nuPlan ``wxyz`` order.
        intrinsics: Native distorted-image camera matrix.
        distortion: Caltech coefficients ``k1, k2, p1, p2, k3``.
        image_size_hw: Native image size as ``(height, width)``.

    The quaternion is not normalized or converted into a matrix. The result
    deliberately retains native intrinsics and distortion for the later raster
    stage.
    """

    model: str
    translation_xyz: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    intrinsics: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    distortion: tuple[float, float, float, float, float]
    image_size_hw: tuple[int, int]


class _NuPlanList(list[Any]):
    """Inert replacement for the three nuPlan list-wrapper pickle classes."""


class _SafeFloat64DType:
    """Inert target for the canonical NumPy float64 dtype pickle."""

    __slots__ = ("byte_order",)

    def __init__(self) -> None:
        self.byte_order: str | None = None

    def __setstate__(self, state: object) -> None:
        expected_tail = (None, None, None, -1, -1, 0)
        if (
            not isinstance(state, tuple)
            or len(state) != 8
            or state[0] != 3
            or state[1] not in {"<", ">"}
            or state[2:] != expected_tail
        ):
            raise pickle.UnpicklingError("unsupported NumPy float64 dtype state")
        self.byte_order = state[1]


def _safe_float64_dtype(*args: object) -> _SafeFloat64DType:
    """Construct only the dtype form emitted for camera float64 scalars."""
    if args != ("f8", False, True):
        raise pickle.UnpicklingError("unsupported NumPy dtype")
    return _SafeFloat64DType()


def _safe_float64_scalar(dtype: object, payload: object) -> float:
    """Decode one canonical finite-or-nonfinite float64 scalar inertly."""
    if (
        not isinstance(dtype, _SafeFloat64DType)
        or dtype.byte_order not in {"<", ">"}
        or not isinstance(payload, bytes)
        or len(payload) != 8
    ):
        raise pickle.UnpicklingError("unsupported NumPy scalar")
    return struct.unpack(f"{dtype.byte_order}d", payload)[0]


class _RestrictedNuPlanUnpickler(pickle.Unpickler):
    """Unpickler allowing only globals used by materialized camera metadata."""

    def find_class(self, module: str, name: str) -> Any:
        if module == _NUPLAN_TYPES_MODULE and name in _NUPLAN_LIST_TYPES:
            return _NuPlanList
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and (
            name == "scalar"
        ):
            return _safe_float64_scalar
        if module == "numpy" and name == "dtype":
            return _safe_float64_dtype
        raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")


def read_ego_poses(frames: Sequence[FrameRef]) -> tuple[RawEgoPose, ...]:
    """Read raw ego poses in frame order.

    Each unique database is opened once in read-only mode for the duration
    of the call. Repeated tokens are queried once per database and remain
    repeated in the returned tuple.

    Args:
        frames: Role-neutral frames with canonical database paths and
            16-digit lowercase hexadecimal ego-pose tokens.

    Returns:
        Raw ego poses in the same order and with the same length as
        ``frames``. An empty input returns an empty tuple.

    Raises:
        NuPlanDataError: A token is malformed, a row is missing or
            non-unique, or a selected value violates the numeric contract.
        sqlite3.Error: SQLite cannot open, query, or close the database.
    """
    grouped: dict[Path, list[tuple[int, str]]] = {}
    for index, frame in enumerate(frames):
        token = frame.ego_pose_token
        if _TOKEN.fullmatch(token) is None:
            raise NuPlanDataError(
                f"invalid ego_pose_token {token!r}: expected 16 lowercase hex digits"
            )
        grouped.setdefault(frame.db_path, []).append((index, token))

    poses: dict[int, RawEgoPose] = {}
    for db_path, requests in grouped.items():
        connection = _open_read_only(db_path)
        cache: dict[str, RawEgoPose] = {}
        try:
            for index, token in requests:
                pose = cache.get(token)
                if pose is None:
                    pose = _read_pose(connection, db_path, token)
                    cache[token] = pose
                poses[index] = pose
        finally:
            connection.close()
    return tuple(poses[index] for index in range(len(frames)))


def read_camera_calibrations(
    frames: Sequence[FrameRef],
) -> tuple[RawCameraCalibration, ...]:
    """Read raw camera calibrations in frame order.

    Each database is opened once in read-only mode. Repeated camera tokens are
    queried and decoded once per database, while repeated results remain in
    their original frame positions.

    Args:
        frames: Role-neutral frames with canonical database paths and
            16-digit lowercase hexadecimal camera tokens.

    Returns:
        Raw calibrations in the same order and with the same length as
        ``frames``. An empty input returns an empty tuple.

    Raises:
        NuPlanDataError: A token is malformed, a row is missing or
            non-unique, the channel disagrees with its frame, a pickle is
            unsafe, or a selected value violates the calibration contract.
        sqlite3.Error: SQLite cannot open, query, or close the database.
    """
    grouped: dict[Path, list[tuple[int, FrameRef]]] = {}
    for index, frame in enumerate(frames):
        token = frame.camera_token
        if _TOKEN.fullmatch(token) is None:
            raise NuPlanDataError(
                f"invalid camera_token {token!r}: expected 16 lowercase hex digits"
            )
        grouped.setdefault(frame.db_path, []).append((index, frame))

    calibrations: dict[int, RawCameraCalibration] = {}
    for db_path, requests in grouped.items():
        connection = _open_read_only(db_path)
        cache: dict[str, tuple[str, RawCameraCalibration]] = {}
        try:
            for index, frame in requests:
                selected = cache.get(frame.camera_token)
                if selected is None:
                    selected = _read_camera(
                        connection,
                        db_path,
                        frame.camera_token,
                    )
                    cache[frame.camera_token] = selected
                channel, calibration = selected
                if channel != frame.channel:
                    raise NuPlanDataError(
                        f"camera_token {frame.camera_token!r} in {db_path} "
                        f"has channel {channel!r}, expected {frame.channel!r}"
                    )
                calibrations[index] = calibration
        finally:
            connection.close()
    return tuple(calibrations[index] for index in range(len(frames)))


def _open_read_only(db_path: Path) -> sqlite3.Connection:
    """Open an existing SQLite database without write permission."""
    return sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)


def _read_pose(
    connection: sqlite3.Connection,
    db_path: Path,
    token: str,
) -> RawEgoPose:
    """Read and validate exactly one ego-pose row."""
    rows = connection.execute(_QUERY, (bytes.fromhex(token),)).fetchmany(2)
    if len(rows) != 1:
        reason = "not found" if not rows else "not unique"
        raise NuPlanDataError(f"ego_pose_token {token!r} is {reason} in {db_path}")

    timestamp, *values, epsg = rows[0]
    numbers = tuple(
        _finite_float(value, field, db_path)
        for field, value in zip(_FLOAT_FIELDS, values, strict=True)
    )
    return RawEgoPose(
        _integer(timestamp, "timestamp", db_path, minimum=0),
        *numbers,
        _integer(epsg, "epsg", db_path, minimum=1),
    )


def _read_camera(
    connection: sqlite3.Connection,
    db_path: Path,
    token: str,
) -> tuple[str, RawCameraCalibration]:
    """Read, safely decode, and validate exactly one camera row."""
    rows = connection.execute(
        _CAMERA_QUERY,
        {
            "blob_fetch_bytes": _MAX_CAMERA_BLOB_BYTES + 1,
            "text_fetch_chars": _MAX_CAMERA_TEXT_CHARS + 1,
            "token": bytes.fromhex(token),
        },
    ).fetchmany(2)
    if len(rows) != 1:
        reason = "not found" if not rows else "not unique"
        raise NuPlanDataError(f"camera_token {token!r} is {reason} in {db_path}")

    (
        channel,
        model,
        translation_blob,
        rotation_blob,
        intrinsics_blob,
        distortion_blob,
        width,
        height,
    ) = rows[0]
    translation = cast(
        tuple[float, float, float],
        _numeric_tuple(
            _decode_camera_blob(
                translation_blob,
                field="translation",
                db_path=db_path,
            ),
            field="translation",
            length=3,
            db_path=db_path,
        ),
    )
    rotation = cast(
        tuple[float, float, float, float],
        _numeric_tuple(
            _decode_camera_blob(
                rotation_blob,
                field="rotation",
                db_path=db_path,
            ),
            field="rotation",
            length=4,
            db_path=db_path,
        ),
    )
    intrinsics = _intrinsics(
        _decode_camera_blob(
            intrinsics_blob,
            field="intrinsic",
            db_path=db_path,
        ),
        db_path,
    )
    distortion = cast(
        tuple[float, float, float, float, float],
        _numeric_tuple(
            _decode_camera_blob(
                distortion_blob,
                field="distortion",
                db_path=db_path,
            ),
            field="distortion",
            length=5,
            db_path=db_path,
        ),
    )
    calibration = RawCameraCalibration(
        model=_text(model, "model", db_path),
        translation_xyz=translation,
        rotation_wxyz=rotation,
        intrinsics=intrinsics,
        distortion=distortion,
        image_size_hw=(
            _camera_integer(height, "height", db_path),
            _camera_integer(width, "width", db_path),
        ),
    )
    return _text(channel, "channel", db_path), calibration


def _decode_camera_blob(
    payload: object,
    *,
    field: str,
    db_path: Path,
) -> Any:
    """Decode one bounded pickle while denying all unrelated globals."""
    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > _MAX_CAMERA_BLOB_BYTES
    ):
        raise NuPlanDataError(
            f"camera.{field} in {db_path} must be a non-empty "
            f"pickle BLOB of at most {_MAX_CAMERA_BLOB_BYTES} bytes"
        )
    stream = io.BytesIO(payload)
    try:
        stop_position: int | None = None
        for opcode, _argument, position in pickletools.genops(payload):
            if opcode.name in _FORBIDDEN_PICKLE_OPCODES:
                raise NuPlanDataError(
                    f"camera.{field} in {db_path} uses forbidden pickle "
                    f"opcode {opcode.name}"
                )
            if opcode.name == "STOP":
                stop_position = position
        if stop_position is None:
            raise NuPlanDataError(
                f"camera.{field} in {db_path} has no pickle STOP"
            )
        if stop_position != len(payload) - 1:
            raise NuPlanDataError(
                f"camera.{field} in {db_path} has trailing pickle bytes"
            )
        value = _RestrictedNuPlanUnpickler(
            stream,
            fix_imports=False,
        ).load()
    except NuPlanDataError:
        raise
    except MemoryError:
        raise
    except Exception as error:
        raise NuPlanDataError(
            f"cannot safely decode camera.{field} in {db_path}"
        ) from error
    return value


def _numeric_tuple(
    value: object,
    *,
    field: str,
    length: int,
    db_path: Path,
) -> tuple[float, ...]:
    """Validate a fixed-length sequence of finite real numbers."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise NuPlanDataError(
            f"camera.{field} in {db_path} must have length {length}"
        )
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise NuPlanDataError(
                f"camera.{field} in {db_path} must contain only real numbers"
            )
        try:
            number = float(item)
        except (OverflowError, ValueError) as error:
            raise NuPlanDataError(
                f"camera.{field} in {db_path} must contain representable numbers"
            ) from error
        if not math.isfinite(number):
            raise NuPlanDataError(
                f"camera.{field} in {db_path} must contain only finite numbers"
            )
        result.append(number)
    return tuple(result)


def _intrinsics(
    value: object,
    db_path: Path,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Validate a native finite pinhole matrix without changing its values."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise NuPlanDataError(f"camera.intrinsic in {db_path} must have shape (3, 3)")
    rows = tuple(
        _numeric_tuple(
            row,
            field="intrinsic",
            length=3,
            db_path=db_path,
        )
        for row in value
    )
    if rows[0][0] <= 0.0 or rows[1][1] <= 0.0:
        raise NuPlanDataError(
            f"camera.intrinsic in {db_path} must have positive focal lengths"
        )
    if not (
        math.isclose(rows[1][0], 0.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(rows[2][0], 0.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(rows[2][1], 0.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(rows[2][2], 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise NuPlanDataError(
            f"camera.intrinsic in {db_path} has an invalid pinhole structure"
        )
    return cast(
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        rows,
    )


def _text(value: object, field: str, db_path: Path) -> str:
    """Validate meaningful database text without normalizing it."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_CAMERA_TEXT_CHARS
    ):
        raise NuPlanDataError(
            f"camera.{field} in {db_path} must be non-empty text of at most "
            f"{_MAX_CAMERA_TEXT_CHARS} characters"
        )
    return value


def _camera_integer(value: object, field: str, db_path: Path) -> int:
    """Validate a positive SQLite integer from a camera row."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NuPlanDataError(
            f"camera.{field} in {db_path} must be a positive integer"
        )
    return value


def _integer(value: object, field: str, db_path: Path, *, minimum: int) -> int:
    """Validate a SQLite integer against an inclusive lower bound."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NuPlanDataError(
            f"ego_pose.{field} in {db_path} must be an integer >= {minimum}"
        )
    return value


def _finite_float(value: object, field: str, db_path: Path) -> float:
    """Convert a SQLite numeric value to a finite Python float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NuPlanDataError(f"ego_pose.{field} in {db_path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise NuPlanDataError(f"ego_pose.{field} in {db_path} must be finite")
    return result

"""Temporary NumPy helpers still used by Waymo depth and DDW code."""

from __future__ import annotations

from numbers import Real
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt

from novel_view.inputs.waymo.types import WaymoContractError


_ArrayScalar = TypeVar("_ArrayScalar", bound=np.generic)


def readonly_owned(value: npt.NDArray[_ArrayScalar]) -> npt.NDArray[_ArrayScalar]:
    """Return an owned C-contiguous read-only copy."""
    result = np.array(value, dtype=value.dtype, order="C", copy=True)
    result.setflags(write=False)
    return cast(npt.NDArray[_ArrayScalar], result)


def require_array(
    value: object,
    name: str,
    *,
    dtype: np.dtype[np.generic],
    shape: tuple[int | None, ...],
) -> npt.NDArray[np.generic]:
    """Validate an array at an existing Waymo depth or DDW boundary."""
    if not isinstance(value, np.ndarray):
        raise WaymoContractError(f"{name} must be a numpy.ndarray")
    if value.dtype != dtype:
        raise WaymoContractError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise WaymoContractError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.owndata or not value.flags.c_contiguous or value.flags.writeable:
        raise WaymoContractError(
            f"{name} must be an owned C-contiguous read-only array"
        )
    return value


def finite_float(value: object, name: str) -> float:
    """Normalize one finite real scalar while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise WaymoContractError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise WaymoContractError(f"{name} must be a finite real number")
    return result


__all__ = ["finite_float", "readonly_owned", "require_array"]

"""Literal Python interpreters inside the supported container image."""

from pathlib import Path


CORE_PYTHON = Path("/opt/envs/core/bin/python")
GEN3C_PYTHON = Path("/opt/envs/gen3c/bin/python")
VGGT_PYTHON = Path("/opt/envs/vggt/bin/python")


__all__ = ["CORE_PYTHON", "GEN3C_PYTHON", "VGGT_PYTHON"]

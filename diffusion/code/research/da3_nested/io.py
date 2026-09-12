"""Write transient source arrays for the frozen DA3 research worker."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from novel_view.inputs.euvs.rgb import EuvsRasterizedSequence


def write_source_rgb(
    source: EuvsRasterizedSequence,
    destination: Path,
) -> None:
    """Write ordered uint8 RGB once as a transient memory-mappable NPY."""
    height, width = source.frames[0].raster.plan.output_size_hw
    memory_map = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.uint8,
        shape=(len(source.frames), height, width, 3),
    )
    try:
        for index, frame in enumerate(source.frames):
            memory_map[index] = frame.raster.rgb
        memory_map.flush()
    finally:
        memory_map._mmap.close()

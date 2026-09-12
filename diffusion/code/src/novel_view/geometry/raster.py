"""Чистая геометрия resize-to-cover и центрального crop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class CoverCropTransform:
    """Resize-to-cover и центральный crop в координатах границ пикселей."""

    source_size_hw: tuple[int, int]
    output_size_hw: tuple[int, int]

    @property
    def resized_size_hw(self) -> tuple[int, int]:
        """Вернуть минимальный целочисленный растр, покрывающий выход."""
        return _cover_size(self.source_size_hw, self.output_size_hw)

    @property
    def scale_x(self) -> float:
        """Вернуть фактический горизонтальный масштаб после округления."""
        return self.resized_size_hw[1] / self.source_size_hw[1]

    @property
    def scale_y(self) -> float:
        """Вернуть фактический вертикальный масштаб после округления."""
        return self.resized_size_hw[0] / self.source_size_hw[0]

    @property
    def crop_left(self) -> int:
        """Вернуть число столбцов, удалённых слева."""
        return (self.resized_size_hw[1] - self.output_size_hw[1]) // 2

    @property
    def crop_top(self) -> int:
        """Вернуть число строк, удалённых сверху."""
        return (self.resized_size_hw[0] - self.output_size_hw[0]) // 2

    @property
    def pixel_transform(self) -> npt.NDArray[np.float64]:
        """Вернуть однородное преобразование source pixels → output pixels."""
        transform = np.array(
            [
                [self.scale_x, 0.0, -float(self.crop_left)],
                [0.0, self.scale_y, -float(self.crop_top)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        transform.flags.writeable = False
        return transform

    def transform_intrinsics(
        self,
        rectified_intrinsics: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Выразить уже выпрямленную pinhole-матрицу на выходном растре."""
        transformed = self.pixel_transform @ np.asarray(
            rectified_intrinsics,
            dtype=np.float64,
        )
        transformed.flags.writeable = False
        return transformed


def _cover_size(
    source_size_hw: tuple[int, int],
    output_size_hw: tuple[int, int],
) -> tuple[int, int]:
    """Вычислить минимальный целочисленный cover без float-отношений."""
    source_height, source_width = source_size_hw
    output_height, output_width = output_size_hw

    width_scaled_height = source_height * output_width
    height_scaled_width = source_width * output_height
    if width_scaled_height >= height_scaled_width:
        return _ceil_div(width_scaled_height, source_width), output_width
    return output_height, _ceil_div(height_scaled_width, source_height)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator

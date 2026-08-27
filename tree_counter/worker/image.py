"""Reading a tile and fitting it to a model input.

Tiles are raw 8-bit RGB bytes. The protocol message already carries the
exact dimensions, so no image codec is needed on either side: QGIS writes
the bytes it read from the raster provider, and the worker reshapes them.
That keeps an image decoder out of the runtime for a file the plugin wrote
itself moments earlier.

Every coordinate the model produces is mapped back through the same
transform that produced its input, so a padded edge tile reports boxes in
the tile's own pixel space.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_counter.errors import ErrorCode, TreeCounterError

CHANNELS = 3
PAD_VALUE = 114
MAX_TILE_BYTES = 64 * 1024 * 1024


class TileError(TreeCounterError):
    """A tile could not be read or does not match its declared shape."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            ErrorCode.WORKER_PROCESS_FAILURE, diagnostic_detail=detail
        )


@dataclass(frozen=True)
class LetterboxTransform:
    """How a tile was scaled and padded to reach the model input."""

    scale: float
    pad_x: float
    pad_y: float

    def to_tile(self, x: float, y: float) -> tuple[float, float]:
        """Map a model-space coordinate back to tile-local pixels."""

        return ((x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale)

    def box_to_tile(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> tuple[float, float, float, float]:
        """Map a model-space box back to tile-local pixels."""

        left, top = self.to_tile(x_min, y_min)
        right, bottom = self.to_tile(x_max, y_max)
        return (left, top, right, bottom)


def read_rgb_tile(path: str | Path, width: int, height: int) -> Any:
    """Return an ``(height, width, 3)`` uint8 array for a raw RGB tile."""

    import numpy

    if width <= 0 or height <= 0:
        raise TileError("tile dimensions must be positive")
    expected = width * height * CHANNELS
    if expected > MAX_TILE_BYTES:
        raise TileError("the tile is larger than the supported maximum")
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise TileError(f"the tile could not be read: {exc}") from exc
    if len(data) != expected:
        raise TileError(
            f"the tile holds {len(data)} bytes but {expected} were declared"
        )
    return numpy.frombuffer(data, dtype=numpy.uint8).reshape(
        height, width, CHANNELS
    )


def letterbox(
    image: Any,
    target_width: int,
    target_height: int,
    pad_value: int = PAD_VALUE,
    center: bool = False,
) -> tuple[Any, LetterboxTransform]:
    """Fit an RGB tile into the model input without distorting it.

    A tile smaller than the model input is padded rather than stretched, so
    pixels keep their scale and a detection's size stays meaningful. Only a
    tile larger than the input is scaled down, and never up.
    """

    import numpy

    if image.ndim != 3 or image.shape[2] != CHANNELS:
        raise TileError("a tile must be a three-channel RGB array")
    if target_width <= 0 or target_height <= 0:
        raise TileError("model input dimensions must be positive")
    height, width = int(image.shape[0]), int(image.shape[1])
    scale = min(target_width / width, target_height / height, 1.0)
    scaled_width = max(1, min(target_width, int(round(width * scale))))
    scaled_height = max(1, min(target_height, int(round(height * scale))))

    if (scaled_width, scaled_height) != (width, height):
        rows = (
            numpy.linspace(0, height - 1, scaled_height)
            .round()
            .astype(numpy.int64)
        )
        columns = (
            numpy.linspace(0, width - 1, scaled_width)
            .round()
            .astype(numpy.int64)
        )
        resized = image[rows][:, columns]
    else:
        resized = image

    canvas = numpy.full(
        (target_height, target_width, CHANNELS), pad_value, dtype=numpy.uint8
    )
    if center:
        pad_x = (target_width - scaled_width) // 2
        pad_y = (target_height - scaled_height) // 2
    else:
        pad_x = 0
        pad_y = 0
    canvas[
        pad_y:pad_y + scaled_height, pad_x:pad_x + scaled_width
    ] = resized
    return canvas, LetterboxTransform(
        scale=scaled_width / width, pad_x=float(pad_x), pad_y=float(pad_y)
    )


def to_model_input(image: Any) -> Any:
    """Return a normalized ``(1, 3, H, W)`` float32 batch for the model."""

    import numpy

    if image.ndim != 3 or image.shape[2] != CHANNELS:
        raise TileError("a model input must be a three-channel RGB array")
    batch = image.astype(numpy.float32) / 255.0
    return numpy.ascontiguousarray(batch.transpose(2, 0, 1)[None, ...])


__all__ = [
    "CHANNELS",
    "MAX_TILE_BYTES",
    "PAD_VALUE",
    "LetterboxTransform",
    "TileError",
    "letterbox",
    "read_rgb_tile",
    "to_model_input",
]

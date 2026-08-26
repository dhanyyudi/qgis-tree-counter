"""Deterministic, QGIS-free raster tiling helpers."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from tree_counter.core.types import TileWindow
from tree_counter.core.validation import validate_overlap_percent
from tree_counter.errors import ValidationError


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a non-negative integer")
    return value


def _axis_starts(size: int, tile_size: int, overlap_percent: int) -> list[int]:
    """Return ordered starts whose fixed-size windows cover one axis."""

    if size <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap_percent / 100)))
    final_start = size - tile_size
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def iter_tile_windows(
    width: int,
    height: int,
    tile_size: int,
    overlap_percent: int,
    origin_x: int = 0,
    origin_y: int = 0,
) -> tuple[TileWindow, ...]:
    """Return a deterministic row-major set of source tile windows.

    ``read_width`` and ``read_height`` describe the source pixels available at
    an edge.  ``model_width`` and ``model_height`` describe the padded model
    input and are always ``tile_size`` for an edge smaller than the model.
    Origins are offsets in the caller's global pixel coordinate system.
    """

    width = _positive_integer(width, "width")
    height = _positive_integer(height, "height")
    tile_size = _positive_integer(tile_size, "tile_size")
    origin_x = _non_negative_integer(origin_x, "origin_x")
    origin_y = _non_negative_integer(origin_y, "origin_y")
    overlap_percent = validate_overlap_percent(overlap_percent)

    x_starts = _axis_starts(width, tile_size, overlap_percent)
    y_starts = _axis_starts(height, tile_size, overlap_percent)
    windows: list[TileWindow] = []
    for row_index, y_start in enumerate(y_starts):
        for column_index, x_start in enumerate(x_starts):
            read_width = min(tile_size, width - x_start)
            read_height = min(tile_size, height - y_start)
            windows.append(
                TileWindow(
                    tile_id=f"r{row_index:05d}_c{column_index:05d}",
                    x_offset=origin_x + x_start,
                    y_offset=origin_y + y_start,
                    read_width=read_width,
                    read_height=read_height,
                    model_width=tile_size,
                    model_height=tile_size,
                )
            )
    return tuple(windows)


__all__ = ["iter_tile_windows"]

"""Turning global pixel detections into map coordinates.

Pixel space has its origin at the raster's top-left with y increasing
downwards; map space has y increasing upwards. The conversion is pure
arithmetic over the raster's own extent and size, so it is testable
without QGIS and identical on every platform.

A detection's position is its box centre. The box itself is also carried
through, because the optional polygon layer draws exactly the box the
model predicted rather than a square derived from the centre.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from typing import Any

from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.qgis_adapter.raster import RasterInfo

MapPoint = tuple[float, float]
MapRect = tuple[float, float, float, float]


class GeoreferenceError(TreeCounterError):
    """A detection could not be placed in map coordinates."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.OUTPUT_FAILURE, diagnostic_detail=detail)


def pixel_to_map(info: RasterInfo, x: float, y: float) -> MapPoint:
    """Return the map coordinate of a pixel position."""

    for value, name in ((x, "x"), (y, "y")):
        if not isinstance(value, (int, float)) or not math.isfinite(
            float(value)
        ):
            raise GeoreferenceError(f"{name} must be a finite number")
    return (
        info.x_minimum + float(x) * info.pixel_size_x,
        info.y_maximum - float(y) * info.pixel_size_y,
    )


def pixel_box_to_map(info: RasterInfo, box: Any) -> MapRect:
    """Return the map rectangle covering a pixel box."""

    try:
        left, top = pixel_to_map(info, box.x_min, box.y_min)
        right, bottom = pixel_to_map(info, box.x_max, box.y_max)
    except AttributeError as exc:
        raise GeoreferenceError("a detection box is required") from exc
    # A pixel box grows downwards, so its lower map y comes from y_max.
    return (left, bottom, right, top)


def detection_center_map(info: RasterInfo, detection: Any) -> MapPoint:
    """Return the map coordinate of a detection's centre."""

    box = getattr(detection, "box", None)
    if box is None:
        raise GeoreferenceError("a detection box is required")
    return pixel_to_map(
        info, (box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0
    )


def map_point(x: float, y: float) -> Any:
    """Return a QGIS point geometry for a map coordinate."""

    from qgis.core import QgsGeometry, QgsPointXY

    return QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y)))


def map_box(rect: MapRect) -> Any:
    """Return a QGIS polygon geometry for a map rectangle."""

    from qgis.core import QgsGeometry, QgsRectangle

    x_min, y_min, x_max, y_max = rect
    return QgsGeometry.fromRect(
        QgsRectangle(float(x_min), float(y_min), float(x_max), float(y_max))
    )


__all__ = [
    "GeoreferenceError",
    "MapPoint",
    "MapRect",
    "detection_center_map",
    "map_box",
    "map_point",
    "pixel_box_to_map",
    "pixel_to_map",
]

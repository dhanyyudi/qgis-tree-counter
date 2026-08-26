"""QGIS-free affine, polygon, and detection scope geometry."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from tree_counter.core.validation import (
    validate_pixel_box,
    validate_pixel_coordinate,
)
from tree_counter.errors import ValidationError

PixelRect = tuple[int, int, int, int]
Point = tuple[float, float]


def _finite(value: object, name: str) -> float:
    return validate_pixel_coordinate(value, name)


def _dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def _rectangle(rectangle: Sequence[object], name: str) -> tuple[float, ...]:
    if isinstance(rectangle, (str, bytes)):
        raise ValidationError(f"{name} must contain four coordinates")
    try:
        values = tuple(rectangle)
    except TypeError as exc:
        raise ValidationError(f"{name} must contain four coordinates") from exc
    if len(values) != 4:
        raise ValidationError(f"{name} must contain four coordinates")
    return tuple(
        _finite(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    )


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def map_rect_to_pixel_rect(
    map_rect: Sequence[object],
    origin_x: float,
    origin_y: float,
    pixel_width: float,
    pixel_height: float,
    raster_width: int,
    raster_height: int,
) -> PixelRect:
    """Convert a map-space rectangle to clamped exclusive pixel bounds.

    The affine is north-up: x resolution is positive and y resolution is
    negative. Map coordinates may be supplied in either order. Floor/ceil
    preserve every touched source pixel at the exclusive integer edges.
    """

    x_min, y_min, x_max, y_max = _rectangle(map_rect, "map_rect")
    origin_x = _finite(origin_x, "origin_x")
    origin_y = _finite(origin_y, "origin_y")
    pixel_width = _finite(pixel_width, "pixel_width")
    pixel_height = _finite(pixel_height, "pixel_height")
    if pixel_width <= 0:
        raise ValidationError("pixel_width must be positive")
    if pixel_height >= 0:
        raise ValidationError(
            "pixel_height must be negative for north-up data"
        )
    raster_width = _dimension(raster_width, "raster_width")
    raster_height = _dimension(raster_height, "raster_height")

    map_x_min, map_x_max = sorted((x_min, x_max))
    map_y_min, map_y_max = sorted((y_min, y_max))
    raw_columns = (
        (map_x_min - origin_x) / pixel_width,
        (map_x_max - origin_x) / pixel_width,
    )
    raw_rows = (
        (map_y_min - origin_y) / pixel_height,
        (map_y_max - origin_y) / pixel_height,
    )
    column_min = math.floor(min(raw_columns))
    column_max = math.ceil(max(raw_columns))
    row_min = math.floor(min(raw_rows))
    row_max = math.ceil(max(raw_rows))

    column_min = _clamp(column_min, 0, raster_width)
    column_max = _clamp(column_max, 0, raster_width)
    row_min = _clamp(row_min, 0, raster_height)
    row_max = _clamp(row_max, 0, raster_height)
    if column_max < column_min:
        column_max = column_min
    if row_max < row_min:
        row_max = row_min
    return (column_min, row_min, column_max, row_max)


map_extent_to_pixel_rect = map_rect_to_pixel_rect


def _point(point: Sequence[object]) -> Point:
    if isinstance(point, (str, bytes)):
        raise ValidationError("point must contain two finite coordinates")
    try:
        values = tuple(point)
    except TypeError as exc:
        raise ValidationError(
            "point must contain two finite coordinates"
        ) from exc
    if len(values) != 2:
        raise ValidationError("point must contain two finite coordinates")
    return (_finite(values[0], "point.x"), _finite(values[1], "point.y"))


def _rings(
    rings: Iterable[Sequence[Sequence[object]]],
) -> tuple[tuple[Point, ...], ...]:
    if isinstance(rings, (str, bytes)):
        raise ValidationError("rings must contain polygon rings")
    try:
        ring_values = tuple(rings)
    except TypeError as exc:
        raise ValidationError("rings must contain polygon rings") from exc
    if not ring_values:
        raise ValidationError("rings must contain an outer ring")
    normalized: list[tuple[Point, ...]] = []
    for ring_index, ring in enumerate(ring_values):
        if isinstance(ring, (str, bytes)):
            raise ValidationError(f"ring {ring_index} is invalid")
        try:
            points = tuple(_point(item) for item in ring)
        except TypeError as exc:
            raise ValidationError(f"ring {ring_index} is invalid") from exc
        if len(points) < 3:
            raise ValidationError(
                f"ring {ring_index} must have at least three points"
            )
        area = sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        if area == 0:
            raise ValidationError(f"ring {ring_index} must enclose an area")
        normalized.append(points)
    return tuple(normalized)


def _on_segment(point: Point, start: Point, end: Point) -> bool:
    cross = (point[0] - start[0]) * (end[1] - start[1]) - (
        point[1] - start[1]
    ) * (end[0] - start[0])
    if cross != 0.0:
        return False
    return (
        min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
        and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])
    )


def _point_in_ring(point: Point, ring: tuple[Point, ...]) -> tuple[bool, bool]:
    inside = False
    for index, start in enumerate(ring):
        end = ring[(index + 1) % len(ring)]
        if _on_segment(point, start, end):
            return (True, True)
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = (end[0] - start[0]) * (point[1] - start[1]) / (
                end[1] - start[1]
            ) + start[0]
            if point[0] < crossing_x:
                inside = not inside
    return (inside, False)


def point_in_polygon(
    point: Sequence[object], rings: Iterable[Sequence[Sequence[object]]]
) -> bool:
    """Return whether a point is in an outer ring and not a hole.

    Boundary points are included, including boundaries of holes. Ring
    orientation is irrelevant; the first ring is the outer ring and later
    rings are holes.
    """

    normalized_point = _point(point)
    normalized_rings = _rings(rings)
    outer_inside, outer_boundary = _point_in_ring(
        normalized_point, normalized_rings[0]
    )
    if outer_boundary:
        return True
    if not outer_inside:
        return False
    for hole in normalized_rings[1:]:
        hole_inside, hole_boundary = _point_in_ring(normalized_point, hole)
        if hole_boundary:
            return True
        if hole_inside:
            return False
    return True


def _pixel_rect(rect: Sequence[object]) -> PixelRect:
    try:
        original_values = tuple(rect)
    except TypeError as exc:
        raise ValidationError(
            "pixel_rect must contain four coordinates"
        ) from exc
    values = _rectangle(original_values, "pixel_rect")
    integer_values: list[int] = []
    for index, value in enumerate(values):
        original = original_values[index]
        if not isinstance(original, int) or isinstance(original, bool):
            raise ValidationError(f"pixel_rect[{index}] must be an integer")
        integer_values.append(int(value))
    col_min, row_min, col_max, row_max = integer_values
    if min(col_min, row_min, col_max, row_max) < 0:
        raise ValidationError("pixel_rect bounds must be non-negative")
    if col_max < col_min or row_max < row_min:
        raise ValidationError("pixel_rect bounds must be ordered")
    return (col_min, row_min, col_max, row_max)


def pixel_box_center(box: Any) -> Point:
    """Return a validated PixelBox-like center point."""

    validate_pixel_box(box)
    return ((box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2)


def _detection_box(detection: Any) -> Any:
    try:
        return detection.box
    except AttributeError as exc:
        raise ValidationError("detection must expose a PixelBox box") from exc


def filter_detections_by_pixel_rect(
    detections: Iterable[Any], pixel_rect: Sequence[object]
) -> tuple[Any, ...]:
    """Keep detections whose PixelBox center lies in an inclusive rect."""

    col_min, row_min, col_max, row_max = _pixel_rect(pixel_rect)
    filtered: list[Any] = []
    for detection in detections:
        center_x, center_y = pixel_box_center(_detection_box(detection))
        if col_min <= center_x <= col_max and row_min <= center_y <= row_max:
            filtered.append(detection)
    return tuple(filtered)


def filter_detections_by_polygon(
    detections: Iterable[Any], rings: Iterable[Sequence[Sequence[object]]]
) -> tuple[Any, ...]:
    """Keep detections whose PixelBox center lies in a polygon mask."""

    normalized_rings = _rings(rings)
    return tuple(
        detection
        for detection in detections
        if point_in_polygon(
            pixel_box_center(_detection_box(detection)), normalized_rings
        )
    )


__all__ = [
    "PixelRect",
    "filter_detections_by_pixel_rect",
    "filter_detections_by_polygon",
    "map_extent_to_pixel_rect",
    "map_rect_to_pixel_rect",
    "pixel_box_center",
    "point_in_polygon",
]

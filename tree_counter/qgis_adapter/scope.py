"""Turning a user's chosen processing scope into raster pixel space.

Three scopes are offered: the whole raster, the current map extent, or a
polygon layer or its selected features. Whatever the user picks is
transformed into the raster's own CRS and clamped to the raster, because
every later step - tiling, detection, deduplication - works in pixels.

A polygon scope also keeps its rings in pixel space so a detection can be
kept or discarded by whether its centre falls inside the polygon. Boxes may
extend past the boundary; centres decide membership.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tree_counter.core.geometry import map_rect_to_pixel_rect, point_in_polygon
from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.qgis_adapter.raster import RasterInfo

Ring = tuple[tuple[float, float], ...]
# One polygon part: its outer ring first, then any holes. A scope holds
# several parts, because a multipart geometry and several selected
# features are independent areas, not holes in one another.
Polygon = tuple[Ring, ...]


class ScopeRejected(TreeCounterError):
    """The processing scope is empty, invalid, or misses the raster."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.INVALID_SCOPE, diagnostic_detail=detail)


class ScopeKind(str, Enum):
    """The processing scopes offered in v1."""

    WHOLE_RASTER = "whole_raster"
    MAP_EXTENT = "map_extent"
    POLYGON = "polygon"


@dataclass(frozen=True)
class PixelScope:
    """The pixel window to process, and any polygon mask inside it."""

    kind: ScopeKind
    column_min: int
    row_min: int
    column_max: int
    row_max: int
    polygons: tuple[Polygon, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.column_max <= self.column_min or self.row_max <= self.row_min:
            raise ScopeRejected(
                "the processing scope does not cover any pixels"
            )

    @property
    def width(self) -> int:
        """Return the scope width in pixels."""

        return self.column_max - self.column_min

    @property
    def height(self) -> int:
        """Return the scope height in pixels."""

        return self.row_max - self.row_min

    @property
    def pixel_count(self) -> int:
        """Return how many source pixels the scope covers."""

        return self.width * self.height

    def contains_center(self, x: float, y: float) -> bool:
        """Return whether a global pixel centre belongs to this scope.

        Without a polygon this is the rectangle test; with one the centre
        must also fall inside the mask.
        """

        if not (
            self.column_min <= x <= self.column_max
            and self.row_min <= y <= self.row_max
        ):
            return False
        if not self.polygons:
            return True
        return any(
            point_in_polygon((x, y), polygon) for polygon in self.polygons
        )


def whole_raster_scope(info: RasterInfo) -> PixelScope:
    """Return the scope covering every pixel of the raster."""

    return PixelScope(
        kind=ScopeKind.WHOLE_RASTER,
        column_min=0,
        row_min=0,
        column_max=int(info.width),
        row_max=int(info.height),
    )


def scope_from_map_rect(
    info: RasterInfo,
    map_rect: Sequence[float],
    kind: ScopeKind = ScopeKind.MAP_EXTENT,
    polygons: Iterable[Iterable[Sequence[Sequence[float]]]] = (),
) -> PixelScope:
    """Return a clamped pixel scope for a rectangle in the raster's CRS."""

    column_min, row_min, column_max, row_max = map_rect_to_pixel_rect(
        map_rect,
        origin_x=info.x_minimum,
        origin_y=info.y_maximum,
        pixel_width=info.pixel_size_x,
        pixel_height=-info.pixel_size_y,
        raster_width=int(info.width),
        raster_height=int(info.height),
    )
    if column_max <= column_min or row_max <= row_min:
        raise ScopeRejected(
            "the selected area does not overlap the raster"
        )
    return PixelScope(
        kind=kind,
        column_min=column_min,
        row_min=row_min,
        column_max=column_max,
        row_max=row_max,
        polygons=tuple(
            tuple(
                tuple((float(x), float(y)) for x, y in ring)
                for ring in polygon
            )
            for polygon in polygons
        ),
    )


def map_ring_to_pixels(
    info: RasterInfo, ring: Sequence[Sequence[float]]
) -> Ring:
    """Convert a ring in the raster's CRS to global pixel coordinates."""

    pixel_x = info.pixel_size_x
    pixel_y = info.pixel_size_y
    if pixel_x <= 0 or pixel_y <= 0:
        raise ScopeRejected("the raster has no usable pixel size")
    converted = tuple(
        (
            (float(x) - info.x_minimum) / pixel_x,
            (info.y_maximum - float(y)) / pixel_y,
        )
        for x, y in ring
    )
    if len(converted) < 3:
        raise ScopeRejected("a polygon ring needs at least three points")
    return converted


# -- QGIS-facing helpers ------------------------------------------------


def _transform(source_crs: Any, target_crs: Any, project: Any = None) -> Any:
    from qgis.core import QgsCoordinateTransform, QgsProject

    context = (QgsProject.instance() if project is None else project)
    return QgsCoordinateTransform(source_crs, target_crs, context)


def transform_rect_to_raster(
    rect: Any, source_crs: Any, info_crs: Any, project: Any = None
) -> tuple[float, float, float, float]:
    """Return a map rectangle transformed into the raster's CRS."""

    if source_crs == info_crs:
        transformed = rect
    else:
        try:
            transformed = _transform(
                source_crs, info_crs, project
            ).transformBoundingBox(rect)
        except Exception as exc:
            raise ScopeRejected(
                f"the selected area could not be transformed: "
                f"{type(exc).__name__}"
            ) from exc
    return (
        float(transformed.xMinimum()),
        float(transformed.yMinimum()),
        float(transformed.xMaximum()),
        float(transformed.yMaximum()),
    )


def scope_from_map_extent(
    info: RasterInfo,
    rect: Any,
    source_crs: Any,
    raster_crs: Any,
    project: Any = None,
) -> PixelScope:
    """Return the pixel scope for the current map extent."""

    map_rect = transform_rect_to_raster(rect, source_crs, raster_crs, project)
    return scope_from_map_rect(info, map_rect, ScopeKind.MAP_EXTENT)


def _geometry_parts(geometry: Any) -> list[list[list[tuple[float, float]]]]:
    """Return each polygon part as its own list of rings.

    Parts are kept separate because the first ring of a part is its outer
    boundary and the rest are its holes. Flattening every part into one
    list would turn the second part into a hole in the first.
    """

    parts: list[list[list[tuple[float, float]]]] = []
    if geometry.isMultipart():
        polygons = geometry.asMultiPolygon()
    else:
        single = geometry.asPolygon()
        polygons = [single] if single else []
    for polygon in polygons:
        rings: list[list[tuple[float, float]]] = []
        for ring in polygon:
            points = [(float(point.x()), float(point.y())) for point in ring]
            if len(points) >= 3:
                rings.append(points)
        if rings:
            parts.append(rings)
    return parts


def scope_from_polygon_layer(
    info: RasterInfo,
    layer: Any,
    raster_crs: Any,
    selected_only: bool = True,
    project: Any = None,
) -> PixelScope:
    """Return the pixel scope for a polygon layer or its selection.

    When the user has a selection, only the selected features count; that
    is what "selected feature" means in the dock. An empty selection with
    ``selected_only`` falls back to nothing rather than silently counting
    the whole layer.
    """

    from qgis.core import QgsCoordinateTransform, QgsGeometry, QgsProject

    if layer is None or not layer.isValid():
        raise ScopeRejected("the polygon layer could not be read")
    features = (
        list(layer.getSelectedFeatures())
        if selected_only and layer.selectedFeatureCount()
        else list(layer.getFeatures())
    )
    if not features:
        raise ScopeRejected("the polygon layer has no features to process")

    source_crs = layer.crs()
    context = QgsProject.instance() if project is None else project
    transform = (
        None
        if source_crs == raster_crs
        else QgsCoordinateTransform(source_crs, raster_crs, context)
    )

    polygons: list[Polygon] = []
    for feature in features:
        geometry: Any = feature.geometry()
        if geometry is None or geometry.isEmpty():
            continue
        if transform is not None:
            working = QgsGeometry(geometry)
            if working.transform(transform) != 0:
                raise ScopeRejected(
                    "the polygon could not be transformed to the raster CRS"
                )
            geometry = working
        for part in _geometry_parts(geometry):
            polygons.append(
                tuple(map_ring_to_pixels(info, ring) for ring in part)
            )
    if not polygons:
        raise ScopeRejected("the selected polygons enclose no area")

    xs = [x for polygon in polygons for ring in polygon for x, _ in ring]
    ys = [y for polygon in polygons for ring in polygon for _, y in ring]
    column_min = max(0, int(min(xs)))
    row_min = max(0, int(min(ys)))
    column_max = min(int(info.width), int(max(xs)) + 1)
    row_max = min(int(info.height), int(max(ys)) + 1)
    if column_max <= column_min or row_max <= row_min:
        raise ScopeRejected("the selected polygons do not overlap the raster")
    return PixelScope(
        kind=ScopeKind.POLYGON,
        column_min=column_min,
        row_min=row_min,
        column_max=column_max,
        row_max=row_max,
        polygons=tuple(polygons),
    )


__all__ = [
    "PixelScope",
    "Polygon",
    "Ring",
    "ScopeKind",
    "ScopeRejected",
    "map_ring_to_pixels",
    "scope_from_map_extent",
    "scope_from_map_rect",
    "scope_from_polygon_layer",
    "transform_rect_to_raster",
    "whole_raster_scope",
]

"""Tests for the QGIS-free processing scope arithmetic."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


def _info(**overrides):
    from tree_counter.qgis_adapter.raster import RasterInfo

    values = {
        "name": "aerial",
        "provider_type": "gdal",
        "width": 1000,
        "height": 800,
        "band_count": 3,
        "is_byte": True,
        "crs_authid": "EPSG:3857",
        "crs_is_valid": True,
        # One map unit per pixel keeps the arithmetic readable.
        "x_minimum": 0.0,
        "y_minimum": 0.0,
        "x_maximum": 1000.0,
        "y_maximum": 800.0,
    }
    values.update(overrides)
    return RasterInfo(**values)


def test_the_whole_raster_scope_covers_every_pixel() -> None:
    from tree_counter.qgis_adapter.scope import ScopeKind, whole_raster_scope

    scope = whole_raster_scope(_info())

    assert (scope.column_min, scope.row_min) == (0, 0)
    assert (scope.column_max, scope.row_max) == (1000, 800)
    assert scope.pixel_count == 800000
    assert scope.kind is ScopeKind.WHOLE_RASTER


def test_a_map_rectangle_becomes_a_pixel_window() -> None:
    from tree_counter.qgis_adapter.scope import scope_from_map_rect

    # y is measured down from the top of the raster extent.
    scope = scope_from_map_rect(_info(), (100.0, 600.0, 300.0, 700.0))

    assert (scope.column_min, scope.column_max) == (100, 300)
    assert (scope.row_min, scope.row_max) == (100, 200)


def test_a_map_rectangle_is_clamped_to_the_raster() -> None:
    from tree_counter.qgis_adapter.scope import scope_from_map_rect

    scope = scope_from_map_rect(_info(), (-500.0, -500.0, 5000.0, 5000.0))

    assert (scope.column_min, scope.row_min) == (0, 0)
    assert (scope.column_max, scope.row_max) == (1000, 800)


def test_a_rectangle_that_misses_the_raster_is_rejected() -> None:
    from tree_counter.qgis_adapter.scope import (
        ScopeRejected,
        scope_from_map_rect,
    )

    with pytest.raises(ScopeRejected):
        scope_from_map_rect(_info(), (5000.0, 5000.0, 6000.0, 6000.0))


def test_a_degenerate_rectangle_is_rejected() -> None:
    from tree_counter.qgis_adapter.scope import (
        ScopeRejected,
        scope_from_map_rect,
    )

    with pytest.raises(ScopeRejected):
        scope_from_map_rect(_info(), (100.0, 100.0, 100.0, 100.0))


def test_an_empty_scope_cannot_be_constructed() -> None:
    from tree_counter.qgis_adapter.scope import (
        PixelScope,
        ScopeKind,
        ScopeRejected,
    )

    with pytest.raises(ScopeRejected):
        PixelScope(ScopeKind.MAP_EXTENT, 10, 10, 10, 20)


def test_a_ring_converts_to_pixel_coordinates() -> None:
    from tree_counter.qgis_adapter.scope import map_ring_to_pixels

    ring = map_ring_to_pixels(
        _info(), [(0.0, 800.0), (100.0, 800.0), (100.0, 700.0)]
    )

    # The raster's top-left corner is pixel (0, 0).
    assert ring[0] == (0.0, 0.0)
    assert ring[1] == (100.0, 0.0)
    assert ring[2] == (100.0, 100.0)


def test_a_ring_with_too_few_points_is_rejected() -> None:
    from tree_counter.qgis_adapter.scope import (
        ScopeRejected,
        map_ring_to_pixels,
    )

    with pytest.raises(ScopeRejected):
        map_ring_to_pixels(_info(), [(0.0, 0.0), (1.0, 1.0)])


class TestMembership:
    """A detection belongs to a scope by where its centre falls."""

    def _rect_scope(self):
        from tree_counter.qgis_adapter.scope import ScopeKind, PixelScope

        return PixelScope(ScopeKind.MAP_EXTENT, 100, 100, 200, 200)

    def _masked_scope(self):
        from tree_counter.qgis_adapter.scope import ScopeKind, PixelScope

        return PixelScope(
            ScopeKind.POLYGON,
            0,
            0,
            100,
            100,
            polygons=(
                (((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)),),
            ),
        )

    def test_a_centre_inside_the_rectangle_belongs(self) -> None:
        assert self._rect_scope().contains_center(150.0, 150.0) is True

    def test_a_centre_outside_the_rectangle_does_not(self) -> None:
        assert self._rect_scope().contains_center(250.0, 150.0) is False

    def test_the_rectangle_boundary_is_inclusive(self) -> None:
        scope = self._rect_scope()

        assert scope.contains_center(100.0, 100.0) is True
        assert scope.contains_center(200.0, 200.0) is True

    def test_a_centre_inside_the_polygon_belongs(self) -> None:
        assert self._masked_scope().contains_center(50.0, 50.0) is True

    def test_a_centre_inside_the_box_but_outside_the_polygon_does_not(
        self,
    ) -> None:
        # Inside the bounding rectangle, outside the ring itself.
        assert self._masked_scope().contains_center(5.0, 5.0) is False

    def test_the_polygon_boundary_is_inclusive(self) -> None:
        assert self._masked_scope().contains_center(10.0, 50.0) is True

    def test_a_hole_excludes_its_interior(self) -> None:
        from tree_counter.qgis_adapter.scope import PixelScope, ScopeKind

        scope = PixelScope(
            ScopeKind.POLYGON,
            0,
            0,
            100,
            100,
            polygons=(
                (
                    ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
                    ((40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)),
                ),
            ),
        )

        assert scope.contains_center(20.0, 20.0) is True
        assert scope.contains_center(50.0, 50.0) is False


def test_a_scope_reports_its_size() -> None:
    from tree_counter.qgis_adapter.scope import PixelScope, ScopeKind

    scope = PixelScope(ScopeKind.MAP_EXTENT, 10, 20, 110, 220)

    assert (scope.width, scope.height) == (100, 200)
    assert scope.pixel_count == 20000


def test_a_scope_tiles_into_windows_over_its_own_origin() -> None:
    from tree_counter.core.tiling import iter_tile_windows
    from tree_counter.qgis_adapter.scope import scope_from_map_rect

    scope = scope_from_map_rect(_info(), (100.0, 500.0, 420.0, 700.0))

    windows = iter_tile_windows(
        scope.width,
        scope.height,
        256,
        0,
        origin_x=scope.column_min,
        origin_y=scope.row_min,
    )

    # Offsets are global pixels, so detections need no second translation.
    assert windows[0].x_offset == scope.column_min
    assert windows[0].y_offset == scope.row_min
    assert all(window.x_offset >= scope.column_min for window in windows)


def test_separate_polygon_parts_are_alternatives_not_holes() -> None:
    from tree_counter.qgis_adapter.scope import PixelScope, ScopeKind

    # Two disjoint areas. Treating the second as a hole in the first would
    # silently drop every detection inside it.
    scope = PixelScope(
        ScopeKind.POLYGON,
        0,
        0,
        1000,
        800,
        polygons=(
            (((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),),
            (
                (
                    (800.0, 700.0),
                    (900.0, 700.0),
                    (900.0, 800.0),
                    (800.0, 800.0),
                ),
            ),
        ),
    )

    assert scope.contains_center(50.0, 50.0) is True
    assert scope.contains_center(850.0, 750.0) is True
    assert scope.contains_center(400.0, 400.0) is False

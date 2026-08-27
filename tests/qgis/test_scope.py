"""Scope tests that need real QGIS CRS transforms and geometry."""

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
        "x_minimum": 0.0,
        "y_minimum": 0.0,
        "x_maximum": 1000.0,
        "y_maximum": 800.0,
    }
    values.update(overrides)
    return RasterInfo(**values)


def _crs(authid: str):
    from qgis.core import QgsCoordinateReferenceSystem

    crs = QgsCoordinateReferenceSystem(authid)
    assert crs.isValid(), f"CRS did not resolve: {authid}"
    return crs


def test_an_extent_in_the_same_crs_is_not_transformed() -> None:
    from qgis.core import QgsRectangle

    from tree_counter.qgis_adapter.scope import scope_from_map_extent

    crs = _crs("EPSG:3857")

    scope = scope_from_map_extent(
        _info(), QgsRectangle(100.0, 600.0, 300.0, 700.0), crs, crs
    )

    assert (scope.column_min, scope.column_max) == (100, 300)
    assert (scope.row_min, scope.row_max) == (100, 200)


def test_an_extent_in_another_crs_is_transformed() -> None:
    from qgis.core import (
        QgsCoordinateTransform,
        QgsProject,
        QgsRectangle,
    )

    from tree_counter.qgis_adapter.scope import scope_from_map_extent

    raster_crs = _crs("EPSG:3857")
    canvas_crs = _crs("EPSG:4326")

    # Take a known 3857 window, express it in 4326, and check it returns.
    target = QgsRectangle(100.0, 600.0, 300.0, 700.0)
    inverse = QgsCoordinateTransform(
        raster_crs, canvas_crs, QgsProject.instance()
    )
    in_degrees = inverse.transformBoundingBox(target)

    scope = scope_from_map_extent(
        _info(), in_degrees, canvas_crs, raster_crs
    )

    assert scope.column_min == pytest.approx(100, abs=1)
    assert scope.column_max == pytest.approx(300, abs=1)
    assert scope.row_min == pytest.approx(100, abs=1)
    assert scope.row_max == pytest.approx(200, abs=1)


def test_an_extent_that_misses_the_raster_is_rejected() -> None:
    from qgis.core import QgsRectangle

    from tree_counter.qgis_adapter.scope import (
        ScopeRejected,
        scope_from_map_extent,
    )

    crs = _crs("EPSG:3857")

    with pytest.raises(ScopeRejected):
        scope_from_map_extent(
            _info(), QgsRectangle(9000.0, 9000.0, 9500.0, 9500.0), crs, crs
        )


def _polygon_layer(wkts, crs_authid: str = "EPSG:3857"):
    from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer

    layer = QgsVectorLayer(f"Polygon?crs={crs_authid}", "scope", "memory")
    assert layer.isValid()
    features = []
    for wkt in wkts:
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromWkt(wkt))
        features.append(feature)
    layer.dataProvider().addFeatures(features)
    layer.updateExtents()
    return layer


def test_a_polygon_layer_becomes_a_masked_scope() -> None:
    from tree_counter.qgis_adapter.scope import (
        ScopeKind,
        scope_from_polygon_layer,
    )

    layer = _polygon_layer(
        ["Polygon((100 600, 300 600, 300 700, 100 700, 100 600))"]
    )

    scope = scope_from_polygon_layer(
        _info(), layer, _crs("EPSG:3857"), selected_only=False
    )

    assert scope.kind is ScopeKind.POLYGON
    assert scope.polygons
    assert scope.column_min == 100
    assert scope.row_min == 100
    assert scope.contains_center(200.0, 150.0) is True
    assert scope.contains_center(500.0, 150.0) is False


def test_only_selected_features_are_used_when_a_selection_exists() -> None:
    from tree_counter.qgis_adapter.scope import scope_from_polygon_layer

    layer = _polygon_layer(
        [
            "Polygon((0 700, 100 700, 100 800, 0 800, 0 700))",
            "Polygon((800 0, 900 0, 900 100, 800 100, 800 0))",
        ]
    )
    first = next(layer.getFeatures())
    layer.selectByIds([first.id()])

    scope = scope_from_polygon_layer(
        _info(), layer, _crs("EPSG:3857"), selected_only=True
    )

    # Only the first polygon, at the raster's top-left, is in scope.
    assert scope.column_max <= 101
    assert scope.row_max <= 101


def test_all_features_are_used_when_nothing_is_selected() -> None:
    from tree_counter.qgis_adapter.scope import scope_from_polygon_layer

    layer = _polygon_layer(
        [
            "Polygon((0 700, 100 700, 100 800, 0 800, 0 700))",
            "Polygon((800 0, 900 0, 900 100, 800 100, 800 0))",
        ]
    )

    scope = scope_from_polygon_layer(
        _info(), layer, _crs("EPSG:3857"), selected_only=True
    )

    assert scope.column_min == 0
    assert scope.column_max >= 900


def test_a_polygon_in_another_crs_is_transformed() -> None:
    from qgis.core import (
        QgsCoordinateTransform,
        QgsGeometry,
        QgsProject,
    )

    from tree_counter.qgis_adapter.scope import scope_from_polygon_layer

    raster_crs = _crs("EPSG:3857")
    layer_crs = _crs("EPSG:4326")
    square = QgsGeometry.fromWkt(
        "Polygon((100 600, 300 600, 300 700, 100 700, 100 600))"
    )
    square.transform(
        QgsCoordinateTransform(raster_crs, layer_crs, QgsProject.instance())
    )
    layer = _polygon_layer([square.asWkt()], "EPSG:4326")

    scope = scope_from_polygon_layer(
        _info(), layer, raster_crs, selected_only=False
    )

    assert scope.column_min == pytest.approx(100, abs=1)
    assert scope.row_min == pytest.approx(100, abs=1)


def test_a_polygon_that_misses_the_raster_is_rejected() -> None:
    from tree_counter.qgis_adapter.scope import (
        ScopeRejected,
        scope_from_polygon_layer,
    )

    layer = _polygon_layer(
        ["Polygon((5000 5000, 5100 5000, 5100 5100, 5000 5100, 5000 5000))"]
    )

    with pytest.raises(ScopeRejected):
        scope_from_polygon_layer(
            _info(), layer, _crs("EPSG:3857"), selected_only=False
        )


def test_an_empty_polygon_layer_is_rejected() -> None:
    from tree_counter.qgis_adapter.scope import (
        ScopeRejected,
        scope_from_polygon_layer,
    )

    layer = _polygon_layer([])

    with pytest.raises(ScopeRejected):
        scope_from_polygon_layer(
            _info(), layer, _crs("EPSG:3857"), selected_only=False
        )


def test_a_multipart_polygon_contributes_every_ring() -> None:
    from tree_counter.qgis_adapter.scope import scope_from_polygon_layer

    layer = _polygon_layer(
        [
            "MultiPolygon(((0 700, 100 700, 100 800, 0 800, 0 700)),"
            "((800 0, 900 0, 900 100, 800 100, 800 0)))"
        ]
    )

    scope = scope_from_polygon_layer(
        _info(), layer, _crs("EPSG:3857"), selected_only=False
    )

    assert len(scope.polygons) == 2
    assert scope.contains_center(50.0, 50.0) is True
    assert scope.contains_center(850.0, 750.0) is True
    assert scope.contains_center(400.0, 400.0) is False


def test_an_invalid_layer_is_rejected() -> None:
    from qgis.core import QgsVectorLayer

    from tree_counter.qgis_adapter.scope import (
        ScopeRejected,
        scope_from_polygon_layer,
    )

    layer = QgsVectorLayer("not a source", "broken", "memory")

    with pytest.raises(ScopeRejected):
        scope_from_polygon_layer(
            _info(), layer, _crs("EPSG:3857"), selected_only=False
        )


def test_two_selected_features_are_alternatives_not_holes() -> None:
    from tree_counter.qgis_adapter.scope import scope_from_polygon_layer

    layer = _polygon_layer(
        [
            "Polygon((0 700, 100 700, 100 800, 0 800, 0 700))",
            "Polygon((800 0, 900 0, 900 100, 800 100, 800 0))",
        ]
    )

    scope = scope_from_polygon_layer(
        _info(), layer, _crs("EPSG:3857"), selected_only=False
    )

    assert len(scope.polygons) == 2
    assert scope.contains_center(50.0, 50.0) is True
    assert scope.contains_center(850.0, 750.0) is True
    assert scope.contains_center(400.0, 400.0) is False


def test_a_polygon_with_a_hole_excludes_the_hole() -> None:
    from tree_counter.qgis_adapter.scope import scope_from_polygon_layer

    layer = _polygon_layer(
        [
            "Polygon((0 400, 400 400, 400 800, 0 800, 0 400),"
            "(100 500, 300 500, 300 700, 100 700, 100 500))"
        ]
    )

    scope = scope_from_polygon_layer(
        _info(), layer, _crs("EPSG:3857"), selected_only=False
    )

    assert len(scope.polygons) == 1
    assert len(scope.polygons[0]) == 2
    assert scope.contains_center(50.0, 50.0) is True
    assert scope.contains_center(200.0, 200.0) is False

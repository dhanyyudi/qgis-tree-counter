"""The dock offers the layers actually loaded in the project."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


@pytest.fixture
def project(qgis_application):
    from qgis.core import QgsProject

    instance = QgsProject.instance()
    instance.removeAllMapLayers()
    yield instance
    instance.removeAllMapLayers()


def _raster(tmp_path, name, bands=3):
    from qgis.core import QgsRasterLayer

    from test_raster import _write_geotiff

    path = _write_geotiff(tmp_path / f"{name}.tif", 8, 8, bands)
    return QgsRasterLayer(str(path), name, "gdal")


def _polygons(name):
    from qgis.core import QgsVectorLayer

    return QgsVectorLayer(
        "Polygon?crs=EPSG:3857&field=id:integer", name, "memory"
    )


def test_only_compatible_rasters_are_offered(project, tmp_path) -> None:
    """A raster the plugin cannot read must not look selectable."""

    from tree_counter.qgis_adapter.layers import raster_layer_names

    good = _raster(tmp_path, "aerial")
    single_band = _raster(tmp_path, "elevation", bands=1)
    project.addMapLayers([good, single_band, _polygons("blocks")])

    names = raster_layer_names()

    assert "aerial" in names
    assert "elevation" not in names
    assert "blocks" not in names


def test_only_polygon_layers_are_offered(project, tmp_path) -> None:
    """The scope combo lists polygons, not every vector layer."""

    from qgis.core import QgsVectorLayer

    from tree_counter.qgis_adapter.layers import polygon_layer_names

    points = QgsVectorLayer("Point?crs=EPSG:3857", "sites", "memory")
    project.addMapLayers([_polygons("blocks"), points,
                          _raster(tmp_path, "aerial")])

    names = polygon_layer_names()

    assert names == ("blocks",)


def test_an_empty_project_offers_nothing(project) -> None:
    """No layers is a normal state, not an error."""

    from tree_counter.qgis_adapter.layers import (
        polygon_layer_names,
        raster_layer_names,
    )

    assert raster_layer_names() == ()
    assert polygon_layer_names() == ()

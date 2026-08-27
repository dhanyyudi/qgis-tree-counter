"""Raster adapter tests that need a real QGIS provider.

A small RGB and a small grayscale GeoTIFF are generated per test run, so no
raster is committed and no maintainer path is referenced. The maintainer's
real aerial raster is used only when its path is supplied through the
environment.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

REAL_RASTER_VARIABLE = "TREE_COUNTER_TEST_RASTER"


def _write_geotiff(
    path: Path, width: int, height: int, bands: int, data_type: str = "Byte"
) -> Path:
    """Write a small GeoTIFF with a deterministic pixel pattern."""

    from osgeo import gdal, osr

    type_map = {"Byte": gdal.GDT_Byte, "UInt16": gdal.GDT_UInt16}
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(path), width, height, bands, type_map[data_type]
    )
    dataset.SetGeoTransform((100.0, 1.0, 0.0, 200.0, 0.0, -1.0))
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(3857)
    dataset.SetProjection(reference.ExportToWkt())
    for band in range(1, bands + 1):
        values = bytearray()
        for row in range(height):
            for column in range(width):
                values.append((row * width + column + band * 10) % 251)
        if data_type == "UInt16":
            packed = struct.pack(
                f"<{width * height}H", *[value * 4 for value in values]
            )
        else:
            packed = bytes(values)
        dataset.GetRasterBand(band).WriteRaster(
            0, 0, width, height, packed
        )
    dataset.FlushCache()
    dataset = None
    return path


def _layer(path: Path, name: str = "test"):
    from qgis.core import QgsRasterLayer

    layer = QgsRasterLayer(str(path), name, "gdal")
    assert layer.isValid(), f"fixture raster did not load: {path}"
    return layer


@pytest.fixture
def rgb_layer(tmp_path: Path):
    return _layer(_write_geotiff(tmp_path / "rgb.tif", 8, 6, 3))


def test_an_rgb_geotiff_is_described(rgb_layer) -> None:
    from tree_counter.qgis_adapter.raster import describe_layer

    info = describe_layer(rgb_layer)

    assert (info.width, info.height) == (8, 6)
    assert info.band_count == 3
    assert info.is_byte is True
    assert info.crs_authid == "EPSG:3857"
    assert info.provider_type == "gdal"
    assert info.pixel_size_x == pytest.approx(1.0)


def test_an_rgb_geotiff_validates(rgb_layer) -> None:
    from tree_counter.qgis_adapter.raster import validate_layer

    assert validate_layer(rgb_layer).band_count == 3


def test_an_rgba_geotiff_ignores_the_alpha_band(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.raster import RasterReader

    layer = _layer(_write_geotiff(tmp_path / "rgba.tif", 4, 4, 4))
    reader = RasterReader(layer)

    data = reader.read_rgb(0, 0, 4, 4)

    assert len(data) == 4 * 4 * 3


def test_a_grayscale_geotiff_is_rejected(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.raster import (
        RasterRejected,
        validate_layer,
    )

    layer = _layer(_write_geotiff(tmp_path / "gray.tif", 4, 4, 1))

    with pytest.raises(RasterRejected):
        validate_layer(layer)


def test_a_sixteen_bit_geotiff_is_rejected(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.raster import (
        RasterRejected,
        validate_layer,
    )

    layer = _layer(
        _write_geotiff(tmp_path / "u16.tif", 4, 4, 3, data_type="UInt16")
    )

    with pytest.raises(RasterRejected):
        validate_layer(layer)


def test_read_rgb_matches_the_provider_pixel_values(rgb_layer) -> None:
    from qgis.core import QgsPointXY, QgsRaster

    from tree_counter.qgis_adapter.raster import RasterReader

    reader = RasterReader(rgb_layer)
    info = reader.info
    data = reader.read_rgb(2, 1, 3, 2)

    assert len(data) == 3 * 2 * 3

    provider = rgb_layer.dataProvider()
    for row in range(2):
        for column in range(3):
            x = info.x_minimum + (2 + column + 0.5) * info.pixel_size_x
            y = info.y_maximum - (1 + row + 0.5) * info.pixel_size_y
            results = provider.identify(
                QgsPointXY(x, y), QgsRaster.IdentifyFormatValue
            ).results()
            offset = (row * 3 + column) * 3
            assert list(data[offset:offset + 3]) == [
                int(results[1]),
                int(results[2]),
                int(results[3]),
            ]


def test_reading_the_whole_raster_is_row_major(rgb_layer) -> None:
    from tree_counter.qgis_adapter.raster import RasterReader

    reader = RasterReader(rgb_layer)

    whole = reader.read_rgb(0, 0, 8, 6)
    first_row = reader.read_rgb(0, 0, 8, 1)

    assert whole[: 8 * 3] == first_row


def test_a_window_outside_the_raster_is_refused(rgb_layer) -> None:
    from tree_counter.qgis_adapter.raster import (
        RasterRejected,
        RasterReader,
    )

    reader = RasterReader(rgb_layer)

    with pytest.raises(RasterRejected):
        reader.read_rgb(6, 0, 4, 2)


def test_a_zero_sized_window_is_refused(rgb_layer) -> None:
    from tree_counter.qgis_adapter.raster import (
        RasterRejected,
        RasterReader,
    )

    reader = RasterReader(rgb_layer)

    with pytest.raises(RasterRejected):
        reader.read_rgb(0, 0, 0, 2)


def test_the_window_extent_is_north_up(rgb_layer) -> None:
    from tree_counter.qgis_adapter.raster import describe_layer, window_extent

    info = describe_layer(rgb_layer)

    extent = window_extent(info, 0, 0, 2, 2)

    # The first pixel row sits at the top of the raster extent.
    assert extent.yMaximum() == pytest.approx(info.y_maximum)
    assert extent.xMinimum() == pytest.approx(info.x_minimum)


def test_the_reader_clones_the_provider(rgb_layer) -> None:
    from tree_counter.qgis_adapter.raster import RasterReader

    reader = RasterReader(rgb_layer)

    assert reader._provider is not rgb_layer.dataProvider()


@pytest.mark.skipif(
    not os.environ.get(REAL_RASTER_VARIABLE),
    reason=f"set {REAL_RASTER_VARIABLE} to run against a real aerial raster",
)
def test_a_real_aerial_raster_reads_a_tile() -> None:
    from tree_counter.qgis_adapter.raster import RasterReader, validate_layer

    layer = _layer(Path(os.environ[REAL_RASTER_VARIABLE]), "aerial")
    info = validate_layer(layer)

    assert info.band_count in (3, 4)
    assert info.is_byte is True

    reader = RasterReader(layer, info)
    data = reader.read_rgb(0, 0, 640, 640)

    assert len(data) == 640 * 640 * 3

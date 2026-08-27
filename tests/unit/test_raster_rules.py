"""Tests for the QGIS-free raster acceptance rules."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


def _info(**overrides):
    from tree_counter.qgis_adapter.raster import RasterInfo

    # The real integration raster: 4-band RGB+alpha, EPSG:3857, ~0.037 m/px.
    values = {
        "name": "aerial",
        "provider_type": "gdal",
        "width": 16458,
        "height": 16456,
        "band_count": 4,
        "is_byte": True,
        "crs_authid": "EPSG:3857",
        "crs_is_valid": True,
        "x_minimum": 12790053.673929604,
        "y_minimum": -391970.42484612024,
        "x_maximum": 12790667.932040697,
        "y_maximum": -391356.2413805632,
        "is_valid": True,
    }
    values.update(overrides)
    return RasterInfo(**values)


def _expect_rejected(**overrides):
    from tree_counter.qgis_adapter.raster import (
        RasterRejected,
        validate_raster_info,
    )

    with pytest.raises(RasterRejected):
        validate_raster_info(_info(**overrides))


def test_the_real_integration_raster_is_accepted() -> None:
    from tree_counter.qgis_adapter.raster import validate_raster_info

    validate_raster_info(_info())


def test_a_three_band_rgb_raster_is_accepted() -> None:
    from tree_counter.qgis_adapter.raster import validate_raster_info

    validate_raster_info(_info(band_count=3))


def test_the_pixel_size_matches_the_documented_resolution() -> None:
    info = _info()

    assert info.pixel_size_x == pytest.approx(0.0373228, abs=1e-7)
    assert info.pixel_size_y == pytest.approx(0.0373228, abs=1e-7)


def test_a_grayscale_raster_is_rejected() -> None:
    _expect_rejected(band_count=1)


def test_a_two_band_raster_is_rejected() -> None:
    _expect_rejected(band_count=2)


def test_a_multispectral_raster_is_rejected() -> None:
    _expect_rejected(band_count=8)


def test_a_sixteen_bit_raster_is_rejected() -> None:
    _expect_rejected(is_byte=False)


@pytest.mark.parametrize(
    "provider", ["wms", "WMTS", "xyz", "wcs", "arcgismapserver", "vectortile"]
)
def test_an_online_provider_is_rejected(provider: str) -> None:
    _expect_rejected(provider_type=provider)


@pytest.mark.parametrize("provider", ["gdal", "GDAL"])
def test_a_local_provider_is_accepted(provider: str) -> None:
    from tree_counter.qgis_adapter.raster import validate_raster_info

    # Acceptance is provider-based, not extension-based, so GeoTIFF, COG
    # and VRT all arrive here as the same gdal provider.
    validate_raster_info(_info(provider_type=provider))


def test_an_invalid_layer_is_rejected() -> None:
    _expect_rejected(is_valid=False)


def test_a_raster_without_a_crs_is_rejected() -> None:
    _expect_rejected(crs_is_valid=False)


def test_an_empty_crs_identifier_is_rejected() -> None:
    _expect_rejected(crs_authid="")


@pytest.mark.parametrize("field", ["width", "height"])
def test_a_zero_dimension_is_rejected(field: str) -> None:
    _expect_rejected(**{field: 0})


def test_a_negative_dimension_is_rejected() -> None:
    _expect_rejected(width=-1)


def test_an_empty_extent_is_rejected() -> None:
    _expect_rejected(x_maximum=12790053.673929604)


def test_an_inverted_extent_is_rejected() -> None:
    _expect_rejected(y_maximum=-500000.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_extent_is_rejected(value: float) -> None:
    _expect_rejected(x_maximum=value)


def test_a_non_raster_description_is_rejected() -> None:
    from tree_counter.qgis_adapter.raster import (
        RasterRejected,
        validate_raster_info,
    )

    with pytest.raises(RasterRejected):
        validate_raster_info({"width": 10})


def test_rejection_messages_are_actionable() -> None:
    from tree_counter.qgis_adapter.raster import (
        RasterRejected,
        validate_raster_info,
    )

    with pytest.raises(RasterRejected) as error:
        validate_raster_info(_info(is_byte=False))

    assert "8-bit" in str(error.value.diagnostic_detail)
    assert "Byte" in str(error.value.diagnostic_detail)


class TestInterleave:
    """Band planes become the interleaved RGB the worker expects."""

    def test_three_planes_interleave(self) -> None:
        from tree_counter.qgis_adapter.raster import interleave_rgb

        assert interleave_rgb(b"\x01\x02", b"\x03\x04", b"\x05\x06") == (
            b"\x01\x03\x05\x02\x04\x06"
        )

    def test_the_result_length_is_three_times_a_plane(self) -> None:
        from tree_counter.qgis_adapter.raster import interleave_rgb

        plane = bytes(range(16))

        assert len(interleave_rgb(plane, plane, plane)) == 48

    def test_mismatched_planes_are_rejected(self) -> None:
        from tree_counter.qgis_adapter.raster import (
            RasterRejected,
            interleave_rgb,
        )

        with pytest.raises(RasterRejected):
            interleave_rgb(b"\x01\x02", b"\x03", b"\x05\x06")

    def test_an_empty_plane_gives_an_empty_result(self) -> None:
        from tree_counter.qgis_adapter.raster import interleave_rgb

        assert interleave_rgb(b"", b"", b"") == b""


def test_a_tile_of_the_real_raster_has_the_expected_byte_count() -> None:
    from tree_counter.qgis_adapter.raster import interleave_rgb

    # A 640x640 tile must be exactly what the worker's read_rgb_tile
    # expects for those dimensions.
    plane = bytes(640 * 640)

    assert len(interleave_rgb(plane, plane, plane)) == 640 * 640 * 3

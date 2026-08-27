"""Deciding whether a raster layer may be counted, and reading its pixels.

The acceptance rules are deliberately separated from QGIS: they operate on
a plain :class:`RasterInfo` value, so they can be tested without a QGIS
installation and reasoned about on their own. Only the two functions that
touch a layer or a provider need QGIS.

The v1 pixel contract is 8-bit RGB. Bands 1/2/3 are red, green and blue; a
fourth alpha band is ignored. Anything else is refused with a message that
says what to do, because silently reinterpreting a 16-bit or multispectral
raster as RGB would produce confident, wrong counts.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tree_counter.errors import ErrorCode, TreeCounterError

RGB_BANDS = (1, 2, 3)
MIN_BANDS = 3
MAX_BANDS = 4

# Providers that stream from a server. V1 counts local, georeferenced data
# only: a tiled web service has no authoritative pixel grid to count on.
ONLINE_PROVIDERS = (
    "wms",
    "wmts",
    "xyz",
    "wcs",
    "arcgismapserver",
    "arcgisrestserver",
    "vectortile",
)


class RasterRejected(TreeCounterError):
    """The selected raster does not satisfy the v1 pixel contract."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.INVALID_RASTER, diagnostic_detail=detail)


@dataclass(frozen=True)
class RasterInfo:
    """The facts about a raster that decide whether it can be counted."""

    name: str
    provider_type: str
    width: int
    height: int
    band_count: int
    is_byte: bool
    crs_authid: str
    crs_is_valid: bool
    x_minimum: float
    y_minimum: float
    x_maximum: float
    y_maximum: float
    is_valid: bool = True

    @property
    def pixel_size_x(self) -> float:
        """Return the map units covered by one pixel horizontally."""

        return (self.x_maximum - self.x_minimum) / self.width

    @property
    def pixel_size_y(self) -> float:
        """Return the map units covered by one pixel vertically."""

        return (self.y_maximum - self.y_minimum) / self.height


def validate_raster_info(info: RasterInfo) -> None:
    """Raise :class:`RasterRejected` unless the raster may be counted."""

    if not isinstance(info, RasterInfo):
        raise RasterRejected("a raster description is required")
    if not info.is_valid:
        raise RasterRejected("the layer could not be loaded")
    provider = (info.provider_type or "").casefold()
    if provider in ONLINE_PROVIDERS:
        raise RasterRejected(
            "online map services cannot be counted; use a local "
            "georeferenced raster"
        )
    if info.width <= 0 or info.height <= 0:
        raise RasterRejected("the raster has no pixels")
    if not info.crs_is_valid or not info.crs_authid:
        raise RasterRejected(
            "the raster has no valid coordinate reference system"
        )
    for value in (
        info.x_minimum,
        info.y_minimum,
        info.x_maximum,
        info.y_maximum,
    ):
        if not math.isfinite(value):
            raise RasterRejected("the raster extent is not finite")
    if info.x_maximum <= info.x_minimum or info.y_maximum <= info.y_minimum:
        raise RasterRejected("the raster extent is empty")
    if info.band_count < MIN_BANDS:
        raise RasterRejected(
            "counting needs an 8-bit RGB raster; this layer has "
            f"{info.band_count} band(s)"
        )
    if info.band_count > MAX_BANDS:
        raise RasterRejected(
            "multispectral rasters are not supported in this release; "
            f"this layer has {info.band_count} bands"
        )
    if not info.is_byte:
        raise RasterRejected(
            "counting needs 8-bit bands; convert the raster to Byte first"
        )


def describe_layer(layer: Any) -> RasterInfo:
    """Return a :class:`RasterInfo` for a QGIS raster layer."""

    from qgis.core import Qgis

    provider = layer.dataProvider()
    if provider is None:
        raise RasterRejected("the layer has no data provider")
    extent = provider.extent()
    band_count = int(provider.bandCount())
    is_byte = band_count > 0 and all(
        provider.dataType(band) == Qgis.DataType.Byte
        for band in range(1, band_count + 1)
    )
    crs = provider.crs()
    return RasterInfo(
        name=str(layer.name()),
        provider_type=str(layer.providerType()),
        width=int(provider.xSize()),
        height=int(provider.ySize()),
        band_count=band_count,
        is_byte=is_byte,
        crs_authid=str(crs.authid()),
        crs_is_valid=bool(crs.isValid()),
        x_minimum=float(extent.xMinimum()),
        y_minimum=float(extent.yMinimum()),
        x_maximum=float(extent.xMaximum()),
        y_maximum=float(extent.yMaximum()),
        is_valid=bool(layer.isValid()),
    )


def validate_layer(layer: Any) -> RasterInfo:
    """Describe and validate a QGIS raster layer in one step."""

    info = describe_layer(layer)
    validate_raster_info(info)
    return info


def window_extent(info: RasterInfo, x: int, y: int, width: int, height: int):
    """Return the map rectangle covering a pixel window.

    Pixel space has its origin at the raster's top-left with y increasing
    downwards, so the window's top edge is measured down from the extent's
    maximum y.
    """

    from qgis.core import QgsRectangle

    if width <= 0 or height <= 0:
        raise RasterRejected("a read window must have positive size")
    if x < 0 or y < 0 or x + width > info.width or y + height > info.height:
        raise RasterRejected("the read window falls outside the raster")
    pixel_x = info.pixel_size_x
    pixel_y = info.pixel_size_y
    x_min = info.x_minimum + x * pixel_x
    y_max = info.y_maximum - y * pixel_y
    return QgsRectangle(
        x_min, y_max - height * pixel_y, x_min + width * pixel_x, y_max
    )


class RasterReader:
    """Reads tightly packed RGB8 bytes from a raster provider.

    The provider is cloned so reading runs on the task thread without
    touching the layer the user is interacting with.
    """

    def __init__(self, layer: Any, info: RasterInfo | None = None) -> None:
        self._info = validate_layer(layer) if info is None else info
        provider = layer.dataProvider()
        clone = getattr(provider, "clone", None)
        self._provider = clone() if callable(clone) else provider

    @property
    def info(self) -> RasterInfo:
        """Return the validated description of the source raster."""

        return self._info

    def read_rgb(self, x: int, y: int, width: int, height: int) -> bytes:
        """Return ``width * height * 3`` interleaved RGB bytes."""

        extent = window_extent(self._info, x, y, width, height)
        planes: list[bytes] = []
        for band in RGB_BANDS:
            block = self._provider.block(band, extent, width, height)
            if block is None or not block.isValid():
                raise RasterRejected(
                    "the raster could not be read; the source may be "
                    "missing or locked"
                )
            if block.width() != width or block.height() != height:
                raise RasterRejected(
                    "the raster returned an unexpected block size"
                )
            plane = bytes(block.data())
            if len(plane) != width * height:
                raise RasterRejected(
                    "the raster returned an unexpected block length"
                )
            planes.append(plane)
        return interleave_rgb(planes[0], planes[1], planes[2])

    def close(self) -> None:
        """Release the cloned provider."""

        self._provider = None


def interleave_rgb(red: bytes, green: bytes, blue: bytes) -> bytes:
    """Return interleaved RGB bytes from three equally sized band planes."""

    if not (len(red) == len(green) == len(blue)):
        raise RasterRejected("the RGB band planes have different sizes")
    output = bytearray(len(red) * 3)
    output[0::3] = red
    output[1::3] = green
    output[2::3] = blue
    return bytes(output)


__all__ = [
    "MAX_BANDS",
    "MIN_BANDS",
    "ONLINE_PROVIDERS",
    "RGB_BANDS",
    "RasterInfo",
    "RasterReader",
    "RasterRejected",
    "describe_layer",
    "interleave_rgb",
    "validate_layer",
    "validate_raster_info",
    "window_extent",
]

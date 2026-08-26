"""QGIS-free deterministic domain contracts for Tree Counter."""

# SPDX-License-Identifier: AGPL-3.0-only

from .types import Detection, InferenceSettings, PixelBox, TileWindow
from .geometry import (
    filter_detections_by_pixel_rect,
    filter_detections_by_polygon,
    map_rect_to_pixel_rect,
    pixel_box_center,
    point_in_polygon,
)
from .tiling import iter_tile_windows

__all__ = [
    "Detection",
    "InferenceSettings",
    "PixelBox",
    "TileWindow",
    "filter_detections_by_pixel_rect",
    "filter_detections_by_polygon",
    "iter_tile_windows",
    "map_rect_to_pixel_rect",
    "pixel_box_center",
    "point_in_polygon",
]

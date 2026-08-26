"""Shared constants for the Tree Counter plugin and its core contracts."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

PLUGIN_NAME = "Tree Counter"
QGIS_MINIMUM_VERSION = "3.44"
QGIS_MAXIMUM_VERSION = "4.99"

DEFAULT_CONFIDENCE = 0.25
DEFAULT_NMS_IOU = 0.70
DEFAULT_DUPLICATE_IOU = 0.50
DEFAULT_TILE_SIZE = 640
DEFAULT_OVERLAP_PERCENT = 20

MIN_TILE_SIZE = 256
MAX_TILE_SIZE = 2048
TILE_SIZE_MULTIPLE = 32
MIN_OVERLAP_PERCENT = 0
MAX_OVERLAP_PERCENT = 50

SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps", "coreml")
PROTOCOL_VERSION = 1

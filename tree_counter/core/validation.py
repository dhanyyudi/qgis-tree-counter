"""Small QGIS-free validators for Tree Counter domain inputs."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from typing import TypeVar

from tree_counter.constants import (
    MAX_OVERLAP_PERCENT,
    MAX_TILE_SIZE,
    MIN_OVERLAP_PERCENT,
    MIN_TILE_SIZE,
    TILE_SIZE_MULTIPLE,
)
from tree_counter.errors import ValidationError

Number = TypeVar("Number", int, float)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be a finite number")
    return result


def validate_confidence(value: Number) -> float:
    """Validate an inclusive confidence value in the range [0, 1]."""

    result = _finite(value, "confidence")
    if not 0.0 <= result <= 1.0:
        raise ValidationError("confidence must be between 0 and 1")
    return result


def validate_iou(value: Number, name: str = "IoU") -> float:
    """Validate an inclusive IoU threshold in the range [0, 1]."""

    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValidationError(f"{name} must be between 0 and 1")
    return result


def validate_nms_iou(value: Number) -> float:
    """Validate the per-tile NMS threshold."""

    return validate_iou(value, "nms_iou")


def validate_duplicate_iou(value: Number) -> float:
    """Validate the cross-tile duplicate threshold."""

    return validate_iou(value, "duplicate_iou")


def validate_tile_size(value: int) -> int:
    """Validate a model tile edge in the inclusive v1 range."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not MIN_TILE_SIZE <= value <= MAX_TILE_SIZE
        or value % TILE_SIZE_MULTIPLE
    ):
        raise ValidationError(
            "tile_size must be a multiple of 32 between 256 and 2048"
        )
    return value


def validate_overlap_percent(value: int) -> int:
    """Validate inclusive tile overlap as a percentage in [0, 50]."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not MIN_OVERLAP_PERCENT <= value <= MAX_OVERLAP_PERCENT
    ):
        raise ValidationError("overlap_percent must be between 0 and 50")
    return value


def validate_pixel_coordinate(
    value: Number, name: str = "coordinate"
) -> float:
    """Validate one finite pixel coordinate."""

    return _finite(value, name)


def validate_pixel_box(box: object) -> None:
    """Validate a PixelBox-like object without importing domain types."""

    try:
        x_min = validate_pixel_coordinate(getattr(box, "x_min"), "x_min")
        y_min = validate_pixel_coordinate(getattr(box, "y_min"), "y_min")
        x_max = validate_pixel_coordinate(getattr(box, "x_max"), "x_max")
        y_max = validate_pixel_coordinate(getattr(box, "y_max"), "y_max")
    except AttributeError as exc:
        raise ValidationError("box must expose four pixel edges") from exc
    if x_min >= x_max or y_min >= y_max:
        raise ValidationError("box edges must describe a positive-area box")


def validate_inference_settings(settings: object) -> None:
    """Validate an :class:`InferenceSettings` instance."""

    from tree_counter.core.types import InferenceSettings

    if not isinstance(settings, InferenceSettings):
        raise ValidationError("settings must be an InferenceSettings instance")

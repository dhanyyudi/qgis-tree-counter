"""Immutable, serializable value objects shared by host and worker code."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass

from tree_counter.constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_DUPLICATE_IOU,
    DEFAULT_NMS_IOU,
    DEFAULT_OVERLAP_PERCENT,
    DEFAULT_TILE_SIZE,
)
from tree_counter.constants import SUPPORTED_DEVICES
from tree_counter.errors import ValidationError


def _finite_number(value: object, name: str) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be a finite number")
    return result


def _probability(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValidationError(f"{name} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class PixelBox:
    """A detection box in global pixel-edge coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (
            ("x_min", self.x_min),
            ("y_min", self.y_min),
            ("x_max", self.x_max),
            ("y_max", self.y_max),
        )
        finite = {name: _finite_number(value, name) for name, value in values}
        if finite["x_min"] >= finite["x_max"]:
            raise ValidationError("x_min must be less than x_max")
        if finite["y_min"] >= finite["y_max"]:
            raise ValidationError("y_min must be less than y_max")
        for name, value in finite.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TileWindow:
    """A source window and its model-sized padded dimensions."""

    tile_id: str
    x_offset: int
    y_offset: int
    read_width: int
    read_height: int
    model_width: int
    model_height: int

    def __post_init__(self) -> None:
        if not isinstance(self.tile_id, str) or not self.tile_id:
            raise ValidationError("tile_id must be a non-empty string")
        for name in (
            "x_offset",
            "y_offset",
            "read_width",
            "read_height",
            "model_width",
            "model_height",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(f"{name} must be an integer")
            if value < 0 or (
                name.startswith("read_") or name.startswith("model_")
            ) and value == 0:
                raise ValidationError(f"{name} is out of range")
        if self.read_width > self.model_width:
            raise ValidationError("read_width cannot exceed model_width")
        if self.read_height > self.model_height:
            raise ValidationError("read_height cannot exceed model_height")


@dataclass(frozen=True)
class Detection:
    """A model detection expressed in global pixel coordinates."""

    box: PixelBox
    confidence: float
    class_id: int
    class_name: str
    tile_ids: tuple[str, ...]
    merged_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.box, PixelBox):
            raise ValidationError("box must be a PixelBox")
        confidence = _probability(self.confidence, "confidence")
        if isinstance(self.class_id, bool) or not isinstance(
            self.class_id, int
        ):
            raise ValidationError("class_id must be an integer")
        if self.class_id < 0:
            raise ValidationError("class_id must not be negative")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValidationError("class_name must be non-empty")
        tile_ids = tuple(self.tile_ids)
        if not tile_ids or any(
            not isinstance(tile_id, str) or not tile_id for tile_id in tile_ids
        ):
            raise ValidationError("tile_ids must contain non-empty strings")
        if (
            isinstance(self.merged_count, bool)
            or not isinstance(self.merged_count, int)
            or self.merged_count < 1
        ):
            raise ValidationError("merged_count must be a positive integer")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "tile_ids", tile_ids)


@dataclass(frozen=True)
class InferenceSettings:
    """Validated settings for one deterministic inference run."""

    confidence: float = DEFAULT_CONFIDENCE
    nms_iou: float = DEFAULT_NMS_IOU
    duplicate_iou: float = DEFAULT_DUPLICATE_IOU
    tile_size: int = DEFAULT_TILE_SIZE
    overlap_percent: int = DEFAULT_OVERLAP_PERCENT
    selected_class_ids: tuple[int, ...] = ()
    requested_device: str = "auto"

    def __post_init__(self) -> None:
        confidence = _probability(self.confidence, "confidence")
        nms_iou = _probability(self.nms_iou, "nms_iou")
        duplicate_iou = _probability(self.duplicate_iou, "duplicate_iou")
        if (
            isinstance(self.tile_size, bool)
            or not isinstance(self.tile_size, int)
            or not 256 <= self.tile_size <= 2048
            or self.tile_size % 32
        ):
            raise ValidationError(
                "tile_size must be a multiple of 32 between 256 and 2048"
            )
        if (
            isinstance(self.overlap_percent, bool)
            or not isinstance(self.overlap_percent, int)
            or not 0 <= self.overlap_percent <= 50
        ):
            raise ValidationError("overlap_percent must be between 0 and 50")
        selected = tuple(self.selected_class_ids)
        if any(
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or class_id < 0
            for class_id in selected
        ):
            raise ValidationError(
                "selected_class_ids must be non-negative integers"
            )
        if len(set(selected)) != len(selected):
            raise ValidationError("selected_class_ids must be unique")
        if (
            not isinstance(self.requested_device, str)
            or self.requested_device not in SUPPORTED_DEVICES
        ):
            raise ValidationError(
                "requested_device is not a supported device name"
            )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "nms_iou", nms_iou)
        object.__setattr__(self, "duplicate_iou", duplicate_iou)
        object.__setattr__(self, "selected_class_ids", selected)

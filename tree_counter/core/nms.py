"""Deterministic confidence filtering and class-aware per-tile NMS."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from tree_counter.core.types import Detection
from tree_counter.core.validation import (
    validate_confidence,
    validate_iou,
    validate_pixel_box,
)
from tree_counter.errors import ValidationError

SortKey = tuple[float, int, float, float, float, float, tuple[str, ...]]


def box_iou(left: Any, right: Any) -> float:
    """Return the intersection-over-union of two positive-area boxes.

    Boxes that only touch along an edge or a corner have an IoU of ``0.0``.
    Invalid or zero-area boxes are rejected rather than silently treated as
    non-overlapping.
    """

    validate_pixel_box(left)
    validate_pixel_box(right)
    overlap_width = min(left.x_max, right.x_max) - max(left.x_min, right.x_min)
    overlap_height = min(left.y_max, right.y_max) - max(
        left.y_min, right.y_min
    )
    if overlap_width <= 0.0 or overlap_height <= 0.0:
        return 0.0
    intersection = overlap_width * overlap_height
    left_area = (left.x_max - left.x_min) * (left.y_max - left.y_min)
    right_area = (right.x_max - right.x_min) * (right.y_max - right.y_min)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def detection_sort_key(detection: Detection) -> SortKey:
    """Return the stable ordering key used by suppression and dedup.

    Detections are ordered by descending confidence, then by class and box
    edges so that equal-confidence input is resolved identically regardless
    of the order in which the backend produced it.
    """

    box = detection.box
    return (
        -detection.confidence,
        detection.class_id,
        box.x_min,
        box.y_min,
        box.x_max,
        box.y_max,
        detection.tile_ids,
    )


def _detections(detections: Iterable[Any]) -> tuple[Detection, ...]:
    if isinstance(detections, (str, bytes)):
        raise ValidationError("detections must be a sequence of Detection")
    try:
        items = tuple(detections)
    except TypeError as exc:
        raise ValidationError(
            "detections must be a sequence of Detection"
        ) from exc
    for item in items:
        if not isinstance(item, Detection):
            raise ValidationError("detections must contain Detection values")
    return items


def filter_by_confidence(
    detections: Iterable[Any], confidence_threshold: float
) -> tuple[Detection, ...]:
    """Keep detections at or above *confidence_threshold*, in input order."""

    threshold = validate_confidence(confidence_threshold)
    return tuple(
        detection
        for detection in _detections(detections)
        if detection.confidence >= threshold
    )


def apply_nms(
    detections: Iterable[Any],
    iou_threshold: float,
    confidence_threshold: float | None = None,
) -> tuple[Detection, ...]:
    """Suppress overlapping same-class detections within one tile.

    Confidence filtering always runs before suppression so a discarded
    detection can never suppress a surviving one. Suppression is inclusive:
    a candidate whose IoU equals *iou_threshold* is removed. Detections of
    different classes never suppress each other, and provenance is left
    untouched because cross-tile merging is a separate operation.
    """

    threshold = validate_iou(iou_threshold, "nms_iou")
    candidates = _detections(detections)
    if confidence_threshold is not None:
        candidates = filter_by_confidence(candidates, confidence_threshold)
    kept: list[Detection] = []
    for candidate in sorted(candidates, key=detection_sort_key):
        if any(
            existing.class_id == candidate.class_id
            and box_iou(existing.box, candidate.box) >= threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return tuple(kept)


def apply_nms_per_class(
    detections: Iterable[Any],
    iou_threshold: float,
    selected_class_ids: Sequence[int] = (),
    confidence_threshold: float | None = None,
) -> tuple[Detection, ...]:
    """Run :func:`apply_nms` after restricting to the selected classes.

    An empty *selected_class_ids* keeps every class, which matches the
    single-class auto-selection behaviour of the UI.
    """

    selected = tuple(selected_class_ids)
    candidates = _detections(detections)
    if selected:
        allowed = set(selected)
        candidates = tuple(
            detection
            for detection in candidates
            if detection.class_id in allowed
        )
    return apply_nms(
        candidates, iou_threshold, confidence_threshold=confidence_threshold
    )


__all__ = [
    "apply_nms",
    "apply_nms_per_class",
    "box_iou",
    "detection_sort_key",
    "filter_by_confidence",
]

"""Class-aware, deterministic cross-tile duplicate suppression."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from tree_counter.core.nms import detection_sort_key, overlaps_at_threshold
from tree_counter.core.types import Detection
from tree_counter.core.validation import validate_iou
from tree_counter.errors import ValidationError

CellKey = tuple[int, int, int]

MIN_GRID_CELL = 1.0


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


def _cell_size(detections: tuple[Detection, ...]) -> float:
    """Return a grid pitch at least as large as the widest detection.

    A cell that is never smaller than the largest box keeps the number of
    cells a single box spans bounded, while spanning insertion keeps the
    candidate set exact rather than approximate.
    """

    largest = MIN_GRID_CELL
    for detection in detections:
        box = detection.box
        largest = max(largest, box.x_max - box.x_min, box.y_max - box.y_min)
    return largest


def _spanned_cells(
    detection: Detection, cell_size: float
) -> tuple[CellKey, ...]:
    box = detection.box
    column_min = math.floor(box.x_min / cell_size)
    column_max = math.floor(box.x_max / cell_size)
    row_min = math.floor(box.y_min / cell_size)
    row_max = math.floor(box.y_max / cell_size)
    return tuple(
        (detection.class_id, column, row)
        for row in range(row_min, row_max + 1)
        for column in range(column_min, column_max + 1)
    )


def _merge(existing: Detection, absorbed: Detection) -> Detection:
    """Keep the winning geometry and union the contributing provenance."""

    return Detection(
        box=existing.box,
        confidence=existing.confidence,
        class_id=existing.class_id,
        class_name=existing.class_name,
        tile_ids=tuple(
            sorted(set(existing.tile_ids) | set(absorbed.tile_ids))
        ),
        merged_count=existing.merged_count + absorbed.merged_count,
    )


def deduplicate_detections(
    detections: Iterable[Any], duplicate_iou: float
) -> tuple[Detection, ...]:
    """Merge same-class detections that overlap across tile boundaries.

    Candidates are processed in descending confidence with stable
    tie-breakers, so the surviving geometry is always the highest-confidence
    box and the result does not depend on input order. Merging is inclusive
    at *duplicate_iou* but always requires real overlap, so a zero threshold
    never merges disjoint boxes that happen to share a grid cell. Merging
    unions the sorted contributing tile IDs and accumulates
    ``merged_count``. Detections of different classes are never merged.

    A spatial grid keyed by class and cell restricts comparisons to boxes
    that can actually overlap; because every box is inserted into each cell
    it spans, the result is identical to comparing every pair.
    """

    threshold = validate_iou(duplicate_iou, "duplicate_iou")
    candidates = _detections(detections)
    if not candidates:
        return ()

    cell_size = _cell_size(candidates)
    kept: list[Detection] = []
    grid: dict[CellKey, list[int]] = {}
    for candidate in sorted(candidates, key=detection_sort_key):
        cells = _spanned_cells(candidate, cell_size)
        nearby: set[int] = set()
        for cell in cells:
            nearby.update(grid.get(cell, ()))
        absorbed_by: int | None = None
        for index in sorted(nearby):
            existing = kept[index]
            if overlaps_at_threshold(existing.box, candidate.box, threshold):
                absorbed_by = index
                break
        if absorbed_by is None:
            kept.append(candidate)
            for cell in cells:
                grid.setdefault(cell, []).append(len(kept) - 1)
        else:
            kept[absorbed_by] = _merge(kept[absorbed_by], candidate)
    return tuple(kept)


__all__ = ["deduplicate_detections"]

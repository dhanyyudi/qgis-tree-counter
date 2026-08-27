"""Turning a raw YOLO11 detection head into canonical predictions.

The model emits one row per anchor with a box in centre form followed by a
score per class. Nothing here suppresses anything: NMS and deduplication
belong to the plugin so the user's thresholds always apply. The output of
this module is the same canonical shape for every backend.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tree_counter.worker.backend_base import ModelRejected, RawDetection
from tree_counter.worker.image import LetterboxTransform
from tree_counter.worker.model_info import BOX_COORDINATES, OutputLayout


def _as_predictions(output: Any, layout: OutputLayout) -> Any:
    import numpy

    array = numpy.asarray(output)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ModelRejected(
                f"expected a single-image batch, got {array.shape[0]}"
            )
        array = array[0]
    elif array.ndim != 2:
        raise ModelRejected(
            f"expected a 2- or 3-dimensional output, got {array.ndim}"
        )
    if layout.transposed:
        array = array.T
    attributes = BOX_COORDINATES + layout.class_count
    if array.shape[1] != attributes:
        raise ModelRejected(
            f"the output row has {array.shape[1]} values but "
            f"{attributes} were expected"
        )
    return array.astype(numpy.float32, copy=False)


def decode_predictions(
    output: Any,
    layout: OutputLayout,
    transform: LetterboxTransform,
    confidence_threshold: float,
    selected_class_ids: Sequence[int] = (),
) -> list[RawDetection]:
    """Return canonical tile-local predictions above the threshold.

    Boxes arrive in centre form in model space and leave in corner form in
    tile space. Class filtering happens here so a class the user did not
    select never reaches NMS and cannot suppress one they did.
    """

    import numpy

    predictions = _as_predictions(output, layout)
    if predictions.size == 0:
        return []
    boxes = predictions[:, :BOX_COORDINATES]
    scores = predictions[:, BOX_COORDINATES:]

    class_ids = scores.argmax(axis=1)
    best = scores.max(axis=1)
    keep = best >= float(confidence_threshold)
    if selected_class_ids:
        allowed = numpy.asarray(tuple(selected_class_ids), dtype=numpy.int64)
        keep &= numpy.isin(class_ids, allowed)
    if not bool(keep.any()):
        return []

    boxes = boxes[keep]
    best = best[keep]
    class_ids = class_ids[keep]

    half_width = boxes[:, 2] / 2.0
    half_height = boxes[:, 3] / 2.0
    x_min = boxes[:, 0] - half_width
    y_min = boxes[:, 1] - half_height
    x_max = boxes[:, 0] + half_width
    y_max = boxes[:, 1] + half_height

    results: list[RawDetection] = []
    for index in range(len(best)):
        left, top, right, bottom = transform.box_to_tile(
            float(x_min[index]),
            float(y_min[index]),
            float(x_max[index]),
            float(y_max[index]),
        )
        if right <= left or bottom <= top:
            # A degenerate box carries no area and cannot be counted.
            continue
        results.append(
            RawDetection(
                x_min=left,
                y_min=top,
                x_max=right,
                y_max=bottom,
                confidence=min(1.0, max(0.0, float(best[index]))),
                class_id=int(class_ids[index]),
            )
        )
    return results


__all__ = ["decode_predictions"]

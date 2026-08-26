"""Deciding whether a model is one this plugin will run at all.

The rule throughout is to reject rather than guess. An export whose output
layout is ambiguous, whose task is not detection, or which already applies
its own NMS would still produce numbers if we assumed something, and those
numbers would be silently wrong, so each of those is refused with a reason.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tree_counter.worker.backend_base import (
    SUPPORTED_TASK,
    ModelRejected,
)

# Ultralytics writes these into ONNX metadata and PT checkpoints.
CLASS_MAP_KEYS = ("names", "class_names")
TASK_KEYS = ("task",)
# Names an export with graph-level NMS typically produces.
EMBEDDED_NMS_OUTPUT_NAMES = (
    "num_dets",
    "num_detections",
    "nmsed_boxes",
    "nmsed_scores",
    "nmsed_classes",
    "detection_boxes",
    "selected_indices",
)
BOX_COORDINATES = 4
# A detection head with fewer rows than this cannot hold boxes plus a class.
MINIMUM_PREDICTION_ROWS = BOX_COORDINATES + 1


@dataclass(frozen=True)
class OutputLayout:
    """How a raw detection output tensor is arranged."""

    transposed: bool
    predictions: int
    class_count: int


def parse_class_map(raw: object) -> tuple[str, ...]:
    """Return ordered class names from Ultralytics metadata.

    Ultralytics stores either a mapping of index to name or a plain list.
    Indices must be a complete 0..n-1 run: a gap would silently shift every
    later class name onto the wrong detections.
    """

    if isinstance(raw, str):
        raise ModelRejected("the class map is not a mapping or list")
    if isinstance(raw, Mapping):
        indexed: dict[int, str] = {}
        for key, value in raw.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise ModelRejected(
                    f"class index {key!r} is not an integer"
                ) from exc
            if index < 0:
                raise ModelRejected(f"class index {index} is negative")
            if index in indexed:
                raise ModelRejected(f"class index {index} is duplicated")
            name = str(value).strip()
            if not name:
                raise ModelRejected(f"class {index} has an empty name")
            indexed[index] = name
        if not indexed:
            raise ModelRejected("the model declares no classes")
        expected = set(range(len(indexed)))
        if set(indexed) != expected:
            raise ModelRejected(
                "class indices must run from 0 without gaps"
            )
        return tuple(indexed[index] for index in sorted(indexed))
    if isinstance(raw, Sequence):
        names = [str(item).strip() for item in raw]
        if not names:
            raise ModelRejected("the model declares no classes")
        if any(not name for name in names):
            raise ModelRejected("a class name is empty")
        return tuple(names)
    raise ModelRejected("the class map is not a mapping or list")


def require_detection_task(metadata: Mapping[str, object]) -> str:
    """Return the task, refusing anything that is not detection."""

    for key in TASK_KEYS:
        if key in metadata:
            task = str(metadata[key]).strip().casefold()
            if task != SUPPORTED_TASK:
                raise ModelRejected(
                    f"only detection models are supported, not {task!r}"
                )
            return task
    raise ModelRejected("the model does not declare a task")


def normalize_family(metadata: Mapping[str, object]) -> str:
    """Return a recognized YOLO11 family name for the model."""

    raw = str(
        metadata.get("family")
        or metadata.get("model")
        or metadata.get("description")
        or ""
    ).casefold()
    if "yolo11" in raw or "yolov11" in raw:
        return "yolo11"
    raise ModelRejected(
        "only YOLO11 detection models are validated for this release"
    )


def reject_embedded_nms(output_names: Sequence[str]) -> None:
    """Refuse an export that applies its own NMS.

    The plugin owns NMS so the NMS IoU setting always applies; a graph that
    has already suppressed boxes would make that setting a no-op.
    """

    lowered = [str(name).casefold() for name in output_names]
    for name in lowered:
        for marker in EMBEDDED_NMS_OUTPUT_NAMES:
            if marker in name:
                raise ModelRejected(
                    "this export applies its own NMS; export without NMS "
                    "so the plugin's NMS IoU setting applies"
                )
    if len(lowered) > 1:
        raise ModelRejected(
            f"expected one raw detection output, found {len(lowered)}"
        )


def describe_output_layout(
    shape: Sequence[object], class_count: int
) -> OutputLayout:
    """Return how a raw YOLO detection output is arranged.

    YOLO11 exports emit either ``(1, 4 + nc, N)`` or ``(1, N, 4 + nc)``.
    When both axes could be the attribute axis the layout is genuinely
    ambiguous, and guessing would transpose every box, so it is refused.
    """

    if len(shape) != 3:
        raise ModelRejected(
            f"expected a 3-dimensional detection output, got {list(shape)}"
        )
    batch, first, second = shape
    if isinstance(batch, int) and batch not in (1, -1, 0):
        raise ModelRejected(f"unsupported batch dimension: {batch}")
    attributes = BOX_COORDINATES + class_count
    if not isinstance(first, int) or not isinstance(second, int):
        raise ModelRejected(
            "the detection output has an unknown dynamic shape"
        )
    first_matches = first == attributes
    second_matches = second == attributes
    if first_matches and second_matches:
        raise ModelRejected(
            "the detection output layout is ambiguous; both axes match the "
            "attribute count"
        )
    if first_matches:
        return OutputLayout(
            transposed=True, predictions=second, class_count=class_count
        )
    if second_matches:
        return OutputLayout(
            transposed=False, predictions=first, class_count=class_count
        )
    raise ModelRejected(
        f"the detection output shape {list(shape)} does not match "
        f"{class_count} classes"
    )


def auto_selected_class_ids(class_names: Sequence[str]) -> tuple[int, ...]:
    """Return the classes to preselect for a model.

    A single-class model has nothing to choose, so it is selected for the
    user. A multi-class model returns nothing preselected: silently picking
    classes would change the count without the user asking.
    """

    if len(class_names) == 1:
        return (0,)
    return ()


def validate_selected_class_ids(
    selected: Sequence[int], class_names: Sequence[str]
) -> tuple[int, ...]:
    """Return validated, ordered class selections for a model."""

    result: list[int] = []
    for value in selected:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ModelRejected("a selected class id is not an integer")
        if not 0 <= value < len(class_names):
            raise ModelRejected(
                f"selected class {value} is not in the model's class map"
            )
        if value in result:
            raise ModelRejected(f"selected class {value} is duplicated")
        result.append(value)
    return tuple(result)


__all__ = [
    "BOX_COORDINATES",
    "EMBEDDED_NMS_OUTPUT_NAMES",
    "OutputLayout",
    "auto_selected_class_ids",
    "describe_output_layout",
    "normalize_family",
    "parse_class_map",
    "reject_embedded_nms",
    "require_detection_task",
    "validate_selected_class_ids",
]

"""The contract every detection backend implements.

Backends differ in how they load a model and produce raw predictions, but
they must agree on what a model is described as and what a tile prediction
looks like, so the runner can treat ONNX and PyTorch identically. NMS,
cross-tile deduplication, and scope filtering are never a backend concern.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tree_counter.errors import ErrorCode, TreeCounterError

SUPPORTED_TASK = "detect"
SUPPORTED_FAMILIES = ("yolo11", "yolo11n", "yolo11s", "yolo11m", "yolo11l",
                      "yolo11x")
SUPPORTED_FORMATS = ("onnx", "pt")


class ModelRejected(TreeCounterError):
    """The model is not a supported YOLO11 detection model."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.INVALID_MODEL, diagnostic_detail=detail)


@dataclass(frozen=True)
class ModelDescription:
    """Everything the host may learn about a model.

    The absolute path is deliberately absent: provenance records the
    filename and hash only.
    """

    filename: str
    sha256: str
    model_format: str
    task: str
    family: str
    class_names: tuple[str, ...]
    input_width: int
    input_height: int
    dynamic_shape: bool
    backend: str
    device: str
    warnings: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.model_format not in SUPPORTED_FORMATS:
            raise ModelRejected(f"unsupported format: {self.model_format!r}")
        if self.task != SUPPORTED_TASK:
            raise ModelRejected(
                f"only {SUPPORTED_TASK} models are supported, not "
                f"{self.task!r}"
            )
        if not self.class_names:
            raise ModelRejected("the model declares no classes")
        if "/" in self.filename or "\\" in self.filename:
            raise ModelRejected("filename must not contain a path")

    @property
    def is_single_class(self) -> bool:
        """Return whether the UI should auto-select the only class."""

        return len(self.class_names) == 1

    def as_message(self) -> dict[str, Any]:
        """Return the protocol payload for a ``model_info`` message."""

        payload: dict[str, Any] = {
            "class_names": list(self.class_names),
            "backend": self.backend,
            "device": self.device,
        }
        if not self.dynamic_shape:
            payload["input_size"] = self.input_width
        return payload


@dataclass(frozen=True)
class RawDetection:
    """One prediction in tile-local pixel coordinates, before NMS."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_id: int

    def as_payload(self, class_names: Sequence[str]) -> dict[str, Any]:
        """Return the runner-facing dictionary for this prediction."""

        if not 0 <= self.class_id < len(class_names):
            raise ModelRejected(
                f"class id {self.class_id} is outside the model's class map"
            )
        return {
            "box": [self.x_min, self.y_min, self.x_max, self.y_max],
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": class_names[self.class_id],
        }


@runtime_checkable
class DetectionBackend(Protocol):
    """What the worker runner requires of any backend."""

    name: str

    def inspect(
        self, model_path: str, model_sha256: str
    ) -> ModelDescription:
        """Describe a model without preparing it for inference."""

    def capabilities(self) -> tuple[str, ...]:
        """Return the accelerators this backend can actually use."""

    def start_run(
        self, model_path: str, model_sha256: str, settings: Any
    ) -> Mapping[str, Any]:
        """Load the model once for a run and report backend and device."""

    def infer_tile(
        self, tile_path: str, tile: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Return raw predictions in tile-local pixel coordinates."""

    def close(self) -> None:
        """Release the model and any device resources."""


__all__ = [
    "SUPPORTED_FAMILIES",
    "SUPPORTED_FORMATS",
    "SUPPORTED_TASK",
    "DetectionBackend",
    "ModelDescription",
    "ModelRejected",
    "RawDetection",
]

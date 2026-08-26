"""A deterministic in-process backend used only by the worker tests.

This module is a test fixture. It is never listed in the plugin package
manifest and therefore never reaches the distributable archive.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


class FakeBackendFailure(RuntimeError):
    """A scripted failure used to exercise worker error handling."""


def _class_names() -> tuple[str, ...]:
    raw = os.environ.get("TREE_COUNTER_FAKE_CLASS_NAMES", "oil_palm")
    return tuple(part for part in raw.split(",") if part)


class FakeBackend:
    """Emit fixed tile-local boxes so results are exactly predictable."""

    name = "fake"

    def __init__(self) -> None:
        self.closed = False
        self._class_names = _class_names()

    def inspect(self, model_path: str, model_sha256: str) -> dict[str, Any]:
        if os.environ.get("TREE_COUNTER_FAKE_INSPECT_ERROR"):
            raise FakeBackendFailure("scripted inspect failure")
        return {
            "class_names": list(self._class_names),
            "backend": self.name,
            "device": "cpu",
            "input_size": 640,
        }

    def start_run(
        self, model_path: str, model_sha256: str, settings: Any
    ) -> dict[str, Any]:
        if os.environ.get("TREE_COUNTER_FAKE_START_ERROR"):
            raise FakeBackendFailure("scripted start failure")
        return {"backend": self.name, "device": "cpu"}

    def infer_tile(
        self, tile_path: str, tile: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        failing = os.environ.get("TREE_COUNTER_FAKE_FAIL_ON_TILE")
        if failing and failing == tile["tile_id"]:
            raise FakeBackendFailure("scripted tile failure")
        bulk = int(os.environ.get("TREE_COUNTER_FAKE_BULK_DETECTIONS", "0"))
        if bulk:
            return self._bulk(bulk)
        detections: list[dict[str, Any]] = [
            {
                "box": [1.0, 1.0, 11.0, 11.0],
                "confidence": 0.90,
                "class_id": 0,
                "class_name": self._class_names[0],
            },
            {
                # IoU 0.82 against the first box, so per-tile NMS at the
                # default 0.70 threshold must remove it.
                "box": [1.5, 1.5, 11.5, 11.5],
                "confidence": 0.80,
                "class_id": 0,
                "class_name": self._class_names[0],
            },
            {
                # Below the default confidence threshold.
                "box": [40.0, 40.0, 50.0, 50.0],
                "confidence": 0.10,
                "class_id": 0,
                "class_name": self._class_names[0],
            },
            {
                # Centered strictly outside the valid area, as happens in
                # the padding of an edge tile.
                "box": [
                    float(tile["model_width"]) - 2.0,
                    float(tile["model_height"]) - 2.0,
                    float(tile["model_width"]) + 8.0,
                    float(tile["model_height"]) + 8.0,
                ],
                "confidence": 0.95,
                "class_id": 0,
                "class_name": self._class_names[0],
            },
        ]
        if len(self._class_names) > 1:
            detections.append(
                {
                    # Same geometry as the winner but a different class.
                    "box": [1.0, 1.0, 11.0, 11.0],
                    "confidence": 0.70,
                    "class_id": 1,
                    "class_name": self._class_names[1],
                }
            )
        return detections

    def _bulk(self, count: int) -> list[dict[str, Any]]:
        """Return *count* well-separated boxes on a fixed grid.

        Nothing here overlaps, so every box survives NMS and dedup and the
        result set is exactly *count* detections.
        """

        columns = 100
        return [
            {
                "box": [
                    float((index % columns) * 5),
                    float((index // columns) * 5),
                    float((index % columns) * 5) + 4.0,
                    float((index // columns) * 5) + 4.0,
                ],
                "confidence": 0.9,
                "class_id": 0,
                "class_name": self._class_names[0],
            }
            for index in range(count)
        ]

    def close(self) -> None:
        self.closed = True


def create_backend() -> FakeBackend:
    """Return the backend instance the worker will use."""

    return FakeBackend()

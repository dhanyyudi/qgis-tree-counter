"""Tests for the stable Tree Counter domain value objects."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import pytest


def test_default_inference_settings_are_frozen_and_json_serializable() -> None:
    from tree_counter.core.types import InferenceSettings

    settings = InferenceSettings()

    assert settings.confidence == 0.25
    assert settings.nms_iou == 0.70
    assert settings.duplicate_iou == 0.50
    assert settings.tile_size == 640
    assert settings.overlap_percent == 20
    assert settings.selected_class_ids == ()
    assert settings.requested_device == "auto"
    assert json.dumps(asdict(settings))
    with pytest.raises(FrozenInstanceError):
        settings.confidence = 0.5  # type: ignore[misc]


def test_pixel_box_rejects_non_finite_or_reversed_edges() -> None:
    from tree_counter.core.types import PixelBox

    box = PixelBox(1.0, 2.0, 3.0, 4.0)
    assert asdict(box) == {
        "x_min": 1.0,
        "y_min": 2.0,
        "x_max": 3.0,
        "y_max": 4.0,
    }
    for values in (
        (float("nan"), 0.0, 1.0, 1.0),
        (0.0, float("inf"), 1.0, 1.0),
        (2.0, 0.0, 1.0, 1.0),
        (0.0, 2.0, 1.0, 1.0),
    ):
        with pytest.raises(ValueError):
            PixelBox(*values)


def test_detection_defaults_are_serializable() -> None:
    from tree_counter.core.types import Detection, PixelBox

    detection = Detection(
        box=PixelBox(1.0, 2.0, 3.0, 4.0),
        confidence=0.9,
        class_id=2,
        class_name="oak",
        tile_ids=("tile-1",),
    )

    assert detection.merged_count == 1
    assert json.dumps(asdict(detection))

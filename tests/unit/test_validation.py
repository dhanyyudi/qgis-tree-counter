"""Tests for Tree Counter settings and domain validation."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math

import pytest


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01]
)
def test_validate_confidence_rejects_non_finite_and_out_of_range_values(
    value: float,
) -> None:
    from tree_counter.core.validation import validate_confidence

    with pytest.raises(ValueError):
        validate_confidence(value)


def test_validate_confidence_accepts_inclusive_bounds() -> None:
    from tree_counter.core.validation import validate_confidence

    assert validate_confidence(0.0) == 0.0
    assert validate_confidence(1.0) == 1.0


@pytest.mark.parametrize("tile_size", [255, 257, 2049, 0, -32])
def test_validate_tile_size_rejects_out_of_range_or_non_multiple_values(
    tile_size: int,
) -> None:
    from tree_counter.core.validation import validate_tile_size

    with pytest.raises(ValueError):
        validate_tile_size(tile_size)


def test_validate_tile_size_accepts_inclusive_divisible_bounds() -> None:
    from tree_counter.core.validation import validate_tile_size

    assert validate_tile_size(256) == 256
    assert validate_tile_size(2048) == 2048


@pytest.mark.parametrize("overlap", [-0.01, 50.01, float("nan"), math.inf])
def test_validate_overlap_rejects_values_outside_zero_to_fifty(
    overlap: float,
) -> None:
    from tree_counter.core.validation import validate_overlap_percent

    with pytest.raises(ValueError):
        validate_overlap_percent(overlap)


def test_validate_overlap_accepts_inclusive_bounds() -> None:
    from tree_counter.core.validation import validate_overlap_percent

    assert validate_overlap_percent(0) == 0
    assert validate_overlap_percent(50) == 50


def test_validate_inference_settings_rejects_invalid_nested_values() -> None:
    from tree_counter.core.types import InferenceSettings

    for kwargs in (
        {"confidence": float("nan")},
        {"nms_iou": 1.1},
        {"duplicate_iou": -0.1},
        {"tile_size": 250},
        {"overlap_percent": 51},
    ):
        with pytest.raises(ValueError):
            InferenceSettings(**kwargs)

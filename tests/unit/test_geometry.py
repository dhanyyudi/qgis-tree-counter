"""Tests for QGIS-free affine and detection scope geometry."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math

import pytest


def test_map_rectangle_converts_to_floor_ceil_and_clamps_pixel_edges() -> None:
    from tree_counter.core.geometry import map_rect_to_pixel_rect

    assert map_rect_to_pixel_rect(
        (95.0, 185.0, 145.0, 225.0),
        100.0,
        220.0,
        10.0,
        -10.0,
        4,
        4,
    ) == (0, 0, 4, 4)
    assert map_rect_to_pixel_rect(
        (105.1, 190.1, 129.9, 214.9),
        100.0,
        220.0,
        10.0,
        -10.0,
        4,
        4,
    ) == (0, 0, 3, 3)


def test_map_rectangle_normalizes_coordinate_order() -> None:
    from tree_counter.core.geometry import map_rect_to_pixel_rect

    assert map_rect_to_pixel_rect(
        (130.0, 190.0, 110.0, 220.0),
        100.0,
        220.0,
        10.0,
        -10.0,
        4,
        4,
    ) == (1, 0, 3, 3)


@pytest.mark.parametrize(
    ("rectangle", "expected"),
    [
        ((150.0, 190.0, 160.0, 200.0), (4, 2, 4, 3)),
        ((50.0, 190.0, 60.0, 200.0), (0, 2, 0, 3)),
        ((110.0, 170.0, 120.0, 175.0), (1, 4, 2, 4)),
    ],
)
def test_non_overlapping_map_rectangles_return_zero_area_clamped_tuple(
    rectangle: tuple[float, ...], expected: tuple[int, ...]
) -> None:
    from tree_counter.core.geometry import map_rect_to_pixel_rect

    assert map_rect_to_pixel_rect(
        rectangle, 100.0, 220.0, 10.0, -10.0, 4, 4
    ) == expected


def test_polygon_boundary_is_included_but_hole_interior_is_excluded() -> None:
    from tree_counter.core.geometry import point_in_polygon

    rings = (
        ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
        ((3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)),
    )
    assert point_in_polygon((0.0, 5.0), rings)
    assert point_in_polygon((3.0, 5.0), rings)
    assert not point_in_polygon((5.0, 5.0), rings)
    assert point_in_polygon((1.0, 1.0), rings)
    assert not point_in_polygon((12.0, 5.0), rings)


def test_detection_scope_filtering_uses_box_center_and_input_order() -> None:
    from tree_counter.core.geometry import (
        filter_detections_by_pixel_rect,
        filter_detections_by_polygon,
    )
    from tree_counter.core.types import Detection, PixelBox

    detections = tuple(
        Detection(PixelBox(*edges), 0.5, index, "tree", (f"t{index}",))
        for index, edges in enumerate(
            (
                (0.0, 0.0, 2.0, 2.0),
                (10.0, 8.0, 12.0, 12.0),
                (4.0, 4.0, 6.0, 6.0),
            )
        )
    )
    assert [item.class_id for item in filter_detections_by_pixel_rect(
        detections, (0, 0, 10, 10)
    )] == [0, 2]
    assert [item.class_id for item in filter_detections_by_polygon(
        detections,
        (((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),),
    )] == [0, 2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"origin_x": math.nan},
        {"pixel_width": 0.0},
        {"pixel_height": 1.0},
        {"raster_width": 0},
        {"raster_height": True},
    ],
)
def test_affine_conversion_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    from tree_counter.core.geometry import map_rect_to_pixel_rect

    values: dict[str, object] = {
        "map_rect": (0.0, 0.0, 10.0, 10.0),
        "origin_x": 0.0,
        "origin_y": 10.0,
        "pixel_width": 1.0,
        "pixel_height": -1.0,
        "raster_width": 10,
        "raster_height": 10,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        map_rect_to_pixel_rect(**values)


def test_polygon_validation_rejects_degenerate_or_non_finite_rings() -> None:
    from tree_counter.core.geometry import point_in_polygon

    for rings in (
        (),
        (((0.0, 0.0), (1.0, 1.0)),),
        (((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),),
        (((0.0, 0.0), (math.inf, 0.0), (1.0, 1.0)),),
    ):
        with pytest.raises(ValueError):
            point_in_polygon((0.0, 0.0), rings)

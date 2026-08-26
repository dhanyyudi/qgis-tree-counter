"""Tests for confidence filtering and class-aware per-tile NMS."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


def _detection(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    confidence: float,
    class_id: int = 0,
    tile_id: str = "r00000_c00000",
):
    from tree_counter.core.types import Detection, PixelBox

    return Detection(
        box=PixelBox(x_min, y_min, x_max, y_max),
        confidence=confidence,
        class_id=class_id,
        class_name=f"class_{class_id}",
        tile_ids=(tile_id,),
    )


def _box(x_min: float, y_min: float, x_max: float, y_max: float):
    from tree_counter.core.types import PixelBox

    return PixelBox(x_min, y_min, x_max, y_max)


def test_disjoint_boxes_have_zero_iou() -> None:
    from tree_counter.core.nms import box_iou

    assert box_iou(_box(0, 0, 10, 10), _box(50, 50, 60, 60)) == 0.0


def test_identical_boxes_have_full_iou() -> None:
    from tree_counter.core.nms import box_iou

    assert box_iou(_box(0, 0, 10, 10), _box(0, 0, 10, 10)) == 1.0


def test_edge_touching_boxes_have_zero_iou() -> None:
    from tree_counter.core.nms import box_iou

    assert box_iou(_box(0, 0, 10, 10), _box(10, 0, 20, 10)) == 0.0


def test_half_overlap_iou_is_exact() -> None:
    from tree_counter.core.nms import box_iou

    # Intersection 50, union 150.
    assert box_iou(_box(0, 0, 10, 10), _box(5, 0, 15, 10)) == pytest.approx(
        50.0 / 150.0
    )


def test_iou_is_symmetric() -> None:
    from tree_counter.core.nms import box_iou

    left = _box(0, 0, 10, 10)
    right = _box(3, 4, 12, 20)
    assert box_iou(left, right) == box_iou(right, left)


@pytest.mark.parametrize(
    "value",
    [None, "box", (0, 0, 1, 1), object()],
)
def test_iou_rejects_objects_without_pixel_edges(value: object) -> None:
    from tree_counter.core.nms import box_iou
    from tree_counter.errors import ValidationError

    with pytest.raises(ValidationError):
        box_iou(_box(0, 0, 1, 1), value)


def test_iou_rejects_zero_area_boxes() -> None:
    from tree_counter.core.nms import box_iou
    from tree_counter.errors import ValidationError

    class _ZeroArea:
        x_min = 0.0
        y_min = 0.0
        x_max = 0.0
        y_max = 5.0

    with pytest.raises(ValidationError):
        box_iou(_box(0, 0, 1, 1), _ZeroArea())


def test_confidence_filter_is_inclusive() -> None:
    from tree_counter.core.nms import filter_by_confidence

    detections = (
        _detection(0, 0, 10, 10, 0.24),
        _detection(20, 20, 30, 30, 0.25),
        _detection(40, 40, 50, 50, 0.26),
    )

    kept = filter_by_confidence(detections, 0.25)

    assert tuple(round(item.confidence, 2) for item in kept) == (0.25, 0.26)


def test_confidence_filter_preserves_input_order() -> None:
    from tree_counter.core.nms import filter_by_confidence

    detections = (
        _detection(0, 0, 10, 10, 0.90),
        _detection(20, 20, 30, 30, 0.30),
        _detection(40, 40, 50, 50, 0.60),
    )

    kept = filter_by_confidence(detections, 0.25)

    assert tuple(item.confidence for item in kept) == (0.90, 0.30, 0.60)


def test_confidence_filter_rejects_invalid_threshold() -> None:
    from tree_counter.core.nms import filter_by_confidence
    from tree_counter.errors import ValidationError

    with pytest.raises(ValidationError):
        filter_by_confidence((), 1.5)


def test_nms_applies_confidence_filter_before_suppression() -> None:
    from tree_counter.core.nms import apply_nms

    # The weak detection would survive NMS if filtering ran afterwards,
    # because the strong overlapping detection is removed by confidence.
    detections = (
        _detection(0, 0, 10, 10, 0.10),
        _detection(1, 1, 11, 11, 0.05),
    )

    assert apply_nms(detections, 0.5, confidence_threshold=0.25) == ()


def test_nms_keeps_highest_confidence_of_an_overlapping_pair() -> None:
    from tree_counter.core.nms import apply_nms

    detections = (
        _detection(1, 1, 11, 11, 0.40),
        _detection(0, 0, 10, 10, 0.90),
    )

    kept = apply_nms(detections, 0.5)

    assert len(kept) == 1
    assert kept[0].confidence == pytest.approx(0.90)


def test_nms_threshold_is_inclusive() -> None:
    from tree_counter.core.nms import apply_nms, box_iou

    low = _detection(0, 0, 10, 10, 0.90)
    high = _detection(5, 0, 15, 10, 0.80)
    threshold = box_iou(low.box, high.box)

    assert len(apply_nms((low, high), threshold)) == 1


def test_nms_keeps_pairs_below_the_threshold() -> None:
    from tree_counter.core.nms import apply_nms, box_iou

    first = _detection(0, 0, 10, 10, 0.90)
    second = _detection(5, 0, 15, 10, 0.80)
    threshold = box_iou(first.box, second.box) + 1e-9

    assert len(apply_nms((first, second), threshold)) == 2


def test_nms_never_suppresses_a_different_class() -> None:
    from tree_counter.core.nms import apply_nms

    detections = (
        _detection(0, 0, 10, 10, 0.90, class_id=0),
        _detection(0, 0, 10, 10, 0.80, class_id=1),
    )

    kept = apply_nms(detections, 0.1)

    assert sorted(item.class_id for item in kept) == [0, 1]


def test_nms_output_is_confidence_descending_and_stable() -> None:
    from tree_counter.core.nms import apply_nms

    detections = (
        _detection(100, 100, 110, 110, 0.50),
        _detection(0, 0, 10, 10, 0.90),
        _detection(50, 50, 60, 60, 0.70),
    )

    kept = apply_nms(detections, 0.5)

    assert [item.confidence for item in kept] == [0.90, 0.70, 0.50]


def test_nms_breaks_confidence_ties_by_class_then_coordinates() -> None:
    from tree_counter.core.nms import apply_nms

    detections = (
        _detection(50, 0, 60, 10, 0.60, class_id=1),
        _detection(0, 0, 10, 10, 0.60, class_id=1),
        _detection(0, 0, 10, 10, 0.60, class_id=0),
    )

    kept = apply_nms(detections, 0.9)

    assert [(item.class_id, item.box.x_min) for item in kept] == [
        (0, 0.0),
        (1, 0.0),
        (1, 50.0),
    ]


def test_nms_is_invariant_under_input_permutation() -> None:
    import itertools

    from tree_counter.core.nms import apply_nms

    detections = (
        _detection(0, 0, 10, 10, 0.90),
        _detection(2, 2, 12, 12, 0.80),
        _detection(40, 40, 50, 50, 0.70, class_id=1),
        _detection(41, 41, 51, 51, 0.60, class_id=1),
        _detection(41, 41, 51, 51, 0.60, class_id=0),
    )
    expected = apply_nms(detections, 0.5)

    for permutation in itertools.permutations(detections):
        assert apply_nms(permutation, 0.5) == expected


def test_nms_does_not_merge_provenance() -> None:
    from tree_counter.core.nms import apply_nms

    detections = (
        _detection(0, 0, 10, 10, 0.90, tile_id="r00000_c00000"),
        _detection(1, 1, 11, 11, 0.80, tile_id="r00000_c00000"),
    )

    kept = apply_nms(detections, 0.5)

    assert kept[0].tile_ids == ("r00000_c00000",)
    assert kept[0].merged_count == 1


def test_nms_rejects_an_invalid_threshold() -> None:
    from tree_counter.core.nms import apply_nms
    from tree_counter.errors import ValidationError

    with pytest.raises(ValidationError):
        apply_nms((), -0.1)


def test_nms_rejects_non_detection_items() -> None:
    from tree_counter.core.nms import apply_nms
    from tree_counter.errors import ValidationError

    with pytest.raises(ValidationError):
        apply_nms(({"confidence": 0.9},), 0.5)


def test_nms_of_an_empty_sequence_is_empty() -> None:
    from tree_counter.core.nms import apply_nms

    assert apply_nms((), 0.5) == ()

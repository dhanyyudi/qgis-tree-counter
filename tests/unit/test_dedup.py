"""Tests for class-aware cross-tile duplicate suppression."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import itertools
import random

import pytest


def _detection(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    confidence: float,
    class_id: int = 0,
    tile_ids: tuple[str, ...] = ("r00000_c00000",),
    merged_count: int = 1,
):
    from tree_counter.core.types import Detection, PixelBox

    return Detection(
        box=PixelBox(x_min, y_min, x_max, y_max),
        confidence=confidence,
        class_id=class_id,
        class_name=f"class_{class_id}",
        tile_ids=tile_ids,
        merged_count=merged_count,
    )


def _reference_deduplicate(detections, duplicate_iou):
    """A deliberately simple O(n^2) oracle for the indexed implementation."""

    from tree_counter.core.nms import box_iou
    from tree_counter.core.types import Detection

    ordered = sorted(
        detections,
        key=lambda item: (
            -item.confidence,
            item.class_id,
            item.box.x_min,
            item.box.y_min,
            item.box.x_max,
            item.box.y_max,
            item.tile_ids,
        ),
    )
    kept: list[Detection] = []
    for candidate in ordered:
        absorbed = False
        for index, existing in enumerate(kept):
            if existing.class_id != candidate.class_id:
                continue
            overlap = box_iou(existing.box, candidate.box)
            if overlap > 0.0 and overlap >= duplicate_iou:
                merged_ids = tuple(
                    sorted(set(existing.tile_ids) | set(candidate.tile_ids))
                )
                kept[index] = Detection(
                    box=existing.box,
                    confidence=existing.confidence,
                    class_id=existing.class_id,
                    class_name=existing.class_name,
                    tile_ids=merged_ids,
                    merged_count=existing.merged_count
                    + candidate.merged_count,
                )
                absorbed = True
                break
        if not absorbed:
            kept.append(candidate)
    return tuple(kept)


def test_empty_input_is_empty() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    assert deduplicate_detections((), 0.5) == ()


def test_non_overlapping_detections_are_all_kept() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(0, 0, 10, 10, 0.90),
        _detection(100, 100, 110, 110, 0.80),
    )

    assert len(deduplicate_detections(detections, 0.5)) == 2


def test_overlapping_same_class_keeps_highest_confidence_geometry() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(1, 1, 11, 11, 0.40, tile_ids=("r00000_c00001",)),
        _detection(0, 0, 10, 10, 0.90, tile_ids=("r00000_c00000",)),
    )

    kept = deduplicate_detections(detections, 0.5)

    assert len(kept) == 1
    assert kept[0].confidence == pytest.approx(0.90)
    assert (
        kept[0].box.x_min,
        kept[0].box.y_min,
        kept[0].box.x_max,
        kept[0].box.y_max,
    ) == (0.0, 0.0, 10.0, 10.0)


def test_merging_unions_sorted_tile_ids_and_counts() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(0, 0, 10, 10, 0.90, tile_ids=("r00000_c00001",)),
        _detection(1, 1, 11, 11, 0.80, tile_ids=("r00000_c00000",)),
        _detection(
            1, 1, 11, 11, 0.70, tile_ids=("r00000_c00002", "r00000_c00001")
        ),
    )

    kept = deduplicate_detections(detections, 0.5)

    assert len(kept) == 1
    assert kept[0].tile_ids == (
        "r00000_c00000",
        "r00000_c00001",
        "r00000_c00002",
    )
    assert kept[0].merged_count == 3


def test_merging_accumulates_existing_merged_counts() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(0, 0, 10, 10, 0.90, merged_count=2),
        _detection(1, 1, 11, 11, 0.80, merged_count=3),
    )

    kept = deduplicate_detections(detections, 0.5)

    assert kept[0].merged_count == 5


def test_a_zero_threshold_keeps_disjoint_detections() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(0, 0, 10, 10, 0.90),
        _detection(1000, 0, 1010, 10, 0.80),
        _detection(2000, 0, 2010, 10, 0.70),
    )

    assert len(deduplicate_detections(detections, 0.0)) == 3


def test_a_zero_threshold_result_is_independent_of_grid_placement() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    # These two boxes are disjoint but close enough to share a grid cell.
    # A grid is an index, not a rule: it must not change the outcome.
    adjacent = (
        _detection(0, 0, 10, 10, 0.90),
        _detection(11, 0, 21, 10, 0.80),
    )
    distant = (
        _detection(0, 0, 10, 10, 0.90),
        _detection(5000, 0, 5010, 10, 0.80),
    )

    assert len(deduplicate_detections(adjacent, 0.0)) == 2
    assert len(deduplicate_detections(distant, 0.0)) == 2


def test_a_zero_threshold_still_merges_any_overlap() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(0, 0, 10, 10, 0.90),
        _detection(9, 0, 19, 10, 0.80),
    )

    kept = deduplicate_detections(detections, 0.0)

    assert len(kept) == 1
    assert kept[0].merged_count == 2


def test_different_classes_are_never_merged() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(0, 0, 10, 10, 0.90, class_id=0),
        _detection(0, 0, 10, 10, 0.80, class_id=1),
    )

    kept = deduplicate_detections(detections, 0.1)

    assert sorted(item.class_id for item in kept) == [0, 1]
    assert all(item.merged_count == 1 for item in kept)


def test_duplicate_threshold_is_inclusive() -> None:
    from tree_counter.core.dedup import deduplicate_detections
    from tree_counter.core.nms import box_iou

    first = _detection(0, 0, 10, 10, 0.90)
    second = _detection(5, 0, 15, 10, 0.80)
    threshold = box_iou(first.box, second.box)

    assert len(deduplicate_detections((first, second), threshold)) == 1


def test_detections_below_the_threshold_are_kept_apart() -> None:
    from tree_counter.core.dedup import deduplicate_detections
    from tree_counter.core.nms import box_iou

    first = _detection(0, 0, 10, 10, 0.90)
    second = _detection(5, 0, 15, 10, 0.80)
    threshold = box_iou(first.box, second.box) + 1e-9

    assert len(deduplicate_detections((first, second), threshold)) == 2


def test_output_is_confidence_descending() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(100, 100, 110, 110, 0.50),
        _detection(0, 0, 10, 10, 0.90),
        _detection(50, 50, 60, 60, 0.70),
    )

    kept = deduplicate_detections(detections, 0.5)

    assert [item.confidence for item in kept] == [0.90, 0.70, 0.50]


def test_result_is_invariant_under_input_permutation() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(0, 0, 10, 10, 0.90, tile_ids=("r00000_c00000",)),
        _detection(2, 2, 12, 12, 0.80, tile_ids=("r00000_c00001",)),
        _detection(40, 40, 50, 50, 0.70, class_id=1),
        _detection(41, 41, 51, 51, 0.70, class_id=1),
        _detection(41, 41, 51, 51, 0.70, class_id=0),
    )
    expected = deduplicate_detections(detections, 0.5)

    for permutation in itertools.permutations(detections):
        assert deduplicate_detections(permutation, 0.5) == expected


def test_equal_confidence_ties_resolve_deterministically() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(3, 3, 13, 13, 0.60),
        _detection(0, 0, 10, 10, 0.60),
    )

    # Intersection 49 over union 151 is above the 0.30 threshold, so the
    # pair merges and the lower-coordinate box wins the confidence tie.
    kept = deduplicate_detections(detections, 0.30)

    assert len(kept) == 1
    assert kept[0].box.x_min == 0.0
    assert kept[0].merged_count == 2


def test_indexed_result_matches_the_reference_on_seeded_cases() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    rng = random.Random(42)
    for case in range(20):
        detections = []
        for _ in range(60):
            x_min = rng.uniform(0, 200)
            y_min = rng.uniform(0, 200)
            width = rng.uniform(4, 24)
            height = rng.uniform(4, 24)
            detections.append(
                _detection(
                    x_min,
                    y_min,
                    x_min + width,
                    y_min + height,
                    round(rng.uniform(0.25, 0.99), 4),
                    class_id=rng.randrange(3),
                    tile_ids=(f"r00000_c{rng.randrange(9):05d}",),
                )
            )
        threshold = rng.choice((0.0, 0.1, 0.3, 0.5, 0.7, 0.9))
        assert deduplicate_detections(
            tuple(detections), threshold
        ) == _reference_deduplicate(tuple(detections), threshold), (
            f"case {case} diverged at IoU {threshold}"
        )


def test_far_apart_clusters_do_not_interact() -> None:
    from tree_counter.core.dedup import deduplicate_detections

    detections = (
        _detection(0, 0, 10, 10, 0.90),
        _detection(1, 1, 11, 11, 0.80),
        _detection(10000, 10000, 10010, 10010, 0.70),
        _detection(10001, 10001, 10011, 10011, 0.60),
    )

    kept = deduplicate_detections(detections, 0.5)

    assert len(kept) == 2
    assert [item.merged_count for item in kept] == [2, 2]


def test_rejects_an_invalid_threshold() -> None:
    from tree_counter.core.dedup import deduplicate_detections
    from tree_counter.errors import ValidationError

    with pytest.raises(ValidationError):
        deduplicate_detections((), 1.2)


def test_rejects_non_detection_items() -> None:
    from tree_counter.core.dedup import deduplicate_detections
    from tree_counter.errors import ValidationError

    with pytest.raises(ValidationError):
        deduplicate_detections((object(),), 0.5)

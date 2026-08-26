"""Tests for deterministic source raster tiling."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


def _shape(window: object) -> tuple[object, ...]:
    return (
        window.tile_id,
        window.x_offset,
        window.y_offset,
        window.read_width,
        window.read_height,
        window.model_width,
        window.model_height,
    )


def test_small_raster_uses_one_padded_window() -> None:
    from tree_counter.core.tiling import iter_tile_windows

    windows = iter_tile_windows(3, 2, 4, 20)

    assert tuple(map(_shape, windows)) == (
        ("r00000_c00000", 0, 0, 3, 2, 4, 4),
    )


def test_exact_raster_has_row_major_windows_without_padding() -> None:
    from tree_counter.core.tiling import iter_tile_windows

    windows = iter_tile_windows(8, 8, 4, 0)

    assert tuple(map(_shape, windows)) == (
        ("r00000_c00000", 0, 0, 4, 4, 4, 4),
        ("r00000_c00001", 4, 0, 4, 4, 4, 4),
        ("r00001_c00000", 0, 4, 4, 4, 4, 4),
        ("r00001_c00001", 4, 4, 4, 4, 4, 4),
    )


def test_non_divisible_edges_append_final_fit_window() -> None:
    from tree_counter.core.tiling import iter_tile_windows

    windows = iter_tile_windows(9, 6, 4, 0, origin_x=10, origin_y=20)

    assert tuple(map(_shape, windows)) == (
        ("r00000_c00000", 10, 20, 4, 4, 4, 4),
        ("r00000_c00001", 14, 20, 4, 4, 4, 4),
        ("r00000_c00002", 15, 20, 4, 4, 4, 4),
        ("r00001_c00000", 10, 22, 4, 4, 4, 4),
        ("r00001_c00001", 14, 22, 4, 4, 4, 4),
        ("r00001_c00002", 15, 22, 4, 4, 4, 4),
    )


def test_one_pixel_remainder_with_overlap_is_covered() -> None:
    from tree_counter.core.tiling import iter_tile_windows

    windows = iter_tile_windows(9, 5, 4, 25)

    assert [window.x_offset for window in windows[:3]] == [0, 3, 5]
    assert [window.y_offset for window in windows[::3]] == [0, 1]
    assert all(window.read_width == 4 for window in windows)


def test_tile_windows_cover_each_requested_pixel_and_are_deterministic(
) -> None:
    from tree_counter.core.tiling import iter_tile_windows

    for width in (1, 2, 4, 5, 9, 17, 31):
        for height in (1, 3, 4, 6, 13):
            for overlap in (0, 1, 20, 33, 50):
                windows = iter_tile_windows(width, height, 4, overlap)
                assert windows == iter_tile_windows(width, height, 4, overlap)
                assert windows
                for row in range(height):
                    for col in range(width):
                        assert any(
                            window.x_offset
                            <= col
                            < window.x_offset + window.read_width
                            and window.y_offset
                            <= row
                            < window.y_offset + window.read_height
                            for window in windows
                        )


@pytest.mark.parametrize(
    "args",
    [
        (0, 2, 4, 20),
        (2, 0, 4, 20),
        (2, 2, 0, 20),
        (2, 2, 4, -1),
        (2, 2, 4, 51),
        (2, 2, 4, 12.5),
        (2, 2, 4, True),
        (2, 2, True, 20),
    ],
)
def test_tiling_rejects_invalid_dimensions_and_settings(
    args: tuple[object, ...],
) -> None:
    from tree_counter.core.tiling import iter_tile_windows

    with pytest.raises(ValueError):
        iter_tile_windows(*args)

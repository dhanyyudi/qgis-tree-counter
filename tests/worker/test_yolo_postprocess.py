"""Tests for tile fitting and raw YOLO output decoding."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path

import numpy
import pytest

from tree_counter.worker.image import LetterboxTransform
from tree_counter.worker.model_info import OutputLayout

IDENTITY = LetterboxTransform(scale=1.0, pad_x=0.0, pad_y=0.0)


def _write_tile(tmp_path: Path, width: int, height: int) -> Path:
    path = tmp_path / "tile.raw"
    path.write_bytes(bytes(range(256)) * ((width * height * 3) // 256 + 1))
    data = path.read_bytes()[: width * height * 3]
    path.write_bytes(data)
    return path


class TestReadTile:
    """Raw RGB tiles are read without any image codec."""

    def test_a_tile_is_read_with_its_declared_shape(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.worker.image import read_rgb_tile

        path = _write_tile(tmp_path, 4, 3)

        image = read_rgb_tile(path, 4, 3)

        assert image.shape == (3, 4, 3)
        assert image.dtype == numpy.uint8

    def test_a_short_tile_is_rejected(self, tmp_path: Path) -> None:
        from tree_counter.worker.image import TileError, read_rgb_tile

        path = tmp_path / "tile.raw"
        path.write_bytes(b"\x00" * 10)

        with pytest.raises(TileError):
            read_rgb_tile(path, 4, 3)

    def test_a_long_tile_is_rejected(self, tmp_path: Path) -> None:
        from tree_counter.worker.image import TileError, read_rgb_tile

        path = tmp_path / "tile.raw"
        path.write_bytes(b"\x00" * (4 * 3 * 3 + 1))

        with pytest.raises(TileError):
            read_rgb_tile(path, 4, 3)

    def test_a_missing_tile_is_reported(self, tmp_path: Path) -> None:
        from tree_counter.worker.image import TileError, read_rgb_tile

        with pytest.raises(TileError):
            read_rgb_tile(tmp_path / "absent.raw", 4, 3)

    def test_an_absurd_tile_size_is_refused(self, tmp_path: Path) -> None:
        from tree_counter.worker.image import TileError, read_rgb_tile

        with pytest.raises(TileError):
            read_rgb_tile(tmp_path / "absent.raw", 100000, 100000)


class TestLetterbox:
    """A tile is padded, never stretched, to reach the model input."""

    def test_an_exact_tile_is_unchanged(self) -> None:
        from tree_counter.worker.image import letterbox

        image = numpy.full((640, 640, 3), 7, dtype=numpy.uint8)

        padded, transform = letterbox(image, 640, 640)

        assert padded.shape == (640, 640, 3)
        assert transform.scale == 1.0
        assert (transform.pad_x, transform.pad_y) == (0.0, 0.0)
        assert numpy.array_equal(padded, image)

    def test_a_smaller_edge_tile_is_padded_not_scaled(self) -> None:
        from tree_counter.worker.image import PAD_VALUE, letterbox

        image = numpy.full((100, 200, 3), 9, dtype=numpy.uint8)

        padded, transform = letterbox(image, 640, 640)

        assert padded.shape == (640, 640, 3)
        assert transform.scale == 1.0
        assert numpy.array_equal(padded[:100, :200], image)
        assert padded[300, 300, 0] == PAD_VALUE

    def test_padding_keeps_coordinates_identical(self) -> None:
        from tree_counter.worker.image import letterbox

        image = numpy.zeros((100, 200, 3), dtype=numpy.uint8)

        _, transform = letterbox(image, 640, 640)

        assert transform.box_to_tile(10.0, 20.0, 30.0, 40.0) == (
            10.0,
            20.0,
            30.0,
            40.0,
        )

    def test_a_larger_tile_is_scaled_down(self) -> None:
        from tree_counter.worker.image import letterbox

        image = numpy.zeros((1280, 1280, 3), dtype=numpy.uint8)

        padded, transform = letterbox(image, 640, 640)

        assert padded.shape == (640, 640, 3)
        assert transform.scale == pytest.approx(0.5)
        assert transform.box_to_tile(0.0, 0.0, 320.0, 320.0) == pytest.approx(
            (0.0, 0.0, 640.0, 640.0)
        )

    def test_centered_padding_offsets_coordinates(self) -> None:
        from tree_counter.worker.image import letterbox

        image = numpy.zeros((100, 100, 3), dtype=numpy.uint8)

        _, transform = letterbox(image, 200, 200, center=True)

        assert (transform.pad_x, transform.pad_y) == (50.0, 50.0)
        assert transform.to_tile(50.0, 50.0) == (0.0, 0.0)

    def test_a_non_rgb_array_is_rejected(self) -> None:
        from tree_counter.worker.image import TileError, letterbox

        with pytest.raises(TileError):
            letterbox(numpy.zeros((10, 10), dtype=numpy.uint8), 640, 640)

    def test_the_model_input_is_normalized_and_channel_first(self) -> None:
        from tree_counter.worker.image import to_model_input

        image = numpy.full((4, 4, 3), 255, dtype=numpy.uint8)

        batch = to_model_input(image)

        assert batch.shape == (1, 3, 4, 4)
        assert batch.dtype == numpy.float32
        assert float(batch.max()) == pytest.approx(1.0)


def _transposed_output(rows: list[list[float]], class_count: int):
    """Build a (1, 4 + nc, N) tensor from per-detection rows."""

    array = numpy.asarray(rows, dtype=numpy.float32)
    return array.T[None, ...], OutputLayout(
        transposed=True, predictions=len(rows), class_count=class_count
    )


def _row_major_output(rows: list[list[float]], class_count: int):
    """Build a (1, N, 4 + nc) tensor from per-detection rows."""

    array = numpy.asarray(rows, dtype=numpy.float32)
    return array[None, ...], OutputLayout(
        transposed=False, predictions=len(rows), class_count=class_count
    )


class TestDecode:
    """Raw head output becomes canonical tile-local predictions."""

    def test_a_centre_box_becomes_corner_form(self) -> None:
        from tree_counter.worker.yolo_postprocess import decode_predictions

        output, layout = _transposed_output([[50, 60, 20, 10, 0.9]], 1)

        [detection] = decode_predictions(output, layout, IDENTITY, 0.25)

        assert (detection.x_min, detection.y_min) == (40.0, 55.0)
        assert (detection.x_max, detection.y_max) == (60.0, 65.0)
        assert detection.confidence == pytest.approx(0.9)
        assert detection.class_id == 0

    def test_both_layouts_decode_identically(self) -> None:
        from tree_counter.worker.yolo_postprocess import decode_predictions

        rows = [[50, 60, 20, 10, 0.9], [10, 10, 4, 4, 0.5]]
        transposed, first = _transposed_output(rows, 1)
        row_major, second = _row_major_output(rows, 1)

        assert decode_predictions(
            transposed, first, IDENTITY, 0.25
        ) == decode_predictions(row_major, second, IDENTITY, 0.25)

    def test_confidence_filtering_is_inclusive(self) -> None:
        from tree_counter.worker.yolo_postprocess import decode_predictions

        output, layout = _transposed_output(
            [[50, 60, 20, 10, 0.25], [10, 10, 4, 4, 0.24]], 1
        )

        detections = decode_predictions(output, layout, IDENTITY, 0.25)

        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.25)

    def test_the_highest_scoring_class_wins(self) -> None:
        from tree_counter.worker.yolo_postprocess import decode_predictions

        output, layout = _transposed_output([[50, 60, 20, 10, 0.2, 0.8]], 2)

        [detection] = decode_predictions(output, layout, IDENTITY, 0.25)

        assert detection.class_id == 1
        assert detection.confidence == pytest.approx(0.8)

    def test_unselected_classes_are_dropped(self) -> None:
        from tree_counter.worker.yolo_postprocess import decode_predictions

        output, layout = _transposed_output(
            [[50, 60, 20, 10, 0.9, 0.1], [10, 10, 4, 4, 0.1, 0.9]], 2
        )

        detections = decode_predictions(
            output, layout, IDENTITY, 0.25, selected_class_ids=(1,)
        )

        assert [item.class_id for item in detections] == [1]

    def test_coordinates_are_mapped_back_through_the_transform(self) -> None:
        from tree_counter.worker.yolo_postprocess import decode_predictions

        output, layout = _transposed_output([[100, 100, 20, 20, 0.9]], 1)
        transform = LetterboxTransform(scale=0.5, pad_x=10.0, pad_y=20.0)

        [detection] = decode_predictions(output, layout, transform, 0.25)

        assert detection.x_min == pytest.approx((90 - 10) / 0.5)
        assert detection.y_min == pytest.approx((90 - 20) / 0.5)

    def test_an_empty_result_is_an_empty_list(self) -> None:
        from tree_counter.worker.yolo_postprocess import decode_predictions

        output, layout = _transposed_output([[50, 60, 20, 10, 0.1]], 1)

        assert decode_predictions(output, layout, IDENTITY, 0.25) == []

    def test_a_degenerate_box_is_dropped(self) -> None:
        from tree_counter.worker.yolo_postprocess import decode_predictions

        output, layout = _transposed_output([[50, 60, 0, 10, 0.9]], 1)

        assert decode_predictions(output, layout, IDENTITY, 0.25) == []

    def test_nothing_is_suppressed_during_decoding(self) -> None:
        from tree_counter.worker.yolo_postprocess import decode_predictions

        # Two heavily overlapping boxes must both survive: NMS belongs to
        # the plugin so the user's IoU setting always applies.
        output, layout = _transposed_output(
            [[50, 50, 20, 20, 0.9], [51, 51, 20, 20, 0.8]], 1
        )

        assert len(decode_predictions(output, layout, IDENTITY, 0.25)) == 2

    def test_a_wrong_row_width_is_rejected(self) -> None:
        from tree_counter.worker.backend_base import ModelRejected
        from tree_counter.worker.yolo_postprocess import decode_predictions

        output, _ = _transposed_output([[50, 60, 20, 10, 0.9]], 1)
        layout = OutputLayout(transposed=True, predictions=1, class_count=3)

        with pytest.raises(ModelRejected):
            decode_predictions(output, layout, IDENTITY, 0.25)

    def test_a_multi_image_batch_is_rejected(self) -> None:
        from tree_counter.worker.backend_base import ModelRejected
        from tree_counter.worker.yolo_postprocess import decode_predictions

        output, layout = _transposed_output([[50, 60, 20, 10, 0.9]], 1)
        batched = numpy.concatenate([output, output], axis=0)

        with pytest.raises(ModelRejected):
            decode_predictions(batched, layout, IDENTITY, 0.25)

    def test_a_confidence_above_one_is_clamped(self) -> None:
        from tree_counter.worker.yolo_postprocess import decode_predictions

        output, layout = _transposed_output([[50, 60, 20, 10, 1.4]], 1)

        [detection] = decode_predictions(output, layout, IDENTITY, 0.25)

        assert detection.confidence == 1.0

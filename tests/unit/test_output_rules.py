"""Tests for output naming, provenance, and georeferencing arithmetic."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _info(**overrides):
    from tree_counter.qgis_adapter.raster import RasterInfo

    values = {
        "name": "aerial",
        "provider_type": "gdal",
        "width": 1000,
        "height": 800,
        "band_count": 3,
        "is_byte": True,
        "crs_authid": "EPSG:3857",
        "crs_is_valid": True,
        "x_minimum": 0.0,
        "y_minimum": 0.0,
        "x_maximum": 1000.0,
        "y_maximum": 800.0,
    }
    values.update(overrides)
    return RasterInfo(**values)


def _detection(x_min=10.0, y_min=20.0, x_max=30.0, y_max=40.0, **overrides):
    from tree_counter.core.types import Detection, PixelBox

    values = {
        "box": PixelBox(x_min, y_min, x_max, y_max),
        "confidence": 0.9,
        "class_id": 0,
        "class_name": "oil_palm",
        "tile_ids": ("r00000_c00001", "r00000_c00000"),
    }
    values.update(overrides)
    return Detection(**values)


class TestGeoreference:
    """Pixel space grows downwards; map space grows upwards."""

    def test_the_origin_pixel_is_the_top_left_corner(self) -> None:
        from tree_counter.qgis_adapter.georeference import pixel_to_map

        assert pixel_to_map(_info(), 0.0, 0.0) == (0.0, 800.0)

    def test_moving_down_in_pixels_lowers_the_map_y(self) -> None:
        from tree_counter.qgis_adapter.georeference import pixel_to_map

        assert pixel_to_map(_info(), 100.0, 200.0) == (100.0, 600.0)

    def test_the_last_pixel_corner_is_the_extent_corner(self) -> None:
        from tree_counter.qgis_adapter.georeference import pixel_to_map

        assert pixel_to_map(_info(), 1000.0, 800.0) == (1000.0, 0.0)

    def test_a_box_becomes_a_correctly_ordered_rectangle(self) -> None:
        from tree_counter.qgis_adapter.georeference import pixel_box_to_map

        rect = pixel_box_to_map(_info(), _detection().box)

        x_min, y_min, x_max, y_max = rect
        assert (x_min, x_max) == (10.0, 30.0)
        # The pixel box top (y=20) is the higher map y.
        assert (y_min, y_max) == (760.0, 780.0)
        assert y_max > y_min

    def test_a_detection_centre_is_the_box_centre(self) -> None:
        from tree_counter.qgis_adapter.georeference import (
            detection_center_map,
        )

        assert detection_center_map(_info(), _detection()) == (20.0, 770.0)

    def test_a_non_finite_coordinate_is_rejected(self) -> None:
        from tree_counter.qgis_adapter.georeference import (
            GeoreferenceError,
            pixel_to_map,
        )

        with pytest.raises(GeoreferenceError):
            pixel_to_map(_info(), float("nan"), 0.0)

    def test_a_detection_without_a_box_is_rejected(self) -> None:
        from tree_counter.qgis_adapter.georeference import (
            GeoreferenceError,
            detection_center_map,
        )

        with pytest.raises(GeoreferenceError):
            detection_center_map(_info(), object())

    def test_a_real_resolution_places_a_detection_precisely(self) -> None:
        from tree_counter.qgis_adapter.georeference import pixel_to_map

        # The real integration raster's geotransform.
        info = _info(
            width=16458,
            height=16456,
            x_minimum=12790053.673929604,
            y_minimum=-391970.42484612024,
            x_maximum=12790667.932040697,
            y_maximum=-391356.2413805632,
        )

        x, y = pixel_to_map(info, 8000.0, 8000.0)

        offset = 8000 * 0.037322768
        assert x == pytest.approx(12790053.673929604 + offset, abs=1e-3)
        assert y == pytest.approx(-391356.2413805632 - offset, abs=1e-3)


class TestTargetNaming:
    """An existing count is never silently replaced."""

    def _request(self, tmp_path: Path, **overrides):
        from tree_counter.qgis_adapter.output import OutputRequest

        values = {"directory": tmp_path, "raster_stem": "aerial"}
        values.update(overrides)
        return OutputRequest(**values)

    def test_the_default_name_follows_the_raster(self, tmp_path: Path) -> None:
        from tree_counter.qgis_adapter.output import default_output_path

        path = default_output_path(self._request(tmp_path))

        assert path.name == "aerial_tree_counting.gpkg"
        assert path.parent == tmp_path

    def test_a_free_target_is_used_as_is(self, tmp_path: Path) -> None:
        from tree_counter.qgis_adapter.output import resolve_target

        assert resolve_target(self._request(tmp_path)).name == (
            "aerial_tree_counting.gpkg"
        )

    def test_an_existing_target_gets_a_timestamped_sibling(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.qgis_adapter.output import resolve_target

        (tmp_path / "aerial_tree_counting.gpkg").write_text("old")

        target = resolve_target(
            self._request(tmp_path, timestamp="20260827T101500")
        )

        assert target.name == "aerial_tree_counting_20260827T101500.gpkg"
        assert not target.exists()

    def test_an_existing_target_without_a_timestamp_is_refused(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.qgis_adapter.output import (
            OutputError,
            resolve_target,
        )

        (tmp_path / "aerial_tree_counting.gpkg").write_text("old")

        with pytest.raises(OutputError):
            resolve_target(self._request(tmp_path))

    def test_an_awkward_raster_name_is_made_safe(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.qgis_adapter.output import default_output_path

        path = default_output_path(
            self._request(tmp_path, raster_stem="../weird name/../x")
        )

        assert "/" not in path.name
        assert ".." not in path.name

    def test_an_empty_raster_name_is_refused(self, tmp_path: Path) -> None:
        from tree_counter.qgis_adapter.output import OutputError

        with pytest.raises(OutputError):
            self._request(tmp_path, raster_stem="   ")

    def test_staging_sits_beside_the_target(self, tmp_path: Path) -> None:
        from tree_counter.qgis_adapter.output import staging_path

        target = tmp_path / "aerial_tree_counting.gpkg"

        # Same directory means os.replace stays on one filesystem.
        assert staging_path(target).parent == target.parent


class TestSummary:
    """Run provenance records everything except the model's location."""

    def _summary(self, **overrides):
        from tree_counter.core.types import InferenceSettings
        from tree_counter.qgis_adapter.output import build_summary
        from tree_counter.qgis_adapter.scope import PixelScope, ScopeKind
        from tree_counter.qgis_adapter.task import RunResult

        result = RunResult(
            run_id="run-1",
            detections=(
                _detection(),
                _detection(class_id=1, class_name="shade_tree"),
            ),
            warnings=("Falling back to the CPU.",),
            backend="onnxruntime",
            device="coreml",
            provider="CoreMLExecutionProvider",
            duration_seconds=12.5,
            tile_count=42,
        )
        values = {
            "run_id": "run-1",
            "status": "completed",
            "raster_info": _info(),
            "scope": PixelScope(ScopeKind.WHOLE_RASTER, 0, 0, 1000, 800),
            "settings": InferenceSettings(selected_class_ids=(0, 1)),
            "result": result,
            "model_filename": "best.onnx",
            "model_sha256": "a" * 64,
            "started_at": "2026-08-27T10:00:00Z",
            "finished_at": "2026-08-27T10:00:12Z",
        }
        values.update(overrides)
        return build_summary(**values)

    def test_every_declared_field_is_populated(self) -> None:
        from tree_counter.qgis_adapter.output import SUMMARY_FIELDS

        summary = self._summary()

        for name, _, _ in SUMMARY_FIELDS:
            assert name in summary.values, name

    def test_the_row_matches_the_declared_order(self) -> None:
        from tree_counter.qgis_adapter.output import SUMMARY_FIELDS

        row = self._summary().as_row()

        assert len(row) == len(SUMMARY_FIELDS)

    def test_counts_are_recorded_per_class(self) -> None:
        summary = self._summary()

        assert summary.values["total_count"] == 2
        assert json.loads(summary.values["counts_by_class"]) == {
            "oil_palm": 1,
            "shade_tree": 1,
        }

    def test_the_model_is_identified_by_name_and_hash(self) -> None:
        summary = self._summary()

        assert summary.values["model_filename"] == "best.onnx"
        assert summary.values["model_sha256"] == "a" * 64

    def test_no_model_path_reaches_provenance(self) -> None:
        from tree_counter.qgis_adapter.output import (
            summary_contains_no_path,
        )

        summary = self._summary(
            model_filename="best.onnx", model_sha256="a" * 64
        )

        assert summary_contains_no_path(summary) is True
        assert "model_path" not in summary.values

    def test_a_leaked_path_is_detected(self) -> None:
        from tree_counter.qgis_adapter.output import (
            summary_contains_no_path,
        )

        summary = self._summary(model_filename="/home/u/models/best.onnx")

        assert summary_contains_no_path(summary) is False

    def test_every_inference_setting_is_recorded(self) -> None:
        summary = self._summary()

        assert summary.values["confidence_threshold"] == pytest.approx(0.25)
        assert summary.values["nms_iou"] == pytest.approx(0.70)
        assert summary.values["duplicate_iou"] == pytest.approx(0.50)
        assert summary.values["tile_size"] == 640
        assert summary.values["overlap_percent"] == 20
        assert json.loads(summary.values["selected_class_ids"]) == [0, 1]

    def test_backend_device_and_provider_are_recorded(self) -> None:
        summary = self._summary()

        assert summary.values["backend"] == "onnxruntime"
        assert summary.values["device"] == "coreml"
        assert summary.values["provider"] == "CoreMLExecutionProvider"

    def test_warnings_and_duration_are_recorded(self) -> None:
        summary = self._summary()

        assert json.loads(summary.values["warnings"]) == [
            "Falling back to the CPU."
        ]
        assert summary.values["duration_seconds"] == pytest.approx(12.5)
        assert summary.values["tile_count"] == 42

    def test_the_scope_is_recorded(self) -> None:
        summary = self._summary()

        assert summary.values["scope_kind"] == "whole_raster"
        assert summary.values["scope_pixels"] == 800000

"""GeoPackage writing verified by reopening what was written."""

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


def _crs():
    from qgis.core import QgsCoordinateReferenceSystem

    crs = QgsCoordinateReferenceSystem("EPSG:3857")
    assert crs.isValid()
    return crs


def _detections(count=3):
    from tree_counter.core.types import Detection, PixelBox

    return tuple(
        Detection(
            box=PixelBox(
                10.0 + index * 20, 20.0, 30.0 + index * 20, 40.0
            ),
            confidence=0.5 + index * 0.1,
            class_id=index % 2,
            class_name="oil_palm" if index % 2 == 0 else "shade_tree",
            tile_ids=("r00000_c00001", "r00000_c00000"),
        )
        for index in range(count)
    )


def _summary(detections):
    from tree_counter.core.types import InferenceSettings
    from tree_counter.qgis_adapter.output import build_summary
    from tree_counter.qgis_adapter.scope import PixelScope, ScopeKind
    from tree_counter.qgis_adapter.task import RunResult

    result = RunResult(
        run_id="run-1",
        detections=detections,
        backend="onnxruntime",
        device="cpu",
        provider="CPUExecutionProvider",
        duration_seconds=3.5,
        tile_count=4,
    )
    return build_summary(
        run_id="run-1",
        status="completed",
        raster_info=_info(),
        scope=PixelScope(ScopeKind.WHOLE_RASTER, 0, 0, 1000, 800),
        settings=InferenceSettings(),
        result=result,
        model_filename="best.onnx",
        model_sha256="a" * 64,
        started_at="2026-08-27T10:00:00Z",
        finished_at="2026-08-27T10:00:03Z",
    )


def _request(tmp_path: Path, **overrides):
    from tree_counter.qgis_adapter.output import OutputRequest

    values = {"directory": tmp_path, "raster_stem": "aerial"}
    values.update(overrides)
    return OutputRequest(**values)


def _write(tmp_path: Path, detections=None, **request_overrides):
    from tree_counter.qgis_adapter.output import resolve_target, write_results

    detections = _detections() if detections is None else detections
    request = _request(tmp_path, **request_overrides)
    target = resolve_target(request)
    return write_results(
        target, request, _info(), detections, _summary(detections), _crs()
    )


def _layer(path: Path, name: str):
    from qgis.core import QgsVectorLayer

    layer = QgsVectorLayer(f"{path}|layername={name}", name, "ogr")
    assert layer.isValid(), f"{name} did not open"
    return layer


def test_a_run_writes_centers_and_summary(tmp_path: Path) -> None:
    path = _write(tmp_path)

    assert path.is_file()
    assert _layer(path, "tree_centers").featureCount() == 3
    assert _layer(path, "run_summary").featureCount() == 1


def test_the_summary_is_written_even_with_no_detections(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, detections=())

    assert _layer(path, "run_summary").featureCount() == 1


def test_boxes_are_optional(tmp_path: Path) -> None:
    from qgis.core import QgsVectorLayer

    path = _write(tmp_path)

    boxes = QgsVectorLayer(
        f"{path}|layername=detection_boxes", "boxes", "ogr"
    )
    assert boxes.isValid() is False


def test_boxes_are_written_when_requested(tmp_path: Path) -> None:
    path = _write(tmp_path, write_boxes=True)

    boxes = _layer(path, "detection_boxes")
    assert boxes.featureCount() == 3
    assert _layer(path, "tree_centers").featureCount() == 3


def test_only_boxes_can_be_written(tmp_path: Path) -> None:
    path = _write(tmp_path, write_centers=False, write_boxes=True)

    assert _layer(path, "detection_boxes").featureCount() == 3
    assert _layer(path, "run_summary").featureCount() == 1


def test_the_detection_schema_is_as_declared(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.output import DETECTION_FIELDS

    layer = _layer(_write(tmp_path), "tree_centers")

    names = [field.name() for field in layer.fields()]
    for declared, _, _ in DETECTION_FIELDS:
        assert declared in names, declared


def test_the_summary_schema_is_as_declared(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.output import SUMMARY_FIELDS

    layer = _layer(_write(tmp_path), "run_summary")

    names = [field.name() for field in layer.fields()]
    for declared, _, _ in SUMMARY_FIELDS:
        assert declared in names, declared


def test_centers_are_placed_at_the_detection_centres(
    tmp_path: Path
) -> None:
    layer = _layer(_write(tmp_path), "tree_centers")

    features = sorted(
        layer.getFeatures(), key=lambda f: f["detection_id"]
    )
    point = features[0].geometry().asPoint()

    # Pixel centre (20, 30) on a 1 unit/pixel raster 800 units tall.
    assert point.x() == pytest.approx(20.0)
    assert point.y() == pytest.approx(770.0)


def test_boxes_cover_the_predicted_rectangle(tmp_path: Path) -> None:
    layer = _layer(_write(tmp_path, write_boxes=True), "detection_boxes")

    feature = sorted(
        layer.getFeatures(), key=lambda f: f["detection_id"]
    )[0]
    box = feature.geometry().boundingBox()

    assert box.xMinimum() == pytest.approx(10.0)
    assert box.xMaximum() == pytest.approx(30.0)
    assert box.yMinimum() == pytest.approx(760.0)
    assert box.yMaximum() == pytest.approx(780.0)


def test_detection_attributes_round_trip(tmp_path: Path) -> None:
    layer = _layer(_write(tmp_path), "tree_centers")

    feature = sorted(
        layer.getFeatures(), key=lambda f: f["detection_id"]
    )[0]

    assert feature["run_id"] == "run-1"
    assert feature["class_id"] == 0
    assert feature["class_name"] == "oil_palm"
    assert feature["confidence"] == pytest.approx(0.5)
    assert feature["tile_ids"] == "r00000_c00000,r00000_c00001"


def test_the_summary_round_trips(tmp_path: Path) -> None:
    layer = _layer(_write(tmp_path), "run_summary")

    feature = next(layer.getFeatures())

    assert feature["model_filename"] == "best.onnx"
    assert feature["model_sha256"] == "a" * 64
    assert feature["backend"] == "onnxruntime"
    assert feature["total_count"] == 3
    assert json.loads(feature["counts_by_class"]) == {
        "oil_palm": 2,
        "shade_tree": 1,
    }


def test_the_output_carries_the_raster_crs(tmp_path: Path) -> None:
    layer = _layer(_write(tmp_path), "tree_centers")

    assert layer.crs().authid() == "EPSG:3857"


def test_no_staging_file_survives_a_successful_run(
    tmp_path: Path
) -> None:
    _write(tmp_path)

    assert list(tmp_path.glob("*.staging.gpkg")) == []


def test_an_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    first = _write(tmp_path)
    original = first.read_bytes()

    second = _write(tmp_path, timestamp="20260827T120000")

    assert second != first
    assert first.read_bytes() == original
    assert second.is_file()


def test_a_failed_write_leaves_no_output(tmp_path: Path) -> None:
    from tree_counter.errors import ErrorCode, TreeCounterError
    from tree_counter.qgis_adapter.output import (
        resolve_target,
        write_results,
    )

    class Broken:
        """A detection whose geometry cannot be computed."""

        box = None
        class_id = 0
        class_name = "oil_palm"
        confidence = 0.5
        tile_ids = ()

    request = _request(tmp_path)
    target = resolve_target(request)

    with pytest.raises(TreeCounterError) as error:
        write_results(
            target,
            request,
            _info(),
            (Broken(),),
            _summary(()),
            _crs(),
        )

    assert error.value.code is ErrorCode.OUTPUT_FAILURE
    assert not target.exists()
    assert list(tmp_path.glob("*.staging.gpkg")) == []


def test_validation_rejects_a_missing_layer(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.output import (
        OutputError,
        validate_geopackage,
    )

    path = _write(tmp_path)

    with pytest.raises(OutputError):
        validate_geopackage(path, ["tree_centers", "detection_boxes"])


def test_result_layers_load_once(tmp_path: Path) -> None:
    from qgis.core import QgsProject

    from tree_counter.qgis_adapter.output import load_result_layers

    path = _write(tmp_path)
    project = QgsProject()

    first = load_result_layers(path, ["tree_centers"], project)
    second = load_result_layers(path, ["tree_centers"], project)

    assert len(first) == 1
    assert second == []
    assert len(project.mapLayers()) == 1

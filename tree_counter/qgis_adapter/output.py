"""Writing counting results to a GeoPackage, atomically.

Results are written to a staging file beside the target so the replacement
stays on one filesystem, validated by reopening them, and only then moved
into place. A run that fails or is cancelled leaves no final file at all:
a partial count that looks like a finished one is worse than no output.

An existing target is never overwritten. A timestamped sibling is used
instead, because the previous count may be the one the user wanted.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tree_counter.errors import ErrorCode, TreeCounterError

CENTERS_LAYER = "tree_centers"
BOXES_LAYER = "detection_boxes"
SUMMARY_TABLE = "run_summary"
STAGING_SUFFIX = ".staging.gpkg"
OUTPUT_SUFFIX = ".gpkg"
OUTPUT_STEM_SUFFIX = "_tree_counting"

# Field name, type name, and length. Kept explicit so the schema is a
# contract rather than whatever the writer happened to infer.
DETECTION_FIELDS = (
    ("detection_id", "int", 0),
    ("run_id", "string", 64),
    ("class_id", "int", 0),
    ("class_name", "string", 128),
    ("confidence", "double", 0),
    ("tile_ids", "string", 512),
)
SUMMARY_FIELDS = (
    ("run_id", "string", 64),
    ("status", "string", 32),
    ("started_at", "string", 32),
    ("finished_at", "string", 32),
    ("duration_seconds", "double", 0),
    ("raster_name", "string", 255),
    ("raster_width", "int", 0),
    ("raster_height", "int", 0),
    ("raster_crs", "string", 64),
    ("scope_kind", "string", 32),
    ("scope_pixels", "int", 0),
    ("model_filename", "string", 255),
    ("model_sha256", "string", 64),
    ("backend", "string", 64),
    ("device", "string", 32),
    ("provider", "string", 64),
    ("confidence_threshold", "double", 0),
    ("nms_iou", "double", 0),
    ("duplicate_iou", "double", 0),
    ("tile_size", "int", 0),
    ("overlap_percent", "int", 0),
    ("selected_class_ids", "string", 255),
    ("tile_count", "int", 0),
    ("total_count", "int", 0),
    ("counts_by_class", "string", 2048),
    ("warnings", "string", 4096),
)


class OutputError(TreeCounterError):
    """The counting output could not be produced or published."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.OUTPUT_FAILURE, diagnostic_detail=detail)


@dataclass(frozen=True)
class OutputRequest:
    """Where results go and which layers the user asked for."""

    directory: Path
    raster_stem: str
    write_centers: bool = True
    write_boxes: bool = False
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not str(self.raster_stem).strip():
            raise OutputError("the raster name is required")


@dataclass
class RunSummary:
    """Everything recorded about one run, with no private model path."""

    values: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> list[Any]:
        """Return the summary in the declared field order."""

        return [self.values.get(name) for name, _, _ in SUMMARY_FIELDS]


def default_output_path(request: OutputRequest) -> Path:
    """Return the default target file for a run."""

    directory = Path(request.directory)
    stem = _safe_stem(request.raster_stem)
    return directory / f"{stem}{OUTPUT_STEM_SUFFIX}{OUTPUT_SUFFIX}"


def output_timestamp() -> str:
    """Return a compact UTC timestamp for a sibling output name."""

    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _safe_stem(stem: str) -> str:
    """Return a file stem that is safe and readable.

    Dots are replaced along with separators. Replacing separators alone
    already prevents traversal, but a name still containing ".." reads
    like one, and the suffix is appended separately anyway.
    """

    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(stem)
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "raster"


def resolve_target(request: OutputRequest) -> Path:
    """Return a target that does not already exist.

    An existing file is never replaced: the previous count may be the one
    the user still needs, so a timestamped sibling is created instead.
    """

    target = default_output_path(request)
    if not target.exists():
        return target
    if not request.timestamp:
        raise OutputError(
            "the output file already exists and no timestamp was supplied"
        )
    stamped = target.with_name(
        f"{target.stem}_{_safe_stem(request.timestamp)}{OUTPUT_SUFFIX}"
    )
    if stamped.exists():
        raise OutputError("a timestamped output file already exists")
    return stamped


def staging_path(target: Path) -> Path:
    """Return the staging file beside a target, on the same filesystem."""

    return Path(target).with_name(Path(target).name + STAGING_SUFFIX)


def build_summary(
    run_id: str,
    status: str,
    raster_info: Any,
    scope: Any,
    settings: Any,
    result: Any,
    model_filename: str,
    model_sha256: str,
    started_at: str,
    finished_at: str,
) -> RunSummary:
    """Assemble the run provenance row.

    Only the model's filename and hash are recorded. The absolute model
    path is deliberately absent: provenance travels with the output and
    must not disclose where a user keeps private model files.
    """

    counts = (
        result.counts_by_class()
        if hasattr(result, "counts_by_class")
        else {}
    )
    return RunSummary(
        {
            "run_id": str(run_id),
            "status": str(status),
            "started_at": str(started_at),
            "finished_at": str(finished_at),
            "duration_seconds": float(
                getattr(result, "duration_seconds", 0.0)
            ),
            "raster_name": str(getattr(raster_info, "name", "")),
            "raster_width": int(getattr(raster_info, "width", 0)),
            "raster_height": int(getattr(raster_info, "height", 0)),
            "raster_crs": str(getattr(raster_info, "crs_authid", "")),
            "scope_kind": str(
                getattr(getattr(scope, "kind", ""), "value", "")
            ),
            "scope_pixels": int(getattr(scope, "pixel_count", 0)),
            "model_filename": str(model_filename),
            "model_sha256": str(model_sha256),
            "backend": str(getattr(result, "backend", "")),
            "device": str(getattr(result, "device", "")),
            "provider": str(getattr(result, "provider", "")),
            "confidence_threshold": float(getattr(settings, "confidence", 0)),
            "nms_iou": float(getattr(settings, "nms_iou", 0)),
            "duplicate_iou": float(getattr(settings, "duplicate_iou", 0)),
            "tile_size": int(getattr(settings, "tile_size", 0)),
            "overlap_percent": int(getattr(settings, "overlap_percent", 0)),
            "selected_class_ids": json.dumps(
                list(getattr(settings, "selected_class_ids", ()) or [])
            ),
            "tile_count": int(getattr(result, "tile_count", 0)),
            "total_count": int(getattr(result, "total_count", 0)),
            "counts_by_class": json.dumps(counts, sort_keys=True),
            "warnings": json.dumps(
                list(getattr(result, "warnings", ()) or [])
            ),
        }
    )


def summary_contains_no_path(summary: RunSummary) -> bool:
    """Return whether the summary is free of filesystem paths."""

    for name, value in summary.values.items():
        if name in ("counts_by_class", "warnings", "selected_class_ids"):
            continue
        if isinstance(value, str) and ("/" in value or "\\" in value):
            return False
    return True


# -- QGIS-facing writing -------------------------------------------------


def _field(name: str, type_name: str, length: int) -> Any:
    from qgis.PyQt.QtCore import QMetaType

    from qgis.core import QgsField

    mapping = {
        "int": QMetaType.Type.Int,
        "double": QMetaType.Type.Double,
        "string": QMetaType.Type.QString,
    }
    field_type = mapping[type_name]
    if length:
        return QgsField(name, field_type, len=length)
    return QgsField(name, field_type)


def _fields(declared: Sequence[tuple[str, str, int]]) -> Any:
    from qgis.core import QgsFields

    fields = QgsFields()
    for name, type_name, length in declared:
        fields.append(_field(name, type_name, length))
    return fields


def _write_layer(
    path: Path,
    layer_name: str,
    geometry_type: Any,
    crs: Any,
    declared: Sequence[tuple[str, str, int]],
    rows: Sequence[tuple[Any, Sequence[Any]]],
    append: bool,
) -> None:
    from qgis.core import (
        QgsCoordinateTransformContext,
        QgsFeature,
        QgsVectorFileWriter,
    )

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = layer_name
    options.fileEncoding = "UTF-8"
    if append:
        options.actionOnExistingFile = (
            QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
        )
    fields = _fields(declared)
    writer = QgsVectorFileWriter.create(
        str(path),
        fields,
        geometry_type,
        crs,
        QgsCoordinateTransformContext(),
        options,
    )
    if writer.hasError() != QgsVectorFileWriter.WriterError.NoError:
        message = writer.errorMessage()
        del writer
        raise OutputError(f"{layer_name} could not be created: {message}")
    for geometry, attributes in rows:
        feature = QgsFeature(fields)
        if geometry is not None:
            feature.setGeometry(geometry)
        feature.setAttributes(list(attributes))
        if not writer.addFeature(feature):
            message = writer.errorMessage()
            del writer
            raise OutputError(f"{layer_name} could not be written: {message}")
    # Releasing the writer closes the GDAL handle, which must happen before
    # the file is moved or reopened.
    del writer


def write_results(
    target: Path,
    request: OutputRequest,
    raster_info: Any,
    detections: Sequence[Any],
    summary: RunSummary,
    crs: Any,
) -> Path:
    """Write a validated GeoPackage into place, atomically."""

    from qgis.core import Qgis

    from tree_counter.qgis_adapter.georeference import (
        detection_center_map,
        map_box,
        map_point,
        pixel_box_to_map,
    )

    staging = staging_path(target)
    if staging.exists():
        staging.unlink()

    expected: list[str] = []
    try:
        append = False
        if request.write_centers:
            rows = []
            for index, detection in enumerate(detections, start=1):
                x, y = detection_center_map(raster_info, detection)
                rows.append(
                    (
                        map_point(x, y),
                        _detection_attributes(index, detection, summary),
                    )
                )
            _write_layer(
                staging,
                CENTERS_LAYER,
                Qgis.WkbType.Point,
                crs,
                DETECTION_FIELDS,
                rows,
                append,
            )
            expected.append(CENTERS_LAYER)
            append = True
        if request.write_boxes:
            rows = []
            for index, detection in enumerate(detections, start=1):
                rectangle = pixel_box_to_map(raster_info, detection.box)
                rows.append(
                    (
                        map_box(rectangle),
                        _detection_attributes(index, detection, summary),
                    )
                )
            _write_layer(
                staging,
                BOXES_LAYER,
                Qgis.WkbType.Polygon,
                crs,
                DETECTION_FIELDS,
                rows,
                append,
            )
            expected.append(BOXES_LAYER)
            append = True
        # The summary is always written, even for a run that found nothing.
        _write_layer(
            staging,
            SUMMARY_TABLE,
            Qgis.WkbType.NoGeometry,
            crs,
            SUMMARY_FIELDS,
            [(None, summary.as_row())],
            append,
        )
        expected.append(SUMMARY_TABLE)

        validate_geopackage(staging, expected)
        os.replace(staging, target)
    except TreeCounterError:
        staging.unlink(missing_ok=True)
        raise
    except Exception as exc:
        staging.unlink(missing_ok=True)
        raise OutputError(
            f"the output could not be written: {type(exc).__name__}: {exc}"
        ) from exc
    return target


def _detection_attributes(
    index: int, detection: Any, summary: RunSummary
) -> list[Any]:
    return [
        index,
        summary.values.get("run_id", ""),
        int(detection.class_id),
        str(detection.class_name),
        float(detection.confidence),
        ",".join(sorted(detection.tile_ids)),
    ]


def validate_geopackage(path: Path, expected_layers: Sequence[str]) -> None:
    """Reopen a written GeoPackage and check it holds what it should."""

    from qgis.core import QgsVectorLayer

    if not Path(path).is_file():
        raise OutputError("the staged output was not created")
    for name in expected_layers:
        layer = QgsVectorLayer(f"{path}|layername={name}", name, "ogr")
        if not layer.isValid():
            raise OutputError(f"the staged output is missing {name}")
        del layer


def load_result_layers(
    path: Path, layer_names: Sequence[str], project: Any = None
) -> list[Any]:
    """Add the requested result layers to the project exactly once."""

    from qgis.core import QgsProject, QgsVectorLayer

    target_project = QgsProject.instance() if project is None else project
    loaded: list[Any] = []
    existing = {
        layer.source() for layer in target_project.mapLayers().values()
    }
    for name in layer_names:
        source = f"{path}|layername={name}"
        if source in existing:
            continue
        layer = QgsVectorLayer(source, name, "ogr")
        if not layer.isValid():
            raise OutputError(f"the result layer {name} could not be loaded")
        target_project.addMapLayer(layer)
        loaded.append(layer)
    return loaded


__all__ = [
    "BOXES_LAYER",
    "CENTERS_LAYER",
    "DETECTION_FIELDS",
    "OUTPUT_STEM_SUFFIX",
    "SUMMARY_FIELDS",
    "SUMMARY_TABLE",
    "OutputError",
    "OutputRequest",
    "RunSummary",
    "build_summary",
    "default_output_path",
    "load_result_layers",
    "output_timestamp",
    "resolve_target",
    "staging_path",
    "summary_contains_no_path",
    "validate_geopackage",
    "write_results",
]

# Changelog

All notable changes to Tree Counter will be documented here.

## [Unreleased]

- No unreleased changes.

## [0.1.0]

- Added a focused Tree Counter dock for QGIS 3.44 through QGIS 4.x, with
  Indonesian UI translation.
- Added whole-raster, current-extent, and selected-polygon processing scopes
  for georeferenced 8-bit RGB rasters.
- Added isolated Runtime Manager installation for pinned ONNX Runtime and
  optional PyTorch/Ultralytics components.
- Added strict Ultralytics YOLO11 detection-model inspection for ONNX and
  trusted `.pt` checkpoints, including per-hash trust confirmation.
- Added class selection, Confidence, NMS IoU, Duplicate IoU, tile, overlap,
  and compatible device controls.
- Added background tiled inference, cancellation, per-tile NMS, class-aware
  cross-tile deduplication, and responsive progress reporting.
- Added atomic GeoPackage output with `tree_centers`, optional
  `detection_boxes`, an always-present `run_summary`, and automatic loading of
  selected result layers into QGIS.
- Added deterministic source-only packaging, QGIS 3/4 compatibility tests,
  pinned runtime locks, and blocking publication/security gates.

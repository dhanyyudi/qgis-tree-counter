# Tree Counter

Tree Counter is an open-source QGIS plugin for counting trees in georeferenced aerial imagery with user-provided Ultralytics YOLO detection models. Palm counting is the first validated use case, while the project is designed for generic tree counting.

This project is under active development and is not yet ready for production installation. The current repository contains the public plugin foundation; inference and the focused counting workflow are planned for future releases.

## Planned workflow

The plugin is designed to accept local YOLO `.pt` and `.onnx` detection models, read supported raster layers through QGIS, run inference locally in an isolated post-install runtime, and produce georeferenced GeoPackage outputs. Planned outputs include tree centers, detection boxes, and run provenance. Models and imagery will remain on the user's machine.

The runtime will be installed and managed explicitly by the user. The plugin will not bundle model files, raster imagery, runtime binaries, or Python wheels, and it will not download models automatically. It will not send models or raster imagery to remote services, and it will not collect telemetry or analytics or upload crash reports automatically. Network access will occur only after the user explicitly starts an action in Runtime Manager.

## Compatibility

The target is one package for QGIS 3.44 LTR through QGIS 4.x (up to 4.99) on Windows, macOS, and Ubuntu LTS. CPU is the required baseline; compatible hardware acceleration is optional.

## Development status

See [CONTRIBUTING.md](CONTRIBUTING.md) for development expectations and [CHANGELOG.md](CHANGELOG.md) for release history. The project is licensed under the GNU Affero General Public License, version 3 only (AGPL-3.0-only).

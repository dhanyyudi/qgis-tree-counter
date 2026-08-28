# Third-party notices

Tree Counter is distributed under AGPL-3.0-only. The plugin ZIP contains only
the Tree Counter source, its original icon, its Indonesian translation, runtime
dependency manifests, and license/notices files. It does not bundle model
files, raster imagery, Python wheels, native libraries, or runtime binaries.

## Host integration

Tree Counter uses the QGIS Python API and Qt compatibility layer supplied by
the user's QGIS installation. QGIS and Qt are not redistributed in the plugin
ZIP. See the [QGIS licensing page](https://qgis.org/license/)
and the [Qt licensing page](https://www.qt.io/licensing/) for their terms.

The `tree_counter/icons/tree_counter.svg` artwork is original to this project
and is distributed under AGPL-3.0-only with the rest of Tree Counter.

## Optional isolated runtime

Runtime Manager can install the following pinned components into a separate
per-user virtual environment after explicit confirmation. They are downloaded
from PyPI and are not redistributed in the plugin ZIP:

| Component | Pinned version | Upstream license |
| --- | --- | --- |
| [NumPy](https://numpy.org/) | 2.5.2 | BSD-3-Clause |
| [ONNX Runtime](https://onnxruntime.ai/) | 1.29.0 | MIT |
| [PyTorch](https://pytorch.org/) | 2.13.0 | BSD-3-Clause |
| [Ultralytics](https://www.ultralytics.com/) | 8.4.120 | AGPL-3.0 |

Their transitive Python dependencies, exact filenames, versions, download
origins, and SHA-256 hashes are recorded in the platform-specific lock files
under `tree_counter/runtime/locks/`. Each installed distribution retains its
own metadata and license terms in the isolated runtime environment.

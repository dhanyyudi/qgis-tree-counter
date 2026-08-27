# Tree Counter

[Bahasa Indonesia](README.id.md)

Tree Counter is an open-source QGIS plugin that counts trees in
georeferenced aerial imagery using detection models you supply. Oil palm is
the first validated use case; nothing in the plugin is specific to it.

Everything runs on your machine. Imagery and models are never uploaded, no
telemetry is collected, and the only feature that touches the network is
the Runtime Manager, and only after you start it.

**This is an experimental release under active development.** It counts
trees and it has been tested against a real checkpoint and a real raster,
but its accuracy on your data is your call to make — see
[Limitations](#limitations).

## What you need

1. **QGIS 3.44 LTR or newer**, up to QGIS 4.x. One package covers both.
2. **A detection model of your own.** Tree Counter downloads no models.
3. **The detection runtime**, installed once from inside the plugin.

## Installing

Install the plugin from the QGIS plugin repository, or from a ZIP through
**Plugins → Manage and Install Plugins → Install from ZIP**.

Then open **Runtime Manager** in the Tree Counter panel and press
**Install**. This is the one action that uses the network: it downloads
ONNX Runtime, and optionally PyTorch and Ultralytics, from PyPI into a
per-user directory outside QGIS. Nothing is written into your QGIS
installation, and every package is verified against a pinned hash before
it is activated.

The runtime needs a **Python 3.12** interpreter on your machine. It is
never built from QGIS's own Python, so the two can never interfere with
each other. If no suitable interpreter is found, the Runtime Manager says
so and the install log lists every candidate it tried.

The runtime is available for Windows x86_64, macOS Apple Silicon, and
Linux x86_64. On Intel Macs the plugin loads but the runtime cannot be
installed, because the pinned PyTorch and ONNX Runtime releases no longer
publish macOS x86_64 wheels for Python 3.12.

## Models

Tree Counter accepts **Ultralytics YOLO11 detection models**:

| Format | Runtime component | Notes |
| --- | --- | --- |
| `.onnx` | ONNX Runtime (CPU) | Recommended. Export without built-in NMS. |
| `.pt` | PyTorch and Ultralytics | Executes code when loaded — see below. |

Segmentation, classification and pose models are refused, as are exports
that apply their own NMS: the plugin owns non-maximum suppression so the
**NMS IoU** setting always means something.

A model is identified by its architecture, not by its filename, so a
checkpoint you trained and renamed is accepted while a YOLOv8 file named
`yolo11-something.pt` is not.

### Why a `.pt` checkpoint asks for confirmation

Loading a PyTorch checkpoint **executes code contained in that file**. A
malicious `.pt` can do anything your user account can do. Tree Counter
therefore refuses to load one until you confirm its exact SHA-256 hash,
and it remembers that confirmation per file. If the file changes, you are
asked again.

Only use `.pt` files you produced yourself or obtained from a source you
trust. If you have a choice, export to `.onnx` and use that instead — an
ONNX graph is data, not code.

## Imagery

Tree Counter reads any raster QGIS can open, provided it is:

- **georeferenced**, with a CRS QGIS recognises;
- **8-bit**, with at least three bands read as RGB.

The raster is processed in tiles at its native resolution. Tile size and
overlap are yours to set; overlapping tiles are deduplicated so a tree
straddling a seam is counted once.

**Resolution matters more than anything else.** A model trained on 10 cm
imagery will find little in a 2 cm orthophoto, because the trees are the
wrong size in pixels. If you get no detections on imagery that obviously
contains trees, this is the first thing to check.

## Output

Results are written to one GeoPackage containing:

- `tree_centers` — one point per tree, with class, confidence, and the
  tiles it came from;
- `detection_boxes` — the detection rectangles, if you asked for them;
- `run_summary` — what was run: model hash, settings, backend, device,
  tile count, duration, and any warnings.

You choose which layers to write. The run refuses to start if you turn
both off, rather than producing a file with no detections in it.

## Devices

CPU is the baseline and always works. Where the hardware and the installed
component support it, CUDA, MPS (Apple Silicon) and CoreML are offered.
The device actually used is recorded in `run_summary`, so a run is never
ambiguous after the fact.

## Limitations

Read these before trusting a count.

- **Accuracy is unmeasured.** The plugin reports how many detections your
  model produced. Whether that equals the number of trees on the ground
  depends entirely on your model and your imagery. There is no ground
  truth in this project and no accuracy claim is made.
- Results depend on the resolution match between your model and your
  imagery, and on the confidence and NMS thresholds you choose.
- Large rasters take time. Progress is reported per tile and a run can be
  cancelled at any point; a cancelled run writes no output.
- Windows has not had a manual end-to-end test in this release.
- The plugin is experimental. Interfaces and outputs may change.

## Troubleshooting

**"No supported Python 3.12 interpreter was found."** Install Python 3.12
and try again. On macOS, note that QGIS launched from the Dock sees a
minimal `PATH`; Tree Counter also searches the usual install locations, so
a Homebrew or python.org installation is found either way.

**The model is rejected.** The message names the reason: not a detection
model, not YOLO11, an export with built-in NMS, or a checkpoint whose hash
you have not confirmed. Each is a real incompatibility, not a warning to
click through.

**No detections at all.** Check that the window you selected actually
contains trees, then check the resolution match described above, then
lower the confidence threshold.

**Something failed mid-run.** Open **Runtime Manager → Open logs**. Logs
stay on your machine and record what ran, never your imagery.

## Privacy and network use

- Imagery and models are read locally and never uploaded.
- No telemetry, no analytics, no automatic crash reports.
- The only network access is the Runtime Manager downloading packages from
  PyPI, and only after you start it.
- Nothing is downloaded automatically, including models.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development expectations,
[CHANGELOG.md](CHANGELOG.md) for release history,
[SECURITY.md](SECURITY.md) for how to report a vulnerability, and
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for the blocking pre-release
gates.

```bash
python3 -m pytest -q
python3 -m flake8 tree_counter scripts tests
python3 scripts/check_publication.py .
python3 scripts/package_plugin.py
```

The QGIS-dependent tests run inside a QGIS Python:

```bash
python3 scripts/run_qgis_tests.py --qgis-app /path/to/QGIS.app -- tests/qgis -q
```

## License

GNU Affero General Public License, version 3 only (AGPL-3.0-only). See
[LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

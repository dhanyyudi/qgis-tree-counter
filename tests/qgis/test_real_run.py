"""Count trees in a real aerial raster with a real model.

Everything here is real: the maintainer's YOLO11 checkpoint, their
aerial raster, the isolated ML runtime, a genuine child process, real
tiles and real deduplication. The test is opt-in and skips with an
explicit reason whenever the assets or the runtime are absent, so it
never runs in CI.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import gc
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from scripts.run_local_integration import selected_backends, selected_scope

REPO_ROOT = Path(__file__).resolve().parents[2]
RASTER_VARIABLE = "TREE_COUNTER_TEST_RASTER"
MODEL_VARIABLE = "TREE_COUNTER_TEST_MODEL_PT"
ONNX_VARIABLE = "TREE_COUNTER_TEST_MODEL_ONNX"
# Counts may differ slightly between PT and an ONNX export of the same
# weights: the graph is frozen at one input size and the arithmetic is not
# bit-identical. Observed on this raster and checkpoint: 14 both ways.
# One detection of tolerance leaves room for that without hiding a real
# divergence.
PARITY_TOLERANCE = 1
# A planted block of this raster, as (column_min, row_min, column_max,
# row_max). Much of the scene is swamp and natural scrub with no palms at
# all, so the window is chosen inside the plantation - a window with no
# trees would make a zero count look like a failure when it is correct.
WINDOW = (9600, 7000, 10880, 8280)
TILE_SIZE = 640
OVERLAP_PERCENT = 20
GRACE_MS = 5000
SELECTED_SCOPE = selected_scope()
SELECTED_BACKENDS = selected_backends()
# Qt5 segfaults when a QgsRasterLayer is collected during interpreter
# teardown, so the layer is kept alive for the whole session.
_LAYERS: list = []


@pytest.fixture(scope="module", autouse=True)
def release_real_raster_layers(qgis_application):
    """Release GDAL layers while QGIS is still alive on Qt5."""

    yield
    _LAYERS.clear()
    gc.collect()


def _runtime_python() -> Path | None:
    from tree_counter.runtime.paths import default_runtime_root

    candidate = default_runtime_root() / "active" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _require(variable: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        pytest.skip(f"set {variable} to run against real assets")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"{variable} does not point to a file")
    return path


class RuntimeTransport:
    """Run the real worker inside the isolated ML runtime."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None

    def start(self, program, arguments) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment.pop("TREE_COUNTER_WORKER_BACKEND", None)
        self._process = subprocess.Popen(
            [program, *arguments],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            cwd=str(REPO_ROOT),
        )

    def write_line(self, line: bytes) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(line)
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise OSError("the worker is no longer accepting input") from exc

    def read_line(self, timeout_ms: int):
        assert self._process is not None and self._process.stdout is not None
        return self._process.stdout.readline() or None

    def read_stderr(self) -> bytes:
        return b""

    def terminate(self, grace_ms: int) -> None:
        if self._process is None:
            return
        for stream in (
            self._process.stdin,
            self._process.stdout,
            self._process.stderr,
        ):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=grace_ms / 1000 or 1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=60)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def exit_code(self):
        return None if self.is_running() else self._process.returncode


@pytest.fixture
def make_run(qgis_application, tmp_path):
    """Return a factory building a run bound to the real raster."""

    from tree_counter.qgis_adapter.process import WorkerChannel
    from tree_counter.qgis_adapter.raster import RasterReader, validate_layer
    from tree_counter.qgis_adapter.task import CountingRun
    from tree_counter.qgis_adapter.workspace import RunWorkspace

    raster = _require(RASTER_VARIABLE)
    interpreter = _runtime_python()
    if interpreter is None:
        pytest.skip("the Tree Counter ML runtime is not installed")

    from qgis.core import QgsRasterLayer

    layer = QgsRasterLayer(str(raster), "aerial", "gdal")
    assert layer.isValid(), "the raster layer did not load"
    _LAYERS.append(layer)
    info = validate_layer(layer)
    opened: list = []

    def build(index: int = 0):
        channel = WorkerChannel(RuntimeTransport())
        channel.start(
            str(interpreter),
            [
                "-I",
                str(REPO_ROOT / "tree_counter/runtime/worker_bootstrap.py"),
            ],
        )
        parent = tmp_path / f"run{index}"
        parent.mkdir(parents=True, exist_ok=True)
        workspace = RunWorkspace.create(parent=parent)
        opened.append((channel, workspace))
        return CountingRun(channel, RasterReader(layer, info), workspace)

    try:
        yield build, info, layer.crs()
    finally:
        for channel, workspace in opened:
            workspace.close()
            channel.close()


@pytest.fixture
def real_run(make_run):
    """Yield a run bound to the real raster, model and runtime."""

    build, _info, _crs = make_run
    backend = SELECTED_BACKENDS[0]
    variable = MODEL_VARIABLE if backend == "pt" else ONNX_VARIABLE
    return build(), _require(variable), backend


def _request(
    model: Path,
    run_id: str = "run-real",
    window: tuple[int, int, int, int] = WINDOW,
):
    from tree_counter.core.types import InferenceSettings
    from tree_counter.qgis_adapter.scope import PixelScope, ScopeKind
    from tree_counter.qgis_adapter.task import RunRequest

    return RunRequest(
        scope=PixelScope(ScopeKind.WHOLE_RASTER, *window),
        settings=InferenceSettings(
            tile_size=TILE_SIZE, overlap_percent=OVERLAP_PERCENT
        ),
        model_path=str(model),
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        run_id=run_id,
    )


@pytest.mark.skipif(
    SELECTED_SCOPE != "bounded", reason="the full-raster soak was selected"
)
def test_a_real_model_counts_trees_in_a_real_raster(real_run) -> None:
    """The whole stack produces real detections on real imagery."""

    run, model, backend = real_run

    result = run.execute(_request(model))

    assert result.total_count > 0, "the model found no trees at all"
    expected_backend = "ultralytics" if backend == "pt" else "onnxruntime"
    assert result.backend == expected_backend
    assert result.tile_count == 9
    print(
        f"\nREAL RUN: {result.total_count} trees, "
        f"{result.tile_count} tiles, device={result.device}, "
        f"{result.duration_seconds:.1f}s"
    )
    counts = result.counts_by_class()
    assert set(counts) == {"oil_palm"}
    assert counts["oil_palm"] == result.total_count


@pytest.mark.skipif(
    SELECTED_SCOPE != "bounded", reason="the full-raster soak was selected"
)
def test_every_detection_falls_inside_the_requested_window(
    real_run,
) -> None:
    """A centre outside the scope would place a tree on the wrong ground."""

    run, model, _backend = real_run
    column_min, row_min, column_max, row_max = WINDOW

    result = run.execute(_request(model))

    for detection in result.detections:
        box = detection.box
        center_x = (box.x_min + box.x_max) / 2
        center_y = (box.y_min + box.y_max) / 2
        assert column_min <= center_x <= column_max
        assert row_min <= center_y <= row_max


@pytest.mark.skipif(
    SELECTED_SCOPE != "bounded", reason="the full-raster soak was selected"
)
def test_no_duplicate_survives_the_iou_threshold(real_run) -> None:
    """Overlapping tiles must not count the same palm twice."""

    from tree_counter.core.nms import overlaps_at_threshold

    run, model, _backend = real_run
    request = _request(model)

    result = run.execute(request)
    boxes = [detection.box for detection in result.detections]
    threshold = request.settings.duplicate_iou

    for index, left in enumerate(boxes):
        for right in boxes[index + 1:]:
            assert not overlaps_at_threshold(left, right, threshold)


@pytest.mark.skipif(
    SELECTED_SCOPE != "bounded", reason="the full-raster soak was selected"
)
def test_the_onnx_export_agrees_with_the_checkpoint(make_run) -> None:
    """The same weights must count the same trees through either backend.

    An ONNX export is frozen at one input size and its arithmetic is not
    bit-identical to PyTorch, so the counts are compared within a
    documented tolerance rather than for exact equality.
    """

    if set(SELECTED_BACKENDS) != {"pt", "onnx"}:
        pytest.skip("select both pt and onnx to run backend parity")
    build, _info, _crs = make_run
    checkpoint = _require(MODEL_VARIABLE)
    exported = _require(ONNX_VARIABLE)

    torch_result = build(0).execute(_request(checkpoint, "run-pt"))
    onnx_result = build(1).execute(_request(exported, "run-onnx"))

    print(
        f"\nPARITY: pt={torch_result.total_count} "
        f"({torch_result.backend}/{torch_result.device}) "
        f"onnx={onnx_result.total_count} "
        f"({onnx_result.backend}/{onnx_result.device})"
    )
    assert torch_result.backend == "ultralytics"
    assert onnx_result.backend == "onnxruntime"
    assert onnx_result.total_count > 0
    assert abs(onnx_result.total_count - torch_result.total_count) <= (
        PARITY_TOLERANCE
    )


@pytest.mark.skipif(
    SELECTED_SCOPE != "full", reason="the bounded integration was selected"
)
@pytest.mark.parametrize("backend", SELECTED_BACKENDS)
def test_full_raster_soak_writes_a_valid_geopackage(
    make_run, backend: str
) -> None:
    """Process the complete raster and publish auditable spatial output."""

    from datetime import datetime, timezone

    from qgis.core import QgsVectorLayer

    from tree_counter.qgis_adapter.output import (
        OutputRequest,
        build_summary,
        output_timestamp,
        resolve_target,
        validate_geopackage,
        write_results,
    )

    build, info, crs = make_run
    variable = MODEL_VARIABLE if backend == "pt" else ONNX_VARIABLE
    model = _require(variable)
    window = (0, 0, info.width, info.height)
    request = _request(model, f"run-full-{backend}", window)
    result = build().execute(request)

    output_value = os.environ.get("TREE_COUNTER_TEST_OUTPUT_DIR", "").strip()
    if not output_value:
        pytest.skip("set TREE_COUNTER_TEST_OUTPUT_DIR for the full soak")
    output_directory = Path(output_value)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_request = OutputRequest(
        directory=output_directory,
        raster_stem=f"tree_counter_full_{backend}",
        write_centers=True,
        write_boxes=True,
        timestamp=output_timestamp(),
    )
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = build_summary(
        run_id=result.run_id,
        status="completed",
        raster_info=info,
        scope=request.scope,
        settings=request.settings,
        result=result,
        model_filename=model.name,
        model_sha256=request.model_sha256,
        started_at=now,
        finished_at=now,
    )
    target = write_results(
        resolve_target(output_request),
        output_request,
        info,
        result.detections,
        summary,
        crs,
    )

    validate_geopackage(
        target, ["tree_centers", "detection_boxes", "run_summary"]
    )
    centers = QgsVectorLayer(
        f"{target}|layername=tree_centers", "tree_centers", "ogr"
    )
    assert centers.isValid()
    assert result.total_count > 0
    assert centers.featureCount() == result.total_count
    assert result.tile_count > 9
    print(
        f"\nFULL SOAK: backend={result.backend}, device={result.device}, "
        f"trees={result.total_count}, tiles={result.tile_count}, "
        f"duration={result.duration_seconds:.1f}s"
    )

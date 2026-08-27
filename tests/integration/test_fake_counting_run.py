"""End-to-end counting against the real worker process.

The backend is the test fake, but everything else is real: a genuine child
process, real pipes, the real protocol, real tile files on disk, and the
real deduplication. Only the model inference itself is scripted.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
TIMEOUT_SECONDS = 60


class PipeTransport:
    """A worker transport backed by an ordinary child process."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None

    def start(self, program, arguments) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(REPO_ROOT), str(FIXTURES), environment.get("PYTHONPATH", ""))
        ).rstrip(os.pathsep)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["TREE_COUNTER_WORKER_BACKEND"] = "fake"
        environment.setdefault("TREE_COUNTER_FAKE_BULK_DETECTIONS", "0")
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
            # A worker that already exited must surface as a clean failure,
            # not an unraisable exception during interpreter shutdown.
            raise OSError("the worker is no longer accepting input") from exc

    def read_line(self, timeout_ms: int):
        assert self._process is not None and self._process.stdout is not None
        line = self._process.stdout.readline()
        return line or None

    def read_stderr(self) -> bytes:
        return b""

    def terminate(self, grace_ms: int) -> None:
        if self._process is None:
            return
        for stream in (self._process.stdin, self._process.stdout,
                       self._process.stderr):
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
                self._process.wait(timeout=TIMEOUT_SECONDS)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def exit_code(self):
        return None if self.is_running() else self._process.returncode


class GradientTiles:
    """Supplies real RGB bytes for whatever window is asked for."""

    def __init__(self) -> None:
        self.windows: list[tuple[int, int, int, int]] = []

    def read_rgb(self, x: int, y: int, width: int, height: int) -> bytes:
        self.windows.append((x, y, width, height))
        return bytes(
            (x + y + index) % 256 for index in range(width * height * 3)
        )


@pytest.fixture
def channel():
    from tree_counter.qgis_adapter.process import WorkerChannel

    transport = PipeTransport()
    worker = WorkerChannel(transport)
    worker.start(sys.executable, ["-m", "tree_counter.worker"])
    yield worker
    worker.close()


def _request(tmp_path: Path, width=1024, height=512, **overrides):
    from tree_counter.core.types import InferenceSettings
    from tree_counter.qgis_adapter.scope import PixelScope, ScopeKind
    from tree_counter.qgis_adapter.task import RunRequest

    settings = {"tile_size": 256, "overlap_percent": 0}
    settings.update(overrides)
    return RunRequest(
        scope=PixelScope(ScopeKind.WHOLE_RASTER, 0, 0, width, height),
        settings=InferenceSettings(**settings),
        model_path="/models/best.onnx",
        model_sha256="c" * 64,
        run_id="run-e2e",
    )


def _run(channel, tmp_path: Path, tiles=None, **kwargs):
    from tree_counter.qgis_adapter.task import CountingRun
    from tree_counter.qgis_adapter.workspace import RunWorkspace

    workspace = RunWorkspace.create(parent=tmp_path)
    run = CountingRun(
        channel, tiles or GradientTiles(), workspace, **kwargs
    )
    return run, workspace


def test_a_real_worker_completes_a_multi_tile_run(
    channel, tmp_path: Path
) -> None:
    tiles = GradientTiles()
    run, workspace = _run(channel, tmp_path, tiles=tiles)

    result = run.execute(_request(tmp_path))

    # 1024x512 at tile 256 with no overlap is an 4x2 grid.
    assert len(tiles.windows) == 8
    assert result.tile_count == 8
    assert result.total_count > 0
    assert result.backend == "fake"
    assert result.device == "cpu"
    workspace.close()


def test_progress_is_monotonic_and_complete(
    channel, tmp_path: Path
) -> None:
    events: list[dict] = []
    run, workspace = _run(channel, tmp_path, on_event=events.append)

    run.execute(_request(tmp_path))

    completed = [
        event["completed_tiles"]
        for event in events
        if event["type"] == "progress"
    ]
    assert completed == sorted(completed)
    assert completed[-1] == 8
    workspace.close()


def test_the_workspace_is_empty_after_a_successful_run(
    channel, tmp_path: Path
) -> None:
    run, workspace = _run(channel, tmp_path)

    run.execute(_request(tmp_path))

    assert workspace.resident_tiles() == ()
    workspace.close()


def test_detections_carry_global_pixel_coordinates(
    channel, tmp_path: Path
) -> None:
    run, workspace = _run(channel, tmp_path)

    result = run.execute(_request(tmp_path))

    # The fake backend reports one box near each tile origin, so at least
    # one detection must sit beyond the first tile.
    assert any(
        detection.box.x_min >= 256.0 for detection in result.detections
    )
    assert all(
        0.0 <= detection.confidence <= 1.0
        for detection in result.detections
    )
    workspace.close()


def test_a_large_result_arrives_in_batches(channel, tmp_path: Path) -> None:
    # Prove the batched result transport survives a real pipe.
    os.environ["TREE_COUNTER_FAKE_BULK_DETECTIONS"] = "400"
    try:
        from tree_counter.qgis_adapter.process import WorkerChannel

        transport = PipeTransport()
        worker = WorkerChannel(transport)
        worker.start(sys.executable, ["-m", "tree_counter.worker"])
        run, workspace = _run(worker, tmp_path)

        result = run.execute(_request(tmp_path, width=2048, height=1024))

        assert result.total_count > 1000
        worker.close()
        workspace.close()
    finally:
        os.environ["TREE_COUNTER_FAKE_BULK_DETECTIONS"] = "0"


def test_cancelling_a_real_run_stops_the_worker(
    channel, tmp_path: Path
) -> None:
    from tree_counter.qgis_adapter.task import RunCancelled

    state = {"calls": 0}

    def cancel_soon() -> bool:
        state["calls"] += 1
        return state["calls"] > 8

    run, workspace = _run(channel, tmp_path, should_cancel=cancel_soon)

    with pytest.raises(RunCancelled):
        run.execute(_request(tmp_path, width=4096, height=4096))

    channel.cancel()
    assert workspace.resident_tiles() == ()
    workspace.close()


def test_a_worker_that_dies_fails_the_run_cleanly(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.process import (
        WorkerChannel,
        WorkerProcessError,
    )

    transport = PipeTransport()
    worker = WorkerChannel(transport)
    worker.start(sys.executable, ["-m", "tree_counter.worker"])
    run, workspace = _run(worker, tmp_path)

    # The worker is gone before the run begins, which is what a crash
    # looks like from here.
    transport.terminate(100)

    with pytest.raises(WorkerProcessError):
        run.execute(_request(tmp_path))

    assert workspace.resident_tiles() == ()
    worker.close()
    workspace.close()

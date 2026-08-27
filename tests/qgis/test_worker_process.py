"""The QProcess transport, exercised against the real worker.

The orchestration itself is covered without QGIS in tests/unit; what needs
the application is that QProcess starts the worker, moves protocol lines
both ways, and stops it on demand.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture
def worker_environment(monkeypatch):
    """Point the worker at the checkout and the fake backend."""

    # Append, never replace: a QGIS bundled interpreter needs the
    # PYTHONPATH it was launched with to find its own standard library,
    # so overwriting it stops the child starting at all.
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT), str(FIXTURES)]
    if existing:
        parts.append(existing)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(parts))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("TREE_COUNTER_WORKER_BACKEND", "fake")


@pytest.fixture
def channel(worker_environment):
    from tree_counter.qgis_adapter.process import (
        QProcessTransport,
        WorkerChannel,
    )

    transport = QProcessTransport()
    worker = WorkerChannel(transport)
    worker.start(sys.executable, ["-m", "tree_counter.worker"])
    yield worker
    worker.close()


def test_qprocess_completes_a_handshake(channel) -> None:
    channel.send(
        {"type": "hello", "protocol_version": 1, "request_id": "hello"}
    )

    reply = channel.receive()

    assert reply["type"] == "hello"
    assert reply["request_id"] == "hello"


def test_qprocess_inspects_a_model(channel) -> None:
    channel.send(
        {"type": "hello", "protocol_version": 1, "request_id": "hello"}
    )
    channel.receive()
    channel.send(
        {
            "type": "inspect_model",
            "protocol_version": 1,
            "request_id": "inspect",
            "model_path": "/models/best.onnx",
            "model_sha256": "c" * 64,
        }
    )

    info = channel.receive()

    assert info["type"] == "model_info"
    assert info["class_names"] == ["oil_palm"]
    assert info["task"] == "detect"


def test_qprocess_cancellation_stops_the_worker(channel) -> None:
    channel.send(
        {"type": "hello", "protocol_version": 1, "request_id": "hello"}
    )
    channel.receive()

    channel.cancel()

    assert channel._transport.is_running() is False


def test_qprocess_reports_a_closed_stream(channel) -> None:
    from tree_counter.qgis_adapter.process import WorkerProcessError

    channel.send(
        {"type": "hello", "protocol_version": 1, "request_id": "hello"}
    )
    channel.receive()
    channel._transport.terminate(1000)

    with pytest.raises(WorkerProcessError):
        channel.receive()


def test_a_full_run_completes_through_qprocess(
    channel, tmp_path: Path
) -> None:
    from tree_counter.core.types import InferenceSettings
    from tree_counter.qgis_adapter.scope import PixelScope, ScopeKind
    from tree_counter.qgis_adapter.task import (
        CountingRun,
        RunRequest,
    )
    from tree_counter.qgis_adapter.workspace import RunWorkspace

    class Tiles:
        def read_rgb(self, x, y, width, height):
            return bytes(width * height * 3)

    workspace = RunWorkspace.create(parent=tmp_path)
    run = CountingRun(channel, Tiles(), workspace)
    request = RunRequest(
        scope=PixelScope(ScopeKind.WHOLE_RASTER, 0, 0, 512, 512),
        settings=InferenceSettings(tile_size=256, overlap_percent=0),
        model_path="/models/best.onnx",
        model_sha256="c" * 64,
        run_id="run-qprocess",
    )

    result = run.execute(request)

    assert result.tile_count == 4
    assert result.total_count > 0
    assert workspace.resident_tiles() == ()
    workspace.close()

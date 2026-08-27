"""The QgsTask wrapper drives a CountingRun and publishes its result."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path

import pytest

PROTOCOL = 1


class ScriptedTransport:
    """A worker that replies from a script and records what it received."""

    def __init__(self, replies=None) -> None:
        from tree_counter.core.protocol import encode_message

        self.sent: list[dict] = []
        self.started: tuple[str, list[str]] | None = None
        self.terminated = False
        self.running = True
        self._encode = encode_message
        self._queue: list[bytes] = []
        self._replies = replies or {}
        self._stderr = bytearray()

    def start(self, program, arguments) -> None:
        self.started = (program, list(arguments))

    def write_line(self, line: bytes) -> None:
        from tree_counter.core.protocol import decode_message

        message = decode_message(line)
        self.sent.append(message)
        handler = self._replies.get(message["type"])
        if handler is not None:
            for reply in handler(message):
                self._queue.append(self._encode(reply))

    def read_line(self, timeout_ms: int):
        if not self._queue:
            return None
        return self._queue.pop(0)

    def read_stderr(self) -> bytes:
        chunk = bytes(self._stderr)
        self._stderr.clear()
        return chunk

    def terminate(self, grace_ms: int) -> None:
        self.terminated = True
        self.running = False

    def is_running(self) -> bool:
        return self.running

    def exit_code(self):
        return None if self.running else 0


def _detection_payload(index: int = 0) -> dict:
    return {
        "box": [float(index), 0.0, float(index) + 8.0, 8.0],
        "confidence": 0.9,
        "class_id": 0,
        "class_name": "oil_palm",
        "tile_ids": ["r00000_c00000"],
        "merged_count": 1,
    }


def _happy_replies(detections=None):
    payloads = detections if detections is not None else [_detection_payload()]

    def hello(message):
        return [
            {
                "type": "hello",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
            }
        ]

    def start_run(message):
        return [
            {
                "type": "run_started",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "run_id": message["run_id"],
                "backend": "fake",
                "device": "cpu",
            }
        ]

    def tile(message):
        return [
            {
                "type": "tile_completed",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "run_id": message["run_id"],
                "tile_id": message["tile_id"],
                "detection_count": 1,
            }
        ]

    def finish(message):
        return [
            {
                "type": "detections",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "run_id": message["run_id"],
                "batch_index": 0,
                "detections": payloads,
            },
            {
                "type": "run_completed",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "run_id": message["run_id"],
                "detection_count": len(payloads),
                "batch_count": 1,
                "duration_seconds": 1.0,
            },
        ]

    return {
        "hello": hello,
        "start_run": start_run,
        "tile": tile,
        "finish_tiles": finish,
    }


class FakeTiles:
    def read_rgb(self, x: int, y: int, width: int, height: int) -> bytes:
        return bytes(width * height * 3)


def _info():
    from tree_counter.qgis_adapter.raster import RasterInfo

    return RasterInfo(
        name="aerial",
        provider_type="gdal",
        width=1000,
        height=800,
        band_count=3,
        is_byte=True,
        crs_authid="EPSG:3857",
        crs_is_valid=True,
        x_minimum=0.0,
        y_minimum=0.0,
        x_maximum=1000.0,
        y_maximum=800.0,
    )


def _request():
    from tree_counter.core.types import InferenceSettings
    from tree_counter.qgis_adapter.scope import PixelScope, ScopeKind
    from tree_counter.qgis_adapter.task import RunRequest

    return RunRequest(
        scope=PixelScope(ScopeKind.WHOLE_RASTER, 0, 0, 512, 512),
        settings=InferenceSettings(tile_size=256, overlap_percent=0),
        model_path="/models/best.onnx",
        model_sha256="a" * 64,
        run_id="run-1",
    )


def _output_request(tmp_path: Path):
    from tree_counter.qgis_adapter.output import OutputRequest

    return OutputRequest(directory=tmp_path, raster_stem="aerial")


def _crs():
    from qgis.core import QgsCoordinateReferenceSystem

    crs = QgsCoordinateReferenceSystem("EPSG:3857")
    assert crs.isValid()
    return crs


def _task(qgis_application, tmp_path: Path, transport, **overrides):
    from tree_counter.qgis_adapter.process import WorkerChannel
    from tree_counter.qgis_adapter.task import CountingTask
    from tree_counter.qgis_adapter.workspace import RunWorkspace

    values = {
        "channel": WorkerChannel(transport),
        "command": ("python3", ["-I", "worker_bootstrap.py"]),
        "tiles": FakeTiles(),
        "workspace": RunWorkspace.create(parent=tmp_path),
        "raster_info": _info(),
        "output_request": _output_request(tmp_path),
        "crs": _crs(),
    }
    values.update(overrides)
    task = CountingTask("Count aerial", _request(), **values)
    events: list[dict] = []
    task.terminal_event.connect(events.append)
    return task, events


def test_counting_task_subclasses_qgs_task(qgis_application) -> None:
    from qgis.core import QgsTask

    from tree_counter.qgis_adapter.task import CountingTask

    assert issubclass(CountingTask, QgsTask)


def test_run_executes_a_run_and_returns_true(
    qgis_application, tmp_path: Path
) -> None:
    task, events = _task(
        qgis_application, tmp_path, ScriptedTransport(_happy_replies())
    )

    assert task.run() is True

    assert events == [] or events[0]["type"] == "completed"
    assert events[0]["type"] == "completed"
    assert Path(events[0]["output_path"]).is_file()
    assert isinstance(events[0], dict)


def test_cancel_makes_should_cancel_return_true(
    qgis_application, tmp_path: Path
) -> None:
    task, _ = _task(
        qgis_application, tmp_path, ScriptedTransport(_happy_replies())
    )

    assert task._should_cancel() is False

    task.cancel()

    assert task._should_cancel() is True


def test_a_failed_run_writes_no_output(
    qgis_application, tmp_path: Path
) -> None:
    replies = _happy_replies()
    replies["start_run"] = lambda message: [
        {
            "type": "error",
            "protocol_version": PROTOCOL,
            "request_id": message["request_id"],
            "code": "invalid_model",
            "message": "The selected model is not supported.",
        }
    ]
    task, events = _task(
        qgis_application, tmp_path, ScriptedTransport(replies)
    )

    assert task.run() is False

    assert events[0]["type"] == "failed"
    assert list(tmp_path.glob("*.gpkg")) == []


def test_a_cancelled_run_writes_no_output(
    qgis_application, tmp_path: Path
) -> None:
    transport = ScriptedTransport(_happy_replies())

    def cancel_immediately() -> bool:
        return True

    task, events = _task(
        qgis_application,
        tmp_path,
        transport,
        should_cancel=cancel_immediately,
    )

    assert task.run() is False

    assert events[0]["type"] == "cancelled"
    assert list(tmp_path.glob("*.gpkg")) == []


def test_events_reaching_the_controller_are_plain_dicts(
    qgis_application, tmp_path: Path
) -> None:
    task, events = _task(
        qgis_application, tmp_path, ScriptedTransport(_happy_replies())
    )

    task.run()

    assert events
    assert all(isinstance(event, dict) for event in events)


def test_cancelling_interrupts_a_blocking_worker_read(
    qgis_application, tmp_path
) -> None:
    """Cancel must not wait out the 120-second read timeout.

    During tile inference the run sits inside a blocking
    ``WorkerChannel.receive()``. Setting a flag alone cannot be observed
    until that read returns, so Cancel would appear stuck and then be
    reported as a worker failure rather than a cancellation.
    """

    closed: list[int] = []

    class SpyChannel:
        def close(self) -> None:
            closed.append(1)

    task, _events = _task(
        qgis_application, tmp_path, None, channel=SpyChannel()
    )

    assert task.cancel() is True
    assert closed == [1], "cancel left the blocking read running"


def test_an_interrupted_read_is_reported_as_a_cancellation(
    qgis_application, tmp_path
) -> None:
    """A read that fails because we cancelled is not a worker failure."""

    from tree_counter.errors import ErrorCode, TreeCounterError
    from tree_counter.qgis_adapter.task import CountingRun, RunCancelled

    class BrokenChannel:
        def receive(self, timeout_ms=None):
            raise TreeCounterError(ErrorCode.WORKER_PROCESS_FAILURE)

    cancelled = {"value": False}
    run = CountingRun(
        BrokenChannel(),
        None,
        None,
        should_cancel=lambda: cancelled["value"],
    )
    cancelled["value"] = True

    with pytest.raises(RunCancelled):
        run._await("run-1", "model_info", None)

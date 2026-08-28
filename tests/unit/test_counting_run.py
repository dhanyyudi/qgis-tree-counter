"""Tests for the counting run lifecycle, with a scripted worker."""

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

    # -- scripting -------------------------------------------------------

    def reply_with(self, message: dict) -> None:
        self._queue.append(self._encode(message))

    def emit_stderr(self, text: str) -> None:
        self._stderr.extend(text.encode("utf-8"))

    # -- transport -------------------------------------------------------

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


class FakeTiles:
    """Returns deterministic RGB bytes and records the windows asked for."""

    def __init__(self, fail_on: int | None = None) -> None:
        self.windows: list[tuple[int, int, int, int]] = []
        self._fail_on = fail_on

    def read_rgb(self, x: int, y: int, width: int, height: int) -> bytes:
        self.windows.append((x, y, width, height))
        if self._fail_on is not None and len(self.windows) == self._fail_on:
            raise OSError("the raster went away")
        return bytes([len(self.windows) % 256]) * (width * height * 3)


def _detection_payload(index: int = 0) -> dict:
    return {
        "box": [float(index), 0.0, float(index) + 8.0, 8.0],
        "confidence": 0.9,
        "class_id": 0,
        "class_name": "oil_palm",
        "tile_ids": ["r00000_c00000"],
        "merged_count": 1,
    }


def _happy_replies(detections=None, batches=1):
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
        chunks = [payloads[i::batches] for i in range(batches)]
        replies = [
            {
                "type": "detections",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "run_id": message["run_id"],
                "batch_index": index,
                "detections": chunk,
            }
            for index, chunk in enumerate(chunks)
        ]
        replies.append(
            {
                "type": "run_completed",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "run_id": message["run_id"],
                "detection_count": len(payloads),
                "batch_count": batches,
                "duration_seconds": 1.25,
            }
        )
        return replies

    return {
        "hello": hello,
        "start_run": start_run,
        "tile": tile,
        "finish_tiles": finish,
    }


def _scope(width=64, height=64, column_min=0, row_min=0):
    from tree_counter.qgis_adapter.scope import PixelScope, ScopeKind

    return PixelScope(
        ScopeKind.WHOLE_RASTER,
        column_min,
        row_min,
        column_min + width,
        row_min + height,
    )


def _request(scope=None, **settings_overrides):
    from tree_counter.core.types import InferenceSettings
    from tree_counter.qgis_adapter.task import RunRequest

    values = {"tile_size": 256, "overlap_percent": 0}
    values.update(settings_overrides)
    return RunRequest(
        scope=scope or _scope(),
        settings=InferenceSettings(**values),
        model_path="/models/best.onnx",
        model_sha256="a" * 64,
        run_id="run-1",
    )


def _run(tmp_path: Path, transport, tiles=None, **kwargs):
    from tree_counter.qgis_adapter.process import WorkerChannel
    from tree_counter.qgis_adapter.task import CountingRun
    from tree_counter.qgis_adapter.workspace import RunWorkspace

    channel = WorkerChannel(transport)
    channel.start("/usr/bin/python3", ["-m", "tree_counter.worker"])
    workspace = RunWorkspace.create(parent=tmp_path)
    return (
        CountingRun(
            channel, tiles or FakeTiles(), workspace, **kwargs
        ),
        workspace,
    )


def test_a_full_run_returns_detections(tmp_path: Path) -> None:
    transport = ScriptedTransport(_happy_replies())
    run, workspace = _run(tmp_path, transport)

    result = run.execute(_request())

    assert result.run_id == "run-1"
    assert result.total_count == 1
    assert result.backend == "fake"
    assert result.device == "cpu"
    assert result.duration_seconds == pytest.approx(1.25)
    workspace.close()


def test_the_worker_receives_the_expected_message_order(
    tmp_path: Path,
) -> None:
    transport = ScriptedTransport(_happy_replies())
    run, workspace = _run(tmp_path, transport)

    run.execute(_request())

    assert [m["type"] for m in transport.sent] == [
        "hello",
        "start_run",
        "tile",
        "finish_tiles",
    ]
    workspace.close()


def test_tiles_declare_their_encoding(tmp_path: Path) -> None:
    transport = ScriptedTransport(_happy_replies())
    run, workspace = _run(tmp_path, transport)

    run.execute(_request())

    tile = next(m for m in transport.sent if m["type"] == "tile")
    assert tile["tile_encoding"] == "rgb8"
    workspace.close()


def test_tile_windows_are_offset_by_the_scope(tmp_path: Path) -> None:
    transport = ScriptedTransport(_happy_replies())
    tiles = FakeTiles()
    run, workspace = _run(
        tmp_path, transport, tiles=tiles
    )

    run.execute(_request(scope=_scope(64, 64, column_min=500, row_min=300)))

    assert tiles.windows[0][:2] == (500, 300)
    workspace.close()


def test_progress_is_reported_for_every_tile(tmp_path: Path) -> None:
    events: list[dict] = []
    transport = ScriptedTransport(_happy_replies())
    run, workspace = _run(
        tmp_path, transport, on_event=events.append
    )

    run.execute(_request(scope=_scope(512, 256), tile_size=256))

    progress = [e for e in events if e["type"] == "progress"]
    assert [e["completed_tiles"] for e in progress] == [1, 2]
    assert all(e["total_tiles"] == 2 for e in progress)
    workspace.close()


def test_only_one_tile_file_exists_at_a_time(tmp_path: Path) -> None:
    resident: list[int] = []

    class Watching(ScriptedTransport):
        def write_line(self, line: bytes) -> None:
            super().write_line(line)
            resident.append(len(workspace.resident_tiles()))

    transport = Watching(_happy_replies())
    run, workspace = _run(tmp_path, transport)

    run.execute(_request(scope=_scope(1024, 512), tile_size=256))

    # Disk use must not grow with the size of the raster.
    assert max(resident) <= 1
    assert workspace.resident_tiles() == ()
    workspace.close()


def test_detections_arriving_in_several_batches_are_reassembled(
    tmp_path: Path,
) -> None:
    payloads = [_detection_payload(index) for index in range(7)]
    transport = ScriptedTransport(_happy_replies(payloads, batches=3))
    run, workspace = _run(tmp_path, transport)

    result = run.execute(_request())

    assert result.total_count == 7
    workspace.close()


def test_per_class_totals_are_reported(tmp_path: Path) -> None:
    payloads = [_detection_payload(0), _detection_payload(20)]
    payloads[1]["class_id"] = 1
    payloads[1]["class_name"] = "shade_tree"
    transport = ScriptedTransport(_happy_replies(payloads))
    run, workspace = _run(tmp_path, transport)

    result = run.execute(_request())

    assert result.counts_by_class() == {"oil_palm": 1, "shade_tree": 1}
    workspace.close()


def test_a_warning_is_collected_without_ending_the_run(
    tmp_path: Path,
) -> None:
    replies = _happy_replies()
    original = replies["start_run"]

    def with_warning(message):
        return [
            {
                "type": "warning",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "code": "cpu_fallback",
                "message": "Falling back to the CPU.",
            },
            *original(message),
        ]

    replies["start_run"] = with_warning
    transport = ScriptedTransport(replies)
    run, workspace = _run(tmp_path, transport)

    result = run.execute(_request())

    assert result.warnings == ("Falling back to the CPU.",)
    assert result.total_count == 1
    workspace.close()


class TestFailures:
    """A run that does not complete never returns a partial count."""

    def test_a_worker_error_fails_the_run(self, tmp_path: Path) -> None:
        from tree_counter.errors import ErrorCode
        from tree_counter.qgis_adapter.task import RunFailed

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
        transport = ScriptedTransport(replies)
        run, workspace = _run(tmp_path, transport)

        with pytest.raises(RunFailed) as error:
            run.execute(_request())

        assert error.value.code is ErrorCode.INVALID_MODEL
        workspace.close()

    def test_a_worker_that_stops_responding_fails_the_run(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.qgis_adapter.process import WorkerProcessError

        replies = _happy_replies()
        replies["start_run"] = lambda message: []
        transport = ScriptedTransport(replies)
        run, workspace = _run(tmp_path, transport)

        with pytest.raises(WorkerProcessError):
            run.execute(_request())

        workspace.close()

    def test_an_unexpected_message_fails_the_run(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.qgis_adapter.task import RunFailed

        replies = _happy_replies()
        replies["start_run"] = lambda message: [
            {
                "type": "model_info",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "filename": "best.onnx",
                "sha256": "a" * 64,
                "model_format": "onnx",
                "task": "detect",
                "family": "yolo11",
                "class_names": ["oil_palm"],
                "input_width": 640,
                "input_height": 640,
                "dynamic_shape": False,
                "backend": "onnxruntime",
                "device": "cpu",
                "provider": "CPUExecutionProvider",
                "warnings": [],
            }
        ]
        transport = ScriptedTransport(replies)
        run, workspace = _run(tmp_path, transport)

        with pytest.raises(RunFailed):
            run.execute(_request())

        workspace.close()

    def test_run_started_records_provider_and_warnings(
        self, tmp_path: Path
    ) -> None:
        replies = _happy_replies()

        def started(message):
            return [
                {
                    "type": "run_started",
                    "protocol_version": PROTOCOL,
                    "request_id": message["request_id"],
                    "run_id": message["run_id"],
                    "backend": "onnxruntime",
                    "device": "cpu",
                    "provider": "CPUExecutionProvider",
                    "warnings": ["CoreML was unavailable; using CPU."],
                }
            ]

        replies["start_run"] = started
        events: list[dict] = []
        transport = ScriptedTransport(replies)
        run, workspace = _run(tmp_path, transport, on_event=events.append)

        result = run.execute(_request())

        assert result.provider == "CPUExecutionProvider"
        assert result.warnings == ("CoreML was unavailable; using CPU.",)
        warnings = [event for event in events if event["type"] == "warning"]
        assert warnings == [
            {
                "type": "warning",
                "message": "CoreML was unavailable; using CPU.",
            }
        ]
        workspace.close()

    def test_a_reply_about_another_run_fails(self, tmp_path: Path) -> None:
        from tree_counter.qgis_adapter.task import RunFailed

        replies = _happy_replies()
        replies["start_run"] = lambda message: [
            {
                "type": "run_started",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "run_id": "some-other-run",
            }
        ]
        transport = ScriptedTransport(replies)
        run, workspace = _run(tmp_path, transport)

        with pytest.raises(RunFailed):
            run.execute(_request())

        workspace.close()

    def test_a_detection_count_mismatch_fails(self, tmp_path: Path) -> None:
        from tree_counter.qgis_adapter.task import RunFailed

        replies = _happy_replies()

        def lying_finish(message):
            return [
                {
                    "type": "detections",
                    "protocol_version": PROTOCOL,
                    "request_id": message["request_id"],
                    "run_id": message["run_id"],
                    "batch_index": 0,
                    "detections": [_detection_payload()],
                },
                {
                    "type": "run_completed",
                    "protocol_version": PROTOCOL,
                    "request_id": message["request_id"],
                    "run_id": message["run_id"],
                    "detection_count": 99,
                    "batch_count": 1,
                    "duration_seconds": 1.0,
                },
            ]

        replies["finish_tiles"] = lying_finish
        transport = ScriptedTransport(replies)
        run, workspace = _run(tmp_path, transport)

        with pytest.raises(RunFailed):
            run.execute(_request())

        workspace.close()

    def test_an_unusable_detection_fails(self, tmp_path: Path) -> None:
        from tree_counter.qgis_adapter.task import RunFailed

        replies = _happy_replies([{"box": [1.0, 2.0], "confidence": 0.5}])
        transport = ScriptedTransport(replies)
        run, workspace = _run(tmp_path, transport)

        with pytest.raises(RunFailed):
            run.execute(_request())

        workspace.close()

    def test_a_raster_read_failure_fails_the_run(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.errors import ErrorCode
        from tree_counter.qgis_adapter.task import RunFailed

        transport = ScriptedTransport(_happy_replies())
        run, workspace = _run(
            tmp_path, transport, tiles=FakeTiles(fail_on=1)
        )

        with pytest.raises(RunFailed) as error:
            run.execute(_request())

        assert error.value.code is ErrorCode.INVALID_RASTER
        workspace.close()

    def test_a_failed_run_leaves_no_tiles_behind(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.qgis_adapter.task import RunFailed

        replies = _happy_replies()
        replies["tile"] = lambda message: [
            {
                "type": "error",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "code": "worker_process_failure",
                "message": "The counting worker could not complete.",
            }
        ]
        transport = ScriptedTransport(replies)
        run, workspace = _run(tmp_path, transport)

        with pytest.raises(RunFailed):
            run.execute(_request())

        assert workspace.resident_tiles() == ()
        workspace.close()

    def test_an_empty_scope_cannot_start(self, tmp_path: Path) -> None:
        from tree_counter.qgis_adapter.task import RunFailed

        transport = ScriptedTransport(_happy_replies())
        run, workspace = _run(tmp_path, transport)

        # A tile size larger than the scope still yields one padded tile,
        # so the guard is about a scope that yields none at all.
        with pytest.raises((RunFailed, Exception)):
            run.execute(_request(scope=_scope(0, 0)))

        workspace.close()


class TestCancellation:
    """Cancellation is checked before every blocking step."""

    def test_cancelling_before_the_handshake_stops_immediately(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.qgis_adapter.task import RunCancelled

        transport = ScriptedTransport(_happy_replies())
        run, workspace = _run(
            tmp_path, transport, should_cancel=lambda: True
        )

        with pytest.raises(RunCancelled):
            run.execute(_request())

        assert transport.sent == []
        workspace.close()

    def test_cancelling_mid_run_stops_before_the_next_tile(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.qgis_adapter.task import RunCancelled

        state = {"calls": 0}

        def cancel_after_a_few() -> bool:
            state["calls"] += 1
            return state["calls"] > 6

        transport = ScriptedTransport(_happy_replies())
        run, workspace = _run(
            tmp_path, transport, should_cancel=cancel_after_a_few
        )

        with pytest.raises(RunCancelled):
            run.execute(_request(scope=_scope(2048, 2048), tile_size=256))

        assert "finish_tiles" not in [m["type"] for m in transport.sent]
        assert workspace.resident_tiles() == ()
        workspace.close()

    def test_a_worker_cancellation_is_reported_as_cancelled(
        self, tmp_path: Path
    ) -> None:
        from tree_counter.qgis_adapter.task import RunCancelled

        replies = _happy_replies()
        replies["start_run"] = lambda message: [
            {
                "type": "cancelled",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
            }
        ]
        transport = ScriptedTransport(replies)
        run, workspace = _run(tmp_path, transport)

        with pytest.raises(RunCancelled):
            run.execute(_request())

        workspace.close()


def test_worker_diagnostics_are_bounded(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.process import (
        MAX_STDERR_BYTES,
        WorkerChannel,
    )

    transport = ScriptedTransport(_happy_replies())
    transport.emit_stderr("x" * (MAX_STDERR_BYTES * 2))
    channel = WorkerChannel(transport)
    channel.start("/usr/bin/python3", [])
    transport.reply_with(
        {"type": "hello", "protocol_version": PROTOCOL, "request_id": "h"}
    )

    channel.receive()

    assert len(channel.diagnostics) <= MAX_STDERR_BYTES

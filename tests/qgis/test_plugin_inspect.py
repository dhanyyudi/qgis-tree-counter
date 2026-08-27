"""Model inspection surfaces the worker's own rejection reason."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

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


def _identity():
    from tree_counter.settings.trust import ModelIdentity

    return ModelIdentity("best.onnx", "a" * 64, ".onnx")


def _plugin(monkeypatch, replies):
    from tree_counter.plugin import TreeCounterPlugin
    from tree_counter.qgis_adapter import process as process_module

    transport = ScriptedTransport(replies)
    plugin = TreeCounterPlugin(None)
    plugin._model_path = "/models/best.onnx"
    monkeypatch.setattr(
        process_module, "QProcessTransport", lambda: transport
    )
    monkeypatch.setattr(
        plugin,
        "_worker_command",
        lambda: ["python3", "-I", "worker_bootstrap.py"],
    )
    return plugin, transport


def _hello(message):
    return [
        {
            "type": "hello",
            "protocol_version": PROTOCOL,
            "request_id": message["request_id"],
        }
    ]


def _error(code, message):
    def reply(message):
        return [
            {
                "type": "error",
                "protocol_version": PROTOCOL,
                "request_id": message["request_id"],
                "code": code,
                "message": message_text,
            }
        ]

    message_text = message
    return reply


def test_inspection_returns_the_model_info(monkeypatch) -> None:
    def info(message):
        return [
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
                "provider": "CPUExecutionProvider",
                "device": "cpu",
                "warnings": [],
            }
        ]

    plugin, transport = _plugin(
        monkeypatch, {"hello": _hello, "inspect_model": info}
    )

    result = plugin._inspect_model(_identity())

    assert result["class_names"] == ["oil_palm"]
    assert transport.terminated is True


def test_a_rejection_raises_the_workers_reason(monkeypatch) -> None:
    from tree_counter.errors import ErrorCode, TreeCounterError

    plugin, transport = _plugin(
        monkeypatch,
        {
            "hello": _hello,
            "inspect_model": _error(
                "invalid_model",
                "only detection models are supported",
            ),
        },
    )

    with pytest.raises(TreeCounterError) as error:
        plugin._inspect_model(_identity())

    assert error.value.code is ErrorCode.INVALID_MODEL
    assert "detection" in error.value.user_message
    assert transport.terminated is True


def test_an_unexpected_reply_is_a_protocol_error(monkeypatch) -> None:
    from tree_counter.errors import ErrorCode, TreeCounterError

    plugin, transport = _plugin(
        monkeypatch, {"hello": _hello, "inspect_model": _hello}
    )

    with pytest.raises(TreeCounterError) as error:
        plugin._inspect_model(_identity())

    assert error.value.code is ErrorCode.WORKER_PROTOCOL_FAILURE
    assert transport.terminated is True


def test_an_error_to_hello_fails_without_inspecting(monkeypatch) -> None:
    from tree_counter.errors import TreeCounterError

    plugin, transport = _plugin(
        monkeypatch,
        {"hello": _error("missing_runtime", "the runtime is missing")},
    )

    with pytest.raises(TreeCounterError) as error:
        plugin._inspect_model(_identity())

    assert error.value.user_message == "the runtime is missing"
    assert not any(
        message["type"] == "inspect_model" for message in transport.sent
    )
    assert transport.terminated is True

"""Tests for the versioned JSONL host/worker protocol."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json

import pytest


def _host_hello() -> dict[str, object]:
    return {
        "type": "hello",
        "protocol_version": 1,
        "request_id": "req-1",
    }


def _start_run() -> dict[str, object]:
    return {
        "type": "start_run",
        "protocol_version": 1,
        "request_id": "req-2",
        "run_id": "run-1",
        "workspace": "/private/run",
        "model_path": "/models/best.onnx",
        "model_sha256": "a" * 64,
        "tile_count": 2,
        "settings": {
            "confidence": 0.25,
            "nms_iou": 0.7,
            "duplicate_iou": 0.5,
            "tile_size": 640,
            "overlap_percent": 20,
            "selected_class_ids": [0],
            "requested_device": "cpu",
        },
    }


def _tile(tile_path: str = "tile_r00000_c00000.png") -> dict[str, object]:
    return {
        "type": "tile",
        "protocol_version": 1,
        "request_id": "req-3",
        "run_id": "run-1",
        "tile_id": "r00000_c00000",
        "tile_path": tile_path,
        "x_offset": 0,
        "y_offset": 0,
        "valid_width": 640,
        "valid_height": 640,
        "model_width": 640,
        "model_height": 640,
    }


HOST_MESSAGES: tuple[dict[str, object], ...] = (
    _host_hello(),
    {
        "type": "inspect_model",
        "protocol_version": 1,
        "request_id": "req-1",
        "model_path": "/models/best.onnx",
        "model_sha256": "b" * 64,
    },
    _start_run(),
    _tile(),
    {
        "type": "finish_tiles",
        "protocol_version": 1,
        "request_id": "req-4",
        "run_id": "run-1",
    },
    {"type": "cancel", "protocol_version": 1, "request_id": "req-5"},
)

WORKER_MESSAGES: tuple[dict[str, object], ...] = (
    {"type": "hello", "protocol_version": 1, "request_id": "req-1"},
    {
        "type": "model_info",
        "protocol_version": 1,
        "request_id": "req-1",
        "class_names": ["oil_palm"],
        "backend": "fake",
        "device": "cpu",
    },
    {
        "type": "run_started",
        "protocol_version": 1,
        "request_id": "req-2",
        "run_id": "run-1",
    },
    {
        "type": "tile_completed",
        "protocol_version": 1,
        "request_id": "req-3",
        "run_id": "run-1",
        "tile_id": "r00000_c00000",
        "detection_count": 1,
    },
    {
        "type": "progress",
        "protocol_version": 1,
        "request_id": "req-3",
        "run_id": "run-1",
        "completed_tiles": 1,
        "total_tiles": 2,
    },
    {
        "type": "warning",
        "protocol_version": 1,
        "request_id": "req-3",
        "code": "cpu_fallback",
        "message": "Falling back to CPU.",
    },
    {
        "type": "detections",
        "protocol_version": 1,
        "request_id": "req-4",
        "run_id": "run-1",
        "batch_index": 0,
        "detections": [],
    },
    {
        "type": "run_completed",
        "protocol_version": 1,
        "request_id": "req-4",
        "run_id": "run-1",
        "detection_count": 0,
        "batch_count": 1,
        "duration_seconds": 1.5,
    },
    {"type": "cancelled", "protocol_version": 1, "request_id": "req-5"},
    {
        "type": "error",
        "protocol_version": 1,
        "request_id": "req-5",
        "code": "invalid_model",
        "message": "The selected model is not supported.",
    },
)


@pytest.mark.parametrize("message", HOST_MESSAGES)
def test_host_messages_round_trip(message: dict[str, object]) -> None:
    from tree_counter.core.protocol import (
        decode_message,
        encode_message,
        validate_host_message,
    )

    validate_host_message(message)
    line = encode_message(message)
    assert line.endswith(b"\n")
    assert decode_message(line) == message


@pytest.mark.parametrize("message", WORKER_MESSAGES)
def test_worker_messages_round_trip(message: dict[str, object]) -> None:
    from tree_counter.core.protocol import (
        decode_message,
        encode_message,
        validate_worker_message,
    )

    validate_worker_message(message)
    assert decode_message(encode_message(message)) == message


def test_every_message_type_is_covered_by_the_tests() -> None:
    from tree_counter.core.protocol import (
        HOST_MESSAGE_TYPES,
        WORKER_MESSAGE_TYPES,
    )

    assert {item["type"] for item in HOST_MESSAGES} == set(HOST_MESSAGE_TYPES)
    assert {item["type"] for item in WORKER_MESSAGES} == set(
        WORKER_MESSAGE_TYPES
    )


def test_encoding_is_deterministic_ascii_single_line() -> None:
    from tree_counter.core.protocol import encode_message

    message = dict(_host_hello())
    message["request_id"] = "req-\u00e9"
    line = encode_message(message)

    assert line.count(b"\n") == 1
    assert line.decode("ascii")
    assert encode_message(dict(reversed(list(message.items())))) == line


def test_encode_rejects_non_finite_numbers() -> None:
    from tree_counter.core.protocol import ProtocolError, encode_message

    message = dict(_host_hello())
    message["extra"] = float("inf")
    with pytest.raises(ProtocolError):
        encode_message(message)


def test_encode_rejects_an_oversized_message() -> None:
    from tree_counter.core.protocol import (
        MAX_MESSAGE_BYTES,
        ProtocolError,
        encode_message,
    )

    message = dict(_host_hello())
    message["request_id"] = "r" * (MAX_MESSAGE_BYTES + 1)
    with pytest.raises(ProtocolError):
        encode_message(message)


def test_decode_rejects_an_oversized_line() -> None:
    from tree_counter.core.protocol import (
        MAX_MESSAGE_BYTES,
        ProtocolError,
        decode_message,
    )

    line = b'{"padding": "' + b"p" * MAX_MESSAGE_BYTES + b'"}\n'
    with pytest.raises(ProtocolError):
        decode_message(line)


@pytest.mark.parametrize(
    "line",
    [
        b"",
        b"\n",
        b"not json\n",
        b"[1, 2, 3]\n",
        b'"text"\n',
        b"null\n",
        b"{unquoted: 1}\n",
        b'{"type": "hello"',
    ],
)
def test_decode_rejects_malformed_lines(line: bytes) -> None:
    from tree_counter.core.protocol import ProtocolError, decode_message

    with pytest.raises(ProtocolError):
        decode_message(line)


def test_decode_rejects_invalid_utf8() -> None:
    from tree_counter.core.protocol import ProtocolError, decode_message

    with pytest.raises(ProtocolError):
        decode_message(b'{"type": "\xff\xfe"}\n')


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity"])
def test_decode_rejects_non_finite_json_literals(literal: bytes) -> None:
    from tree_counter.core.protocol import ProtocolError, decode_message

    line = b'{"type": "progress", "completed_tiles": ' + literal + b"}\n"
    with pytest.raises(ProtocolError):
        decode_message(line)


def test_decode_rejects_duplicate_keys() -> None:
    from tree_counter.core.protocol import ProtocolError, decode_message

    line = b'{"type": "hello", "type": "cancel"}\n'
    with pytest.raises(ProtocolError):
        decode_message(line)


def test_decode_rejects_nested_duplicate_keys() -> None:
    from tree_counter.core.protocol import ProtocolError, decode_message

    line = b'{"type": "hello", "s": {"a": 1, "a": 2}}\n'
    with pytest.raises(ProtocolError):
        decode_message(line)


def test_validation_rejects_an_unknown_protocol_version() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    message = dict(_host_hello())
    message["protocol_version"] = 2
    with pytest.raises(ProtocolError):
        validate_host_message(message)


@pytest.mark.parametrize("version", [True, 1.0, "1", None])
def test_validation_rejects_a_non_integer_protocol_version(
    version: object,
) -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    message = dict(_host_hello())
    message["protocol_version"] = version
    with pytest.raises(ProtocolError):
        validate_host_message(message)


def test_validation_rejects_an_unknown_message_type() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    message = dict(_host_hello())
    message["type"] = "shutdown"
    with pytest.raises(ProtocolError):
        validate_host_message(message)


def test_validation_rejects_a_worker_type_on_the_host_channel() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    message = dict(_host_hello())
    message["type"] = "run_completed"
    with pytest.raises(ProtocolError):
        validate_host_message(message)


def test_validation_rejects_a_host_type_on_the_worker_channel() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_worker_message,
    )

    with pytest.raises(ProtocolError):
        validate_worker_message(_start_run())


def test_validation_rejects_unknown_fields() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    message = dict(_host_hello())
    message["shell_command"] = "rm -rf /"
    with pytest.raises(ProtocolError):
        validate_host_message(message)


@pytest.mark.parametrize("field", ["type", "protocol_version", "request_id"])
def test_validation_rejects_missing_required_fields(field: str) -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    message = dict(_host_hello())
    del message[field]
    with pytest.raises(ProtocolError):
        validate_host_message(message)


@pytest.mark.parametrize(
    "field, value",
    [
        ("request_id", ""),
        ("request_id", 5),
        ("run_id", ""),
        ("tile_count", -1),
        ("tile_count", 1.5),
        ("tile_count", True),
        ("settings", []),
        ("model_sha256", "abc"),
        ("model_sha256", "Z" * 64),
        ("workspace", ""),
    ],
)
def test_validation_rejects_wrong_field_types(
    field: str, value: object
) -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    message = _start_run()
    message[field] = value
    with pytest.raises(ProtocolError):
        validate_host_message(message)


def test_validation_accepts_an_uppercase_hash_case_insensitively() -> None:
    from tree_counter.core.protocol import validate_host_message

    message = _start_run()
    message["model_sha256"] = "A" * 64
    validate_host_message(message)


@pytest.mark.parametrize(
    "tile_path",
    [
        "../escape.png",
        "nested/../../escape.png",
        "/absolute/tile.png",
        "C:\\Windows\\tile.png",
        "..\\escape.png",
        "sub\\dir\\tile.png",
        "tile\x00.png",
        "",
        ".",
        "..",
    ],
)
def test_validation_rejects_unsafe_tile_paths(tile_path: str) -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    with pytest.raises(ProtocolError):
        validate_host_message(_tile(tile_path))


def test_validation_accepts_a_nested_relative_tile_path() -> None:
    from tree_counter.core.protocol import validate_host_message

    validate_host_message(_tile("tiles/tile_r00000_c00000.png"))


def test_validation_rejects_negative_tile_geometry() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    message = _tile()
    message["x_offset"] = -1
    with pytest.raises(ProtocolError):
        validate_host_message(message)


def test_validation_rejects_valid_size_larger_than_the_model_size() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    message = _tile()
    message["valid_width"] = 641
    with pytest.raises(ProtocolError):
        validate_host_message(message)


def test_validation_rejects_non_finite_worker_numbers() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_worker_message,
    )

    message = {
        "type": "run_completed",
        "protocol_version": 1,
        "request_id": "req-4",
        "run_id": "run-1",
        "detection_count": 0,
        "batch_count": 0,
        "duration_seconds": float("nan"),
    }
    with pytest.raises(ProtocolError):
        validate_worker_message(message)


def test_validation_rejects_a_non_mapping_message() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_host_message,
    )

    with pytest.raises(ProtocolError):
        validate_host_message([("type", "hello")])


def test_protocol_version_matches_the_shared_constant() -> None:
    from tree_counter.constants import PROTOCOL_VERSION as constant
    from tree_counter.core.protocol import PROTOCOL_VERSION

    assert PROTOCOL_VERSION == constant == 1


def test_encoded_worker_error_never_carries_the_diagnostic_detail() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_worker_message,
    )

    message = {
        "type": "error",
        "protocol_version": 1,
        "request_id": "req-5",
        "code": "invalid_model",
        "message": "The selected model is not supported.",
        "traceback": "private diagnostic detail",
    }
    with pytest.raises(ProtocolError):
        validate_worker_message(message)


class TestWorkerStateMachine:
    """The worker-side accept order is strict and fails closed."""

    def test_hello_must_come_first(self) -> None:
        from tree_counter.core.protocol import (
            ProtocolError,
            WorkerStateMachine,
        )

        machine = WorkerStateMachine()
        with pytest.raises(ProtocolError):
            machine.accept("start_run")

    def test_hello_cannot_be_repeated(self) -> None:
        from tree_counter.core.protocol import (
            ProtocolError,
            WorkerStateMachine,
        )

        machine = WorkerStateMachine()
        machine.accept("hello")
        with pytest.raises(ProtocolError):
            machine.accept("hello")

    def test_inspect_then_run_is_allowed(self) -> None:
        from tree_counter.core.protocol import WorkerStateMachine

        machine = WorkerStateMachine()
        machine.accept("hello")
        machine.accept("inspect_model")
        machine.accept("start_run")
        machine.accept("tile")
        machine.accept("tile")
        machine.accept("finish_tiles")

        assert machine.is_terminal

    def test_tile_before_start_run_is_rejected(self) -> None:
        from tree_counter.core.protocol import (
            ProtocolError,
            WorkerStateMachine,
        )

        machine = WorkerStateMachine()
        machine.accept("hello")
        with pytest.raises(ProtocolError):
            machine.accept("tile")

    def test_a_second_run_is_rejected(self) -> None:
        from tree_counter.core.protocol import (
            ProtocolError,
            WorkerStateMachine,
        )

        machine = WorkerStateMachine()
        machine.accept("hello")
        machine.accept("start_run")
        with pytest.raises(ProtocolError):
            machine.accept("start_run")

    def test_messages_after_a_terminal_state_are_rejected(self) -> None:
        from tree_counter.core.protocol import (
            ProtocolError,
            WorkerStateMachine,
        )

        machine = WorkerStateMachine()
        machine.accept("hello")
        machine.accept("start_run")
        machine.accept("finish_tiles")
        with pytest.raises(ProtocolError):
            machine.accept("tile")

    def test_cancel_is_accepted_at_any_point_after_hello(self) -> None:
        from tree_counter.core.protocol import WorkerStateMachine

        machine = WorkerStateMachine()
        machine.accept("hello")
        machine.accept("start_run")
        machine.accept("tile")
        machine.accept("cancel")

        assert machine.is_terminal
        assert machine.is_cancelled

    def test_finish_tiles_without_a_run_is_rejected(self) -> None:
        from tree_counter.core.protocol import (
            ProtocolError,
            WorkerStateMachine,
        )

        machine = WorkerStateMachine()
        machine.accept("hello")
        with pytest.raises(ProtocolError):
            machine.accept("finish_tiles")

    def test_an_unknown_type_is_rejected(self) -> None:
        from tree_counter.core.protocol import (
            ProtocolError,
            WorkerStateMachine,
        )

        machine = WorkerStateMachine()
        with pytest.raises(ProtocolError):
            machine.accept("shutdown")


def test_protocol_error_uses_the_stable_worker_protocol_code() -> None:
    from tree_counter.core.protocol import ProtocolError
    from tree_counter.errors import ErrorCode, TreeCounterError

    error = ProtocolError("private detail")

    assert isinstance(error, TreeCounterError)
    assert error.code is ErrorCode.WORKER_PROTOCOL_FAILURE
    assert "private detail" not in error.user_message
    assert error.diagnostic_detail == "private detail"


def test_decode_ignores_trailing_whitespace_only() -> None:
    from tree_counter.core.protocol import decode_message

    payload = json.dumps({"type": "hello"}).encode("utf-8")
    assert decode_message(payload + b"\r\n") == {"type": "hello"}


def test_a_detection_batch_budget_fits_inside_one_line() -> None:
    from tree_counter.core.protocol import (
        MAX_BATCH_PAYLOAD_BYTES,
        MAX_MESSAGE_BYTES,
    )

    # The budget must leave room for the envelope around the batch.
    assert 0 < MAX_BATCH_PAYLOAD_BYTES < MAX_MESSAGE_BYTES


def test_run_completed_no_longer_carries_the_result_set() -> None:
    from tree_counter.core.protocol import (
        ProtocolError,
        validate_worker_message,
    )

    # A result set has no size ceiling; it arrives in detections batches.
    message = {
        "type": "run_completed",
        "protocol_version": 1,
        "request_id": "req-4",
        "run_id": "run-1",
        "detection_count": 2,
        "batch_count": 1,
        "duration_seconds": 1.5,
        "detections": [],
    }
    with pytest.raises(ProtocolError):
        validate_worker_message(message)

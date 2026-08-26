"""Versioned JSONL protocol shared by the QGIS host and the worker.

The transport is one UTF-8 JSON object per line on stdin/stdout. Every
message carries ``protocol_version`` and ``request_id``; run messages also
carry ``run_id``. Validation fails closed: unknown versions, unknown types,
unknown fields, non-finite numbers, oversized lines, duplicate JSON keys,
and unsafe tile paths are all rejected rather than interpreted.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence

from tree_counter.constants import PROTOCOL_VERSION
from tree_counter.errors import ErrorCode, TreeCounterError

MAX_MESSAGE_BYTES = 1024 * 1024
MAX_IDENTIFIER_LENGTH = 128
MAX_TEXT_LENGTH = 4096

_SHA256_PATTERN = re.compile(r"\A[0-9a-fA-F]{64}\Z")
_WINDOWS_DRIVE_PATTERN = re.compile(r"\A[A-Za-z]:")


class ProtocolError(TreeCounterError):
    """A protocol violation carrying a private diagnostic detail."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            ErrorCode.WORKER_PROTOCOL_FAILURE, diagnostic_detail=detail
        )


class _Field:
    """One declared field: its checker and whether it is required."""

    __slots__ = ("check", "required")

    def __init__(self, check: str, required: bool = True) -> None:
        self.check = check
        self.required = required


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name} must be a non-empty string")
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise ProtocolError(f"{name} is too long")


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name} must be a non-empty string")
    if len(value) > MAX_TEXT_LENGTH:
        raise ProtocolError(f"{name} is too long")


def _non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer")
    if value < 0:
        raise ProtocolError(f"{name} must not be negative")


def _positive_int(value: object, name: str) -> None:
    _non_negative_int(value, name)
    if value == 0:
        raise ProtocolError(f"{name} must be positive")


def _finite_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ProtocolError(f"{name} must be a finite number")


def _mapping(value: object, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")


def _list(value: object, name: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"{name} must be an array")


def _string_list(value: object, name: str) -> None:
    _list(value, name)
    for index, item in enumerate(value):
        _text(item, f"{name}[{index}]")


def _sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.match(value):
        raise ProtocolError(f"{name} must be a hexadecimal SHA-256 digest")


def _model_path(value: object, name: str) -> None:
    """Accept an opaque host-side path without echoing it anywhere public."""

    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ProtocolError(f"{name} must not contain a null byte")


def _tile_path(value: object, name: str) -> None:
    """Accept only a safe workspace-relative tile file path.

    The worker joins this with the run workspace supplied in ``start_run``,
    so absolute paths, drive letters, backslashes, and any ``..`` or ``.``
    component are rejected instead of normalized.
    """

    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ProtocolError(f"{name} must not contain a null byte")
    if value.startswith("/") or value.startswith("\\"):
        raise ProtocolError(f"{name} must be workspace-relative")
    if "\\" in value or _WINDOWS_DRIVE_PATTERN.match(value):
        raise ProtocolError(f"{name} must use forward slashes")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ProtocolError(f"{name} must not contain traversal components")


_CHECKS = {
    "identifier": _identifier,
    "text": _text,
    "non_negative_int": _non_negative_int,
    "positive_int": _positive_int,
    "finite": _finite_number,
    "mapping": _mapping,
    "list": _list,
    "string_list": _string_list,
    "sha256": _sha256,
    "model_path": _model_path,
    "tile_path": _tile_path,
}

_COMMON_FIELDS = {
    "type": _Field("identifier"),
    "protocol_version": _Field("non_negative_int"),
    "request_id": _Field("identifier"),
}

HOST_SCHEMAS: dict[str, dict[str, _Field]] = {
    "hello": {
        "host_name": _Field("text", required=False),
        "host_version": _Field("text", required=False),
    },
    "inspect_model": {
        "model_path": _Field("model_path"),
        "model_sha256": _Field("sha256"),
        "model_suffix": _Field("text", required=False),
    },
    "start_run": {
        "run_id": _Field("identifier"),
        "workspace": _Field("model_path"),
        "model_path": _Field("model_path"),
        "model_sha256": _Field("sha256"),
        "tile_count": _Field("non_negative_int"),
        "settings": _Field("mapping"),
    },
    "tile": {
        "run_id": _Field("identifier"),
        "tile_id": _Field("identifier"),
        "tile_path": _Field("tile_path"),
        "x_offset": _Field("non_negative_int"),
        "y_offset": _Field("non_negative_int"),
        "valid_width": _Field("positive_int"),
        "valid_height": _Field("positive_int"),
        "model_width": _Field("positive_int"),
        "model_height": _Field("positive_int"),
    },
    "finish_tiles": {"run_id": _Field("identifier")},
    "cancel": {"run_id": _Field("identifier", required=False)},
}

WORKER_SCHEMAS: dict[str, dict[str, _Field]] = {
    "hello": {
        "worker_version": _Field("text", required=False),
        "python_version": _Field("text", required=False),
    },
    "model_info": {
        "class_names": _Field("string_list"),
        "backend": _Field("identifier"),
        "device": _Field("identifier"),
        "input_size": _Field("positive_int", required=False),
    },
    "run_started": {
        "run_id": _Field("identifier"),
        "backend": _Field("identifier", required=False),
        "device": _Field("identifier", required=False),
    },
    "tile_completed": {
        "run_id": _Field("identifier"),
        "tile_id": _Field("identifier"),
        "detection_count": _Field("non_negative_int"),
    },
    "progress": {
        "run_id": _Field("identifier"),
        "completed_tiles": _Field("non_negative_int"),
        "total_tiles": _Field("non_negative_int"),
    },
    "warning": {
        "code": _Field("identifier"),
        "message": _Field("text"),
        "run_id": _Field("identifier", required=False),
    },
    "run_completed": {
        "run_id": _Field("identifier"),
        "detections": _Field("list"),
        "duration_seconds": _Field("finite"),
    },
    "cancelled": {"run_id": _Field("identifier", required=False)},
    "error": {
        "code": _Field("identifier"),
        "message": _Field("text"),
        "run_id": _Field("identifier", required=False),
    },
}

HOST_MESSAGE_TYPES = tuple(sorted(HOST_SCHEMAS))
WORKER_MESSAGE_TYPES = tuple(sorted(WORKER_SCHEMAS))


def _no_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(name: str) -> None:
    raise ProtocolError(f"non-finite JSON literal: {name}")


def encode_message(message: Mapping[str, object]) -> bytes:
    """Serialize *message* to one deterministic ASCII JSONL line."""

    _mapping(message, "message")
    try:
        payload = json.dumps(
            dict(message),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"message is not serializable: {exc}") from exc
    line = payload.encode("ascii") + b"\n"
    if len(line) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message exceeds the maximum line size")
    return line


def decode_message(line: bytes | str) -> dict[str, object]:
    """Parse one JSONL line into a mapping, failing closed on any doubt."""

    if isinstance(line, str):
        raw = line.encode("utf-8", errors="strict")
    elif isinstance(line, (bytes, bytearray)):
        raw = bytes(line)
    else:
        raise ProtocolError("line must be bytes or str")
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError("line exceeds the maximum message size")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"line is not valid UTF-8: {exc}") from exc
    text = text.strip()
    if not text:
        raise ProtocolError("line is empty")
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ProtocolError:
        raise
    except ValueError as exc:
        raise ProtocolError(f"line is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ProtocolError("message must be a JSON object")
    return decoded


def _validate(
    message: Mapping[str, object],
    schemas: Mapping[str, Mapping[str, _Field]],
    channel: str,
) -> None:
    _mapping(message, "message")
    for name, field in _COMMON_FIELDS.items():
        if name not in message:
            raise ProtocolError(f"missing required field: {name}")
        _CHECKS[field.check](message[name], name)
    version = message["protocol_version"]
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version!r}")
    message_type = message["type"]
    if message_type not in schemas:
        raise ProtocolError(
            f"unknown {channel} message type: {message_type!r}"
        )
    schema = schemas[message_type]
    allowed = set(_COMMON_FIELDS) | set(schema)
    for name in message:
        if name not in allowed:
            raise ProtocolError(
                f"unknown field for {message_type}: {name!r}"
            )
    for name, field in schema.items():
        if name not in message:
            if field.required:
                raise ProtocolError(
                    f"missing required field for {message_type}: {name}"
                )
            continue
        _CHECKS[field.check](message[name], name)
    if message_type == "tile":
        if message["valid_width"] > message["model_width"]:
            raise ProtocolError("valid_width cannot exceed model_width")
        if message["valid_height"] > message["model_height"]:
            raise ProtocolError("valid_height cannot exceed model_height")
    if message_type == "progress":
        if message["completed_tiles"] > message["total_tiles"]:
            raise ProtocolError("completed_tiles cannot exceed total_tiles")


def validate_host_message(message: Mapping[str, object]) -> None:
    """Validate a message the QGIS host sends to the worker."""

    _validate(message, HOST_SCHEMAS, "host")


def validate_worker_message(message: Mapping[str, object]) -> None:
    """Validate a message the worker sends to the QGIS host."""

    _validate(message, WORKER_SCHEMAS, "worker")


class WorkerStateMachine:
    """The strict order in which a worker accepts host messages.

    ``hello`` happens exactly once. ``inspect_model`` may repeat while idle.
    Exactly one run is allowed per process; ``finish_tiles`` and ``cancel``
    are terminal, and nothing is accepted afterwards.
    """

    _IDLE = "idle"
    _READY = "ready"
    _RUNNING = "running"
    _FINISHED = "finished"
    _CANCELLED = "cancelled"

    def __init__(self) -> None:
        self._state = self._IDLE

    @property
    def state(self) -> str:
        """Return the current state name."""

        return self._state

    @property
    def is_terminal(self) -> bool:
        """Return whether the session refuses any further host message."""

        return self._state in (self._FINISHED, self._CANCELLED)

    @property
    def is_cancelled(self) -> bool:
        """Return whether the session ended through cancellation."""

        return self._state == self._CANCELLED

    def accept(self, message_type: object) -> None:
        """Advance the session or raise :class:`ProtocolError`."""

        if message_type not in HOST_SCHEMAS:
            raise ProtocolError(
                f"unknown host message type: {message_type!r}"
            )
        if self.is_terminal:
            raise ProtocolError(
                f"{message_type} arrived after the session ended"
            )
        if message_type == "hello":
            if self._state != self._IDLE:
                raise ProtocolError("hello may only be sent once")
            self._state = self._READY
            return
        if self._state == self._IDLE:
            raise ProtocolError(f"{message_type} arrived before hello")
        if message_type == "cancel":
            self._state = self._CANCELLED
            return
        if message_type == "inspect_model":
            if self._state != self._READY:
                raise ProtocolError(
                    "inspect_model is not allowed during a run"
                )
            return
        if message_type == "start_run":
            if self._state != self._READY:
                raise ProtocolError("only one run is allowed per worker")
            self._state = self._RUNNING
            return
        if self._state != self._RUNNING:
            raise ProtocolError(f"{message_type} requires an active run")
        if message_type == "finish_tiles":
            self._state = self._FINISHED
        # ``tile`` keeps the session running.


__all__ = [
    "HOST_MESSAGE_TYPES",
    "HOST_SCHEMAS",
    "MAX_IDENTIFIER_LENGTH",
    "MAX_MESSAGE_BYTES",
    "MAX_TEXT_LENGTH",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "WORKER_MESSAGE_TYPES",
    "WORKER_SCHEMAS",
    "WorkerStateMachine",
    "decode_message",
    "encode_message",
    "validate_host_message",
    "validate_worker_message",
]

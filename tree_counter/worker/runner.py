"""The worker session loop: protocol handling and detection assembly.

The runner owns the strict host-message order, converts tile-local backend
output into global pixel detections, applies per-tile NMS and class-aware
cross-tile deduplication, and emits only validated protocol lines. Human
diagnostics never reach stdout.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from tree_counter.constants import PROTOCOL_VERSION
from tree_counter.core.dedup import deduplicate_detections
from tree_counter.core.nms import apply_nms_per_class
from tree_counter.core.protocol import (
    MAX_BATCH_PAYLOAD_BYTES,
    MAX_DETECTIONS_PER_BATCH,
    ProtocolError,
    WorkerStateMachine,
    decode_message,
    encode_message,
    validate_host_message,
    validate_worker_message,
)
from tree_counter.core.types import Detection, InferenceSettings, PixelBox
from tree_counter.errors import ErrorCode, TreeCounterError

WriteLine = Callable[[bytes], None]
Log = Callable[[str], None]
BackendFactory = Callable[[], Any]

BACKEND_ENVIRONMENT_VARIABLE = "TREE_COUNTER_WORKER_BACKEND"
# Accepted values are hard-coded so the environment can never name an
# arbitrary import target. ``fake`` exists for the process-level protocol
# tests and its module is never part of the plugin package.
FAKE_BACKEND_NAME = "fake"
FAKE_BACKEND_MODULE = "tree_counter_fake_backend"
ONNX_BACKEND_NAME = "onnx"
ULTRALYTICS_BACKEND_NAME = "ultralytics"
SUPPORTED_BACKEND_NAMES = (
    "auto",
    ONNX_BACKEND_NAME,
    ULTRALYTICS_BACKEND_NAME,
    FAKE_BACKEND_NAME,
)
SUFFIX_BACKENDS = {
    ".onnx": ONNX_BACKEND_NAME,
    ".pt": ULTRALYTICS_BACKEND_NAME,
}


def _build_backend(name: str) -> Any:
    if name == ONNX_BACKEND_NAME:
        from tree_counter.worker.backend_onnx import create_backend

        return create_backend()
    if name == ULTRALYTICS_BACKEND_NAME:
        from tree_counter.worker.backend_ultralytics import create_backend

        return create_backend()
    raise BackendUnavailableError(f"unsupported backend name: {name!r}")


class SelectingBackend:
    """Chooses the real backend from the model's format on first use.

    The worker is started before the model is known, so the concrete
    backend is resolved from the file suffix at the first call and reused
    for the rest of the session.
    """

    name = "auto"

    def __init__(self) -> None:
        self._backend: Any = None

    def _for(self, model_path: str) -> Any:
        suffix = Path(model_path).suffix.casefold()
        if suffix not in SUFFIX_BACKENDS:
            raise BackendUnavailableError(
                f"no backend handles {suffix!r} models"
            )
        chosen = SUFFIX_BACKENDS[suffix]
        if self._backend is None:
            self._backend = _build_backend(chosen)
            self.name = getattr(self._backend, "name", chosen)
        return self._backend

    def capabilities(self) -> tuple[str, ...]:
        """Return accelerators once a backend has been chosen."""

        if self._backend is None:
            return ()
        return tuple(self._backend.capabilities())

    def inspect(self, model_path: str, model_sha256: str) -> Any:
        """Describe a model using the backend its format requires."""

        return self._for(model_path).inspect(model_path, model_sha256)

    def start_run(
        self, model_path: str, model_sha256: str, settings: Any
    ) -> Mapping[str, Any]:
        """Load a model using the backend its format requires."""

        return self._for(model_path).start_run(
            model_path, model_sha256, settings
        )

    def infer_tile(
        self, tile_path: str, tile: Mapping[str, Any]
    ) -> list[Any]:
        """Delegate tile inference to the chosen backend."""

        if self._backend is None:
            raise BackendUnavailableError("no model has been loaded")
        return self._backend.infer_tile(tile_path, tile)

    def close(self) -> None:
        """Close the chosen backend, if one was ever built."""

        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()


class BackendUnavailableError(TreeCounterError):
    """No inference backend is available for the requested selection."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.MISSING_RUNTIME, diagnostic_detail=detail)


def resolve_backend_factory(
    environment: Mapping[str, str] | None = None,
) -> BackendFactory:
    """Return a factory for the backend named by the environment.

    Only hard-coded names are accepted, so the environment can never name
    an arbitrary import target. ``auto`` defers the choice until the model
    is known and then picks by file format.
    """

    env = os.environ if environment is None else environment
    name = env.get(BACKEND_ENVIRONMENT_VARIABLE, "auto").strip().casefold()
    if name not in SUPPORTED_BACKEND_NAMES:
        raise BackendUnavailableError(f"unsupported backend name: {name!r}")
    if name == "auto":
        return SelectingBackend
    if name != FAKE_BACKEND_NAME:
        return lambda: _build_backend(name)

    def _factory() -> Any:
        try:
            module = importlib.import_module(FAKE_BACKEND_MODULE)
        except ImportError as exc:
            raise BackendUnavailableError(
                f"test backend module is not importable: {exc}"
            ) from exc
        return module.create_backend()

    return _factory


def _batch_detections(
    detections: Sequence[Detection],
) -> Iterator[list[dict[str, object]]]:
    """Split detections into batches that each fit in one protocol line.

    A run over a large area can produce far more detections than a single
    JSONL message may carry, so batches are accumulated against a byte
    budget as well as a count cap. A batch always holds at least one
    detection, so no detection is ever split across messages.
    """

    batch: list[dict[str, object]] = []
    size = 0
    for detection in detections:
        payload = _detection_payload(detection)
        payload_size = len(json.dumps(payload, ensure_ascii=True)) + 1
        if batch and (
            size + payload_size > MAX_BATCH_PAYLOAD_BYTES
            or len(batch) >= MAX_DETECTIONS_PER_BATCH
        ):
            yield batch
            batch = []
            size = 0
        batch.append(payload)
        size += payload_size
    if batch:
        yield batch


def _detection_payload(detection: Detection) -> dict[str, object]:
    box = detection.box
    return {
        "box": [box.x_min, box.y_min, box.x_max, box.y_max],
        "confidence": detection.confidence,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "tile_ids": list(detection.tile_ids),
        "merged_count": detection.merged_count,
    }


class WorkerRunner:
    """One worker session over a stream of host JSONL lines."""

    def __init__(
        self,
        write_line: WriteLine,
        log: Log,
        backend_factory: BackendFactory,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._write_line = write_line
        self._log = log
        self._backend_factory = backend_factory
        self._clock = clock
        self._machine = WorkerStateMachine()
        self._backend: Any | None = None
        self._settings: InferenceSettings | None = None
        self._run_id = ""
        self._workspace: Path | None = None
        self._total_tiles = 0
        self._completed_tiles = 0
        self._started_at = 0.0
        self._detections: list[Detection] = []

    def serve(self, lines: Iterable[bytes]) -> int:
        """Consume host lines and return the process exit status."""

        try:
            for raw in lines:
                if not raw.strip():
                    continue
                self._handle_line(raw)
                if self._machine.is_terminal:
                    return 0
            return self._handle_eof()
        except TreeCounterError as error:
            self._emit_error(error)
            return 1
        except Exception as error:  # Never leak a traceback to stdout.
            self._emit_error(
                TreeCounterError(
                    ErrorCode.WORKER_PROCESS_FAILURE,
                    diagnostic_detail=f"{type(error).__name__}: {error}",
                )
            )
            return 1
        finally:
            self._close_backend()

    def _handle_eof(self) -> int:
        if self._machine.is_terminal:
            return 0
        raise ProtocolError(
            "host closed the connection before a terminal event"
        )

    def _handle_line(self, raw: bytes) -> None:
        message = decode_message(raw)
        validate_host_message(message)
        message_type = str(message["type"])
        self._machine.accept(message_type)
        handler = {
            "hello": self._on_hello,
            "inspect_model": self._on_inspect_model,
            "start_run": self._on_start_run,
            "tile": self._on_tile,
            "finish_tiles": self._on_finish_tiles,
            "cancel": self._on_cancel,
        }[message_type]
        handler(message)

    def _send(self, message: Mapping[str, object]) -> None:
        validate_worker_message(message)
        self._write_line(encode_message(message))

    def _emit_error(self, error: TreeCounterError) -> None:
        detail = error.diagnostic_detail or "no diagnostic detail"
        self._log(f"error {error.code.value}: {detail}")
        try:
            self._send(
                {
                    "type": "error",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": "terminal",
                    "code": error.code.value,
                    "message": error.user_message,
                }
            )
        except TreeCounterError as send_error:
            self._log(f"error emission failed: {send_error}")

    def _close_backend(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is None:
            return
        close = getattr(backend, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception as error:  # A failed close must not mask a result.
            self._log(f"backend close failed: {type(error).__name__}: {error}")

    def _ensure_backend(self) -> Any:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def _on_hello(self, message: Mapping[str, object]) -> None:
        import sys

        self._send(
            {
                "type": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": message["request_id"],
                "python_version": sys.version.split()[0],
            }
        )

    def _on_inspect_model(self, message: Mapping[str, object]) -> None:
        backend = self._ensure_backend()
        info = backend.inspect(
            str(message["model_path"]), str(message["model_sha256"])
        )
        payload: dict[str, object] = {
            "type": "model_info",
            "protocol_version": PROTOCOL_VERSION,
            "request_id": message["request_id"],
            "class_names": list(info["class_names"]),
            "backend": str(info["backend"]),
            "device": str(info["device"]),
        }
        if info.get("input_size") is not None:
            payload["input_size"] = int(info["input_size"])
        self._send(payload)

    def _on_start_run(self, message: Mapping[str, object]) -> None:
        settings_payload = message["settings"]
        if not isinstance(settings_payload, Mapping):
            raise ProtocolError("settings must be an object")
        self._settings = self._build_settings(settings_payload)
        self._run_id = str(message["run_id"])
        self._workspace = Path(str(message["workspace"])).resolve()
        if not self._workspace.is_dir():
            raise TreeCounterError(
                ErrorCode.WORKER_PROCESS_FAILURE,
                diagnostic_detail="run workspace does not exist",
            )
        self._total_tiles = int(message["tile_count"])
        self._completed_tiles = 0
        self._detections = []
        self._started_at = self._clock()
        backend = self._ensure_backend()
        started = backend.start_run(
            str(message["model_path"]),
            str(message["model_sha256"]),
            self._settings,
        )
        payload: dict[str, object] = {
            "type": "run_started",
            "protocol_version": PROTOCOL_VERSION,
            "request_id": message["request_id"],
            "run_id": self._run_id,
        }
        if isinstance(started, Mapping):
            for key in ("backend", "device"):
                if started.get(key) is not None:
                    payload[key] = str(started[key])
        self._send(payload)

    @staticmethod
    def _build_settings(payload: Mapping[str, object]) -> InferenceSettings:
        allowed = {
            "confidence",
            "nms_iou",
            "duplicate_iou",
            "tile_size",
            "overlap_percent",
            "selected_class_ids",
            "requested_device",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ProtocolError(f"unknown settings fields: {sorted(unknown)}")
        values = dict(payload)
        if "selected_class_ids" in values:
            class_ids = values["selected_class_ids"]
            if isinstance(class_ids, (str, bytes)) or not isinstance(
                class_ids, Iterable
            ):
                raise ProtocolError("selected_class_ids must be an array")
            values["selected_class_ids"] = tuple(class_ids)
        return InferenceSettings(**values)

    def _require_run(self, message: Mapping[str, object]) -> InferenceSettings:
        if self._settings is None or self._workspace is None:
            raise ProtocolError("no active run")
        if str(message["run_id"]) != self._run_id:
            raise ProtocolError("run_id does not match the active run")
        return self._settings

    def _resolve_tile_path(self, tile_path: str) -> Path:
        workspace = self._workspace
        if workspace is None:
            raise ProtocolError("no active run")
        candidate = (workspace / tile_path).resolve()
        if candidate != workspace and workspace not in candidate.parents:
            raise ProtocolError("tile path escapes the run workspace")
        if not candidate.is_file():
            raise TreeCounterError(
                ErrorCode.WORKER_PROCESS_FAILURE,
                diagnostic_detail="tile file is missing",
            )
        return candidate

    def _on_tile(self, message: Mapping[str, object]) -> None:
        settings = self._require_run(message)
        tile_id = str(message["tile_id"])
        path = self._resolve_tile_path(str(message["tile_path"]))
        backend = self._ensure_backend()
        raw_detections = backend.infer_tile(str(path), dict(message))
        detections = self._tile_detections(
            raw_detections, message, tile_id, settings
        )
        self._detections.extend(detections)
        self._completed_tiles += 1
        self._send(
            {
                "type": "tile_completed",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": message["request_id"],
                "run_id": self._run_id,
                "tile_id": tile_id,
                "detection_count": len(detections),
            }
        )
        self._send(
            {
                "type": "progress",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": message["request_id"],
                "run_id": self._run_id,
                "completed_tiles": self._completed_tiles,
                "total_tiles": max(self._total_tiles, self._completed_tiles),
            }
        )

    def _tile_detections(
        self,
        raw_detections: object,
        message: Mapping[str, object],
        tile_id: str,
        settings: InferenceSettings,
    ) -> tuple[Detection, ...]:
        """Convert tile-local backend output into global detections."""

        if isinstance(raw_detections, (str, bytes)) or not isinstance(
            raw_detections, Iterable
        ):
            raise ProtocolError("backend returned a non-iterable result")
        valid_width = int(message["valid_width"])
        valid_height = int(message["valid_height"])
        x_offset = int(message["x_offset"])
        y_offset = int(message["y_offset"])
        local: list[Detection] = []
        for item in raw_detections:
            if not isinstance(item, Mapping):
                raise ProtocolError("backend detection must be an object")
            box_values = item["box"]
            if isinstance(box_values, (str, bytes)) or not isinstance(
                box_values, Iterable
            ):
                raise ProtocolError("backend detection box must be an array")
            edges = tuple(box_values)
            if len(edges) != 4:
                raise ProtocolError("backend detection box needs four edges")
            box = PixelBox(*(float(edge) for edge in edges))
            center_x = (box.x_min + box.x_max) / 2
            center_y = (box.y_min + box.y_max) / 2
            # Padded edge tiles produce detections outside the real pixels.
            if not 0.0 <= center_x <= float(valid_width):
                continue
            if not 0.0 <= center_y <= float(valid_height):
                continue
            local.append(
                Detection(
                    box=box,
                    confidence=float(item["confidence"]),
                    class_id=int(item["class_id"]),
                    class_name=str(item["class_name"]),
                    tile_ids=(tile_id,),
                )
            )
        suppressed = apply_nms_per_class(
            local,
            settings.nms_iou,
            settings.selected_class_ids,
            confidence_threshold=settings.confidence,
        )
        return tuple(
            Detection(
                box=PixelBox(
                    detection.box.x_min + x_offset,
                    detection.box.y_min + y_offset,
                    detection.box.x_max + x_offset,
                    detection.box.y_max + y_offset,
                ),
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name,
                tile_ids=detection.tile_ids,
            )
            for detection in suppressed
        )

    def _on_finish_tiles(self, message: Mapping[str, object]) -> None:
        settings = self._require_run(message)
        if self._completed_tiles != self._total_tiles:
            self._send(
                {
                    "type": "warning",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": message["request_id"],
                    "run_id": self._run_id,
                    "code": "tile_count_mismatch",
                    "message": (
                        "The number of processed tiles did not match the "
                        "announced tile count."
                    ),
                }
            )
        final = deduplicate_detections(
            self._detections, settings.duplicate_iou
        )
        batch_count = 0
        for batch in _batch_detections(final):
            self._send(
                {
                    "type": "detections",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": message["request_id"],
                    "run_id": self._run_id,
                    "batch_index": batch_count,
                    "detections": batch,
                }
            )
            batch_count += 1
        self._send(
            {
                "type": "run_completed",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": message["request_id"],
                "run_id": self._run_id,
                "detection_count": len(final),
                "batch_count": batch_count,
                "duration_seconds": max(
                    0.0, self._clock() - self._started_at
                ),
            }
        )

    def _on_cancel(self, message: Mapping[str, object]) -> None:
        self._log(f"cancelled after {self._completed_tiles} tiles")
        payload: dict[str, object] = {
            "type": "cancelled",
            "protocol_version": PROTOCOL_VERSION,
            "request_id": message["request_id"],
        }
        if self._run_id:
            payload["run_id"] = self._run_id
        self._send(payload)


__all__ = [
    "BACKEND_ENVIRONMENT_VARIABLE",
    "BackendUnavailableError",
    "SelectingBackend",
    "SUPPORTED_BACKEND_NAMES",
    "WorkerRunner",
    "resolve_backend_factory",
]

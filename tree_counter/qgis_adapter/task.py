"""Driving one counting run from validated inputs to final detections.

``CountingRun`` owns the whole exchange and is deliberately free of QGIS:
it takes a channel, a tile source, and a workspace, so the entire lifecycle
- including cancellation, worker death, and malformed replies - is testable
without an application. ``CountingTask`` is the thin ``QgsTask`` wrapper
that runs it on a background thread.

Two rules shape the design. Cancellation is checked before every blocking
step, so a run stops promptly rather than at the end of a long tile. And a
run that does not reach ``run_completed`` is a failure: partial detections
are discarded rather than presented as a smaller count.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tree_counter.constants import PROTOCOL_VERSION
from tree_counter.core.tiling import iter_tile_windows
from tree_counter.core.types import Detection, InferenceSettings, PixelBox
from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.qgis_adapter.process import WorkerChannel
from tree_counter.qgis_adapter.scope import PixelScope
from tree_counter.qgis_adapter.workspace import RunWorkspace

# ``CountingTask`` needs QGIS, but ``CountingRun`` and friends must still be
# importable in ordinary Python for the QGIS-free tests.
try:
    from qgis.PyQt.QtCore import pyqtSignal
    from qgis.core import QgsTask
except ImportError:  # QGIS is absent; the QGIS-free core still loads.
    pyqtSignal = None  # type: ignore[assignment]
    QgsTask = object  # type: ignore[assignment,misc]

TILE_ENCODING = "rgb8"
ProgressCallback = Callable[[Mapping[str, Any]], None]
CancelCheck = Callable[[], bool]


class RunCancelled(TreeCounterError):
    """The user cancelled the counting run."""

    def __init__(self, detail: str = "cancelled by the user") -> None:
        super().__init__(ErrorCode.CANCELLATION, diagnostic_detail=detail)


class RunFailed(TreeCounterError):
    """The counting run could not be completed."""

    def __init__(self, code: ErrorCode, detail: str) -> None:
        super().__init__(code, diagnostic_detail=detail)


@runtime_checkable
class TileSource(Protocol):
    """Supplies raw RGB bytes for a pixel window."""

    def read_rgb(self, x: int, y: int, width: int, height: int) -> bytes:
        """Return tightly packed RGB8 bytes for a window."""


@dataclass(frozen=True)
class RunRequest:
    """Everything one counting run needs, already validated."""

    scope: PixelScope
    settings: InferenceSettings
    model_path: str
    model_sha256: str
    run_id: str = ""

    def identifier(self) -> str:
        """Return the run identifier, generating one when absent."""

        return self.run_id or f"run-{uuid.uuid4().hex[:12]}"


@dataclass
class RunResult:
    """The outcome of a completed counting run."""

    run_id: str
    detections: tuple[Detection, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())
    backend: str = ""
    device: str = ""
    provider: str = ""
    duration_seconds: float = 0.0
    tile_count: int = 0

    @property
    def total_count(self) -> int:
        """Return how many trees were counted."""

        return len(self.detections)

    def counts_by_class(self) -> dict[str, int]:
        """Return the per-class totals, ordered by class name."""

        totals: dict[str, int] = {}
        for detection in self.detections:
            totals[detection.class_name] = (
                totals.get(detection.class_name, 0) + 1
            )
        return dict(sorted(totals.items()))


def _never_cancel() -> bool:
    """Report that no cancellation was requested."""

    return False


def _ignore(event: Mapping[str, Any]) -> None:
    """Discard an event when the caller supplied no handler."""


class CountingRun:
    """One counting run over a worker channel."""

    def __init__(
        self,
        channel: WorkerChannel,
        tiles: TileSource,
        workspace: RunWorkspace,
        on_event: ProgressCallback = _ignore,
        should_cancel: CancelCheck = _never_cancel,
    ) -> None:
        self._channel = channel
        self._tiles = tiles
        self._workspace = workspace
        self._on_event = on_event
        self._should_cancel = should_cancel
        self._batch_total = 0

    def execute(self, request: RunRequest) -> RunResult:
        """Run the whole exchange and return the final detections."""

        run_id = request.identifier()
        windows = iter_tile_windows(
            request.scope.width,
            request.scope.height,
            request.settings.tile_size,
            request.settings.overlap_percent,
            origin_x=request.scope.column_min,
            origin_y=request.scope.row_min,
        )
        if not windows:
            raise RunFailed(
                ErrorCode.INVALID_SCOPE, "the scope produced no tiles"
            )

        result = RunResult(run_id=run_id, tile_count=len(windows))
        self._guard()
        self._handshake()
        self._start_run(request, run_id, len(windows), result)

        for index, window in enumerate(windows, start=1):
            self._guard()
            self._process_tile(window, run_id, index, len(windows), result)

        self._guard()
        self._finish(run_id, request, result)
        return result

    # -- steps -----------------------------------------------------------

    def _handshake(self) -> None:
        self._channel.send(
            {
                "type": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": "hello",
            }
        )
        reply = self._channel.receive()
        self._require(reply, "hello")

    def _start_run(
        self,
        request: RunRequest,
        run_id: str,
        tile_count: int,
        result: RunResult,
    ) -> None:
        settings = request.settings
        self._channel.send(
            {
                "type": "start_run",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": "start",
                "run_id": run_id,
                "workspace": str(self._workspace.tiles),
                "model_path": request.model_path,
                "model_sha256": request.model_sha256,
                "tile_count": tile_count,
                "settings": {
                    "confidence": settings.confidence,
                    "nms_iou": settings.nms_iou,
                    "duplicate_iou": settings.duplicate_iou,
                    "tile_size": settings.tile_size,
                    "overlap_percent": settings.overlap_percent,
                    "selected_class_ids": list(settings.selected_class_ids),
                    "requested_device": settings.requested_device,
                },
            }
        )
        started = self._await(run_id, "run_started", result)
        result.backend = str(started.get("backend", ""))
        result.device = str(started.get("device", ""))

    def _process_tile(
        self,
        window: Any,
        run_id: str,
        index: int,
        total: int,
        result: RunResult,
    ) -> None:
        name = f"{window.tile_id}.raw"
        try:
            data = self._tiles.read_rgb(
                window.x_offset,
                window.y_offset,
                window.read_width,
                window.read_height,
            )
        except TreeCounterError:
            raise
        except Exception as exc:
            raise RunFailed(
                ErrorCode.INVALID_RASTER,
                f"a tile could not be read: {type(exc).__name__}",
            ) from exc
        self._workspace.write_tile(name, data)
        try:
            self._guard()
            self._channel.send(
                {
                    "type": "tile",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": f"tile-{index}",
                    "run_id": run_id,
                    "tile_id": window.tile_id,
                    "tile_path": name,
                    "tile_encoding": TILE_ENCODING,
                    "x_offset": window.x_offset,
                    "y_offset": window.y_offset,
                    "valid_width": window.read_width,
                    "valid_height": window.read_height,
                    "model_width": window.model_width,
                    "model_height": window.model_height,
                }
            )
            self._await(run_id, "tile_completed", result)
        finally:
            # The worker has read the tile, so disk use stays bounded even
            # if the exchange failed.
            self._workspace.discard_tile(name)
        self._on_event(
            {
                "type": "progress",
                "run_id": run_id,
                "completed_tiles": index,
                "total_tiles": total,
            }
        )

    def _finish(
        self, run_id: str, request: RunRequest, result: RunResult
    ) -> None:
        self._channel.send(
            {
                "type": "finish_tiles",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": "finish",
                "run_id": run_id,
            }
        )
        batches: list[dict[str, Any]] = []
        while True:
            message = self._await(
                run_id, ("detections", "run_completed"), result
            )
            if message["type"] == "detections":
                batches.extend(message["detections"])
                continue
            expected = int(message.get("batch_count", 0))
            if expected and expected != self._batch_total:
                raise RunFailed(
                    ErrorCode.WORKER_PROTOCOL_FAILURE,
                    "the worker reported a different number of batches",
                )
            declared = int(message.get("detection_count", len(batches)))
            if declared != len(batches):
                raise RunFailed(
                    ErrorCode.WORKER_PROTOCOL_FAILURE,
                    "the worker reported a different number of detections",
                )
            result.detections = tuple(
                self._detection(payload, request) for payload in batches
            )
            result.duration_seconds = float(
                message.get("duration_seconds", 0.0)
            )
            return

    # -- helpers ---------------------------------------------------------

    def _detection(
        self, payload: Mapping[str, Any], request: RunRequest
    ) -> Detection:
        try:
            box = payload["box"]
            detection = Detection(
                box=PixelBox(
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ),
                confidence=float(payload["confidence"]),
                class_id=int(payload["class_id"]),
                class_name=str(payload["class_name"]),
                tile_ids=tuple(str(t) for t in payload.get("tile_ids", ())),
                merged_count=int(payload.get("merged_count", 1)),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RunFailed(
                ErrorCode.WORKER_PROTOCOL_FAILURE,
                f"the worker sent an unusable detection: {exc}",
            ) from exc
        return detection

    def _await(
        self, run_id: str, expected: str | Sequence[str], result: RunResult
    ) -> dict[str, Any]:
        """Return the next expected message, handling the ones in between.

        Warnings and progress are informational and never terminate a run;
        an error or a cancellation from the worker always does.
        """

        wanted = (expected,) if isinstance(expected, str) else tuple(expected)
        while True:
            self._guard()
            message = self._channel.receive()
            kind = str(message["type"])
            if kind == "warning":
                text = str(message.get("message", ""))
                result.warnings = result.warnings + (text,)
                self._on_event({"type": "warning", "message": text})
                continue
            if kind == "progress":
                self._on_event(dict(message))
                continue
            if kind == "error":
                raise RunFailed(
                    self._error_code(str(message.get("code", ""))),
                    str(
                        message.get(
                            "message", "the worker reported an error"
                        )
                    ),
                )
            if kind == "cancelled":
                raise RunCancelled("the worker acknowledged cancellation")
            if kind in wanted:
                if kind == "detections":
                    self._batch_total += 1
                self._require(message, kind, run_id)
                return message
            raise RunFailed(
                ErrorCode.WORKER_PROTOCOL_FAILURE,
                f"expected {wanted}, the worker sent {kind!r}",
            )

    @staticmethod
    def _error_code(code: str) -> ErrorCode:
        try:
            return ErrorCode(code)
        except ValueError:
            return ErrorCode.WORKER_PROTOCOL_FAILURE

    @staticmethod
    def _require(
        message: Mapping[str, Any], kind: str, run_id: str | None = None
    ) -> None:
        if str(message["type"]) != kind:
            raise RunFailed(
                ErrorCode.WORKER_PROTOCOL_FAILURE,
                f"expected {kind}, the worker sent {message['type']!r}",
            )
        if run_id is not None and "run_id" in message:
            if str(message["run_id"]) != run_id:
                raise RunFailed(
                    ErrorCode.WORKER_PROTOCOL_FAILURE,
                    "the worker replied about a different run",
                )

    def _guard(self) -> None:
        if self._should_cancel():
            raise RunCancelled()


class CountingTask(QgsTask):
    """A ``QgsTask`` that runs one counting run and publishes its output.

    ``run`` executes a :class:`CountingRun` on the task thread and returns
    ``True`` only when the run completed and the GeoPackage was published.
    A cancelled or failed run emits a terminal event and writes nothing, so
    a partial count is never mistaken for a finished one. Events are plain
    dictionaries: no QGIS layer object crosses into the controller before
    publication has succeeded.
    """

    if pyqtSignal is not None:
        progress_event = pyqtSignal(dict)
        warning_event = pyqtSignal(dict)
        terminal_event = pyqtSignal(dict)

    def __init__(
        self,
        description: str,
        request: RunRequest,
        channel: WorkerChannel,
        command: tuple[str, Sequence[str]],
        tiles: TileSource,
        workspace: RunWorkspace,
        raster_info: Any,
        output_request: Any,
        crs: Any,
        should_cancel: CancelCheck | None = None,
    ) -> None:
        if pyqtSignal is None:
            raise RuntimeError("CountingTask requires QGIS")
        super().__init__(description, QgsTask.Flag.CanCancel)
        self._request = request
        self._channel = channel
        self._program, self._arguments = command
        self._tiles = tiles
        self._workspace = workspace
        self._raster_info = raster_info
        self._output_request = output_request
        self._crs = crs
        self._cancelled = False
        self._should_cancel = (
            should_cancel
            if should_cancel is not None
            else (lambda: self._cancelled)
        )
        self._started_at = ""

    def cancel(self) -> bool:
        """Ask the run to stop at its next cancellation check."""

        self._cancelled = True
        return True

    def run(self) -> bool:
        """Execute the run, publish its output, and report the outcome."""

        self._started_at = self._now()
        keep_log = False
        try:
            self._channel.start(self._program, list(self._arguments))
            run = CountingRun(
                self._channel,
                self._tiles,
                self._workspace,
                on_event=self._on_event,
                should_cancel=self._should_cancel,
            )
            result = run.execute(self._request)
            published = self._publish(result)
        except RunCancelled:
            self.terminal_event.emit({"type": "cancelled"})
            return False
        except TreeCounterError as error:
            keep_log = True
            self.terminal_event.emit({"type": "failed", "error": error})
            return False
        except Exception as error:  # A worker bug must not crash QGIS.
            keep_log = True
            self.terminal_event.emit(
                {
                    "type": "failed",
                    "error": RunFailed(
                        ErrorCode.WORKER_PROCESS_FAILURE,
                        f"{type(error).__name__}: {error}",
                    ),
                }
            )
            return False
        finally:
            self._channel.close()
            self._workspace.close(keep_log=keep_log)
        self.terminal_event.emit(
            {
                "type": "completed",
                "result": result,
                "output_path": str(published),
            }
        )
        return True

    # -- helpers ---------------------------------------------------------

    def _on_event(self, event: Mapping[str, Any]) -> None:
        kind = str(event.get("type", ""))
        if kind == "progress":
            self.progress_event.emit(dict(event))
        elif kind == "warning":
            self.warning_event.emit(dict(event))

    def _publish(self, result: RunResult) -> Path:
        from tree_counter.qgis_adapter.output import (
            build_summary,
            resolve_target,
            write_results,
        )

        target = resolve_target(self._output_request)
        summary = build_summary(
            run_id=result.run_id,
            status="completed",
            raster_info=self._raster_info,
            scope=self._request.scope,
            settings=self._request.settings,
            result=result,
            model_filename=Path(self._request.model_path).name,
            model_sha256=self._request.model_sha256,
            started_at=self._started_at,
            finished_at=self._now(),
        )
        return write_results(
            target,
            self._output_request,
            self._raster_info,
            result.detections,
            summary,
            self._crs,
        )

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "TILE_ENCODING",
    "CountingRun",
    "CountingTask",
    "RunCancelled",
    "RunFailed",
    "RunRequest",
    "RunResult",
    "TileSource",
]

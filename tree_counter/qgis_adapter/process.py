"""The channel to one isolated worker process.

The transport is injectable. In QGIS the worker is launched with
``QProcess`` and a fixed argument vector; tests use a plain pipe-backed
process or a scripted fake. Either way this module only moves validated
protocol lines and never interprets them.

Every read is bounded: a line longer than the protocol maximum, a worker
that says nothing for too long, or a flood of diagnostics on stderr all end
the exchange rather than being absorbed indefinitely.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from tree_counter.core.protocol import (
    MAX_MESSAGE_BYTES,
    ProtocolError,
    decode_message,
    encode_message,
    validate_host_message,
    validate_worker_message,
)
from tree_counter.errors import ErrorCode, TreeCounterError

DEFAULT_READ_TIMEOUT_MS = 120_000
DEFAULT_START_TIMEOUT_MS = 60_000
CANCEL_GRACE_MS = 5_000
MAX_STDERR_BYTES = 256 * 1024


class WorkerProcessError(TreeCounterError):
    """The worker process could not be started, read, or stopped."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            ErrorCode.WORKER_PROCESS_FAILURE, diagnostic_detail=detail
        )


@runtime_checkable
class Transport(Protocol):
    """The minimum a worker transport must provide."""

    def start(self, program: str, arguments: Sequence[str]) -> None:
        """Start the process."""

    def write_line(self, line: bytes) -> None:
        """Write one complete protocol line."""

    def read_line(self, timeout_ms: int) -> bytes | None:
        """Return one line, or ``None`` when the worker closed its output."""

    def read_stderr(self) -> bytes:
        """Return whatever diagnostics are available without blocking."""

    def terminate(self, grace_ms: int) -> None:
        """Ask the process to stop, then kill it if it does not."""

    def is_running(self) -> bool:
        """Return whether the process is still alive."""

    def exit_code(self) -> int | None:
        """Return the exit status, or ``None`` while still running."""


class WorkerChannel:
    """A validated protocol conversation with one worker process."""

    def __init__(
        self,
        transport: Transport,
        read_timeout_ms: int = DEFAULT_READ_TIMEOUT_MS,
    ) -> None:
        self._transport = transport
        self._read_timeout_ms = read_timeout_ms
        self._stderr = bytearray()
        self._started = False

    @property
    def diagnostics(self) -> str:
        """Return the bounded worker diagnostics collected so far."""

        return self._stderr.decode("utf-8", errors="replace")

    def start(self, program: str, arguments: Sequence[str]) -> None:
        """Start the worker with a fixed argument vector."""

        if self._started:
            raise WorkerProcessError("the worker is already running")
        if not program:
            raise WorkerProcessError("no worker program was given")
        try:
            self._transport.start(str(program), [str(a) for a in arguments])
        except TreeCounterError:
            raise
        except Exception as exc:
            raise WorkerProcessError(
                f"the worker could not be started: {type(exc).__name__}"
            ) from exc
        self._started = True

    def send(self, message: Mapping[str, Any]) -> None:
        """Validate and write one host message."""

        validate_host_message(message)
        line = encode_message(message)
        try:
            self._transport.write_line(line)
        except TreeCounterError:
            raise
        except Exception as exc:
            self._drain_stderr()
            raise WorkerProcessError(
                f"the worker stopped accepting input: {type(exc).__name__}"
            ) from exc

    def receive(self, timeout_ms: int | None = None) -> dict[str, Any]:
        """Return the next validated worker message.

        A closed stream, an oversized line, or a message that does not
        validate is a terminal failure: a partially understood exchange
        must never be treated as a result.
        """

        timeout = self._read_timeout_ms if timeout_ms is None else timeout_ms
        try:
            raw = self._transport.read_line(timeout)
        except TreeCounterError:
            raise
        except Exception as exc:
            self._drain_stderr()
            raise WorkerProcessError(
                f"the worker could not be read: {type(exc).__name__}"
            ) from exc
        self._drain_stderr()
        if raw is None:
            raise WorkerProcessError(
                "the worker stopped responding before finishing"
            )
        if len(raw) > MAX_MESSAGE_BYTES:
            raise ProtocolError("the worker sent an oversized line")
        message = decode_message(raw)
        validate_worker_message(message)
        return message

    def cancel(self, request_id: str = "cancel") -> None:
        """Ask the worker to stop, then stop it if it will not.

        The protocol cancel is tried first so the worker can shut down
        cleanly; termination is the fallback, never the first move.
        """

        if not self._started:
            return
        try:
            self.send(
                {
                    "type": "cancel",
                    "protocol_version": 1,
                    "request_id": request_id,
                }
            )
        except TreeCounterError:
            pass
        self._transport.terminate(CANCEL_GRACE_MS)

    def close(self) -> None:
        """Stop the worker and release the transport."""

        if not self._started:
            return
        try:
            if self._transport.is_running():
                self._transport.terminate(CANCEL_GRACE_MS)
        finally:
            self._drain_stderr()
            self._started = False

    def _drain_stderr(self) -> None:
        try:
            chunk = self._transport.read_stderr()
        except Exception:  # Diagnostics must never break the exchange.
            return
        if not chunk:
            return
        remaining = MAX_STDERR_BYTES - len(self._stderr)
        if remaining > 0:
            self._stderr.extend(chunk[:remaining])


class QProcessTransport:
    """A worker transport backed by ``QProcess``.

    QGIS owns the event loop, so the process is driven with the blocking
    ``waitFor`` API from inside a task thread rather than with signals.
    """

    def __init__(self) -> None:
        self._process: Any = None
        self._buffer = bytearray()

    def start(self, program: str, arguments: Sequence[str]) -> None:
        """Start the worker without going through a shell."""

        from qgis.PyQt.QtCore import QProcess

        process = QProcess()
        process.setProgram(program)
        process.setArguments(list(arguments))
        process.setProcessChannelMode(
            QProcess.ProcessChannelMode.SeparateChannels
        )
        process.start()
        if not process.waitForStarted(DEFAULT_START_TIMEOUT_MS):
            raise WorkerProcessError("the worker did not start")
        self._process = process

    def write_line(self, line: bytes) -> None:
        """Write one line and flush it to the worker."""

        if self._process is None:
            raise WorkerProcessError("the worker is not running")
        written = self._process.write(line)
        if written != len(line):
            raise WorkerProcessError("the worker input could not be written")
        self._process.waitForBytesWritten(DEFAULT_START_TIMEOUT_MS)

    def read_line(self, timeout_ms: int) -> bytes | None:
        """Return one newline-terminated line, or ``None`` at end of stream."""

        if self._process is None:
            raise WorkerProcessError("the worker is not running")
        while True:
            index = self._buffer.find(b"\n")
            if index >= 0:
                line = bytes(self._buffer[: index + 1])
                del self._buffer[: index + 1]
                return line
            if len(self._buffer) > MAX_MESSAGE_BYTES:
                raise ProtocolError("the worker sent an oversized line")
            if not self._process.waitForReadyRead(timeout_ms):
                chunk = bytes(self._process.readAllStandardOutput())
                if chunk:
                    self._buffer.extend(chunk)
                    continue
                return None
            self._buffer.extend(bytes(self._process.readAllStandardOutput()))

    def read_stderr(self) -> bytes:
        """Return any diagnostics already buffered by Qt."""

        if self._process is None:
            return b""
        return bytes(self._process.readAllStandardError())

    def terminate(self, grace_ms: int) -> None:
        """Terminate, then kill only if the worker ignores termination."""

        if self._process is None:
            return
        if self._process.state() == self._process.ProcessState.NotRunning:
            return
        self._process.closeWriteChannel()
        self._process.terminate()
        if not self._process.waitForFinished(grace_ms):
            self._process.kill()
            self._process.waitForFinished(grace_ms)

    def is_running(self) -> bool:
        """Return whether the worker is still alive."""

        if self._process is None:
            return False
        return self._process.state() != self._process.ProcessState.NotRunning

    def exit_code(self) -> int | None:
        """Return the worker's exit status once it has finished."""

        if self._process is None or self.is_running():
            return None
        return int(self._process.exitCode())


__all__ = [
    "CANCEL_GRACE_MS",
    "DEFAULT_READ_TIMEOUT_MS",
    "MAX_STDERR_BYTES",
    "QProcessTransport",
    "Transport",
    "WorkerChannel",
    "WorkerProcessError",
]

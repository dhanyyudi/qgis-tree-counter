"""Worker entry point: JSONL on stdin/stdout, diagnostics on stderr."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from typing import Any

from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.worker.runner import WorkerRunner, resolve_backend_factory


def _lines(stream: Any) -> Iterator[bytes]:
    for raw in stream:
        yield raw


def _log(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def _failing_factory(error: TreeCounterError) -> Callable[[], Any]:
    """Defer a resolution failure until the host actually needs a backend.

    The handshake still succeeds so the host receives a structured protocol
    error instead of an opaque non-zero exit.
    """

    def _factory() -> Any:
        raise error

    return _factory


def main(argv: list[str] | None = None) -> int:
    """Serve one worker session and return the process exit status."""

    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments:
        _log("the worker does not accept command-line arguments")
        return 2

    stdout = sys.stdout.buffer

    def write_line(line: bytes) -> None:
        stdout.write(line)
        stdout.flush()

    try:
        backend_factory = resolve_backend_factory()
    except TreeCounterError as error:
        backend_factory = _failing_factory(error)
    except Exception as error:  # pragma: no cover - defensive guard.
        _log(f"backend resolution failed: {type(error).__name__}")
        backend_factory = _failing_factory(
            TreeCounterError(
                ErrorCode.WORKER_PROCESS_FAILURE,
                diagnostic_detail="backend resolution failed",
            )
        )

    runner = WorkerRunner(write_line, _log, backend_factory)
    return runner.serve(_lines(sys.stdin.buffer))


if __name__ == "__main__":
    raise SystemExit(main())

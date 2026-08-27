"""Run Runtime Manager processes through QGIS's ``QProcess`` API."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence

from tree_counter.runtime.installer import ProcessResult

START_TIMEOUT_MS = 30_000
KILL_TIMEOUT_MS = 1_000
TIMEOUT_RETURN_CODE = -1


def _decode(value: object) -> str:
    """Decode a QProcess byte array without assuming its encoding is valid."""

    return bytes(value).decode("utf-8", errors="replace")


class QProcessRunner:
    """Execute one fixed argument vector and capture both output streams."""

    def __call__(
        self, argv: Sequence[str], timeout: float
    ) -> ProcessResult:
        """Run *argv*, killing it and returning failure on timeout."""

        from qgis.PyQt.QtCore import QProcess

        command = [str(part) for part in argv]
        if not command:
            return ProcessResult(
                TIMEOUT_RETURN_CODE, "", "no process executable was given"
            )
        process = QProcess()
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setProcessChannelMode(
            QProcess.ProcessChannelMode.SeparateChannels
        )
        process.start()
        if not process.waitForStarted(START_TIMEOUT_MS):
            error = _decode(process.readAllStandardError())
            if not error:
                error = process.errorString()
            process.close()
            return ProcessResult(TIMEOUT_RETURN_CODE, "", error)

        timeout_ms = max(1, int(float(timeout) * 1000))
        finished = process.waitForFinished(timeout_ms)
        if not finished:
            process.kill()
            process.waitForFinished(KILL_TIMEOUT_MS)
            stdout = _decode(process.readAllStandardOutput())
            stderr = _decode(process.readAllStandardError())
            if stderr:
                stderr += "\n"
            stderr += "process timed out"
            process.close()
            return ProcessResult(TIMEOUT_RETURN_CODE, stdout, stderr)

        stdout = _decode(process.readAllStandardOutput())
        stderr = _decode(process.readAllStandardError())
        returncode = int(process.exitCode())
        process.close()
        return ProcessResult(returncode, stdout, stderr)


__all__ = [
    "KILL_TIMEOUT_MS",
    "QProcessRunner",
    "START_TIMEOUT_MS",
    "TIMEOUT_RETURN_CODE",
]

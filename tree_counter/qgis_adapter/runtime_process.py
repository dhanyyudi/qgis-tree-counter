"""Run Runtime Manager processes through QGIS's ``QProcess`` API."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence

from tree_counter.runtime.installer import ProcessResult

START_TIMEOUT_MS = 30_000
KILL_TIMEOUT_MS = 1_000
POLL_INTERVAL_MS = 100
TIMEOUT_RETURN_CODE = -1
CRASH_RETURN_CODE = -2


def _append(stderr: str, note: str) -> str:
    """Add a diagnostic note to whatever the process already reported."""

    return f"{stderr}\n{note}" if stderr else note


def _decode(value: object) -> str:
    """Decode a QProcess byte array without assuming its encoding is valid."""

    return bytes(value).decode("utf-8", errors="replace")


class QProcessRunner:
    """Execute one fixed argument vector and capture both output streams."""

    def __call__(
        self, argv: Sequence[str], timeout: float
    ) -> ProcessResult:
        """Run *argv*, killing it and returning failure on timeout."""

        from qgis.PyQt.QtCore import (
            QCoreApplication,
            QEventLoop,
            QProcess,
        )

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
        finished = self._wait(
            process, timeout_ms, QProcess, QCoreApplication, QEventLoop
        )
        if not finished:
            process.kill()
            process.waitForFinished(KILL_TIMEOUT_MS)
            stdout = _decode(process.readAllStandardOutput())
            stderr = _decode(process.readAllStandardError())
            process.close()
            return ProcessResult(
                TIMEOUT_RETURN_CODE,
                stdout,
                _append(stderr, "process timed out"),
            )

        stdout = _decode(process.readAllStandardOutput())
        stderr = _decode(process.readAllStandardError())
        # Compare by value, not identity: Qt5 returns a plain int here
        # while Qt6 returns an enum member.
        crashed = (
            process.exitStatus() != QProcess.ExitStatus.NormalExit
        )
        returncode = CRASH_RETURN_CODE if crashed else int(process.exitCode())
        if crashed:
            stderr = _append(stderr, "process crashed")
        process.close()
        return ProcessResult(returncode, stdout, stderr)

    @staticmethod
    def _wait(process, timeout_ms, QProcess, QCoreApplication, QEventLoop):
        """Wait for *process*, keeping the QGIS event loop alive.

        A runtime install downloads hundreds of megabytes, so waiting in
        one blocking call would freeze the whole QGIS window for minutes
        and look exactly like the failure this runner exists to fix. User
        input stays excluded so a second action cannot start on top of the
        one already running.
        """

        elapsed = 0
        while True:
            slice_ms = min(POLL_INTERVAL_MS, max(0, timeout_ms - elapsed))
            if process.waitForFinished(slice_ms):
                return True
            if process.state() == QProcess.ProcessState.NotRunning:
                return True
            elapsed += POLL_INTERVAL_MS
            if elapsed >= timeout_ms:
                return False
            if QCoreApplication.instance() is not None:
                QCoreApplication.processEvents(
                    QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                )


__all__ = [
    "CRASH_RETURN_CODE",
    "KILL_TIMEOUT_MS",
    "POLL_INTERVAL_MS",
    "QProcessRunner",
    "START_TIMEOUT_MS",
    "TIMEOUT_RETURN_CODE",
]

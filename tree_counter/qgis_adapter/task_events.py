"""Queue task events onto the QGIS application thread."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

from qgis.PyQt.QtCore import QObject, Qt, pyqtSignal, pyqtSlot


class TaskEventBridge(QObject):
    """Forward worker-thread signals from a main-thread QObject."""

    progress_event = pyqtSignal(dict)
    warning_event = pyqtSignal(dict)
    terminal_event = pyqtSignal(dict)

    def connect_task(self, task: Any) -> None:
        """Connect one task using explicitly queued delivery."""

        connection = Qt.ConnectionType.QueuedConnection
        self._connect(task.progress_event, self._forward_progress, connection)
        self._connect(task.warning_event, self._forward_warning, connection)
        self._connect(task.terminal_event, self._forward_terminal, connection)

    def disconnect_task(self, task: Any) -> None:
        """Remove every connection made by :meth:`connect_task`."""

        for signal, slot in (
            (task.progress_event, self._forward_progress),
            (task.warning_event, self._forward_warning),
            (task.terminal_event, self._forward_terminal),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def dispose(self, task: Any) -> None:
        """Disconnect task and receivers, then release this QObject."""

        self.disconnect_task(task)
        for signal in (
            self.progress_event,
            self.warning_event,
            self.terminal_event,
        ):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass
        self.setParent(None)
        self.deleteLater()

    @staticmethod
    def _connect(signal: Any, slot: Any, connection: Any) -> None:
        try:
            signal.connect(slot, connection)
        except TypeError:
            # Tiny signal fakes used by import-level tests have no Qt
            # connection-type argument; real Qt signals always use it.
            signal.connect(slot)

    @pyqtSlot(dict)
    def _forward_progress(self, event: dict) -> None:
        self.progress_event.emit(dict(event))

    @pyqtSlot(dict)
    def _forward_warning(self, event: dict) -> None:
        self.warning_event.emit(dict(event))

    @pyqtSlot(dict)
    def _forward_terminal(self, event: dict) -> None:
        self.terminal_event.emit(dict(event))


__all__ = ["TaskEventBridge"]

"""Task events are always delivered on the QGIS application thread."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import threading
import time


def test_worker_signal_is_queued_to_the_application_thread(
    qgis_application,
) -> None:
    from qgis.PyQt.QtCore import QObject, pyqtSignal

    from tree_counter.qgis_adapter.task_events import TaskEventBridge

    class Emitter(QObject):
        progress_event = pyqtSignal(dict)
        warning_event = pyqtSignal(dict)
        terminal_event = pyqtSignal(dict)

    emitter = Emitter()
    bridge = TaskEventBridge()
    received: list[tuple[dict, object]] = []
    bridge.progress_event.connect(
        lambda event: received.append((event, threading.current_thread()))
    )
    bridge.connect_task(emitter)

    worker = threading.Thread(
        target=lambda: emitter.progress_event.emit({"type": "progress"})
    )
    worker.start()
    worker.join()
    deadline = time.monotonic() + 2
    while not received and time.monotonic() < deadline:
        qgis_application.processEvents()

    assert received == [
        ({"type": "progress"}, threading.main_thread())
    ]
    bridge.disconnect_task(emitter)

"""Submit background tasks without importing QGIS in the plugin module."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any


def add_task(task: Any) -> None:
    """Hand *task* to the QGIS application task manager."""

    from qgis.core import QgsApplication

    QgsApplication.taskManager().addTask(task)


__all__ = ["add_task"]

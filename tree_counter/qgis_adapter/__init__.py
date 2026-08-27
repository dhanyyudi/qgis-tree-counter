"""QGIS-facing adapters. Every QGIS import in the plugin lives here.

The plugin entry point must never import ``qgis.core`` itself: the
foundation gate checks that ``plugin.py`` only reaches Qt through
``qgis.PyQt``. The two host services the plugin needs but cannot import
are exposed here, with their QGIS imports deferred into the call.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any


def map_layers_named(name: str) -> list[Any]:
    """Return every loaded map layer with the given name."""

    from qgis.core import QgsProject

    return QgsProject.instance().mapLayersByName(str(name))


def add_task(task: Any) -> None:
    """Hand a task to the QGIS task manager."""

    from qgis.core import QgsApplication

    QgsApplication.taskManager().addTask(task)


__all__ = ["add_task", "map_layers_named"]

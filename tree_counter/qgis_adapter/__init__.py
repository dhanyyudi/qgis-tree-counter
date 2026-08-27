"""QGIS-facing adapters. Every QGIS import in the plugin lives here.

The plugin entry point must never import ``qgis.core`` itself: the
foundation gate checks that ``plugin.py`` only reaches Qt through
``qgis.PyQt``. The two host services the plugin needs but cannot import
are exposed here, with their QGIS imports deferred into the call.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

from tree_counter.qgis_adapter.task_manager import add_task


def map_layers_named(name: str) -> list[Any]:
    """Return every loaded map layer with the given name."""

    from qgis.core import QgsProject

    return QgsProject.instance().mapLayersByName(str(name))


__all__ = ["add_task", "map_layers_named"]

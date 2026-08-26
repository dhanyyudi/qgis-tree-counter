"""Tree Counter QGIS plugin package."""

# SPDX-License-Identifier: AGPL-3.0-only


def classFactory(iface):
    """Create the plugin instance for QGIS without eager imports."""
    from .plugin import TreeCounterPlugin

    return TreeCounterPlugin(iface)

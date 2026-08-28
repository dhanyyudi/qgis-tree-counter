"""List the project layers the dock can offer.

Only layers Tree Counter can actually use are offered. A raster it cannot
read, or a vector layer with no polygons, would look selectable and then
fail at the moment the user pressed Start.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

from tree_counter.qgis_adapter.raster import validate_layer


def _project_layers() -> list[Any]:
    from qgis.core import QgsProject

    return list(QgsProject.instance().mapLayers().values())


def is_usable_raster(layer: Any) -> bool:
    """Return whether the plugin can read this layer as RGB imagery."""

    from qgis.core import QgsMapLayer

    if layer is None or layer.type() != QgsMapLayer.LayerType.RasterLayer:
        return False
    if not layer.isValid():
        return False
    try:
        validate_layer(layer)
    except Exception:
        # Provider plugins can raise their own exception types while a
        # layer is opening or disappearing. Eligibility is a UI boundary:
        # an unusable layer is omitted rather than crashing the dock.
        return False
    return True


def is_polygon_layer(layer: Any) -> bool:
    """Return whether the layer is a vector layer holding polygons."""

    from qgis.core import QgsMapLayer, QgsWkbTypes

    if layer is None or layer.type() != QgsMapLayer.LayerType.VectorLayer:
        return False
    if not layer.isValid():
        return False
    return layer.geometryType() == QgsWkbTypes.GeometryType.PolygonGeometry


def raster_layer_names() -> tuple[str, ...]:
    """Return the names of every raster the plugin can count trees in."""

    return tuple(
        layer.name()
        for layer in _project_layers()
        if is_usable_raster(layer)
    )


def polygon_layer_names() -> tuple[str, ...]:
    """Return the names of every polygon layer that can define a scope."""

    return tuple(
        layer.name()
        for layer in _project_layers()
        if is_polygon_layer(layer)
    )


def project_file_name() -> str:
    """Return the current saved project filename, or an empty string."""

    from qgis.core import QgsProject

    return str(QgsProject.instance().fileName() or "")


def connect_layer_changes(callback: Any) -> tuple[tuple[Any, Any], ...]:
    """Call *callback* whenever the set of project layers changes."""

    from qgis.core import QgsProject

    project = QgsProject.instance()

    def handler(*_args: Any) -> None:
        callback()

    connections = []
    for signal in (
        project.layersAdded,
        project.layersRemoved,
        project.cleared,
    ):
        signal.connect(handler)
        connections.append((signal, handler))
    return tuple(connections)


def disconnect_layer_changes(connections: Any) -> None:
    """Disconnect handles returned by :func:`connect_layer_changes`."""

    for signal, handler in tuple(connections or ()):
        try:
            signal.disconnect(handler)
        except (TypeError, RuntimeError):
            pass


__all__ = [
    "connect_layer_changes",
    "disconnect_layer_changes",
    "is_polygon_layer",
    "is_usable_raster",
    "polygon_layer_names",
    "project_file_name",
    "raster_layer_names",
]

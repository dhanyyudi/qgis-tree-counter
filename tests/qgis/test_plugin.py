"""Plugin entry points create, focus, and unload one dock."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


class FakeIface:
    """Records what the plugin adds to the QGIS window."""

    def __init__(self, window) -> None:
        self._window = window
        self.toolbar_actions: list[object] = []
        self.menu_entries: list[tuple[str, object]] = []
        self.docks: list[object] = []

    def mainWindow(self):
        return self._window

    def addToolBarIcon(self, action) -> None:
        self.toolbar_actions.append(action)

    def removeToolBarIcon(self, action) -> None:
        self.toolbar_actions.remove(action)

    def addPluginToMenu(self, menu, action) -> None:
        self.menu_entries.append((menu, action))

    def removePluginMenu(self, menu, action) -> None:
        self.menu_entries.remove((menu, action))

    def addDockWidget(self, area, dock) -> None:
        self.docks.append(dock)
        self._window.addDockWidget(area, dock)

    def removeDockWidget(self, dock) -> None:
        self.docks.remove(dock)
        self._window.removeDockWidget(dock)


@pytest.fixture
def plugin(qgis_application):
    from qgis.PyQt.QtWidgets import QMainWindow

    from tree_counter.plugin import TreeCounterPlugin

    window = QMainWindow()
    iface = FakeIface(window)
    instance = TreeCounterPlugin(iface)
    instance.initGui()
    yield instance, iface
    instance.unload()
    window.setParent(None)


def test_the_class_factory_returns_the_plugin(qgis_application) -> None:
    from qgis.PyQt.QtWidgets import QMainWindow

    from tree_counter.plugin import TreeCounterPlugin, classFactory

    window = QMainWindow()
    instance = classFactory(FakeIface(window))

    assert isinstance(instance, TreeCounterPlugin)
    window.setParent(None)


def test_init_adds_a_toolbar_icon_and_menu_entry(plugin) -> None:
    instance, iface = plugin

    assert len(iface.toolbar_actions) == 1
    assert len(iface.menu_entries) == 1
    assert iface.menu_entries[0][0] == "&Tree Counter"


def test_the_packaged_icon_exists() -> None:
    from tree_counter.plugin import icon_path

    assert icon_path().is_file()
    assert icon_path().suffix == ".svg"


def test_the_action_opens_one_right_side_dock(plugin) -> None:
    from qgis.PyQt.QtCore import Qt

    from tree_counter.ui.dock import DOCK_OBJECT_NAME

    instance, iface = plugin

    iface.toolbar_actions[0].trigger()

    assert len(iface.docks) == 1
    assert iface.docks[0].objectName() == DOCK_OBJECT_NAME
    assert iface.mainWindow().dockWidgetArea(iface.docks[0]) == (
        Qt.DockWidgetArea.RightDockWidgetArea
    )


def test_activating_twice_reuses_the_same_dock(plugin) -> None:
    instance, iface = plugin

    first = instance.show_dock()
    second = instance.show_dock()

    assert first is second
    assert len(iface.docks) == 1


def test_unloading_removes_everything_it_added(plugin) -> None:
    instance, iface = plugin
    instance.show_dock()

    instance.unload()

    assert iface.toolbar_actions == []
    assert iface.menu_entries == []
    assert iface.docks == []
    assert instance.dock is None


def test_unloading_without_opening_the_dock_is_safe(plugin) -> None:
    instance, iface = plugin

    instance.unload()

    assert iface.toolbar_actions == []


def test_the_runtime_status_never_raises(plugin) -> None:
    instance, _ = plugin

    # A machine with no runtime must still open the dock.
    assert isinstance(instance._runtime_status(), str)

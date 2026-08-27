"""The QGIS entry points for Tree Counter.

The plugin owns one dock and one toolbar action. Both point at the same
dock instance, so activating either shows the one that already exists
rather than creating a second. Unloading removes everything it added.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from typing import Any

MENU_TITLE = "&Tree Counter"
ACTION_TEXT = "Tree Counter"
ICON_RELATIVE = "icons/tree_counter.svg"


def icon_path() -> Path:
    """Return the packaged toolbar icon."""

    return Path(__file__).resolve().parent / ICON_RELATIVE


class TreeCounterPlugin:
    """Creates the dock, the toolbar action, and the menu entry."""

    def __init__(self, iface: Any) -> None:
        self.iface = iface
        self.dock: Any = None
        self.action: Any = None
        self._controller: Any = None

    # -- QGIS lifecycle --------------------------------------------------

    def initGui(self) -> None:
        """Add the toolbar action and the Plugins menu entry."""

        from qgis.PyQt.QtGui import QIcon
        from qgis.PyQt.QtWidgets import QAction

        icon = QIcon(str(icon_path()))
        self.action = QAction(icon, ACTION_TEXT, self.iface.mainWindow())
        self.action.setObjectName("TreeCounterAction")
        self.action.setCheckable(False)
        self.action.triggered.connect(self.show_dock)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu(MENU_TITLE, self.action)

    def unload(self) -> None:
        """Remove everything this plugin added to the QGIS window."""

        if self.action is not None:
            self.iface.removePluginMenu(MENU_TITLE, self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None
        if self.dock is not None:
            self.iface.removeDockWidget(self.dock)
            self.dock.setParent(None)
            self.dock = None
        self._controller = None

    # -- dock ------------------------------------------------------------

    def show_dock(self) -> Any:
        """Create the dock once, then raise it on every later request."""

        from qgis.PyQt.QtCore import Qt

        if self.dock is None:
            self.dock = self._build_dock()
            self.iface.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea, self.dock
            )
        self.dock.show()
        self.dock.raise_()
        return self.dock

    def _build_dock(self) -> Any:
        from tree_counter.ui.dock import build_dock

        self._controller = self._build_controller()
        return build_dock(
            self._controller,
            parent=self.iface.mainWindow(),
            open_runtime_manager=self.show_runtime_manager,
        )

    def _build_controller(self) -> Any:
        from tree_counter.settings.presets import PresetStore
        from tree_counter.settings.store import (
            SettingsStore,
            default_settings_path,
        )
        from tree_counter.settings.trust import TrustStore, identify_model
        from tree_counter.ui.controller import CountingController

        store = SettingsStore(default_settings_path())
        return CountingController(
            identify_model=identify_model,
            inspect_model=self._inspect_model,
            trust_store=TrustStore(store),
            preset_store=PresetStore(store),
            runtime_status=self._runtime_status,
        )

    # -- services --------------------------------------------------------

    def _runtime_status(self) -> str:
        try:
            return self._installer().inspect().state.value
        except Exception:
            return "not_installed"

    def _inspect_model(self, identity: Any) -> dict:
        """Inspect a model in the isolated worker.

        Wired in the release that adds model inspection to the dock; the
        controller already treats a failure here as a rejected model.
        """

        raise NotImplementedError(
            "model inspection is wired with the counting task"
        )

    def _installer(self) -> Any:
        from tree_counter.runtime.installer import RuntimeInstaller
        from tree_counter.runtime.paths import (
            RuntimePaths,
            default_runtime_root,
        )

        return RuntimeInstaller(
            paths=RuntimePaths(default_runtime_root()),
            runner=self._unavailable_runner,
            lock_root=Path(__file__).resolve().parent / "runtime" / "locks",
        )

    @staticmethod
    def _unavailable_runner(argv, timeout):
        """Refuse to run a process outside the Runtime Manager."""

        raise RuntimeError(
            "runtime changes are only made from the Runtime Manager"
        )

    def show_runtime_manager(self) -> Any:
        """Open the Runtime Manager dialog."""

        from tree_counter.ui.runtime_dialog import RuntimeManagerDialog

        dialog = RuntimeManagerDialog(
            self._installer(), parent=self.iface.mainWindow()
        )
        dialog.show()
        return dialog


def classFactory(iface: Any) -> TreeCounterPlugin:
    """Return the plugin instance QGIS loads."""

    return TreeCounterPlugin(iface)


__all__ = ["ACTION_TEXT", "MENU_TITLE", "TreeCounterPlugin", "classFactory"]

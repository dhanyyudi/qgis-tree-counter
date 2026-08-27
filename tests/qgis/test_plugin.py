"""Plugin entry points create, focus, and unload one dock."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

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


def test_the_output_request_honours_the_chosen_layers(
    qgis_application, tmp_path
) -> None:
    from tree_counter.plugin import TreeCounterPlugin
    from tree_counter.ui.controller import ViewState

    instance = TreeCounterPlugin(None)

    class FakeRaster:
        def source(self):
            return "/data/aerial.tif"

        def name(self):
            return "aerial"

    state = ViewState(
        output_path=str(tmp_path),
        write_centers=False,
        write_boxes=True,
    )

    request = instance._output_request(state, FakeRaster())

    assert request.write_centers is False
    assert request.write_boxes is True
    assert request.directory == tmp_path


def test_a_task_terminating_after_unload_does_not_reach_the_controller(
    plugin,
) -> None:
    """Unload drops the controller while the task is still running.

    Cancellation is asynchronous, so the task emits its terminal event
    after unload. Left connected, that called on_cancelled on a
    controller that no longer exists and raised inside QGIS.
    """

    class RunningTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    instance, _iface = plugin
    task = RunningTask()
    task.terminal_event = _FakeSignal()
    task.terminal_event.connect(instance._on_terminal)
    instance._task = task

    instance.unload()
    task.terminal_event.emit({"type": "cancelled"})

    assert task.cancelled is True


class _FakeSignal:
    """A minimal stand-in for a pyqtSignal carrying a dict."""

    def __init__(self) -> None:
        self._slots: list = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def disconnect(self, slot) -> None:
        self._slots.remove(slot)

    def emit(self, payload: dict) -> None:
        for slot in list(self._slots):
            slot(payload)


def test_start_run_does_not_depend_on_a_cached_package_reexport(
    qgis_application, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plugin reload may retain the old qgis_adapter package object."""

    import tree_counter.qgis_adapter as current_adapter
    from tree_counter.plugin import TreeCounterPlugin

    stale_adapter = ModuleType("tree_counter.qgis_adapter")
    stale_adapter.__path__ = list(current_adapter.__path__)
    monkeypatch.setitem(
        sys.modules, "tree_counter.qgis_adapter", stale_adapter
    )

    submitted: list[object] = []
    task_manager = ModuleType("tree_counter.qgis_adapter.task_manager")
    task_manager.add_task = submitted.append
    monkeypatch.setitem(
        sys.modules,
        "tree_counter.qgis_adapter.task_manager",
        task_manager,
    )

    class FakeTask:
        progress_event = _FakeSignal()
        warning_event = _FakeSignal()
        terminal_event = _FakeSignal()

    class FakeController:
        on_event = staticmethod(lambda event: None)
        on_failed = staticmethod(lambda error: None)

    plugin = TreeCounterPlugin(None)
    plugin._controller = FakeController()
    task = FakeTask()
    monkeypatch.setattr(plugin, "_build_task", lambda state: task)

    plugin._start_run(object())

    assert submitted == [task]


def test_the_dedicated_task_manager_adapter_is_packaged() -> None:
    """Task submission must remain importable after a package hot reload."""

    import tree_counter

    adapter = (
        Path(tree_counter.__file__).resolve().parent
        / "qgis_adapter"
        / "task_manager.py"
    )

    assert adapter.is_file()


@pytest.mark.parametrize(
    ("write_centers", "write_boxes", "expected_layers"),
    [
        (True, False, ["tree_centers"]),
        (False, True, ["detection_boxes"]),
        (True, True, ["tree_centers", "detection_boxes"]),
    ],
)
def test_a_completed_run_loads_the_requested_result_layers(
    qgis_application,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_centers: bool,
    write_boxes: bool,
    expected_layers: list[str],
) -> None:
    from tree_counter.plugin import TreeCounterPlugin
    from tree_counter.qgis_adapter import output

    loaded: list[tuple[Path, list[str]]] = []
    monkeypatch.setattr(
        output,
        "load_result_layers",
        lambda path, names: loaded.append((path, list(names))),
    )

    class FakeController:
        state = SimpleNamespace(
            write_centers=write_centers,
            write_boxes=write_boxes,
        )

        def on_completed(self, result, output_path) -> None:
            pass

    plugin = TreeCounterPlugin(None)
    plugin._controller = FakeController()
    output_path = tmp_path / "counting.gpkg"

    plugin._on_terminal(
        {
            "type": "completed",
            "result": object(),
            "output_path": str(output_path),
        }
    )

    assert loaded == [(output_path, expected_layers)]

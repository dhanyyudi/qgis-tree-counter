"""The QGIS entry points for Tree Counter.

The plugin owns one dock and one toolbar action. Both point at the same
dock instance, so activating either shows the one that already exists
rather than creating a second. Unloading removes everything it added.

The controller is given the three service callbacks it needs but is not
allowed to reach the worker, the runtime, or the raster directly: those
live here, where QGIS is available.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from typing import Any

from tree_counter.errors import ErrorCode, TreeCounterError

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
        self._model_path: str | None = None
        self._task: Any = None

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

        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self.action is not None:
            self.iface.removePluginMenu(MENU_TITLE, self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None
        if self.dock is not None:
            self.iface.removeDockWidget(self.dock)
            self.dock.setParent(None)
            self.dock = None
        self._controller = None
        self._model_path = None

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

        def identify_and_remember(path: Any) -> Any:
            self._model_path = str(path)
            return identify_model(path)

        store = SettingsStore(default_settings_path())
        return CountingController(
            identify_model=identify_and_remember,
            inspect_model=self._inspect_model,
            trust_store=TrustStore(store),
            preset_store=PresetStore(store),
            runtime_status=self._runtime_status,
            start_run=self._start_run,
            cancel_run=self._cancel_run,
        )

    # -- services --------------------------------------------------------

    def _runtime_status(self) -> str:
        try:
            return self._installer().inspect().state.value
        except Exception:
            return "not_installed"

    def _runtime_paths(self) -> Any:
        from tree_counter.runtime.paths import (
            RuntimePaths,
            default_runtime_root,
        )

        return RuntimePaths(default_runtime_root())

    def _worker_command(self) -> list[str]:
        from tree_counter.qgis_adapter.launcher import build_worker_command

        return build_worker_command(
            self._installer().inspect(), self._runtime_paths()
        )

    def _inspect_model(self, identity: Any) -> dict:
        """Inspect a model in the isolated worker and return its info.

        A worker that rejects the model answers with an ``error`` message;
        its code and message are surfaced verbatim instead of being
        collapsed into a generic "no classes" reply. The channel is always
        closed, so a rejection or a crash can never leave a worker process
        behind.
        """

        from tree_counter.constants import PROTOCOL_VERSION
        from tree_counter.qgis_adapter.process import (
            QProcessTransport,
            WorkerChannel,
        )

        if not self._model_path:
            raise TreeCounterError(
                ErrorCode.INVALID_MODEL,
                diagnostic_detail="no model path was recorded",
            )
        command = self._worker_command()
        channel = WorkerChannel(QProcessTransport())
        try:
            channel.start(command[0], command[1:])
            channel.send(
                {
                    "type": "hello",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": "hello",
                }
            )
            self._expect(channel.receive(), "hello")
            channel.send(
                {
                    "type": "inspect_model",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": "inspect",
                    "model_path": self._model_path,
                    "model_sha256": identity.sha256,
                }
            )
            return self._expect(channel.receive(), "model_info")
        finally:
            channel.close()

    @staticmethod
    def _expect(message: Any, wanted: str) -> dict:
        """Return a worker reply of the expected type, or raise its error."""

        kind = str(message.get("type", ""))
        if kind == "error":
            code = str(message.get("code", ""))
            try:
                error_code = ErrorCode(code)
            except ValueError:
                error_code = ErrorCode.WORKER_PROTOCOL_FAILURE
            raise TreeCounterError(
                error_code,
                user_message=str(message.get("message") or ""),
            )
        if kind != wanted:
            raise TreeCounterError(
                ErrorCode.WORKER_PROTOCOL_FAILURE,
                diagnostic_detail=(
                    f"expected {wanted!r}, the worker sent {kind!r}"
                ),
            )
        return dict(message)

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

    # -- counting run ----------------------------------------------------

    def _start_run(self, state: Any) -> None:
        """Build a counting task from the dock state and hand it to QGIS."""

        from tree_counter.qgis_adapter import add_task

        try:
            task = self._build_task(state)
        except TreeCounterError as error:
            self._controller.on_failed(error)
            return
        task.progress_event.connect(self._controller.on_event)
        task.warning_event.connect(self._controller.on_event)
        task.terminal_event.connect(self._on_terminal)
        self._task = task
        add_task(task)

    def _cancel_run(self) -> None:
        """Ask the running counting task to stop."""

        if self._task is not None:
            self._task.cancel()

    def _on_terminal(self, event: dict) -> None:
        kind = str(event.get("type", ""))
        self._task = None
        if kind == "completed":
            self._controller.on_completed(
                event.get("result"), event.get("output_path", "")
            )
        elif kind == "cancelled":
            self._controller.on_cancelled()
        else:
            self._controller.on_failed(event.get("error"))

    def _build_task(self, state: Any) -> Any:
        from tree_counter.core.types import with_selected_classes
        from tree_counter.qgis_adapter.output import (
            OutputRequest,
            output_timestamp,
        )
        from tree_counter.qgis_adapter.process import (
            QProcessTransport,
            WorkerChannel,
        )
        from tree_counter.qgis_adapter.raster import (
            RasterReader,
            validate_layer,
        )
        from tree_counter.qgis_adapter.task import CountingTask, RunRequest
        from tree_counter.qgis_adapter.workspace import RunWorkspace

        raster = self._find_layer(state.raster_name)
        info = validate_layer(raster)
        scope = self._build_scope(info, state, raster)
        settings = with_selected_classes(
            state.settings, state.selected_class_ids
        )
        request = RunRequest(
            scope=scope,
            settings=settings,
            model_path=str(self._model_path or ""),
            model_sha256=state.model.sha256,
        )
        output_request = OutputRequest(
            directory=self._output_directory(state),
            raster_stem=self._raster_stem(raster),
            timestamp=output_timestamp(),
        )
        command = self._worker_command()
        return CountingTask(
            description=f"Count trees in {state.raster_name}",
            request=request,
            channel=WorkerChannel(QProcessTransport()),
            command=(command[0], command[1:]),
            tiles=RasterReader(raster, info),
            workspace=RunWorkspace.create(),
            raster_info=info,
            output_request=output_request,
            crs=raster.crs(),
        )

    def _build_scope(self, info: Any, state: Any, raster: Any) -> Any:
        from tree_counter.qgis_adapter.scope import (
            ScopeKind,
            scope_from_map_extent,
            scope_from_polygon_layer,
            whole_raster_scope,
        )

        if state.scope is ScopeKind.WHOLE_RASTER:
            return whole_raster_scope(info)
        if state.scope is ScopeKind.MAP_EXTENT:
            canvas = self.iface.mapCanvas()
            return scope_from_map_extent(
                info,
                canvas.extent(),
                canvas.mapSettings().destinationCrs(),
                raster.crs(),
            )
        layer = self._find_layer(state.polygon_layer_name)
        return scope_from_polygon_layer(
            info, layer, raster.crs(), selected_only=True
        )

    def _find_layer(self, name: str) -> Any:
        from tree_counter.qgis_adapter import map_layers_named

        layers = map_layers_named(str(name))
        if not layers:
            raise TreeCounterError(
                ErrorCode.INVALID_RASTER,
                diagnostic_detail=f"no loaded layer is named {name!r}",
            )
        return layers[0]

    def _output_directory(self, state: Any) -> Path:
        path = Path(str(state.output_path or ""))
        if path.suffix.casefold() == ".gpkg":
            return path.parent
        return path

    @staticmethod
    def _raster_stem(raster: Any) -> str:
        source = raster.source()
        if source:
            return Path(source).stem
        return raster.name()


def classFactory(iface: Any) -> TreeCounterPlugin:
    """Return the plugin instance QGIS loads."""

    return TreeCounterPlugin(iface)


__all__ = ["ACTION_TEXT", "MENU_TITLE", "TreeCounterPlugin", "classFactory"]

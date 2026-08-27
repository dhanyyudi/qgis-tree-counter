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
MODEL_DIALOG_TITLE = "Choose a detection model"
MODEL_FILE_FILTER = "Detection models (*.onnx *.pt)"
OUTPUT_DIALOG_TITLE = "Choose where to write the results"
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
        self._task_bridge: Any = None
        self._inspection_task: Any = None
        self._inspection_bridge: Any = None
        self._layer_connections: Any = ()
        self._translator: Any = None

    # -- QGIS lifecycle --------------------------------------------------

    def initGui(self) -> None:
        """Add the toolbar action and the Plugins menu entry."""

        from qgis.PyQt.QtGui import QIcon
        from qgis.PyQt.QtWidgets import QAction

        self._translator = self._install_translator()
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
            # The task keeps running after unload and will emit its
            # terminal event later. Leaving it connected would call back
            # into a controller this method is about to drop.
            if self._task_bridge is not None:
                self._task_bridge.dispose(self._task)
            self._task.cancel()
            self._task = None
            self._task_bridge = None
        if self._inspection_task is not None:
            if self._inspection_bridge is not None:
                self._inspection_bridge.dispose(self._inspection_task)
            self._inspection_task.cancel()
            self._inspection_task = None
            self._inspection_bridge = None
        if self._layer_connections:
            from tree_counter.qgis_adapter.layers import (
                disconnect_layer_changes,
            )

            disconnect_layer_changes(self._layer_connections)
            self._layer_connections = ()
        if self._translator is not None:
            from qgis.PyQt.QtCore import QCoreApplication

            QCoreApplication.instance().removeTranslator(self._translator)
            self._translator = None
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
        self._set_default_output_directory()
        dock = build_dock(
            self._controller,
            parent=self.iface.mainWindow(),
            open_runtime_manager=self.show_runtime_manager,
            choose_model=self.choose_model,
            choose_output=self.choose_output,
        )
        self._connect_project(dock)
        return dock

    def _connect_project(self, dock: Any) -> None:
        """Keep the layer combos in step with the QGIS project."""

        from tree_counter.qgis_adapter.layers import connect_layer_changes

        self._refresh_layer_choices(dock)
        self._layer_connections = connect_layer_changes(
            lambda: self._refresh_layer_choices(dock)
        )

    def _set_default_output_directory(self) -> None:
        """Use the saved QGIS project's directory when one is available."""

        if self._controller is None or self._controller.state.output_path:
            return
        from tree_counter.qgis_adapter.layers import project_file_name

        filename = project_file_name()
        if filename:
            self._controller.set_output_path(Path(filename).parent)

    @staticmethod
    def _refresh_layer_choices(dock: Any) -> None:
        """Offer exactly the layers the plugin can actually use."""

        from tree_counter.qgis_adapter.layers import (
            polygon_layer_names,
            raster_layer_names,
        )
        from tree_counter.ui.dock import (
            set_polygon_choices,
            set_raster_choices,
        )

        set_raster_choices(dock, raster_layer_names())
        set_polygon_choices(dock, polygon_layer_names())

    def choose_output(self) -> str:
        """Ask where results should be written."""

        from qgis.PyQt.QtWidgets import QFileDialog

        directory = QFileDialog.getExistingDirectory(
            self.iface.mainWindow(), OUTPUT_DIALOG_TITLE, ""
        )
        if directory and self._controller is not None:
            self._controller.set_output_path(directory)
        return str(directory or "")

    def choose_model(self) -> str:
        """Ask for a model file and hand it to the controller."""

        from qgis.PyQt.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getOpenFileName(
            self.iface.mainWindow(),
            MODEL_DIALOG_TITLE,
            "",
            MODEL_FILE_FILTER,
        )
        if path and self._controller is not None:
            self._controller.select_model(path)
        return str(path or "")

    def _build_controller(self) -> Any:
        from tree_counter.settings.presets import PresetStore
        from tree_counter.settings.store import (
            SettingsStore,
            default_settings_path,
        )
        from tree_counter.settings.trust import TrustStore, identify_model
        from tree_counter.ui.controller import CountingController

        def identify_and_remember(path: Any) -> Any:
            # Only a file that identified cleanly may become the path a
            # run uses. Recording it first would pair a new, unreadable
            # path with the previous model's hash.
            identity = identify_model(path)
            self._model_path = str(path)
            return identity

        store = SettingsStore(default_settings_path())
        return CountingController(
            identify_model=identify_and_remember,
            inspect_model=self._inspect_model,
            trust_store=TrustStore(store),
            preset_store=PresetStore(store),
            runtime_status=self._runtime_status,
            start_run=self._start_run,
            cancel_run=self._cancel_run,
            start_model_inspection=self._start_model_inspection,
        )

    # -- services --------------------------------------------------------

    def _install_translator(self) -> Any:
        from qgis.PyQt.QtCore import QCoreApplication

        from tree_counter.i18n import install_translator

        return install_translator(QCoreApplication.instance())

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

    def _inspect_model(
        self, identity: Any, should_cancel: Any = lambda: False
    ) -> dict:
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
        channel.set_cancel_check(should_cancel)
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

    def _start_model_inspection(self, identity: Any) -> None:
        """Submit isolated model inspection without blocking the dock."""

        from tree_counter.qgis_adapter.task import ModelInspectionTask
        from tree_counter.qgis_adapter.task_events import TaskEventBridge
        from tree_counter.qgis_adapter.task_manager import add_task

        task = ModelInspectionTask(
            f"Inspect {identity.filename}", identity, self._inspect_model
        )
        parent = self.iface.mainWindow() if self.iface is not None else None
        bridge = TaskEventBridge(parent)
        bridge.terminal_event.connect(self._on_model_terminal)
        bridge.connect_task(task)
        self._inspection_task = task
        self._inspection_bridge = bridge
        add_task(task)

    def _on_model_terminal(self, event: dict) -> None:
        if self._inspection_task is not None and self._inspection_bridge:
            self._inspection_bridge.dispose(self._inspection_task)
        self._inspection_task = None
        self._inspection_bridge = None
        if self._controller is None:
            return
        kind = str(event.get("type", ""))
        if kind == "completed":
            self._controller.on_model_inspected(event.get("info", {}))
        elif kind == "failed":
            self._controller.on_model_inspection_failed(event.get("error"))

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
        from tree_counter.qgis_adapter.runtime_process import QProcessRunner

        return RuntimeInstaller(
            paths=RuntimePaths(default_runtime_root()),
            runner=QProcessRunner(),
            lock_root=Path(__file__).resolve().parent / "runtime" / "locks",
        )

    def _refresh_runtime_state(self) -> None:
        """Let the dock re-read the runtime after the Manager changed it."""

        if self._controller is not None:
            self._controller.refresh_runtime()

    def show_runtime_manager(self) -> Any:
        """Open the Runtime Manager dialog."""

        from tree_counter.ui.runtime_dialog import RuntimeManagerDialog

        dialog = RuntimeManagerDialog(
            self._installer(),
            parent=self.iface.mainWindow(),
            on_changed=self._refresh_runtime_state,
        )
        dialog.show()
        return dialog

    # -- counting run ----------------------------------------------------

    def _start_run(self, state: Any) -> None:
        """Build a counting task from the dock state and hand it to QGIS."""

        from tree_counter.qgis_adapter.task_manager import add_task
        from tree_counter.qgis_adapter.task_events import TaskEventBridge

        try:
            task = self._build_task(state)
        except TreeCounterError as error:
            self._controller.on_failed(error)
            return
        parent = self.iface.mainWindow() if self.iface is not None else None
        bridge = TaskEventBridge(parent)
        bridge.progress_event.connect(self._controller.on_event)
        bridge.warning_event.connect(self._controller.on_event)
        bridge.terminal_event.connect(self._on_terminal)
        bridge.connect_task(task)
        self._task = task
        self._task_bridge = bridge
        add_task(task)

    def _cancel_run(self) -> None:
        """Ask the running counting task to stop."""

        if self._task is not None:
            self._task.cancel()

    def _on_terminal(self, event: dict) -> None:
        kind = str(event.get("type", ""))
        if self._task is not None and self._task_bridge is not None:
            self._task_bridge.dispose(self._task)
        self._task = None
        self._task_bridge = None
        if self._controller is None:
            return
        if kind == "completed":
            output_path = str(event.get("output_path", ""))
            self._controller.on_completed(event.get("result"), output_path)
            self._load_result_layers(output_path)
        elif kind == "cancelled":
            self._controller.on_cancelled()
        else:
            self._controller.on_failed(event.get("error"))

    def _load_result_layers(self, output_path: str) -> None:
        """Load only the result layers selected for the completed run."""

        from tree_counter.qgis_adapter.output import (
            BOXES_LAYER,
            CENTERS_LAYER,
            load_result_layers,
        )

        state = self._controller.state
        layer_names = []
        if state.write_centers:
            layer_names.append(CENTERS_LAYER)
        if state.write_boxes:
            layer_names.append(BOXES_LAYER)
        try:
            load_result_layers(Path(output_path), layer_names)
        except Exception as error:
            # The GeoPackage has already been atomically published. A QGIS
            # provider error must not turn a completed count into a failed
            # run or hide the saved result from the user.
            self._controller.on_event(
                {
                    "type": "warning",
                    "message": (
                        "The output was saved, but QGIS could not add its "
                        f"result layers automatically: {error}"
                    ),
                }
            )

    def _build_task(self, state: Any) -> Any:
        from tree_counter.core.types import with_selected_classes
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
        output_request = self._output_request(state, raster)
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

    def _output_request(self, state: Any, raster: Any) -> Any:
        from tree_counter.qgis_adapter.output import (
            OutputRequest,
            output_timestamp,
        )

        return OutputRequest(
            directory=self._output_directory(state),
            raster_stem=self._raster_stem(raster),
            write_centers=bool(state.write_centers),
            write_boxes=bool(state.write_boxes),
            timestamp=output_timestamp(),
        )

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

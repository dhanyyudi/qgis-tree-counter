"""The Tree Counter dock.

The dock is a view. It renders whatever the controller reports and hands
user actions straight back; it never calls the worker, the runtime, or the
output services itself. That keeps the workflow in one tested place and
keeps this file to layout and wiring.

Sections follow the approved design: Choose data, Select model, Detection
settings, Output, Start counting. Advanced is collapsed by default. There
is no Task Type control and no Min Distance, by decision.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

from tree_counter.constants import (
    MAX_OVERLAP_PERCENT,
    MAX_TILE_SIZE,
    MIN_OVERLAP_PERCENT,
    MIN_TILE_SIZE,
    TILE_SIZE_MULTIPLE,
)
from tree_counter.core.types import InferenceSettings
from tree_counter.qgis_adapter.scope import ScopeKind
from tree_counter.ui.controller import CountingController, Phase
from tree_counter.ui.widgets import (
    CollapsibleSection,
    checked_class_ids,
    make_class_list,
    make_double_control,
    make_int_control,
    set_class_items,
)

DOCK_OBJECT_NAME = "TreeCounterDock"
DOCK_TITLE = "Tree Counter"
SECTION_TITLES = (
    "Choose data",
    "Select model",
    "Detection settings",
    "Output",
    "Start counting",
)
SCOPE_CHOICES = (
    ("Whole raster", ScopeKind.WHOLE_RASTER),
    ("Current map extent", ScopeKind.MAP_EXTENT),
    ("Polygon layer or selection", ScopeKind.POLYGON),
)


def _dock_base() -> Any:
    from qgis.PyQt.QtWidgets import QDockWidget

    return QDockWidget


def build_dock(
    controller: CountingController,
    parent: Any = None,
    open_runtime_manager: Any = None,
) -> Any:
    """Return the Tree Counter dock bound to *controller*."""

    from qgis.PyQt import QtWidgets

    dock = _dock_base()(DOCK_TITLE, parent)
    dock.setObjectName(DOCK_OBJECT_NAME)

    container = QtWidgets.QWidget()
    outer = QtWidgets.QVBoxLayout(container)
    outer.setContentsMargins(8, 8, 8, 8)

    header = QtWidgets.QHBoxLayout()
    runtime_label = QtWidgets.QLabel("Runtime: unknown")
    runtime_button = QtWidgets.QToolButton()
    runtime_button.setText("Runtime Manager")
    header.addWidget(runtime_label)
    header.addStretch(1)
    header.addWidget(runtime_button)
    outer.addLayout(header)

    data = CollapsibleSection(SECTION_TITLES[0], expanded=True)
    raster_combo = QtWidgets.QComboBox()
    scope_combo = QtWidgets.QComboBox()
    for label, _ in SCOPE_CHOICES:
        scope_combo.addItem(label)
    polygon_combo = QtWidgets.QComboBox()
    polygon_combo.setEnabled(False)
    data.add_row("Raster", raster_combo)
    data.add_row("Scope", scope_combo)
    data.add_row("Polygon layer", polygon_combo)
    outer.addWidget(data.widget)

    model = CollapsibleSection(SECTION_TITLES[1], expanded=True)
    model_path = QtWidgets.QLineEdit()
    model_path.setReadOnly(True)
    model_path.setPlaceholderText("Choose a .onnx or trusted .pt model")
    browse = QtWidgets.QPushButton("Browse...")
    trust_button = QtWidgets.QPushButton("Confirm this checkpoint")
    trust_button.setVisible(False)
    classes = make_class_list()
    model.add_row("Model", model_path)
    model.add_widget(browse)
    model.add_widget(trust_button)
    model.add_row("Classes", classes)
    outer.addWidget(model.widget)

    detection = CollapsibleSection(SECTION_TITLES[2], expanded=True)
    defaults = InferenceSettings()
    confidence = make_double_control(0.0, 1.0, 0.05, defaults.confidence)
    nms_iou = make_double_control(0.0, 1.0, 0.05, defaults.nms_iou)
    tile_size = make_int_control(
        MIN_TILE_SIZE, MAX_TILE_SIZE, TILE_SIZE_MULTIPLE, defaults.tile_size
    )
    overlap = make_int_control(
        MIN_OVERLAP_PERCENT, MAX_OVERLAP_PERCENT, 5, defaults.overlap_percent
    )
    detection.add_row("Confidence", confidence)
    detection.add_row("NMS IoU", nms_iou)
    detection.add_row("Tile size", tile_size)
    detection.add_row("Overlap %", overlap)

    # Advanced stays folded: these are the settings a user should not need.
    advanced = CollapsibleSection("Advanced", expanded=False)
    duplicate_iou = make_double_control(
        0.0, 1.0, 0.05, defaults.duplicate_iou
    )
    device = QtWidgets.QComboBox()
    device.addItems(["auto", "cpu"])
    advanced.add_row("Duplicate IoU", duplicate_iou)
    advanced.add_row("Device", device)
    detection.add_widget(advanced.widget)
    outer.addWidget(detection.widget)

    output = CollapsibleSection(SECTION_TITLES[3], expanded=True)
    output_path = QtWidgets.QLineEdit()
    output_path.setPlaceholderText("Where results are written")
    write_centers = QtWidgets.QCheckBox("Tree centres")
    write_centers.setChecked(True)
    write_boxes = QtWidgets.QCheckBox("Detection boxes")
    output.add_row("Output", output_path)
    output.add_widget(write_centers)
    output.add_widget(write_boxes)
    outer.addWidget(output.widget)

    run = CollapsibleSection(SECTION_TITLES[4], expanded=True)
    primary = QtWidgets.QPushButton("Start counting")
    primary.setEnabled(False)
    progress = QtWidgets.QProgressBar()
    progress.setRange(0, 100)
    status = QtWidgets.QLabel("")
    status.setWordWrap(True)
    results = QtWidgets.QLabel("")
    results.setWordWrap(True)
    run.add_widget(primary)
    run.add_widget(progress)
    run.add_widget(status)
    run.add_widget(results)
    outer.addWidget(run.widget)
    outer.addStretch(1)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(container)
    dock.setWidget(scroll)

    dock.tree_counter = {
        "controller": controller,
        "runtime_label": runtime_label,
        "runtime_button": runtime_button,
        "raster_combo": raster_combo,
        "scope_combo": scope_combo,
        "polygon_combo": polygon_combo,
        "model_path": model_path,
        "browse": browse,
        "trust_button": trust_button,
        "classes": classes,
        "confidence": confidence,
        "nms_iou": nms_iou,
        "tile_size": tile_size,
        "overlap": overlap,
        "duplicate_iou": duplicate_iou,
        "device": device,
        "advanced": advanced,
        "output_path": output_path,
        "write_centers": write_centers,
        "write_boxes": write_boxes,
        "primary": primary,
        "progress": progress,
        "status": status,
        "results": results,
        "sections": (data, model, detection, output, run),
    }

    _wire(dock, controller, open_runtime_manager)
    controller.subscribe(lambda state: render(dock, state))
    return dock


def _wire(
    dock: Any, controller: CountingController, open_runtime: Any
) -> None:
    parts = dock.tree_counter

    def on_scope(index: int) -> None:
        _, kind = SCOPE_CHOICES[max(0, min(index, len(SCOPE_CHOICES) - 1))]
        layer = (
            parts["polygon_combo"].currentText()
            if kind is ScopeKind.POLYGON
            else ""
        )
        controller.set_scope(kind, layer)

    def on_classes() -> None:
        controller.set_selected_classes(checked_class_ids(parts["classes"]))

    def on_settings() -> None:
        controller.set_settings(
            InferenceSettings(
                confidence=parts["confidence"].value(),
                nms_iou=parts["nms_iou"].value(),
                duplicate_iou=parts["duplicate_iou"].value(),
                tile_size=parts["tile_size"].value(),
                overlap_percent=parts["overlap"].value(),
                selected_class_ids=controller.state.selected_class_ids,
                requested_device=parts["device"].currentText(),
            )
        )

    def on_primary() -> None:
        if controller.state.primary_action == "cancel":
            controller.cancel()
        else:
            controller.start()

    parts["raster_combo"].currentTextChanged.connect(controller.set_raster)
    parts["scope_combo"].currentIndexChanged.connect(on_scope)
    parts["polygon_combo"].currentTextChanged.connect(
        lambda name: controller.set_scope(controller.state.scope, name)
    )
    parts["classes"].itemChanged.connect(lambda _item: on_classes())
    for control in ("confidence", "nms_iou", "duplicate_iou"):
        parts[control].valueChanged.connect(lambda _v: on_settings())
    for control in ("tile_size", "overlap"):
        parts[control].valueChanged.connect(lambda _v: on_settings())
    parts["device"].currentTextChanged.connect(lambda _t: on_settings())
    parts["output_path"].textChanged.connect(controller.set_output_path)
    parts["primary"].clicked.connect(on_primary)
    parts["trust_button"].clicked.connect(controller.confirm_trust)
    if open_runtime is not None:
        parts["runtime_button"].clicked.connect(open_runtime)


def render(dock: Any, state: Any) -> None:
    """Update every widget from a controller state."""

    parts = dock.tree_counter
    parts["runtime_label"].setText(f"Runtime: {state.runtime_state}")
    parts["polygon_combo"].setEnabled(state.scope is ScopeKind.POLYGON)

    model = state.model
    parts["model_path"].setText(model.filename if model else "")
    needs_confirmation = bool(
        model and model.needs_trust and not model.trusted
    )
    parts["trust_button"].setVisible(needs_confirmation)

    names = model.class_names if model else ()
    if [
        parts["classes"].item(index).text()
        for index in range(parts["classes"].count())
    ] != list(names):
        parts["classes"].blockSignals(True)
        set_class_items(parts["classes"], names, state.selected_class_ids)
        parts["classes"].blockSignals(False)

    parts["primary"].setText(
        "Cancel" if state.primary_action == "cancel" else "Start counting"
    )
    parts["primary"].setEnabled(
        state.can_start or state.phase is Phase.RUNNING
    )
    parts["progress"].setValue(state.progress_percent)

    lines = [state.message] if state.message else []
    lines.extend(state.warnings)
    parts["status"].setText("\n".join(lines))

    if state.phase is Phase.COMPLETED:
        summary = [f"Total: {state.total_count}"]
        summary.extend(
            f"{name}: {count}"
            for name, count in state.counts_by_class.items()
        )
        summary.append(f"Output: {state.output_path}")
        parts["results"].setText("\n".join(summary))
    elif state.phase is Phase.RUNNING:
        parts["results"].setText(
            f"Tile {state.completed_tiles} of {state.total_tiles}"
        )
    else:
        parts["results"].setText("")

    for name in (
        "raster_combo",
        "scope_combo",
        "browse",
        "confidence",
        "nms_iou",
        "tile_size",
        "overlap",
        "duplicate_iou",
        "device",
        "output_path",
        "write_centers",
        "write_boxes",
        "classes",
    ):
        parts[name].setEnabled(state.controls_enabled)


def set_raster_choices(dock: Any, names: tuple[str, ...]) -> None:
    """Replace the eligible raster list without losing the selection."""

    combo = dock.tree_counter["raster_combo"]
    current = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(list(names))
    if current in names:
        combo.setCurrentText(current)
    combo.blockSignals(False)
    combo.currentTextChanged.emit(combo.currentText())


def set_polygon_choices(dock: Any, names: tuple[str, ...]) -> None:
    """Replace the polygon layer list."""

    combo = dock.tree_counter["polygon_combo"]
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(list(names))
    combo.blockSignals(False)


def set_device_choices(dock: Any, names: tuple[str, ...]) -> None:
    """Replace the device list with what the runtime actually offers."""

    combo = dock.tree_counter["device"]
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(list(names))
    combo.blockSignals(False)


__all__ = [
    "DOCK_OBJECT_NAME",
    "DOCK_TITLE",
    "SCOPE_CHOICES",
    "SECTION_TITLES",
    "build_dock",
    "render",
    "set_device_choices",
    "set_polygon_choices",
    "set_raster_choices",
]

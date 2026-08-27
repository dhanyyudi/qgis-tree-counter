"""The dock renders controller state and forwards user actions."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


class FakeTrustStore:
    def __init__(self, trusted=True) -> None:
        self._trusted = trusted

    def is_trusted(self, identity) -> bool:
        return self._trusted

    def confirm(self, identity) -> None:
        self._trusted = True


def _identity(suffix=".onnx"):
    from tree_counter.settings.trust import ModelIdentity

    return ModelIdentity(f"best{suffix}", "a" * 64, suffix)


def _controller(class_names=("oil_palm",), suffix=".onnx", trusted=True,
                runtime="ready", **kwargs):
    from tree_counter.ui.controller import CountingController

    identity = _identity(suffix)
    return CountingController(
        identify_model=lambda path: identity,
        inspect_model=lambda chosen: {
            "class_names": list(class_names),
            "backend": "onnxruntime",
        },
        trust_store=FakeTrustStore(trusted),
        runtime_status=lambda: runtime,
        **kwargs,
    )


@pytest.fixture
def dock(qgis_application):
    from tree_counter.ui.dock import build_dock

    controller = _controller()
    widget = build_dock(controller)
    yield widget
    widget.setParent(None)


def test_the_dock_has_the_approved_section_order(dock) -> None:
    from tree_counter.ui.dock import SECTION_TITLES

    titles = [
        section.toggle.text() for section in dock.tree_counter["sections"]
    ]

    assert titles == list(SECTION_TITLES)


def test_advanced_is_collapsed_by_default(dock) -> None:
    assert dock.tree_counter["advanced"].expanded is False


def test_advanced_holds_duplicate_iou_and_device(dock) -> None:
    parts = dock.tree_counter

    assert parts["duplicate_iou"] is not None
    assert parts["device"] is not None


def test_there_is_no_task_type_or_min_distance(dock) -> None:
    from qgis.PyQt.QtWidgets import QLabel

    labels = [
        widget.text().casefold()
        for widget in dock.findChildren(QLabel)
    ]

    assert not any("task type" in text for text in labels)
    assert not any("min distance" in text for text in labels)


def test_the_three_scopes_are_offered(dock) -> None:
    combo = dock.tree_counter["scope_combo"]

    assert combo.count() == 3
    assert "Whole raster" == combo.itemText(0)


def test_the_polygon_layer_control_follows_the_scope(dock) -> None:
    from tree_counter.qgis_adapter.scope import ScopeKind

    parts = dock.tree_counter
    assert parts["polygon_combo"].isEnabled() is False

    parts["controller"].set_scope(ScopeKind.POLYGON, "blocks")

    assert parts["polygon_combo"].isEnabled() is True


def test_only_eligible_rasters_are_listed(dock) -> None:
    from tree_counter.ui.dock import set_raster_choices

    set_raster_choices(dock, ("aerial", "orthophoto"))

    combo = dock.tree_counter["raster_combo"]
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "aerial",
        "orthophoto",
    ]
    assert dock.tree_counter["controller"].state.raster_name == "aerial"


def test_start_is_disabled_until_everything_is_chosen(dock) -> None:
    parts = dock.tree_counter
    assert parts["primary"].isEnabled() is False

    controller = parts["controller"]
    controller.set_raster("aerial")
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/best.onnx")

    assert parts["primary"].isEnabled() is True
    assert parts["primary"].text() == "Start counting"


def test_a_single_class_model_shows_one_checked_class(dock) -> None:
    from qgis.PyQt.QtCore import Qt

    controller = dock.tree_counter["controller"]
    controller.set_raster("aerial")
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/best.onnx")

    listing = dock.tree_counter["classes"]
    assert listing.count() == 1
    assert listing.item(0).checkState() == Qt.CheckState.Checked


def test_a_multi_class_model_shows_an_unchecked_checklist(
    qgis_application,
) -> None:
    from qgis.PyQt.QtCore import Qt

    from tree_counter.ui.dock import build_dock

    controller = _controller(class_names=("oil_palm", "shade_tree"))
    dock = build_dock(controller)
    controller.set_raster("aerial")
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/best.onnx")

    listing = dock.tree_counter["classes"]
    assert listing.count() == 2
    assert all(
        listing.item(index).checkState() == Qt.CheckState.Unchecked
        for index in range(2)
    )
    assert dock.tree_counter["primary"].isEnabled() is False
    dock.setParent(None)


def test_ticking_a_class_enables_start(qgis_application) -> None:
    from qgis.PyQt.QtCore import Qt

    from tree_counter.ui.dock import build_dock

    controller = _controller(class_names=("oil_palm", "shade_tree"))
    dock = build_dock(controller)
    controller.set_raster("aerial")
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/best.onnx")

    dock.tree_counter["classes"].item(1).setCheckState(
        Qt.CheckState.Checked
    )

    assert controller.state.selected_class_ids == (1,)
    assert dock.tree_counter["primary"].isEnabled() is True
    dock.setParent(None)


def test_an_untrusted_checkpoint_shows_a_confirm_button(
    qgis_application,
) -> None:
    from tree_counter.ui.dock import build_dock

    controller = _controller(suffix=".pt", trusted=False)
    dock = build_dock(controller)
    controller.set_raster("aerial")
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/best.pt")

    # isVisible() is False while no ancestor is shown, so isHidden() is
    # what reflects the explicit setVisible call under test.
    assert dock.tree_counter["trust_button"].isHidden() is False
    assert dock.tree_counter["primary"].isEnabled() is False

    dock.tree_counter["trust_button"].click()

    assert dock.tree_counter["trust_button"].isHidden() is True
    assert dock.tree_counter["primary"].isEnabled() is True
    dock.setParent(None)


def test_running_switches_the_button_to_cancel(qgis_application) -> None:
    from tree_counter.ui.dock import build_dock

    cancelled: list[int] = []
    controller = _controller(
        start_run=lambda state: None, cancel_run=lambda: cancelled.append(1)
    )
    dock = build_dock(controller)
    controller.set_raster("aerial")
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/best.onnx")

    controller.start()

    parts = dock.tree_counter
    assert parts["primary"].text() == "Cancel"
    assert parts["raster_combo"].isEnabled() is False

    parts["primary"].click()
    assert cancelled == [1]
    dock.setParent(None)


def test_progress_and_results_are_shown(qgis_application) -> None:
    from tree_counter.qgis_adapter.task import RunResult
    from tree_counter.ui.dock import build_dock

    controller = _controller(start_run=lambda state: None)
    dock = build_dock(controller)
    controller.set_raster("aerial")
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/best.onnx")
    controller.start()

    controller.on_event(
        {"type": "progress", "completed_tiles": 5, "total_tiles": 10}
    )
    assert dock.tree_counter["progress"].value() == 50

    result = RunResult(run_id="run-1")
    controller.on_completed(result, "/tmp/out.gpkg")

    text = dock.tree_counter["results"].text()
    assert "Total: 0" in text
    assert "/tmp/out.gpkg" in text
    assert dock.tree_counter["raster_combo"].isEnabled() is True
    dock.setParent(None)


def test_warnings_reach_the_status_area(qgis_application) -> None:
    from tree_counter.ui.dock import build_dock

    controller = _controller(start_run=lambda state: None)
    dock = build_dock(controller)
    controller.set_raster("aerial")
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/best.onnx")
    controller.start()

    controller.on_event(
        {"type": "warning", "message": "Falling back to the CPU."}
    )

    assert "CPU" in dock.tree_counter["status"].text()
    dock.setParent(None)


def test_the_runtime_state_is_shown_in_the_header(dock) -> None:
    assert "ready" in dock.tree_counter["runtime_label"].text()


def test_the_dock_translates_the_runtime_state_value(qgis_application) -> None:
    from qgis.PyQt.QtCore import QCoreApplication

    from tree_counter.i18n import install_translator
    from tree_counter.ui.dock import build_dock

    app = QCoreApplication.instance()
    translator = install_translator(app, locale="id_ID")
    assert translator is not None
    dock = build_dock(_controller(runtime="repair_required"))
    try:
        assert dock.tree_counter["runtime_label"].text() == (
            "Runtime: perlu perbaikan"
        )
    finally:
        app.removeTranslator(translator)
        dock.setParent(None)


def test_the_output_checkboxes_update_the_controller(dock) -> None:
    parts = dock.tree_counter
    controller = parts["controller"]

    parts["write_boxes"].setChecked(True)
    assert controller.state.write_boxes is True

    parts["write_centers"].setChecked(False)
    assert controller.state.write_centers is False


def test_unticking_both_layers_disables_start(dock) -> None:
    parts = dock.tree_counter
    controller = parts["controller"]
    controller.set_raster("aerial")
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/best.onnx")

    assert parts["primary"].isEnabled() is True

    parts["write_centers"].setChecked(False)

    assert parts["primary"].isEnabled() is False


def test_the_runtime_button_opens_the_manager(qgis_application) -> None:
    from tree_counter.ui.dock import build_dock

    opened: list[int] = []
    controller = _controller()
    dock = build_dock(
        controller, open_runtime_manager=lambda: opened.append(1)
    )

    dock.tree_counter["runtime_button"].click()

    assert opened == [1]
    dock.setParent(None)


def test_browse_asks_the_host_to_choose_a_model(qgis_application) -> None:
    """The Browse button was created but never connected to anything.

    Without this the only way to select a model was unavailable, so a run
    could never start no matter what else the user set.
    """

    from tree_counter.ui.dock import build_dock

    chosen: list[int] = []
    dock = build_dock(_controller(), choose_model=lambda: chosen.append(1))
    try:
        dock.tree_counter["browse"].click()
    finally:
        dock.setParent(None)

    assert chosen == [1]


def test_the_dock_lists_the_offered_layers(qgis_application) -> None:
    """Both layer combos are populated from the project."""

    from tree_counter.ui.dock import (
        build_dock,
        set_polygon_choices,
        set_raster_choices,
    )

    dock = build_dock(_controller())
    try:
        set_raster_choices(dock, ("aerial", "orthophoto"))
        set_polygon_choices(dock, ("blocks",))
        raster = dock.tree_counter["raster_combo"]
        polygon = dock.tree_counter["polygon_combo"]

        assert [
            raster.itemText(index) for index in range(raster.count())
        ] == ["aerial", "orthophoto"]
        assert [
            polygon.itemText(index) for index in range(polygon.count())
        ] == ["blocks"]
    finally:
        dock.setParent(None)

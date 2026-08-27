"""Accessible names and keyboard reachability for the dock."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


class _Trust:
    def is_trusted(self, identity) -> bool:
        return True

    def confirm(self, identity) -> None:
        return None


def _controller():
    from tree_counter.settings.trust import ModelIdentity
    from tree_counter.ui.controller import CountingController

    identity = ModelIdentity("best.onnx", "a" * 64, ".onnx")
    return CountingController(
        identify_model=lambda path: identity,
        inspect_model=lambda chosen: {
            "class_names": ["oil_palm"],
            "backend": "onnxruntime",
        },
        trust_store=_Trust(),
        runtime_status=lambda: "ready",
    )


@pytest.fixture
def dock(qgis_application):
    from tree_counter.ui.dock import build_dock

    widget = build_dock(_controller())
    yield widget
    widget.setParent(None)


def _ready(dock):
    controller = dock.tree_counter["controller"]
    controller.set_raster("aerial")
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/best.onnx")


def test_every_control_has_a_label_or_accessible_name(dock) -> None:
    parts = dock.tree_counter
    labelled = (
        "raster_combo",
        "scope_combo",
        "polygon_combo",
        "model_path",
        "classes",
        "confidence",
        "nms_iou",
        "tile_size",
        "overlap",
        "duplicate_iou",
        "device",
        "output_path",
    )
    for name in labelled:
        assert parts[name].accessibleName(), name

    titled = (
        "browse",
        "trust_button",
        "write_centers",
        "write_boxes",
        "primary",
    )
    for name in titled:
        control = parts[name]
        assert control.text() or control.accessibleName(), name


def test_every_actionable_control_is_keyboard_reachable(dock) -> None:
    from qgis.PyQt.QtCore import Qt

    parts = dock.tree_counter
    controls = (
        "raster_combo",
        "scope_combo",
        "browse",
        "trust_button",
        "classes",
        "confidence",
        "nms_iou",
        "tile_size",
        "overlap",
        "duplicate_iou",
        "device",
        "output_path",
        "write_centers",
        "write_boxes",
        "primary",
    )
    for name in controls:
        control = parts[name]
        assert control.focusPolicy() & Qt.FocusPolicy.TabFocus, name


def test_the_primary_button_is_reachable_and_activatable(dock) -> None:
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtTest import QTest

    from tree_counter.ui.controller import Phase

    parts = dock.tree_counter
    controller = parts["controller"]
    primary = parts["primary"]

    _ready(dock)
    assert primary.isEnabled() is True
    assert primary.focusPolicy() & Qt.FocusPolicy.TabFocus

    primary.setFocus()
    QTest.keyClick(primary, Qt.Key.Key_Space)

    assert controller.state.phase is Phase.RUNNING
    assert primary.text() == "Cancel"

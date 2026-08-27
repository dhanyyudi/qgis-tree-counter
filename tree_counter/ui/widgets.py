"""Small reusable widgets built through ``qgis.PyQt``.

Widgets are constructed in code rather than loaded from generated bindings,
because generated files import PyQt directly and would break the single
package that must work on both Qt5 and Qt6.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any


def _widgets() -> Any:
    from qgis.PyQt import QtWidgets

    return QtWidgets


def _core() -> Any:
    from qgis.PyQt import QtCore

    return QtCore


class CollapsibleSection:
    """A titled group whose body can be folded away.

    Built from a checkable tool button and a container rather than a
    QGIS-specific collapsible group box, so it behaves identically on
    QGIS 3 and QGIS 4.
    """

    def __init__(self, title: str, expanded: bool = True) -> None:
        widgets = _widgets()
        core = _core()

        self.widget = widgets.QWidget()
        layout = widgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.toggle = widgets.QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(bool(expanded))
        self.toggle.setStyleSheet("QToolButton { border: none; }")
        self.toggle.setToolButtonStyle(
            core.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.toggle.setArrowType(
            core.Qt.ArrowType.DownArrow
            if expanded
            else core.Qt.ArrowType.RightArrow
        )
        layout.addWidget(self.toggle)

        self.body = widgets.QWidget()
        self.body_layout = widgets.QFormLayout(self.body)
        self.body_layout.setContentsMargins(12, 0, 0, 0)
        self.body.setVisible(bool(expanded))
        layout.addWidget(self.body)

        self.toggle.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        core = _core()
        self.body.setVisible(bool(checked))
        self.toggle.setArrowType(
            core.Qt.ArrowType.DownArrow
            if checked
            else core.Qt.ArrowType.RightArrow
        )

    @property
    def expanded(self) -> bool:
        """Return whether the body is visible."""

        return bool(self.toggle.isChecked())

    def set_expanded(self, expanded: bool) -> None:
        """Fold or unfold the body."""

        self.toggle.setChecked(bool(expanded))

    def add_row(self, label: str, widget: Any) -> None:
        """Add a labelled row to the body.

        The label becomes the field's buddy and its accessible name, so a
        screen reader announces what each control is for.
        """

        self.body_layout.addRow(label, widget)
        widget.setAccessibleName(str(label))

    def add_widget(self, widget: Any) -> None:
        """Add a full-width widget to the body."""

        self.body_layout.addRow(widget)


def make_double_control(
    minimum: float,
    maximum: float,
    step: float,
    value: float,
    decimals: int = 2,
) -> Any:
    """Return a bounded spin box for a fractional setting."""

    widgets = _widgets()
    box = widgets.QDoubleSpinBox()
    box.setRange(float(minimum), float(maximum))
    box.setSingleStep(float(step))
    box.setDecimals(int(decimals))
    box.setValue(float(value))
    return box


def make_int_control(
    minimum: int, maximum: int, step: int, value: int
) -> Any:
    """Return a bounded spin box for a whole-number setting."""

    widgets = _widgets()
    box = widgets.QSpinBox()
    box.setRange(int(minimum), int(maximum))
    box.setSingleStep(int(step))
    box.setValue(int(value))
    return box


def make_class_list() -> Any:
    """Return the multi-class checklist widget."""

    widgets = _widgets()
    listing = widgets.QListWidget()
    listing.setSelectionMode(
        widgets.QAbstractItemView.SelectionMode.NoSelection
    )
    listing.setMaximumHeight(120)
    return listing


def set_class_items(
    listing: Any, names: tuple[str, ...], checked: tuple[int, ...]
) -> None:
    """Fill the checklist, checking the selected classes."""

    core = _core()
    widgets = _widgets()
    listing.clear()
    for index, name in enumerate(names):
        item = widgets.QListWidgetItem(str(name))
        item.setFlags(item.flags() | core.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            core.Qt.CheckState.Checked
            if index in checked
            else core.Qt.CheckState.Unchecked
        )
        listing.addItem(item)


def checked_class_ids(listing: Any) -> tuple[int, ...]:
    """Return the indices the user has ticked."""

    core = _core()
    return tuple(
        index
        for index in range(listing.count())
        if listing.item(index).checkState() == core.Qt.CheckState.Checked
    )


__all__ = [
    "CollapsibleSection",
    "checked_class_ids",
    "make_class_list",
    "make_double_control",
    "make_int_control",
    "set_class_items",
]

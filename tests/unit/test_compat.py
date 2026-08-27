"""Tests for the QGIS 3/4 compatibility boundary."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_compat_source_imports_qt_only_from_qgis_pyqt() -> None:
    source = (ROOT / "tree_counter" / "compat.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    imports = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    for node in imports:
        module_name = getattr(node, "module", "") or ""
        assert not any(
            alias.name in {"PyQt5", "PyQt6", "PySide2", "PySide6"}
            for alias in getattr(node, "names", [])
        )
        if module_name.startswith("qgis"):
            assert module_name == "qgis.PyQt" or module_name.startswith(
                "qgis.PyQt."
            )
    assert "ultralytics" not in source
    assert "torch" not in source
    assert "onnxruntime" not in source


def test_compat_imports_without_qgis_and_exposes_helpers() -> None:
    compat = importlib.import_module("tree_counter.compat")

    assert compat.qt_available() is False
    assert compat.qgis_version() is None
    assert compat.qt_exec(object) is None


# Qt classes the plugin touches, and the attributes it may read straight
# off the class. Anything else that starts with a capital is almost
# certainly an unscoped Qt5 enum member, which the official PyQGIS 4
# checker rejects and which breaks under PyQt6.
QT_CLASS_DIRECT_ATTRIBUTES = {
    "QStandardPaths": {
        "StandardLocation",
        "displayName",
        "findExecutable",
        "locate",
        "standardLocations",
        "writableLocation",
    },
    "QtCore": {
        "pyqtSignal",
        "pyqtSlot",
        "QObject",
        "QStandardPaths",
        "Qt",
    },
}


def _shipped_sources():
    import pathlib

    package = pathlib.Path(__file__).resolve().parents[2] / "tree_counter"
    return sorted(package.rglob("*.py"))


def test_qt_enum_members_are_always_scoped() -> None:
    """Qt5 allows ``QStandardPaths.AppDataLocation``; PyQt6 does not.

    The CI checker catches this, but it only runs remotely, so the same
    rule is enforced here against every shipped source file.
    """

    import ast

    offenders: list[str] = []
    for path in _shipped_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            value = node.value
            if not isinstance(value, ast.Name):
                continue
            allowed = QT_CLASS_DIRECT_ATTRIBUTES.get(value.id)
            if allowed is None:
                continue
            if node.attr in allowed or not node.attr[:1].isupper():
                continue
            offenders.append(
                f"{path.name}:{node.lineno} {value.id}.{node.attr}"
            )

    assert offenders == [], (
        "unscoped Qt enum access; use the scoped form, for example "
        "QStandardPaths.StandardLocation.AppDataLocation: "
        + ", ".join(offenders)
    )


def test_the_guard_detects_an_unscoped_enum() -> None:
    import ast

    tree = ast.parse("QStandardPaths.AppDataLocation\n")
    node = next(
        item for item in ast.walk(tree) if isinstance(item, ast.Attribute)
    )

    allowed = QT_CLASS_DIRECT_ATTRIBUTES["QStandardPaths"]
    assert node.attr not in allowed
    assert node.attr[:1].isupper()


def test_the_settings_path_uses_the_scoped_standard_location() -> None:
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[2]
        / "tree_counter"
        / "settings"
        / "store.py"
    ).read_text(encoding="utf-8")

    assert "QStandardPaths.StandardLocation.AppDataLocation" in source
    assert "QStandardPaths.AppDataLocation" not in source

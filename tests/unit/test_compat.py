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

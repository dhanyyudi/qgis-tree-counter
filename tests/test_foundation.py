"""Tests for the public Tree Counter plugin foundation."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import ast
import configparser
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "tree_counter"


def test_class_factory_constructs_the_plugin_and_preserves_iface() -> None:
    import tree_counter

    from tree_counter.plugin import TreeCounterPlugin

    sentinel_iface = object()
    plugin = tree_counter.classFactory(sentinel_iface)

    # Construction must not touch QGIS; the dock and the toolbar action
    # are created in initGui, which the QGIS tests exercise for real.
    assert isinstance(plugin, TreeCounterPlugin)
    assert plugin.iface is sentinel_iface
    assert plugin.dock is None
    assert plugin.action is None


def test_metadata_uses_the_public_plugin_identity() -> None:
    metadata = configparser.ConfigParser()
    metadata.read(PACKAGE / "metadata.txt", encoding="utf-8")
    values = metadata["general"]

    assert values["name"] == "Tree Counter"
    assert values["qgisminimumversion"] == "3.44"
    assert values["qgismaximumversion"] == "4.99"
    assert values["description"] == (
        "Count trees in georeferenced aerial imagery with user-provided "
        "YOLO models."
    )
    # The exact wording is locked in scripts/check_publication.py; what
    # matters here is that every disclosure a user needs before installing
    # is actually present in the text QGIS shows them.
    from scripts.check_publication import REQUIRED_ABOUT_DISCLOSURES

    about = values["about"].lower()
    missing = [
        phrase
        for phrase, _message in REQUIRED_ABOUT_DISCLOSURES
        if phrase not in about
    ]
    assert missing == [], f"about text is missing: {missing}"
    assert values["version"] == "0.1.0"
    assert values["author"] == "Dhany Yudi Prasetyo"
    assert values["email"] == "dhanyyudi.prasetyo@gmail.com"
    assert values["homepage"] == (
        "https://github.com/dhanyyudi/qgis-tree-counter"
    )
    assert values["repository"] == (
        "https://github.com/dhanyyudi/qgis-tree-counter"
    )
    assert values["tracker"] == (
        "https://github.com/dhanyyudi/qgis-tree-counter/issues"
    )
    assert values["license"] == "AGPL-3.0-only"
    assert values["experimental"] == "True"
    assert values["deprecated"] == "False"
    assert values["hasprocessingprovider"] == "no"
    assert values["server"] == "False"


def test_root_and_package_licenses_are_identical() -> None:
    root_license = ROOT / "LICENSE"
    package_license = PACKAGE / "LICENSE"

    assert root_license.is_file()
    assert package_license.is_file()
    assert root_license.read_bytes() == package_license.read_bytes()


def test_class_factory_imports_plugin_lazily() -> None:
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    module = ast.parse(source)

    assert all(
        not isinstance(node, (ast.Import, ast.ImportFrom))
        or not any(
            alias.name.endswith("plugin")
            for alias in getattr(node, "names", [])
        )
        for node in module.body
    )
    factory = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "classFactory"
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "plugin"
        and node.level == 1
        for node in ast.walk(factory)
    )


def test_the_plugin_imports_qt_only_through_qgis() -> None:
    """Qt must never be imported directly, or the single package breaks.

    One archive has to load under Qt5 and Qt6, which only holds while
    every Qt import goes through qgis.PyQt.
    """

    source = (PACKAGE / "plugin.py").read_text(encoding="utf-8")
    module = ast.parse(source)

    def is_allowed(name: str | None) -> bool:
        if not name:
            return False
        if name == "qgis.PyQt" or name.startswith("qgis.PyQt."):
            return True
        if name == "tree_counter" or name.startswith("tree_counter."):
            return True
        return name in ("__future__", "pathlib", "typing")

    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            assert all(is_allowed(alias.name) for alias in node.names), (
                [alias.name for alias in node.names]
            )
        elif isinstance(node, ast.ImportFrom):
            assert is_allowed(node.module), node.module

    for forbidden in ("PyQt5", "PyQt6", "requests", "urllib", "subprocess"):
        assert forbidden not in source, forbidden

    class_names = {
        node.name
        for node in module.body
        if isinstance(node, ast.ClassDef)
    }
    assert "TreeCounterPlugin" in class_names


def _tracked_public_text() -> list[tuple[Path, str]]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    tracked = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = ROOT / os.fsdecode(raw_path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        tracked.append((path, text))
    return tracked


def _forbidden_public_text(text: str) -> bool:
    lowered = text.casefold()
    return any(
        marker in lowered
        for marker in (
            "/" + "users/",
            "agents" + ".md",
            "docs/" + "internal",
            "." + "superpowers",
            "graphify-" + "out",
        )
    )


def test_all_tracked_public_text_files_do_not_leak_internal_paths() -> None:
    for path, text in _tracked_public_text():
        assert not _forbidden_public_text(text), path


@pytest.mark.parametrize(
    "forbidden_text",
    [
        "/" + "Users/maintainer/private",
        "AGENTS" + ".md",
        "agents" + ".MD",
        "docs/" + "internal/plan.md",
        "." + "superpowers/sdd/plan.md",
        "graphify-" + "out/report.json",
    ],
)
def test_public_text_scanner_catches_forbidden_content(
    forbidden_text: str,
) -> None:
    assert _forbidden_public_text(f"public text with {forbidden_text}")

"""Tests for the public Tree Counter plugin foundation."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import ast
import configparser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "tree_counter"


def test_class_factory_constructs_inert_shell_and_preserves_iface() -> None:
    import tree_counter

    from tree_counter.plugin import TreeCounterPlugin

    sentinel_iface = object()
    plugin = tree_counter.classFactory(sentinel_iface)

    assert isinstance(plugin, TreeCounterPlugin)
    assert plugin.iface is sentinel_iface
    assert plugin.initGui() is None
    assert plugin.unload() is None


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
    assert values["about"] == (
        "Open-source QGIS plugin foundation for local tree counting in "
        "georeferenced aerial imagery. Under active development; not ready "
        "for production installation."
    )
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


def test_shell_is_inert_and_does_not_import_non_qgis_ui_dependencies() -> None:
    source = (PACKAGE / "plugin.py").read_text(encoding="utf-8")
    module = ast.parse(source)

    def is_allowed_qgis_pyqt(name: str | None) -> bool:
        return bool(name) and (
            name == "qgis.PyQt" or name.startswith("qgis.PyQt.")
        )

    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            assert all(
                is_allowed_qgis_pyqt(alias.name) for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert is_allowed_qgis_pyqt(node.module)

    assert "requests" not in source
    assert "urllib" not in source
    assert "subprocess" not in source
    assert "QDockWidget" not in source
    assert "addToolBarIcon" not in source

    class_names = {
        node.name
        for node in module.body
        if isinstance(node, ast.ClassDef)
    }
    assert "TreeCounterPlugin" in class_names


def test_public_files_do_not_leak_internal_paths_or_instructions() -> None:
    public_files = [
        ROOT / "README.md",
        ROOT / "README.id.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "CHANGELOG.md",
        ROOT / "LICENSE",
        PACKAGE / "LICENSE",
        PACKAGE / "metadata.txt",
        PACKAGE / "__init__.py",
        PACKAGE / "plugin.py",
    ]
    forbidden = ("/" + "Users/", "AGENTS" + ".md", "docs/" + "internal")

    for path in public_files:
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path

"""Policy tests for the blocking quality and release workflows."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _workflow(name: str) -> str:
    path = WORKFLOWS / name
    assert path.is_file(), f"missing workflow: {path}"
    return path.read_text(encoding="utf-8")


def _required_commands(workflow: str) -> None:
    for command in (
        "pytest",
        "flake8",
        "bandit",
        "detect-secrets",
        "scripts/check_publication.py",
        "scripts/package_plugin.py",
    ):
        assert command in workflow, f"workflow omits required gate: {command}"


def test_quality_workflow_has_blocking_triggers_permissions_gates_and_artifact(
) -> None:
    workflow = _workflow("quality.yml")

    assert re.search(r"(?m)^\s*pull_request:\s*$", workflow)
    assert re.search(r"(?m)^\s*push:\s*$", workflow)
    assert re.search(r"(?m)^\s*permissions:\s*$", workflow)
    assert re.search(r"(?m)^\s+contents:\s*read\s*$", workflow)
    assert "python-version: '3.12'" in workflow
    _required_commands(workflow)
    assert "actions/upload-artifact" in workflow
    archive_pattern = (
        "dist/tree_counter-${{ steps.version.outputs."
        "version }}.zip"
    )
    assert archive_pattern in workflow


def test_release_workflow_is_tagged_minimal_and_repeats_quality_gates(
) -> None:
    workflow = _workflow("release.yml")

    assert re.search(r"(?m)^\s*tags:\s*$", workflow)
    assert re.search(r"(?m)^\s+-\s+['\"]?v\*['\"]?\s*$", workflow)
    assert re.search(r"(?m)^\s*permissions:\s*$", workflow)
    assert re.search(r"(?m)^\s+contents:\s*write\s*$", workflow)
    assert re.search(r"(?m)^\s*needs:\s*quality\s*$", workflow)
    assert "actions/create-release" not in workflow
    assert "gh release create" in workflow
    assert "tree_counter-${{ steps.version.outputs.version }}.zip" in workflow
    _required_commands(workflow)
    assert "actions/upload-artifact" not in workflow
    assert "plugins.qgis.org" not in workflow.lower()
    assert "OSGEO" not in workflow.upper()
    assert "qgis_password" not in workflow.lower()


def test_only_release_workflow_can_request_contents_write() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if path.name == "release.yml":
            assert re.search(r"(?m)^\s+contents:\s*write\s*$", text)
        else:
            assert not re.search(r"(?m)^\s+contents:\s*write\s*$", text)


@pytest.mark.parametrize(
    "config",
    ["pyproject.toml", ".flake8", ".bandit", ".secrets.baseline"],
)
def test_development_tool_configuration_is_committed(config: str) -> None:
    assert (ROOT / config).is_file()


def test_quality_workflow_has_no_external_publish_target() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = path.read_text(encoding="utf-8").lower()
        assert "plugins.qgis.org" not in workflow
        assert "osgeo" not in workflow

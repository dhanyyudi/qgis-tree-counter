"""Tests for the environment-only local integration entry point."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys

import pytest


def test_cli_selects_only_scope_and_backends() -> None:
    """The CLI accepts control flags without accepting user data paths."""

    from scripts.run_local_integration import parse_args

    arguments = parse_args(
        ["--scope", "full", "--backends", "pt,onnx"]
    )

    assert arguments.scope == "full"
    assert arguments.backends == ("pt", "onnx")


def test_qgis_tests_read_the_selected_scope_and_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.run_local_integration import (
        selected_backends,
        selected_scope,
    )

    monkeypatch.setenv("TREE_COUNTER_TEST_SCOPE", "full")
    monkeypatch.setenv("TREE_COUNTER_TEST_BACKENDS", "onnx")

    assert selected_scope() == "full"
    assert selected_backends() == ("onnx",)


def test_environment_reports_missing_names_without_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing errors identify variables without leaking paths."""

    from scripts.run_local_integration import (
        IntegrationConfigurationError,
        load_environment,
    )

    monkeypatch.delenv("TREE_COUNTER_TEST_MODEL_PT", raising=False)
    monkeypatch.delenv("TREE_COUNTER_TEST_RASTER", raising=False)
    monkeypatch.delenv("TREE_COUNTER_TEST_OUTPUT_DIR", raising=False)

    with pytest.raises(IntegrationConfigurationError) as raised:
        load_environment(("pt",))

    message = str(raised.value)
    assert "TREE_COUNTER_TEST_MODEL_PT" in message
    assert "TREE_COUNTER_TEST_RASTER" in message
    assert "TREE_COUNTER_TEST_OUTPUT_DIR" in message
    assert "/" not in message
    assert "\\" not in message


def test_output_directory_is_not_created_by_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reading the environment does not write an integration workspace."""

    from scripts.run_local_integration import load_environment

    model = tmp_path / "model.pt"
    raster = tmp_path / "raster.tif"
    output = tmp_path / "results"
    model.write_bytes(b"model")
    raster.write_bytes(b"raster")
    monkeypatch.setenv("TREE_COUNTER_TEST_MODEL_PT", str(model))
    monkeypatch.setenv("TREE_COUNTER_TEST_RASTER", str(raster))
    monkeypatch.setenv("TREE_COUNTER_TEST_OUTPUT_DIR", str(output))

    environment = load_environment(("pt",))

    assert environment.output_dir == output
    assert not output.exists()


def test_main_runs_selected_real_tests_inside_qgis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import run_local_integration

    model = tmp_path / "model.onnx"
    raster = tmp_path / "raster.tif"
    output = tmp_path / "results"
    model.write_bytes(b"model")
    raster.write_bytes(b"raster")
    monkeypatch.setenv("TREE_COUNTER_TEST_MODEL_ONNX", str(model))
    monkeypatch.setenv("TREE_COUNTER_TEST_RASTER", str(raster))
    monkeypatch.setenv("TREE_COUNTER_TEST_OUTPUT_DIR", str(output))

    calls: list[list[str]] = []
    runner = ModuleType("scripts.run_qgis_tests")
    runner.main = lambda arguments: calls.append(list(arguments)) or 0
    monkeypatch.setitem(sys.modules, "scripts.run_qgis_tests", runner)
    monkeypatch.setattr(
        pytest,
        "main",
        lambda arguments: pytest.fail(
            "the integration harness ran under the host Python"
        ),
    )

    result = run_local_integration.main(
        ["--scope", "full", "--backends", "onnx"]
    )

    assert result == 0
    assert calls == [
        ["--", "tests/qgis/test_real_run.py", "-q", "-rs"]
    ]
    assert run_local_integration.os.environ[
        "TREE_COUNTER_TEST_SCOPE"
    ] == "full"
    assert run_local_integration.os.environ[
        "TREE_COUNTER_TEST_BACKENDS"
    ] == "onnx"

"""Tests for the environment-only local integration entry point."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path

import pytest


def test_cli_selects_only_scope_and_backends() -> None:
    """The CLI accepts control flags without accepting user data paths."""

    from scripts.run_local_integration import parse_args

    arguments = parse_args(
        ["--scope", "full", "--backends", "pt,onnx"]
    )

    assert arguments.scope == "full"
    assert arguments.backends == ("pt", "onnx")


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

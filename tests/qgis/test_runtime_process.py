"""Tests for the QProcess-backed Runtime Manager runner."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import time
from pathlib import Path

import pytest


def _supported_python(runner) -> str:
    """Return a supported standalone Python for the process tests."""

    import os

    from tree_counter.runtime.python_probe import (
        discover_candidates,
        probe_python,
    )

    for candidate in discover_candidates(environment=os.environ):
        if probe_python(candidate, runner).is_supported:
            return candidate
    pytest.skip("no supported standalone Python interpreter is available")


def test_runner_captures_exit_code_and_separate_streams(
    qgis_application,
) -> None:
    """A short real process returns its code, stdout, and stderr."""

    from tree_counter.qgis_adapter.runtime_process import QProcessRunner

    runner = QProcessRunner()
    result = runner(
        (
            _supported_python(runner),
            "-I",
            "-c",
            "import sys; print('runner-stdout'); "
            "print('runner-stderr', file=sys.stderr)",
        ),
        10.0,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "runner-stdout"
    assert result.stderr.strip() == "runner-stderr"


def test_runner_kills_a_process_that_exceeds_its_timeout(
    qgis_application, tmp_path: Path
) -> None:
    """A timed-out child cannot continue and write after the runner returns."""

    from tree_counter.qgis_adapter.runtime_process import QProcessRunner

    runner = QProcessRunner()
    marker = tmp_path / "child-finished"
    result = runner(
        (
            _supported_python(runner),
            "-I",
            "-c",
            (
                "import pathlib,sys,time; time.sleep(1); "
                "pathlib.Path(sys.argv[1]).write_text('finished')"
            ),
            str(marker),
        ),
        0.05,
    )

    assert result.returncode != 0
    time.sleep(0.2)
    assert not marker.exists()


def test_plugin_installer_uses_a_real_qprocess_runner(
    qgis_application,
) -> None:
    """The production plugin must not wire Runtime Manager to a refusal."""

    from tree_counter.plugin import TreeCounterPlugin
    from tree_counter.qgis_adapter.runtime_process import QProcessRunner

    plugin = TreeCounterPlugin(None)
    installer = plugin._installer()

    assert isinstance(installer._runner, QProcessRunner)
    assert not hasattr(plugin, "_unavailable_runner")

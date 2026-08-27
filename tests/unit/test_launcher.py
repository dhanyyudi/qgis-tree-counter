"""The worker launch command is built only from a verified runtime."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tree_counter.runtime.installer import RuntimeStatus
from tree_counter.runtime.paths import RuntimePaths, RuntimeState


def _status(state: RuntimeState = RuntimeState.READY) -> RuntimeStatus:
    return RuntimeStatus(state=state)


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(tmp_path / "runtime")


def _interpreter(paths: RuntimePaths, windows: bool = False) -> Path:
    exe = (
        paths.active / "Scripts" / "python.exe"
        if windows
        else paths.active / "bin" / "python"
    )
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")
    return exe


def _bootstrap(tmp_path: Path) -> Path:
    path = tmp_path / "worker_bootstrap.py"
    path.write_text("", encoding="utf-8")
    return path


def test_the_command_is_exactly_interpreter_isolated_bootstrap(
    tmp_path: Path,
) -> None:
    from tree_counter.qgis_adapter.launcher import build_worker_command

    paths = _paths(tmp_path)
    exe = _interpreter(paths)
    bootstrap = _bootstrap(tmp_path)

    command = build_worker_command(_status(), paths, bootstrap=bootstrap)

    assert command == [str(exe), "-I", str(bootstrap)]


def test_the_interpreter_comes_from_the_runtime_not_sys_executable(
    tmp_path: Path,
) -> None:
    from tree_counter.qgis_adapter.launcher import build_worker_command

    paths = _paths(tmp_path)
    exe = _interpreter(paths)

    command = build_worker_command(
        _status(), paths, bootstrap=_bootstrap(tmp_path)
    )

    assert command[0] == str(exe)
    assert command[0] != sys.executable


def test_the_windows_interpreter_layout_is_supported(
    tmp_path: Path,
) -> None:
    from tree_counter.qgis_adapter.launcher import build_worker_command

    paths = _paths(tmp_path)
    exe = _interpreter(paths, windows=True)

    command = build_worker_command(
        _status(), paths, bootstrap=_bootstrap(tmp_path)
    )

    assert command[0] == str(exe)


def test_a_runtime_that_is_not_ready_is_refused(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.launcher import (
        LauncherError,
        build_worker_command,
    )

    paths = _paths(tmp_path)
    _interpreter(paths)

    with pytest.raises(LauncherError):
        build_worker_command(
            _status(RuntimeState.NOT_INSTALLED),
            paths,
            bootstrap=_bootstrap(tmp_path),
        )


def test_a_missing_interpreter_is_refused(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.launcher import (
        LauncherError,
        build_worker_command,
    )

    paths = _paths(tmp_path)

    with pytest.raises(LauncherError):
        build_worker_command(
            _status(), paths, bootstrap=_bootstrap(tmp_path)
        )


def test_a_missing_bootstrap_is_refused(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.launcher import (
        LauncherError,
        build_worker_command,
    )

    paths = _paths(tmp_path)
    _interpreter(paths)

    with pytest.raises(LauncherError):
        build_worker_command(
            _status(), paths, bootstrap=tmp_path / "missing.py"
        )


def test_the_default_bootstrap_lives_inside_the_plugin() -> None:
    from tree_counter.qgis_adapter.launcher import plugin_bootstrap

    bootstrap = plugin_bootstrap()

    assert bootstrap.name == "worker_bootstrap.py"
    assert bootstrap.is_file()

"""Building the worker launch command from a verified runtime.

The isolated worker is started with a fixed argument vector built from the
runtime's own interpreter and the plugin's bootstrap script. Nothing is
taken from ``sys.executable``: QGIS's Python must never be used as the
worker, so a runtime that is missing, broken, or not marked ready is
refused here rather than guessed at.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from typing import Any

from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.runtime.paths import RuntimePaths, RuntimeState

BOOTSTRAP_RELATIVE = ("runtime", "worker_bootstrap.py")


class LauncherError(TreeCounterError):
    """A worker command could not be built from the runtime."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            ErrorCode.MISSING_RUNTIME, diagnostic_detail=detail
        )


def venv_python(active: Path) -> Path:
    """Return the interpreter inside an activated runtime directory."""

    windows = Path(active) / "Scripts" / "python.exe"
    if windows.exists():
        return windows
    return Path(active) / "bin" / "python"


def plugin_bootstrap() -> Path:
    """Return the bootstrap script shipped with the plugin package."""

    import tree_counter

    root = Path(tree_counter.__file__).resolve().parent
    return root.joinpath(*BOOTSTRAP_RELATIVE)


def build_worker_command(
    status: Any,
    paths: RuntimePaths,
    bootstrap: Path | str | None = None,
) -> list[str]:
    """Return the worker argv, or raise when the runtime cannot launch it.

    The command is exactly ``[interpreter, "-I", bootstrap]`` with nothing
    else: no shell, no extra arguments, and no interpolation.
    """

    state = getattr(status, "state", None)
    if state is not RuntimeState.READY:
        raise LauncherError("the runtime is not ready to run a worker")
    interpreter = venv_python(paths.active)
    if not interpreter.is_file():
        raise LauncherError("the runtime interpreter is missing")
    script = plugin_bootstrap() if bootstrap is None else Path(bootstrap)
    if not script.is_file():
        raise LauncherError("the worker bootstrap script is missing")
    return [str(interpreter), "-I", str(script)]


__all__ = [
    "BOOTSTRAP_RELATIVE",
    "LauncherError",
    "build_worker_command",
    "plugin_bootstrap",
    "venv_python",
]

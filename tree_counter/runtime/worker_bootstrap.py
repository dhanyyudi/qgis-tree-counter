"""Fixed entry script that launches the worker in an isolated interpreter.

The runtime interpreter is started as
``runtime_python -I .../tree_counter/runtime/worker_bootstrap.py``. This
script resolves its own installed plugin parent directory, verifies that the
expected ``tree_counter`` packages really are children of that parent, puts
only that one directory on ``sys.path``, and runs the worker module. Nothing
is taken from ``PYTHONPATH``, the current directory, a QGIS site-packages
tree, or a dynamically built Python command string.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PACKAGE_NAME = "tree_counter"
WORKER_MODULE = "tree_counter.worker"
REQUIRED_PACKAGE_FILES = (
    "__init__.py",
    "core/__init__.py",
    "core/protocol.py",
    "worker/__init__.py",
    "worker/__main__.py",
    "worker/runner.py",
)


class BootstrapError(RuntimeError):
    """The bootstrap refuses to run against an unexpected layout."""


def resolve_plugin_parent(script_path: str | Path) -> Path:
    """Return the verified directory that contains the plugin package.

    The layout must be ``<parent>/tree_counter/runtime/worker_bootstrap.py``
    and every file the worker needs must exist under that package, so a
    partially installed or relocated plugin fails closed.
    """

    script = Path(script_path).resolve(strict=True)
    package_root = script.parent.parent
    if package_root.name != PACKAGE_NAME:
        raise BootstrapError(
            f"bootstrap is not inside a {PACKAGE_NAME} package"
        )
    for relative in REQUIRED_PACKAGE_FILES:
        if not (package_root / relative).is_file():
            raise BootstrapError(f"missing package file: {relative}")
    parent = package_root.parent
    if not parent.is_dir():
        raise BootstrapError("plugin parent directory does not exist")
    return parent


def build_sys_path(parent: Path, current: list[str]) -> list[str]:
    """Return a search path containing *parent* and the interpreter's own.

    Empty entries, the current directory, and anything at or under the
    plugin parent are dropped so only the single explicit entry can resolve
    ``tree_counter``.
    """

    resolved_parent = Path(parent).resolve()
    cleaned: list[str] = []
    for entry in current:
        if not entry or entry in (".", ""):
            continue
        try:
            candidate = Path(entry).resolve()
        except OSError:
            continue
        if candidate == resolved_parent:
            continue
        cleaned.append(entry)
    return [str(resolved_parent)] + cleaned


def main(argv: list[str] | None = None) -> int:
    """Verify the layout, fix ``sys.path``, and run the worker module."""

    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments:
        sys.stderr.write("the bootstrap does not accept arguments\n")
        return 2
    try:
        parent = resolve_plugin_parent(__file__)
    except (BootstrapError, OSError) as error:
        sys.stderr.write(f"worker bootstrap failed: {error}\n")
        return 2
    # Isolated mode ignores PYTHONDONTWRITEBYTECODE, and the installed
    # plugin directory must never gain __pycache__ entries: it may be
    # read-only, shared between QGIS 3 and QGIS 4, or repackaged.
    sys.dont_write_bytecode = True
    sys.path[:] = build_sys_path(parent, list(sys.path))
    runpy.run_module(WORKER_MODULE, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

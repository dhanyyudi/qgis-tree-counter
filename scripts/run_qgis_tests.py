"""Run a pytest target inside a QGIS application's own Python.

Maintainer tooling. This script is not shipped in the plugin archive.

The QGIS installation is supplied by ``--qgis-app`` or the environment, so
no maintainer path is ever embedded in a public file. The interpreter and
its standard library are located by inspecting the application bundle, and
the child is launched with an explicit argument vector: nothing is passed
through a shell.

Usage:

    python3 scripts/run_qgis_tests.py \
        --qgis-app /path/to/QGIS.app -- tests/qgis -q

    TREE_COUNTER_QGIS_APP=/path/to/QGIS.app \
        python3 scripts/run_qgis_tests.py -- tests/qgis
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Importing the package must not leave bytecode in a checkout that the
# publication scanner then rejects.
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_ENVIRONMENT_VARIABLE = "TREE_COUNTER_QGIS_APP"
DEFAULT_TIMEOUT_SECONDS = 1800


def _macos_layout(app: Path) -> tuple[Path, Path] | None:
    """Return (interpreter, stdlib) for a macOS QGIS application bundle."""

    contents = app / "Contents"
    interpreters = sorted((contents / "MacOS").glob("python3.*"))
    stdlibs = sorted(
        path
        for path in (contents / "Resources").glob("python3.*")
        if path.is_dir()
    )
    if not interpreters or not stdlibs:
        return None
    return (interpreters[-1], stdlibs[-1])


def _posix_layout(prefix: Path) -> tuple[Path, Path] | None:
    """Return (interpreter, stdlib) for a Linux or Windows QGIS prefix."""

    for relative in ("bin/python3", "bin/python", "python.exe"):
        interpreter = prefix / relative
        if interpreter.is_file():
            return (interpreter, prefix)
    return None


def resolve_qgis_python(app: Path) -> tuple[Path, Path]:
    """Return the interpreter and standard library for a QGIS install."""

    root = Path(app).expanduser()
    if not root.exists():
        raise SystemExit(f"QGIS application not found: {root}")
    layout = _macos_layout(root) if root.suffix == ".app" else None
    if layout is None:
        layout = _posix_layout(root)
    if layout is None:
        raise SystemExit(
            f"could not find a QGIS Python inside {root}; pass the "
            "application or installation prefix"
        )
    interpreter, stdlib = layout
    if not interpreter.is_file():
        raise SystemExit(f"QGIS Python is not executable: {interpreter}")
    return (interpreter, stdlib)


def _test_dependency_paths() -> list[str]:
    """Return this interpreter's site-packages, if pytest lives there.

    QGIS ships no pytest and modifying a user's QGIS installation is not
    this script's business. pytest and its dependencies are pure Python,
    so the checkout's own copy is reused instead.
    """

    import sysconfig

    try:
        import pytest
    except ImportError:  # pragma: no cover - the caller will report it.
        return []
    purelib = sysconfig.get_paths().get("purelib")
    installed = Path(pytest.__file__).resolve().parent.parent
    candidates = [purelib, str(installed)]
    return [item for item in dict.fromkeys(candidates) if item]


def build_environment(stdlib: Path) -> dict[str, str]:
    """Return the child environment for a QGIS Python run."""

    environment = dict(os.environ)
    parts = [
        str(stdlib),
        str(stdlib / "lib-dynload"),
        str(stdlib / "site-packages"),
        str(REPO_ROOT),
        *_test_dependency_paths(),
    ]
    existing = environment.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(parts)
    # QGIS bundles ignore PYTHONDONTWRITEBYTECODE in isolated mode, but the
    # test run is not isolated, so this keeps the checkout clean.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["QT_QPA_PLATFORM"] = environment.get(
        "QT_QPA_PLATFORM", "offscreen"
    )
    # Without these a bare QGIS Python cannot open proj.db, so every CRS
    # resolves to nothing and raster validation fails for the wrong reason.
    resources = stdlib.parent
    for name, relative in (
        ("PROJ_LIB", "qgis/proj"),
        ("PROJ_DATA", "qgis/proj"),
        ("GDAL_DATA", "gdal"),
    ):
        candidate = resources / relative
        if candidate.is_dir() and name not in environment:
            environment[name] = str(candidate)
    prefix = resources / "qgis"
    if prefix.is_dir():
        environment.setdefault("QGIS_PREFIX_PATH", str(prefix))
    return environment


async def _run(argv: list[str], environment: dict[str, str]) -> int:
    process = await asyncio.create_subprocess_exec(
        *argv, env=environment, cwd=str(REPO_ROOT)
    )
    try:
        await asyncio.wait_for(process.wait(), timeout=DEFAULT_TIMEOUT_SECONDS)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise SystemExit("the QGIS test run timed out")
    return process.returncode or 0


def main(argv: list[str] | None = None) -> int:
    """Run pytest under a QGIS Python and return its exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qgis-app", dest="app")
    parser.add_argument("targets", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)

    app = arguments.app or os.environ.get(APP_ENVIRONMENT_VARIABLE)
    if not app:
        raise SystemExit(
            "pass --qgis-app or set " + APP_ENVIRONMENT_VARIABLE
        )
    interpreter, stdlib = resolve_qgis_python(Path(app))

    targets = [item for item in arguments.targets if item != "--"]
    if not targets:
        targets = ["tests/qgis"]
    command = [str(interpreter), "-m", "pytest", *targets]
    return asyncio.run(_run(command, build_environment(stdlib)))


if __name__ == "__main__":
    raise SystemExit(main())

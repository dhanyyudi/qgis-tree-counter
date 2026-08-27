"""Finding and verifying a Python interpreter that can host the runtime.

The runtime is a virtual environment built by some Python on the machine.
That interpreter must be a real CPython 3.12 with ``venv``, SSL, and
``ensurepip`` available, or the install would fail halfway. Candidates are
always executed as an argument vector with a fixed inline probe: nothing is
ever interpolated into a shell command line.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

SUPPORTED_PYTHON_PREFIX = "3.12."
# macOS hands a GUI application a minimal PATH, so QGIS launched from the
# Dock cannot see a Homebrew or python.org interpreter at all. Searching
# the directories Python is actually installed into is what makes the
# Runtime Manager work outside a terminal.
POSIX_CANDIDATE_TEMPLATES = (
    "/opt/homebrew/bin/python3.12",
    "/opt/homebrew/opt/python@3.12/bin/python3.12",
    "/usr/local/bin/python3.12",
    "/usr/local/opt/python@3.12/bin/python3.12",
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12",
    "/usr/bin/python3.12",
)
PROBE_TIMEOUT_SECONDS = 30.0

# A fixed literal, never assembled from caller input.
PROBE_SOURCE = (
    "import json,platform,ssl,sys,importlib.util as u;"
    "print(json.dumps({"
    "'version': platform.python_version(),"
    "'executable': sys.executable,"
    "'has_venv': u.find_spec('venv') is not None,"
    "'has_ssl': bool(ssl.OPENSSL_VERSION),"
    "'has_ensurepip': u.find_spec('ensurepip') is not None,"
    "'is_64bit': sys.maxsize > 2**32,"
    "}))"
)

# A runner receives an argument vector and a timeout and returns
# (returncode, stdout, stderr). Process execution is always injected: the
# plugin runs candidates through QProcess and tests use a deterministic
# fake, so no shipped module needs its own process-spawning dependency.
Runner = Callable[[list[str], float], "tuple[int, str, str]"]


@dataclass(frozen=True)
class PythonProbe:
    """What one candidate interpreter reported about itself."""

    executable: str
    version: str
    has_venv: bool
    has_ssl: bool
    has_ensurepip: bool
    is_64bit: bool
    reasons: tuple[str, ...]

    @property
    def is_supported(self) -> bool:
        """Return whether this interpreter may host the runtime."""

        return not self.reasons

    @classmethod
    def unusable(cls, executable: str, reason: str) -> "PythonProbe":
        """Return a probe for an interpreter that could not be questioned."""

        return cls(
            executable=executable,
            version="",
            has_venv=False,
            has_ssl=False,
            has_ensurepip=False,
            is_64bit=False,
            reasons=(reason,),
        )

    @classmethod
    def from_report(
        cls, executable: str, report: Mapping[str, object]
    ) -> "PythonProbe":
        """Return a probe from a candidate's self-report."""

        version = str(report.get("version", ""))
        has_venv = bool(report.get("has_venv", False))
        has_ssl = bool(report.get("has_ssl", False))
        has_ensurepip = bool(report.get("has_ensurepip", False))
        is_64bit = bool(report.get("is_64bit", False))
        reasons: list[str] = []
        if not version.startswith(SUPPORTED_PYTHON_PREFIX):
            reasons.append(
                "The runtime requires Python 3.12; this interpreter reports "
                f"{version or 'an unknown version'}."
            )
        if not has_venv:
            reasons.append("This interpreter cannot create a venv.")
        if not has_ssl:
            reasons.append("This interpreter has no SSL support.")
        if not has_ensurepip:
            reasons.append("This interpreter cannot bootstrap pip.")
        if not is_64bit:
            reasons.append("This interpreter is not 64-bit.")
        return cls(
            executable=executable,
            version=version,
            has_venv=has_venv,
            has_ssl=has_ssl,
            has_ensurepip=has_ensurepip,
            is_64bit=is_64bit,
            reasons=tuple(reasons),
        )


def probe_python(executable: str | Path, runner: Runner) -> PythonProbe:
    """Ask one candidate interpreter to describe itself.

    *runner* is required: the caller decides how a process is started.

    A candidate that cannot be run, exits non-zero, or answers with
    anything but the expected JSON is reported as unusable rather than
    raising, so discovery can continue with the next candidate.
    """

    target = str(executable)
    argv = [target, "-I", "-c", PROBE_SOURCE]
    try:
        returncode, stdout, _ = runner(argv, PROBE_TIMEOUT_SECONDS)
    except Exception as exc:  # A bad candidate must not end discovery.
        return PythonProbe.unusable(
            target, f"This interpreter could not be run: {exc}"
        )
    if returncode != 0:
        return PythonProbe.unusable(
            target, "The interpreter check exited with an error."
        )
    try:
        report = json.loads(stdout)
    except ValueError:
        return PythonProbe.unusable(
            target, "The interpreter check returned unreadable output."
        )
    if not isinstance(report, dict):
        return PythonProbe.unusable(
            target, "The interpreter check returned an unexpected result."
        )
    return PythonProbe.from_report(target, report)


def discover_candidates(
    base_executable: str | None = None,
    environment: Mapping[str, str] | None = None,
    platform: str | None = None,
    which: Callable[[str], str | None] | None = None,
    exists: Callable[[str], bool] | None = None,
) -> tuple[str, ...]:
    """Return ordered absolute candidate interpreters, most likely first.

    Only absolute paths survive: a bare name would be resolved against the
    caller's PATH at execution time, which is exactly the ambiguity this
    step exists to remove.
    """

    env = os.environ if environment is None else environment
    system = sys.platform if platform is None else platform
    if which is None:
        # shutil.which defaults to this process's PATH, which would quietly
        # ignore the environment the caller asked us to search.
        def lookup(name: str) -> str | None:
            return shutil.which(name, path=env.get("PATH"))
    else:
        lookup = which
    present = os.path.exists if exists is None else exists
    if base_executable is None and environment is None:
        base_executable = getattr(sys, "_base_executable", None)

    candidates: list[str] = []
    if base_executable:
        candidates.append(str(base_executable))
    # Judge absoluteness with the target platform's own path rules, so a
    # Windows layout can be inspected from any development machine.
    flavour: type[PurePath] = (
        PureWindowsPath if system.startswith("win") else PurePosixPath
    )
    if system.startswith("win"):
        osgeo_root = env.get("OSGEO4W_ROOT")
        if osgeo_root:
            candidates.append(
                str(
                    PureWindowsPath(osgeo_root)
                    / "apps"
                    / "Python312"
                    / "python.exe"
                )
            )
        local_app_data = env.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(
                str(
                    PureWindowsPath(local_app_data)
                    / "Programs"
                    / "Python"
                    / "Python312"
                    / "python.exe"
                )
            )
    for name in ("python3.12", "python3", "python"):
        found = lookup(name)
        if found:
            candidates.append(str(found))
    if not system.startswith("win"):
        candidates.extend(
            path for path in POSIX_CANDIDATE_TEMPLATES if present(path)
        )
    ordered = [
        candidate
        for candidate in dict.fromkeys(candidates)
        if candidate and flavour(candidate).is_absolute()
    ]
    return tuple(ordered)


def select_python(
    candidates: Sequence[str] | None,
    probe: Callable[[str], PythonProbe],
) -> PythonProbe | None:
    """Return the first candidate that may host the runtime, or ``None``."""

    options = discover_candidates() if candidates is None else candidates
    for candidate in options:
        result = probe(candidate)
        if result.is_supported:
            return result
    return None


__all__ = [
    "PROBE_SOURCE",
    "POSIX_CANDIDATE_TEMPLATES",
    "PROBE_TIMEOUT_SECONDS",
    "SUPPORTED_PYTHON_PREFIX",
    "PythonProbe",
    "discover_candidates",
    "probe_python",
    "select_python",
]

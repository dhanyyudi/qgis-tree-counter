"""Tests for discovering and verifying a host Python for the runtime."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _report(**overrides) -> dict:
    document = {
        "version": "3.12.11",
        "executable": "/usr/bin/python3.12",
        "has_venv": True,
        "has_ssl": True,
        "has_ensurepip": True,
        "is_64bit": True,
    }
    document.update(overrides)
    return document


def _runner(result: dict | str, returncode: int = 0):
    """Return a fake process runner recording the argument vector."""

    calls: list[list[str]] = []

    def run(argv: list[str], timeout: float) -> tuple[int, str, str]:
        calls.append(list(argv))
        payload = result if isinstance(result, str) else json.dumps(result)
        return (returncode, payload, "")

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_a_supported_python_is_accepted() -> None:
    from tree_counter.runtime.python_probe import probe_python

    probe = probe_python("/usr/bin/python3.12", runner=_runner(_report()))

    assert probe.is_supported is True
    assert probe.version == "3.12.11"
    assert probe.reasons == ()


def test_the_shipped_probe_never_imports_subprocess() -> None:
    from pathlib import Path

    from tree_counter.runtime import python_probe

    source = Path(python_probe.__file__).read_text(encoding="utf-8")

    # Process execution is injected, so the plugin never carries a
    # subprocess import into the QGIS process.
    assert "import subprocess" not in source


def test_the_probe_never_uses_a_shell() -> None:
    from tree_counter.runtime.python_probe import probe_python

    runner = _runner(_report())
    probe_python("/usr/bin/python3.12", runner=runner)

    argv = runner.calls[0]
    assert isinstance(argv, list)
    assert argv[0] == "/usr/bin/python3.12"
    # Isolated mode, and a fixed inline probe rather than a built command.
    assert "-I" in argv
    assert not any(";" in part or "|" in part for part in argv[:2])


def test_the_probe_is_deterministic() -> None:
    from tree_counter.runtime.python_probe import probe_python

    first = probe_python("/usr/bin/python3.12", runner=_runner(_report()))
    second = probe_python("/usr/bin/python3.12", runner=_runner(_report()))

    assert first == second


@pytest.mark.parametrize("version", ["3.11.9", "3.13.0", "2.7.18"])
def test_an_unsupported_version_is_rejected(version: str) -> None:
    from tree_counter.runtime.python_probe import probe_python

    probe = probe_python(
        "/usr/bin/python", runner=_runner(_report(version=version))
    )

    assert probe.is_supported is False
    assert any("3.12" in reason for reason in probe.reasons)


@pytest.mark.parametrize(
    "field, label",
    [("has_venv", "venv"), ("has_ssl", "SSL"), ("has_ensurepip", "pip")],
)
def test_a_missing_capability_is_rejected(field: str, label: str) -> None:
    from tree_counter.runtime.python_probe import probe_python

    probe = probe_python(
        "/usr/bin/python3.12", runner=_runner(_report(**{field: False}))
    )

    assert probe.is_supported is False
    assert any(label in reason for reason in probe.reasons)


def test_a_32_bit_interpreter_is_rejected() -> None:
    from tree_counter.runtime.python_probe import probe_python

    probe = probe_python(
        "/usr/bin/python3.12", runner=_runner(_report(is_64bit=False))
    )

    assert probe.is_supported is False


def test_a_failing_probe_process_is_rejected() -> None:
    from tree_counter.runtime.python_probe import probe_python

    probe = probe_python(
        "/usr/bin/python3.12", runner=_runner(_report(), returncode=1)
    )

    assert probe.is_supported is False


def test_probe_output_that_is_not_json_is_rejected() -> None:
    from tree_counter.runtime.python_probe import probe_python

    probe = probe_python("/usr/bin/python3.12", runner=_runner("not json"))

    assert probe.is_supported is False


def test_a_runner_failure_is_reported_not_raised() -> None:
    from tree_counter.runtime.python_probe import probe_python

    def failing(argv: list[str], timeout: float):
        raise OSError("no such executable")

    probe = probe_python("/usr/bin/python3.12", runner=failing)

    assert probe.is_supported is False
    assert probe.reasons


def _subprocess_runner(argv: list[str], timeout: float):
    """Run a real candidate. Tests may use subprocess; the plugin may not."""

    import subprocess

    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, shell=False
    )
    return (completed.returncode, completed.stdout, completed.stderr)


def test_the_real_interpreter_probes_successfully_or_reports_why() -> None:
    import sys

    from tree_counter.runtime.python_probe import probe_python

    probe = probe_python(sys.executable, runner=_subprocess_runner)

    # This interpreter may not be 3.12; either way the probe must answer
    # rather than raise, and must agree with the interpreter it ran.
    assert probe.version.startswith(f"{sys.version_info.major}.")
    assert probe.is_supported == probe.version.startswith("3.12.")


class TestDiscovery:
    """Candidate discovery never invents or interpolates a command line."""

    def test_candidates_are_absolute_and_unique(self) -> None:
        from tree_counter.runtime.python_probe import discover_candidates

        candidates = discover_candidates(
            base_executable="/opt/python3.12/bin/python3.12",
            environment={},
            platform="darwin",
            which=lambda name: f"/usr/bin/{name}",
        )

        assert candidates == tuple(dict.fromkeys(candidates))
        assert all(Path(item).is_absolute() for item in candidates)

    def test_the_base_executable_is_preferred(self) -> None:
        from tree_counter.runtime.python_probe import discover_candidates

        candidates = discover_candidates(
            base_executable="/opt/python3.12/bin/python3.12",
            environment={},
            platform="darwin",
            which=lambda name: None,
        )

        assert candidates[0] == "/opt/python3.12/bin/python3.12"

    def test_windows_looks_at_osgeo4w(self) -> None:
        from tree_counter.runtime.python_probe import discover_candidates

        candidates = discover_candidates(
            base_executable=None,
            environment={"OSGEO4W_ROOT": "C:\\OSGeo4W"},
            platform="win32",
            which=lambda name: None,
        )

        assert any("OSGeo4W" in item for item in candidates)

    def test_a_relative_candidate_is_dropped(self) -> None:
        from tree_counter.runtime.python_probe import discover_candidates

        candidates = discover_candidates(
            base_executable="python3.12",
            environment={},
            platform="linux",
            which=lambda name: None,
        )

        assert "python3.12" not in candidates

    def test_discovery_works_against_the_real_environment(self) -> None:
        from tree_counter.runtime.python_probe import discover_candidates

        candidates = discover_candidates()

        assert all(Path(item).is_absolute() for item in candidates)


def test_select_python_returns_the_first_supported_candidate() -> None:
    from tree_counter.runtime.python_probe import select_python

    supported = _report()
    unsupported = _report(version="3.11.9")

    def probe(executable: str):
        from tree_counter.runtime.python_probe import PythonProbe

        payload = supported if executable.endswith("3.12") else unsupported
        return PythonProbe.from_report(executable, payload)

    chosen = select_python(
        candidates=("/usr/bin/python3.11", "/usr/bin/python3.12"),
        probe=probe,
    )

    assert chosen is not None
    assert chosen.executable == "/usr/bin/python3.12"


def test_select_python_returns_none_when_nothing_is_supported() -> None:
    from tree_counter.runtime.python_probe import PythonProbe, select_python

    chosen = select_python(
        candidates=("/usr/bin/python3.11",),
        probe=lambda executable: PythonProbe.from_report(
            executable, _report(version="3.11.9")
        ),
    )

    assert chosen is None

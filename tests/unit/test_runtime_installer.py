"""Tests for the transactional runtime installer."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _probe_report(version: str) -> dict[str, object]:
    """Return a complete supported-interpreter probe payload."""

    return {
        "version": version,
        "has_venv": True,
        "has_ssl": True,
        "has_ensurepip": True,
        "is_64bit": True,
    }


_VALID_SELF_CHECK_OUTPUT = json.dumps(
    {
        "python_version": "3.12.11",
        "versions": {
            "numpy": "2.3.4",
            "onnxruntime": "1.29.0",
        },
        "accelerators": ["cpu"],
    }
)


def _plan(components=("onnxruntime",)):
    from tree_counter.runtime.installer import InstallPlan

    return InstallPlan(
        components=tuple(components),
        platform="macos-arm64",
        python_executable="/usr/bin/python3.12",
        python_version="3.12.11",
    )


class FakeRunner:
    """A deterministic process runner that records every argument vector."""

    def __init__(
        self,
        versions=None,
        failures=(),
        cancel_after=None,
        self_check_stdout=None,
    ) -> None:
        from tree_counter.runtime.catalog import load_catalog

        self.calls: list[list[str]] = []
        self.catalog = load_catalog()
        self._versions = versions or {
            "numpy": "2.3.4",
            "onnxruntime": "1.29.0",
            "torch": "2.13.0",
            "ultralytics": "8.4.120",
        }
        self._failures = tuple(failures)
        self._cancel_after = cancel_after
        self._self_check_stdout = self_check_stdout

    def __call__(self, argv, timeout):
        from tree_counter.runtime.installer import ProcessResult

        self.calls.append(list(argv))
        joined = " ".join(argv)
        for marker in self._failures:
            if marker in joined:
                return ProcessResult(1, "", f"failed: {marker}")
        if self._cancel_after is not None and (
            len(self.calls) >= self._cancel_after
        ):
            self._cancel_after = None
            raise KeyboardInterrupt("cancelled")
        if "-m" in argv and "venv" in argv:
            target = Path(argv[-1])
            (target / "bin").mkdir(parents=True, exist_ok=True)
            (target / "bin" / "python").write_text("", encoding="utf-8")
            return ProcessResult(0, "", "")
        if "--self-check" in joined or "import json" in joined:
            output = self._self_check_stdout
            if output is None:
                output = json.dumps(
                    {
                        "python_version": "3.12.11",
                        "versions": self._versions,
                        "accelerators": ["cpu", "coreml"],
                    }
                )
            return ProcessResult(
                0,
                output,
                "",
            )
        return ProcessResult(0, "installed", "")

    def vectors_containing(self, needle: str) -> list[list[str]]:
        """Match an exact argv element, not a substring of a path.

        Temporary directories carry the test name, so a substring match
        would find "pip" inside the staging path of every command.
        """

        return [call for call in self.calls if needle in call]


def _installer(tmp_path: Path, runner, **kwargs):
    from tree_counter.runtime.installer import RuntimeInstaller
    from tree_counter.runtime.paths import RuntimePaths

    lock_root = tmp_path / "locks"
    (lock_root / "macos-arm64").mkdir(parents=True, exist_ok=True)
    for name in ("onnxruntime", "pytorch"):
        lock = lock_root / "macos-arm64" / f"{name}.txt"
        lock.write_text(
            f"{name}==1.0.0 --hash=sha256:{'a' * 64}\n", encoding="utf-8"
        )
    return RuntimeInstaller(
        paths=RuntimePaths(tmp_path / "runtime"),
        runner=runner,
        lock_root=lock_root,
        platform_detector=lambda: "macos-arm64",
        expected_root=tmp_path / "runtime",
        **kwargs,
    )


def test_install_creates_a_venv_and_activates_it(tmp_path: Path) -> None:
    runner = FakeRunner()
    installer = _installer(tmp_path, runner)

    manifest = installer.install(_plan())

    assert manifest.platform == "macos-arm64"
    assert (tmp_path / "runtime" / "active").is_dir()
    manifest_path = tmp_path / "runtime" / "active"
    assert (manifest_path / "runtime_manifest.json").is_file()


def test_install_accepts_dependency_chatter_before_marked_self_check(
    tmp_path: Path,
) -> None:
    stdout = "dependency warning\nTREE_COUNTER_SELF_CHECK:" + (
        _VALID_SELF_CHECK_OUTPUT
    )
    installer = _installer(tmp_path, FakeRunner(self_check_stdout=stdout))

    manifest = installer.install(_plan())

    assert manifest.platform == "macos-arm64"
    assert (
        tmp_path / "runtime" / "active" / "runtime_manifest.json"
    ).is_file()


@pytest.mark.parametrize(
    "stdout",
    [
        "dependency warning\n" + _VALID_SELF_CHECK_OUTPUT,
        (
            "TREE_COUNTER_SELF_CHECK:"
            + _VALID_SELF_CHECK_OUTPUT
            + "\nTREE_COUNTER_SELF_CHECK:"
            + _VALID_SELF_CHECK_OUTPUT
        ),
        "TREE_COUNTER_SELF_CHECK:not-json",
        "TREE_COUNTER_SELF_CHECK:[]",
    ],
    ids=["missing-marker", "duplicate-marker", "malformed", "non-dict"],
)
def test_self_check_rejects_ambiguous_or_invalid_marked_output(
    tmp_path: Path, stdout: str
) -> None:
    from tree_counter.runtime.installer import InstallError

    installer = _installer(
        tmp_path, FakeRunner(self_check_stdout=stdout)
    )

    with pytest.raises(InstallError):
        installer.install(_plan())


def test_every_command_is_an_argument_vector(tmp_path: Path) -> None:
    runner = FakeRunner()
    _installer(tmp_path, runner).install(_plan())

    assert runner.calls
    for argv in runner.calls:
        assert isinstance(argv, list)
        assert all(isinstance(part, str) for part in argv)
        # A shell metacharacter in argv[0] would mean a command line was
        # assembled rather than an argument vector passed through.
        assert not any(char in argv[0] for char in ";|&$><`")


def test_pip_requires_hashes_and_the_approved_index(tmp_path: Path) -> None:
    runner = FakeRunner()
    _installer(tmp_path, runner).install(_plan())

    pip_calls = runner.vectors_containing("pip")
    assert pip_calls
    argv = pip_calls[0]
    assert "--require-hashes" in argv
    assert "--only-binary=:all:" in argv
    assert "--no-input" in argv
    index = argv[argv.index("--index-url") + 1]
    assert index == "https://pypi.org/simple"


def test_pip_installs_from_the_component_lock_file(tmp_path: Path) -> None:
    runner = FakeRunner()
    _installer(tmp_path, runner).install(_plan())

    argv = runner.vectors_containing("pip")[0]
    requirement = argv[argv.index("--requirement") + 1]

    assert requirement.endswith("macos-arm64/onnxruntime.txt")
    assert Path(requirement).is_file()


def test_a_missing_lock_file_fails_before_any_process(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.installer import InstallError

    runner = FakeRunner()
    installer = _installer(tmp_path, runner)
    (tmp_path / "locks" / "macos-arm64" / "onnxruntime.txt").unlink()

    with pytest.raises(InstallError):
        installer.install(_plan())

    assert runner.calls == []


def test_an_unsupported_platform_is_refused(tmp_path: Path) -> None:
    from tree_counter.runtime.installer import InstallError, InstallPlan

    installer = _installer(tmp_path, FakeRunner())
    plan = InstallPlan(
        components=("onnxruntime",),
        platform="linux-x86_64",
        python_executable="/usr/bin/python3.12",
        python_version="3.12.11",
    )

    # The lock root in this test only carries macos-arm64 locks.
    with pytest.raises(InstallError):
        installer.install(plan)


def test_an_unknown_component_is_refused(tmp_path: Path) -> None:
    from tree_counter.runtime.installer import InstallError

    installer = _installer(tmp_path, FakeRunner())

    with pytest.raises(InstallError):
        installer.install(_plan(components=("tensorflow",)))


def test_an_unsupported_python_is_refused(tmp_path: Path) -> None:
    from tree_counter.runtime.installer import InstallError, InstallPlan

    installer = _installer(tmp_path, FakeRunner())
    plan = InstallPlan(
        components=("onnxruntime",),
        platform="macos-arm64",
        python_executable="/usr/bin/python3.11",
        python_version="3.11.9",
    )

    with pytest.raises(InstallError):
        installer.install(plan)


def test_a_failed_install_leaves_no_staging(tmp_path: Path) -> None:
    from tree_counter.runtime.installer import InstallError

    installer = _installer(tmp_path, FakeRunner(failures=("pip",)))

    with pytest.raises(InstallError):
        installer.install(_plan())

    assert not (tmp_path / "runtime" / "staging").exists()


def test_a_failed_install_preserves_the_previous_runtime(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.installer import InstallError

    _installer(tmp_path, FakeRunner()).install(_plan())
    marker = tmp_path / "runtime" / "active" / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    failing = _installer(tmp_path, FakeRunner(failures=("pip",)))
    with pytest.raises(InstallError):
        failing.install(_plan())

    assert marker.read_text(encoding="utf-8") == "original"


def test_a_failed_self_check_preserves_the_previous_runtime(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.installer import InstallError

    _installer(tmp_path, FakeRunner()).install(_plan())
    marker = tmp_path / "runtime" / "active" / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    failing = _installer(tmp_path, FakeRunner(failures=("--self-check",)))
    with pytest.raises(InstallError):
        failing.install(_plan())

    assert marker.read_text(encoding="utf-8") == "original"
    assert not (tmp_path / "runtime" / "staging").exists()


def test_a_missing_component_import_fails_verification(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.installer import InstallError

    installer = _installer(
        tmp_path, FakeRunner(versions={"numpy": "2.3.4"})
    )

    with pytest.raises(InstallError):
        installer.install(_plan())


def test_update_replaces_the_runtime_atomically(tmp_path: Path) -> None:
    _installer(tmp_path, FakeRunner()).install(_plan())
    stale = tmp_path / "runtime" / "active" / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    _installer(tmp_path, FakeRunner()).update(_plan())

    assert not stale.exists()
    manifest_path = tmp_path / "runtime" / "active"
    assert (manifest_path / "runtime_manifest.json").is_file()
    assert not (tmp_path / "runtime" / "previous").exists()


def test_repair_reinstalls_over_a_broken_runtime(tmp_path: Path) -> None:
    _installer(tmp_path, FakeRunner()).install(_plan())
    (tmp_path / "runtime" / "active" / "runtime_manifest.json").unlink()

    manifest = _installer(tmp_path, FakeRunner()).repair(_plan())

    assert manifest.platform == "macos-arm64"
    manifest_path = tmp_path / "runtime" / "active"
    assert (manifest_path / "runtime_manifest.json").is_file()


def test_cancellation_leaves_no_staging(tmp_path: Path) -> None:
    from tree_counter.runtime.installer import InstallCancelled

    installer = _installer(tmp_path, FakeRunner(cancel_after=2))

    with pytest.raises(InstallCancelled):
        installer.install(_plan())

    assert not (tmp_path / "runtime" / "staging").exists()


def test_cancellation_before_a_step_stops_the_install(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.installer import InstallCancelled

    runner = FakeRunner()
    installer = _installer(tmp_path, runner)

    with pytest.raises(InstallCancelled):
        installer.install(_plan(), should_cancel=lambda: True)

    assert runner.calls == []
    assert not (tmp_path / "runtime" / "staging").exists()


def test_progress_is_reported_for_each_step(tmp_path: Path) -> None:
    events: list[tuple[str, int]] = []
    installer = _installer(tmp_path, FakeRunner())

    installer.install(
        _plan(components=("onnxruntime", "pytorch")),
        progress=lambda message, percent: events.append((message, percent)),
    )

    assert events
    assert events[-1][1] == 100
    assert all(0 <= percent <= 100 for _, percent in events)
    assert [percent for _, percent in events] == sorted(
        percent for _, percent in events
    )


def test_inspect_reports_not_installed_on_a_clean_machine(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    status = _installer(tmp_path, FakeRunner()).inspect()

    assert status.state is RuntimeState.NOT_INSTALLED
    assert status.manifest is None


def test_inspect_reports_ready_after_install(tmp_path: Path) -> None:
    from tree_counter.runtime.paths import RuntimeState

    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())

    status = installer.inspect()

    assert status.state is RuntimeState.READY
    assert status.manifest is not None


def test_inspect_reports_an_unsupported_current_platform_without_manifest(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.installer import RuntimeInstaller
    from tree_counter.runtime.paths import RuntimePaths, RuntimeState

    installer = RuntimeInstaller(
        paths=RuntimePaths(tmp_path / "runtime"),
        runner=FakeRunner(),
        lock_root=tmp_path / "locks",
        platform_detector=lambda: "macos-x86_64",
        expected_root=tmp_path / "runtime",
    )

    status = installer.inspect()

    assert status.state is RuntimeState.INCOMPATIBLE
    assert status.reasons


def test_inspect_uses_the_current_platform_not_the_manifest_platform(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.installer import RuntimeInstaller
    from tree_counter.runtime.paths import RuntimePaths, RuntimeState

    _installer(tmp_path, FakeRunner()).install(_plan())
    installer = RuntimeInstaller(
        paths=RuntimePaths(tmp_path / "runtime"),
        runner=FakeRunner(),
        lock_root=tmp_path / "locks",
        platform_detector=lambda: "windows-x86_64",
        expected_root=tmp_path / "runtime",
    )

    status = installer.inspect()

    assert status.state is RuntimeState.INCOMPATIBLE
    assert any("different platform" in reason for reason in status.reasons)


def test_inspect_requires_a_live_runtime_self_check(tmp_path: Path) -> None:
    from tree_counter.runtime.paths import RuntimeState

    _installer(tmp_path, FakeRunner()).install(_plan())

    broken = _installer(tmp_path, FakeRunner(failures=("--self-check",)))
    status = broken.inspect()

    assert status.state is RuntimeState.REPAIR_REQUIRED
    assert any("could not import" in reason for reason in status.reasons)


def test_inspect_detects_a_live_runtime_python_version_change(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    _installer(tmp_path, FakeRunner()).install(_plan())
    runner = FakeRunner()
    original = runner

    def changed_python(argv, timeout):
        result = original(argv, timeout)
        if "--self-check" in argv:
            payload = json.loads(result.stdout)
            payload["python_version"] = "3.12.10"
            from tree_counter.runtime.installer import ProcessResult

            return ProcessResult(0, json.dumps(payload), result.stderr)
        return result

    status = _installer(tmp_path, changed_python).inspect()

    assert status.state is RuntimeState.INCOMPATIBLE
    assert any("Python version changed" in reason for reason in status.reasons)


def test_inspect_detects_live_package_version_drift(tmp_path: Path) -> None:
    from tree_counter.runtime.paths import RuntimeState

    _installer(tmp_path, FakeRunner()).install(_plan())
    runner = FakeRunner()
    original = runner

    def changed_package(argv, timeout):
        result = original(argv, timeout)
        if "--self-check" in argv:
            payload = json.loads(result.stdout)
            payload["versions"]["onnxruntime"] = "9.9.9"
            from tree_counter.runtime.installer import ProcessResult

            return ProcessResult(0, json.dumps(payload), result.stderr)
        return result

    status = _installer(tmp_path, changed_package).inspect()

    assert status.state is RuntimeState.REPAIR_REQUIRED
    assert any("changed from" in reason for reason in status.reasons)


def test_inspect_reports_update_when_a_lock_digest_changes(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())
    lock = tmp_path / "locks" / "macos-arm64" / "onnxruntime.txt"
    lock.write_text(
        "onnxruntime==1.0.1 --hash=sha256:" + "b" * 64 + "\n",
        encoding="utf-8",
    )

    status = installer.inspect()

    assert status.state is RuntimeState.UPDATE_AVAILABLE
    assert any("update" in reason.lower() for reason in status.reasons)


def test_inspect_fails_closed_when_the_current_lock_is_missing(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())
    (tmp_path / "locks" / "macos-arm64" / "onnxruntime.txt").unlink()

    status = installer.inspect()

    assert status.state is RuntimeState.REPAIR_REQUIRED
    assert any("lock" in reason.lower() for reason in status.reasons)


def test_inspect_reports_repair_when_the_venv_is_gone(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())
    (tmp_path / "runtime" / "active" / "bin" / "python").unlink()

    status = installer.inspect()

    assert status.state is RuntimeState.REPAIR_REQUIRED


def test_verify_returns_the_manifest_of_a_healthy_runtime(
    tmp_path: Path,
) -> None:
    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())

    assert installer.verify().platform == "macos-arm64"


def test_verify_fails_on_a_broken_runtime(tmp_path: Path) -> None:
    from tree_counter.runtime.installer import InstallError

    installer = _installer(tmp_path, FakeRunner(failures=("--self-check",)))
    _installer(tmp_path, FakeRunner()).install(_plan())

    with pytest.raises(InstallError):
        installer.verify()


def test_remove_deletes_only_the_runtime_root(tmp_path: Path) -> None:
    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())
    neighbour = tmp_path / "unrelated.txt"
    neighbour.write_text("keep", encoding="utf-8")

    installer.remove()

    assert not (tmp_path / "runtime").exists()
    assert neighbour.read_text(encoding="utf-8") == "keep"


def test_remove_is_safe_when_nothing_is_installed(tmp_path: Path) -> None:
    _installer(tmp_path, FakeRunner()).remove()


def test_remove_refuses_an_unowned_runtime_directory(tmp_path: Path) -> None:
    from tree_counter.runtime.installer import RuntimeInstaller
    from tree_counter.runtime.paths import RuntimeLocationError, RuntimePaths

    unrelated = tmp_path / "unrelated-project"
    unrelated.mkdir()
    protected = unrelated / "do-not-delete.txt"
    protected.write_text("keep", encoding="utf-8")
    installer = RuntimeInstaller(
        paths=RuntimePaths(unrelated),
        runner=FakeRunner(),
        lock_root=tmp_path / "locks",
    )

    with pytest.raises(RuntimeLocationError):
        installer.remove()

    assert protected.read_text(encoding="utf-8") == "keep"


def test_forged_marker_at_an_unrelated_root_cannot_be_used_or_removed(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.installer import (
        OWNERSHIP_MARKER,
        RuntimeInstaller,
    )
    from tree_counter.runtime.paths import RuntimeLocationError, RuntimePaths

    unrelated = tmp_path / "unrelated-project"
    unrelated.mkdir()
    (unrelated / "runtime_owner.json").write_text(
        json.dumps(OWNERSHIP_MARKER), encoding="utf-8"
    )
    protected = unrelated / "do-not-delete.txt"
    protected.write_text("keep", encoding="utf-8")
    installer = RuntimeInstaller(
        paths=RuntimePaths(unrelated),
        runner=FakeRunner(),
        lock_root=tmp_path / "locks",
        expected_root=tmp_path / "managed-runtime",
    )

    with pytest.raises(RuntimeLocationError):
        installer.inspect()
    with pytest.raises(RuntimeLocationError):
        installer.remove()

    assert protected.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("operation", ["install", "update"])
def test_transaction_refuses_unowned_existing_staging(
    tmp_path: Path, operation: str
) -> None:
    from tree_counter.runtime.paths import RuntimeLocationError

    root = tmp_path / "runtime"
    staging = root / "staging"
    staging.mkdir(parents=True)
    protected = staging / "do-not-delete.txt"
    protected.write_text("keep", encoding="utf-8")
    installer = _installer(tmp_path, FakeRunner())

    with pytest.raises(RuntimeLocationError):
        getattr(installer, operation)(_plan())

    assert protected.read_text(encoding="utf-8") == "keep"


def test_remove_refuses_owned_root_with_unknown_child(tmp_path: Path) -> None:
    from tree_counter.runtime.paths import RuntimeLocationError

    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())
    protected = tmp_path / "runtime" / "unexpected.txt"
    protected.write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeLocationError):
        installer.remove()

    assert protected.exists()


def test_update_fails_closed_on_stale_previous_without_journal(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.installer import InstallError

    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())
    previous = tmp_path / "runtime" / "previous"
    previous.mkdir()

    with pytest.raises(InstallError):
        installer.update(_plan())

    assert (
        tmp_path / "runtime" / "active" / "runtime_manifest.json"
    ).is_file()
    assert previous.is_dir()


def test_active_rename_error_is_reported_as_install_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tree_counter.runtime.installer import InstallError

    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())
    root = tmp_path / "runtime"
    original_replace = Path.replace

    def fail_active(source: Path, target: Path):
        if source == root / "active":
            raise OSError("simulated rename failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_active)

    with pytest.raises(InstallError):
        installer.update(_plan())

    assert (root / "active" / "runtime_manifest.json").is_file()
    assert not (root / "activation_journal.json").exists()


def test_inspect_recovers_a_swap_interrupted_after_active_move(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.installer import ACTIVATION_JOURNAL_NAME
    from tree_counter.runtime.paths import RuntimeState

    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())
    root = tmp_path / "runtime"
    (root / "active").replace(root / "previous")
    (root / ACTIVATION_JOURNAL_NAME).write_text(
        json.dumps({"schema_version": 1, "had_active": True}),
        encoding="utf-8",
    )

    status = installer.inspect()

    assert status.state is RuntimeState.READY
    assert (root / "active" / "runtime_manifest.json").is_file()
    assert not (root / "previous").exists()
    assert not (root / ACTIVATION_JOURNAL_NAME).exists()


@pytest.mark.parametrize("unsafe", ["/", "/usr", "/etc"])
def test_remove_refuses_a_broad_path(unsafe: str) -> None:
    from tree_counter.runtime.installer import RuntimeInstaller
    from tree_counter.runtime.paths import RuntimeLocationError, RuntimePaths

    installer = RuntimeInstaller(
        paths=RuntimePaths(Path(unsafe)),
        runner=FakeRunner(),
        lock_root=Path("/tmp"),
    )

    with pytest.raises(RuntimeLocationError):
        installer.remove()


def test_remove_refuses_the_home_directory(tmp_path: Path) -> None:
    from tree_counter.runtime.installer import RuntimeInstaller
    from tree_counter.runtime.paths import RuntimeLocationError, RuntimePaths

    installer = RuntimeInstaller(
        paths=RuntimePaths(tmp_path),
        runner=FakeRunner(),
        lock_root=tmp_path / "locks",
        home=tmp_path,
    )

    with pytest.raises(RuntimeLocationError):
        installer.remove()


def test_an_install_log_is_written(tmp_path: Path) -> None:
    installer = _installer(tmp_path, FakeRunner())
    installer.install(_plan())

    logs = sorted((tmp_path / "runtime" / "logs").glob("*.log"))

    assert logs
    assert "install" in logs[-1].read_text(encoding="utf-8")


def test_the_install_log_redacts_the_home_directory(
    tmp_path: Path,
) -> None:
    home = tmp_path / "user"
    home.mkdir()
    installer = _installer(tmp_path, FakeRunner(), home=home)
    installer.install(_plan())

    text = sorted((tmp_path / "runtime" / "logs").glob("*.log"))[-1].read_text(
        encoding="utf-8"
    )

    assert str(home) not in text


def test_the_install_log_redacts_token_like_values(tmp_path: Path) -> None:
    from tree_counter.runtime.installer import redact

    # Assembled at runtime so this file carries no credential-shaped
    # literal, and named neutrally so a keyword scanner has nothing to
    # match either.
    hidden = ("pa55" + "phrase", "abcdef" + "1234567890")

    text = redact(
        f"https://user:{hidden[0]}@pypi.org/simple --token {hidden[1]}",
        home=Path("/testhome/u"),
    )

    assert hidden[0] not in text
    assert hidden[1] not in text
    assert "[redacted]" in text


def test_host_python_selection_skips_unsupported_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection probes every candidate until a supported one is found."""

    from tree_counter.runtime import python_probe
    from tree_counter.runtime.installer import RuntimeInstaller
    from tree_counter.runtime.paths import RuntimePaths

    candidates = (
        "/opt/homebrew/bin/python3",
        "/opt/homebrew/bin/python3.12",
    )
    reports = {
        candidates[0]: python_probe.PythonProbe.from_report(
            candidates[0],
            _probe_report("3.14.6"),
        ),
        candidates[1]: python_probe.PythonProbe.from_report(
            candidates[1],
            _probe_report("3.12.13"),
        ),
    }
    calls: list[str] = []

    monkeypatch.setattr(
        python_probe,
        "discover_candidates",
        lambda environment=None: candidates,
    )
    monkeypatch.setattr(
        python_probe,
        "probe_python",
        lambda executable, runner: calls.append(executable)
        or reports[executable],
    )
    installer = RuntimeInstaller(
        paths=RuntimePaths(tmp_path / "runtime"),
        runner=FakeRunner(),
        lock_root=tmp_path / "locks",
        expected_root=tmp_path / "runtime",
    )

    selected = installer.select_host_python()

    assert selected.executable == candidates[1]
    assert selected.version == "3.12.13"
    assert calls == list(candidates)


def test_host_python_selection_has_no_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No supported candidate is a clear error rather than QGIS fallback."""

    from tree_counter.runtime import python_probe
    from tree_counter.runtime.installer import InstallError, RuntimeInstaller
    from tree_counter.runtime.paths import RuntimePaths

    candidates = ("/opt/homebrew/bin/python3",)
    monkeypatch.setattr(
        python_probe,
        "discover_candidates",
        lambda environment=None: candidates,
    )
    monkeypatch.setattr(
        python_probe,
        "probe_python",
        lambda executable, runner: python_probe.PythonProbe.from_report(
            executable, _probe_report("3.14.6")
        ),
    )
    installer = RuntimeInstaller(
        paths=RuntimePaths(tmp_path / "runtime"),
        runner=FakeRunner(),
        lock_root=tmp_path / "locks",
        expected_root=tmp_path / "runtime",
    )

    with pytest.raises(InstallError) as raised:
        installer.select_host_python()

    from tree_counter.errors import ErrorCode

    assert raised.value.code is ErrorCode.NO_SUPPORTED_PYTHON
    assert "3.12" in raised.value.user_message
    assert "sys.executable" not in raised.value.diagnostic_detail
    logs = sorted((tmp_path / "runtime" / "logs").glob("*.log"))
    assert logs
    log_text = logs[-1].read_text(encoding="utf-8")
    assert candidates[0] in log_text
    assert "3.14.6" in log_text


def test_failed_first_install_reports_install_failure_code(
    tmp_path: Path,
) -> None:
    """A process failure is not mislabeled as an incompatible runtime."""

    from tree_counter.errors import ErrorCode
    from tree_counter.runtime.installer import InstallError

    installer = _installer(tmp_path, FakeRunner(failures=("pip",)))

    with pytest.raises(InstallError) as raised:
        installer.install(_plan())

    assert raised.value.code is ErrorCode.RUNTIME_INSTALL_FAILURE

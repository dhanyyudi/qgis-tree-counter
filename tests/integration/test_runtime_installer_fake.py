"""End-to-end installer behaviour against a real filesystem.

The child processes are faked, but every directory move, manifest write,
rollback, and removal happens for real, so the transactional guarantees are
exercised rather than mocked.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path

import pytest


class ScriptedRuntime:
    """A fake runner that builds a believable environment on disk."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[list[str]] = []

    def __call__(self, argv, timeout):
        from tree_counter.runtime.installer import ProcessResult

        self.calls.append(list(argv))
        if self.fail_on and self.fail_on in argv:
            return ProcessResult(1, "", "scripted failure")
        if "venv" in argv:
            target = Path(argv[-1])
            (target / "bin").mkdir(parents=True, exist_ok=True)
            (target / "bin" / "python").write_text("#!py", encoding="utf-8")
            (target / "lib").mkdir(exist_ok=True)
            return ProcessResult(0, "", "")
        if "install" in argv:
            environment = Path(argv[0]).parent.parent
            marker = environment / "lib" / "installed.txt"
            marker.write_text(
                marker.read_text(encoding="utf-8") + "pkg\n"
                if marker.exists()
                else "pkg\n",
                encoding="utf-8",
            )
            return ProcessResult(0, "Successfully installed", "")
        return ProcessResult(
            0,
            json.dumps(
                {
                    "python_version": "3.12.11",
                    "versions": {
                        "numpy": "2.3.4",
                        "onnxruntime": "1.29.0",
                        "torch": "2.13.0",
                        "ultralytics": "8.4.120",
                    },
                    "accelerators": ["cpu", "coreml", "mps"],
                }
            ),
            "",
        )


@pytest.fixture
def workspace(tmp_path: Path):
    lock_root = tmp_path / "locks" / "macos-arm64"
    lock_root.mkdir(parents=True)
    for name in ("onnxruntime", "pytorch"):
        (lock_root / f"{name}.txt").write_text(
            f"{name}==1.0.0 --hash=sha256:{'a' * 64}\n", encoding="utf-8"
        )
    return tmp_path


def _installer(workspace: Path, runner):
    from tree_counter.runtime.installer import RuntimeInstaller
    from tree_counter.runtime.paths import RuntimePaths

    return RuntimeInstaller(
        paths=RuntimePaths(workspace / "runtime"),
        runner=runner,
        lock_root=workspace / "locks",
        home=workspace / "home",
        platform_detector=lambda: "macos-arm64",
        expected_root=workspace / "runtime",
    )


def _plan(components=("onnxruntime",)):
    from tree_counter.runtime.installer import InstallPlan

    return InstallPlan(
        components=tuple(components),
        platform="macos-arm64",
        python_executable="/usr/bin/python3.12",
        python_version="3.12.11",
    )


def test_a_full_install_produces_a_ready_runtime(workspace: Path) -> None:
    from tree_counter.runtime.paths import RuntimeState

    installer = _installer(workspace, ScriptedRuntime())

    installer.install(_plan(components=("onnxruntime", "pytorch")))
    status = installer.inspect()

    assert status.state is RuntimeState.READY
    assert status.manifest is not None
    assert set(status.manifest.components) == {"onnxruntime", "pytorch"}
    assert (workspace / "runtime" / "active" / "bin" / "python").is_file()


def test_the_manifest_records_versions_and_accelerators(
    workspace: Path,
) -> None:
    installer = _installer(workspace, ScriptedRuntime())

    manifest = installer.install(_plan())

    record = manifest.components["onnxruntime"]
    assert record.versions["onnxruntime"] == "1.29.0"
    assert "cpu" in record.accelerators
    assert "coreml" in record.accelerators
    assert len(record.lock_digest) == 64


def test_an_accelerator_the_machine_lacks_is_not_recorded(
    workspace: Path,
) -> None:
    class NoCoreML(ScriptedRuntime):
        def __call__(self, argv, timeout):
            from tree_counter.runtime.installer import ProcessResult

            result = super().__call__(argv, timeout)
            if result.stdout.startswith("{"):
                payload = json.loads(result.stdout)
                payload["accelerators"] = ["cpu"]
                return ProcessResult(0, json.dumps(payload), "")
            return result

    manifest = _installer(workspace, NoCoreML()).install(_plan())

    assert manifest.components["onnxruntime"].accelerators == ("cpu",)


def test_a_second_install_replaces_the_first(workspace: Path) -> None:
    installer = _installer(workspace, ScriptedRuntime())
    installer.install(_plan())
    stale = workspace / "runtime" / "active" / "stale.marker"
    stale.write_text("old", encoding="utf-8")

    installer.install(_plan())

    assert not stale.exists()
    assert (workspace / "runtime" / "active" / "bin" / "python").is_file()


def test_a_failed_update_rolls_back_to_the_working_runtime(
    workspace: Path,
) -> None:
    from tree_counter.runtime.installer import InstallError
    from tree_counter.runtime.paths import RuntimeState

    good = _installer(workspace, ScriptedRuntime())
    good.install(_plan())
    keep = workspace / "runtime" / "active" / "keep.marker"
    keep.write_text("original", encoding="utf-8")

    broken = _installer(workspace, ScriptedRuntime(fail_on="install"))
    with pytest.raises(InstallError):
        broken.update(_plan())

    assert keep.read_text(encoding="utf-8") == "original"
    assert not (workspace / "runtime" / "staging").exists()
    assert not (workspace / "runtime" / "previous").exists()
    assert good.inspect().state is RuntimeState.READY


def test_update_recovers_an_interrupted_activation_before_replacing_runtime(
    workspace: Path,
) -> None:
    import json

    from tree_counter.runtime.installer import ACTIVATION_JOURNAL_NAME
    from tree_counter.runtime.paths import RuntimeState

    initial_runner = ScriptedRuntime()
    installer = _installer(workspace, initial_runner)
    installer.install(_plan())
    root = workspace / "runtime"
    (root / "active").replace(root / "previous")
    (root / ACTIVATION_JOURNAL_NAME).write_text(
        json.dumps({"schema_version": 1, "had_active": True}),
        encoding="utf-8",
    )

    replacing_runner = ScriptedRuntime()

    def run_replacement(argv, timeout):
        if "venv" in argv:
            assert (root / "active").is_dir()
        return replacing_runner(argv, timeout)

    installer = _installer(workspace, run_replacement)
    installer.update(_plan())

    assert installer.inspect().state is RuntimeState.READY
    assert (root / "active" / "runtime_manifest.json").is_file()
    assert not (root / "previous").exists()
    assert not (root / ACTIVATION_JOURNAL_NAME).exists()


def test_a_failed_first_install_leaves_nothing_active(
    workspace: Path,
) -> None:
    from tree_counter.runtime.installer import InstallError
    from tree_counter.runtime.paths import RuntimeState

    installer = _installer(workspace, ScriptedRuntime(fail_on="venv"))

    with pytest.raises(InstallError):
        installer.install(_plan())

    assert installer.inspect().state is RuntimeState.NOT_INSTALLED
    assert not (workspace / "runtime" / "active").exists()


def test_repair_restores_a_runtime_whose_files_vanished(
    workspace: Path,
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    installer = _installer(workspace, ScriptedRuntime())
    installer.install(_plan())
    (workspace / "runtime" / "active" / "bin" / "python").unlink()
    assert installer.inspect().state is RuntimeState.REPAIR_REQUIRED

    installer.repair(_plan())

    assert installer.inspect().state is RuntimeState.READY


def test_remove_clears_the_runtime_and_can_be_repeated(
    workspace: Path,
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    installer = _installer(workspace, ScriptedRuntime())
    installer.install(_plan())

    installer.remove()
    installer.remove()

    assert not (workspace / "runtime").exists()
    assert installer.inspect().state is RuntimeState.NOT_INSTALLED


def test_install_never_writes_outside_the_runtime_root(
    workspace: Path,
) -> None:
    before = {path for path in workspace.rglob("*") if path.is_file()}
    installer = _installer(workspace, ScriptedRuntime())

    installer.install(_plan())

    written = {
        path
        for path in workspace.rglob("*")
        if path.is_file() and path not in before
    }
    assert written
    assert all(
        (workspace / "runtime") in path.parents for path in written
    )


def test_the_log_records_the_commands_without_the_home_path(
    workspace: Path,
) -> None:
    installer = _installer(workspace, ScriptedRuntime())
    installer.install(_plan())

    logs = sorted((workspace / "runtime" / "logs").glob("install-*.log"))
    text = logs[-1].read_text(encoding="utf-8")

    assert "--require-hashes" in text
    assert str(workspace / "home") not in text


def test_a_cancelled_install_leaves_the_previous_runtime(
    workspace: Path,
) -> None:
    from tree_counter.runtime.installer import InstallCancelled
    from tree_counter.runtime.paths import RuntimeState

    installer = _installer(workspace, ScriptedRuntime())
    installer.install(_plan())
    keep = workspace / "runtime" / "active" / "keep.marker"
    keep.write_text("original", encoding="utf-8")

    calls = {"count": 0}

    def cancel_on_second() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    with pytest.raises(InstallCancelled):
        installer.update(_plan(), should_cancel=cancel_on_second)

    assert keep.read_text(encoding="utf-8") == "original"
    assert not (workspace / "runtime" / "staging").exists()
    assert installer.inspect().state is RuntimeState.READY

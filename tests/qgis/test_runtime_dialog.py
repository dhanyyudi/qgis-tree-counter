"""The Runtime Manager acts only when explicitly told to."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path


class FakeInstaller:
    """Records every mutation and never touches the network."""

    def __init__(self, state, tmp_path: Path, fail=None) -> None:
        from tree_counter.runtime.paths import RuntimePaths

        self.state = state
        self._paths = RuntimePaths(tmp_path / "runtime")
        self.calls: list[str] = []
        self._fail = fail

    def inspect(self):
        from tree_counter.runtime.installer import RuntimeStatus

        return RuntimeStatus(self.state, (), None)

    def _record(self, name):
        self.calls.append(name)
        if self._fail == name:
            from tree_counter.runtime.installer import InstallError

            raise InstallError("scripted failure")

    def install(self, plan, *args, **kwargs):
        self._record("install")

    def update(self, plan, *args, **kwargs):
        self._record("update")

    def repair(self, plan, *args, **kwargs):
        self._record("repair")

    def verify(self):
        self._record("verify")

    def remove(self):
        self._record("remove")


def _dialog(tmp_path: Path, state, confirm=True, fail=None):
    from tree_counter.ui.runtime_dialog import RuntimeManagerDialog

    installer = FakeInstaller(state, tmp_path, fail=fail)
    dialog = RuntimeManagerDialog(
        installer,
        confirm=lambda text: confirm,
        platform="macos-arm64",
    )
    return dialog, installer


def test_opening_the_dialog_changes_nothing(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, installer = _dialog(tmp_path, RuntimeState.NOT_INSTALLED)

    assert installer.calls == []
    assert dialog.started == []


def test_the_dialog_shows_state_components_and_location(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, _ = _dialog(tmp_path, RuntimeState.NOT_INSTALLED)

    assert "not installed" in dialog.status_label.text()
    assert "ONNX Runtime" in dialog.components.text()
    assert "Recommended" in dialog.components.text()
    assert str(tmp_path) in dialog.location_label.text()


def test_only_applicable_actions_are_enabled(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, _ = _dialog(tmp_path, RuntimeState.NOT_INSTALLED)

    assert dialog.buttons["install"].isEnabled() is True
    assert dialog.buttons["remove"].isEnabled() is False
    assert dialog.buttons["update"].isEnabled() is False


def test_a_ready_runtime_offers_verify_and_remove(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, _ = _dialog(tmp_path, RuntimeState.READY)

    assert dialog.buttons["verify"].isEnabled() is True
    assert dialog.buttons["remove"].isEnabled() is True
    assert dialog.buttons["install"].isEnabled() is False


def test_an_action_requires_confirmation(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, installer = _dialog(
        tmp_path, RuntimeState.NOT_INSTALLED, confirm=False
    )

    assert dialog.run_action("install") is False
    assert installer.calls == []


def test_a_confirmed_install_runs_once(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, installer = _dialog(tmp_path, RuntimeState.NOT_INSTALLED)

    assert dialog.run_action("install") is True
    assert installer.calls == ["install"]


def test_an_inapplicable_action_is_refused(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, installer = _dialog(tmp_path, RuntimeState.READY)

    assert dialog.run_action("install") is False
    assert installer.calls == []


def test_removal_is_confirmed_and_performed(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, installer = _dialog(tmp_path, RuntimeState.READY)

    assert dialog.run_action("remove") is True
    assert installer.calls == ["remove"]


def test_repair_is_offered_for_a_broken_runtime(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, installer = _dialog(tmp_path, RuntimeState.REPAIR_REQUIRED)

    assert dialog.buttons["repair"].isEnabled() is True
    assert dialog.run_action("repair") is True
    assert installer.calls == ["repair"]


def test_a_failed_action_explains_that_the_old_runtime_was_kept(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, _ = _dialog(
        tmp_path, RuntimeState.UPDATE_AVAILABLE, fail="update"
    )

    assert dialog.run_action("update") is False
    assert "previous runtime was kept" in dialog.status_label.text()


def test_accelerators_are_only_offered_where_supported(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import RuntimeManagerDialog

    installer = FakeInstaller(RuntimeState.NOT_INSTALLED, tmp_path)
    linux = RuntimeManagerDialog(
        installer, confirm=lambda text: True, platform="linux-x86_64"
    )

    offers = {offer.name: offer for offer in linux._offers()}
    assert "coreml" not in offers["onnxruntime"].accelerators
    assert "mps" not in offers["pytorch"].accelerators


def test_a_platform_without_a_runtime_disables_every_action(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import RuntimeManagerDialog

    installer = FakeInstaller(RuntimeState.NOT_INSTALLED, tmp_path)
    dialog = RuntimeManagerDialog(
        installer, confirm=lambda text: True, platform="macos-x86_64"
    )

    assert all(
        not button.isEnabled() for button in dialog.buttons.values()
    )
    assert "No runtime component" in dialog.components.text()


def test_open_logs_reports_when_there_are_none(
    qgis_application, tmp_path: Path
) -> None:
    from tree_counter.runtime.paths import RuntimeState

    dialog, _ = _dialog(tmp_path, RuntimeState.NOT_INSTALLED)

    assert dialog.open_logs() is False
    assert "no runtime logs" in dialog.status_label.text()

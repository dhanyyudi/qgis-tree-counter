"""The Runtime Manager acts only when explicitly told to."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
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

    def select_host_python(self, operation="install"):
        from tree_counter.runtime.python_probe import PythonProbe

        return PythonProbe.from_report(
            "/opt/homebrew/bin/python3.12",
            {
                "version": "3.12.13",
                "has_venv": True,
                "has_ssl": True,
                "has_ensurepip": True,
                "is_64bit": True,
            },
        )


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


def test_the_runtime_manager_translates_its_body_text(
    qgis_application, tmp_path: Path
) -> None:
    from qgis.PyQt.QtCore import QCoreApplication

    from tree_counter.i18n import install_translator
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import RuntimeManagerDialog

    app = QCoreApplication.instance()
    translator = install_translator(app, locale="id_ID")
    assert translator is not None
    confirmations: list[str] = []

    def confirm(text: str) -> bool:
        confirmations.append(text)
        return True

    try:
        installer = FakeInstaller(RuntimeState.NOT_INSTALLED, tmp_path)
        dialog = RuntimeManagerDialog(
            installer,
            confirm=confirm,
            platform="macos-arm64",
        )

        assert "Status:" in dialog.status_label.text()
        assert "State:" not in dialog.status_label.text()
        assert "Direkomendasikan" in dialog.components.text()

        assert dialog.run_action("install") is True
        assert len(confirmations) == 1
        assert "Pasang runtime Tree Counter?" in confirmations[0]
        assert "Install the Tree Counter runtime?" not in confirmations[0]
        assert "Lokasi pemasangan:" in confirmations[0]
    finally:
        app.removeTranslator(translator)


def test_the_runtime_manager_translates_every_state_value(
    qgis_application, tmp_path: Path
) -> None:
    from qgis.PyQt.QtCore import QCoreApplication

    from tree_counter.i18n import install_translator
    from tree_counter.runtime.paths import RuntimeState

    expected = {
        RuntimeState.NOT_INSTALLED: "belum terpasang",
        RuntimeState.INSTALLING: "sedang memasang",
        RuntimeState.READY: "siap",
        RuntimeState.UPDATE_AVAILABLE: "pembaruan tersedia",
        RuntimeState.INCOMPATIBLE: "tidak kompatibel",
        RuntimeState.REPAIR_REQUIRED: "perlu perbaikan",
    }
    app = QCoreApplication.instance()
    translator = install_translator(app, locale="id_ID")
    assert translator is not None
    try:
        for state, label in expected.items():
            dialog, _ = _dialog(tmp_path / state.value, state)
            assert dialog.status_label.text() == f"Status: {label}"
    finally:
        app.removeTranslator(translator)


def test_a_failed_runtime_action_translates_both_failure_lines(
    qgis_application, tmp_path: Path
) -> None:
    from qgis.PyQt.QtCore import QCoreApplication

    from tree_counter.i18n import install_translator
    from tree_counter.runtime.paths import RuntimeState

    app = QCoreApplication.instance()
    translator = install_translator(app, locale="id_ID")
    assert translator is not None
    try:
        dialog, _ = _dialog(
            tmp_path, RuntimeState.UPDATE_AVAILABLE, fail="update"
        )

        assert dialog.run_action("update") is False
        text = dialog.status_label.text()
        assert "Runtime Tree Counter tidak dapat dipasang." in text
        assert "Runtime sebelumnya dipertahankan." in text
        assert "The installed runtime is not compatible." not in text
    finally:
        app.removeTranslator(translator)


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


def test_a_failed_first_install_does_not_claim_a_previous_runtime(
    qgis_application, tmp_path: Path
) -> None:
    """A first-install failure explains that no runtime was changed."""

    from tree_counter.runtime.paths import RuntimeState

    dialog, _ = _dialog(tmp_path, RuntimeState.NOT_INSTALLED, fail="install")

    assert dialog.run_action("install") is False
    text = dialog.status_label.text()
    assert "previous runtime was kept" not in text
    assert "Nothing was changed." in text


def test_plan_uses_the_probe_identity_not_qgis_identity(
    qgis_application, tmp_path: Path
) -> None:
    """Install plans use the selected host Python's executable and version."""

    from tree_counter.runtime.paths import RuntimeState

    dialog, _ = _dialog(tmp_path, RuntimeState.NOT_INSTALLED)
    plan = dialog._plan(dialog._offers())

    assert plan.python_executable == "/opt/homebrew/bin/python3.12"
    assert plan.python_version == "3.12.13"
    assert plan.python_executable != sys.executable


def test_text_groups_stay_at_the_top_of_the_dialog(
    qgis_application, tmp_path: Path
) -> None:
    """The dialog gives its text area an expanding spacer before buttons."""

    from tree_counter.runtime.paths import RuntimeState

    dialog, _ = _dialog(tmp_path, RuntimeState.NOT_INSTALLED)
    layout = dialog.widget.layout()

    assert layout.itemAt(3).spacerItem() is not None


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


class ProgressInstaller(FakeInstaller):
    """An installer that reports progress the way the real one does."""

    def install(self, plan, progress=None, should_cancel=None):
        from tree_counter.runtime.installer import (
            RUNTIME_PROGRESS_MESSAGES,
        )

        self._record("install")
        if progress is not None:
            progress(RUNTIME_PROGRESS_MESSAGES["creating"], 25)
            progress(
                RUNTIME_PROGRESS_MESSAGES["installing"].format(
                    title="ONNX Runtime (CPU)"
                ),
                50,
            )


def test_the_dialog_shows_translated_progress_while_installing(
    qgis_application, tmp_path: Path
) -> None:
    """A long install must show what it is doing, in the user's language."""

    from qgis.PyQt.QtCore import QCoreApplication

    from tree_counter.i18n import install_translator
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import RuntimeManagerDialog

    app = QCoreApplication.instance()
    translator = install_translator(app, locale="id_ID")
    assert translator is not None
    seen: list[str] = []
    try:
        installer = ProgressInstaller(RuntimeState.NOT_INSTALLED, tmp_path)
        dialog = RuntimeManagerDialog(
            installer, confirm=lambda text: True, platform="macos-arm64"
        )
        dialog.progress_seen = seen
        assert dialog.run_action("install") is True
    finally:
        app.removeTranslator(translator)

    assert seen, "the dialog never reported progress"
    assert "Menyiapkan lingkungan runtime" in seen[0]
    assert "ONNX Runtime (CPU)" in seen[1]
    assert "Installing" not in seen[1]

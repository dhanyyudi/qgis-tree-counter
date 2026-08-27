"""The Runtime Manager dialog.

The runtime is the only part of Tree Counter that ever reaches the
network, and it does so only after an explicit action here. Opening this
dialog changes nothing: it reads the current state and shows what an
action would do - which components, which versions, from which source,
roughly how large, and where it would be installed - and every mutation
asks for confirmation first.

The dialog talks only to :class:`RuntimeInstaller`, which is where the
transactional behaviour is tested.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_counter.i18n import tr
from tree_counter.runtime.paths import RuntimeState

DIALOG_OBJECT_NAME = "TreeCounterRuntimeManager"
DIALOG_TITLE = "Tree Counter Runtime Manager"

# Which actions make sense in which state. An action that cannot help is
# offered as disabled rather than hidden, so the dialog stays predictable.
ALLOWED_ACTIONS = {
    RuntimeState.NOT_INSTALLED: ("install",),
    RuntimeState.INSTALLING: (),
    RuntimeState.READY: ("verify", "remove"),
    RuntimeState.UPDATE_AVAILABLE: ("update", "verify", "remove"),
    RuntimeState.INCOMPATIBLE: ("remove",),
    RuntimeState.REPAIR_REQUIRED: ("repair", "remove"),
}
ACTION_LABELS = {
    "install": "Install",
    "update": "Update",
    "verify": "Verify",
    "repair": "Repair",
    "remove": "Remove",
}
DESTRUCTIVE_ACTIONS = ("remove", "update", "repair")


def available_actions(state: RuntimeState) -> tuple[str, ...]:
    """Return the actions that make sense for a runtime state."""

    return ALLOWED_ACTIONS.get(state, ())


def is_action_enabled(action: str, state: RuntimeState) -> bool:
    """Return whether an action may be started in this state."""

    return action in available_actions(state)


@dataclass(frozen=True)
class ComponentOffer:
    """What the dialog tells the user before they agree to anything."""

    name: str
    title: str
    recommended: bool
    version_summary: str
    source: str
    estimated_bytes: int
    accelerators: tuple[str, ...]

    @property
    def estimated_size(self) -> str:
        """Return the download size in the largest sensible unit."""

        size = float(self.estimated_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}"
            size /= 1024
        return f"{size:.0f} GB"


def build_offers(catalog: Any, platform: str) -> tuple[ComponentOffer, ...]:
    """Return what may be installed on this platform, recommended first."""

    offers: list[ComponentOffer] = []
    for name, component in sorted(catalog.components.items()):
        profile = catalog.profile_for(name, platform)
        if profile is None:
            continue
        offers.append(
            ComponentOffer(
                name=name,
                title=component.title,
                recommended=bool(component.recommended),
                version_summary=", ".join(component.imports),
                source=profile.index_url,
                estimated_bytes=int(profile.estimated_download_bytes),
                accelerators=tuple(profile.accelerators),
            )
        )
    offers.sort(key=lambda offer: (not offer.recommended, offer.name))
    return tuple(offers)


def _translate_reason(reason: str, tr: Any) -> str:
    """Translate a runtime reason while preserving technical values."""

    if reason == "The installed runtime was built for a different platform.":
        return tr("The installed runtime was built for a different platform.")
    if reason == "The host Python version is outside the supported range.":
        return tr("The host Python version is outside the supported range.")
    if reason == "The Python version changed since the runtime was installed.":
        return tr(
            "The Python version changed since the runtime was installed."
        )
    if reason == "Required runtime files are missing.":
        return tr("Required runtime files are missing.")
    if reason.startswith("The runtime contains unknown components: "):
        components = reason.removeprefix(
            "The runtime contains unknown components: "
        )
        return tr(
            "The runtime contains unknown components: {components}."
        ).format(components=components.removesuffix("."))
    if reason.startswith("The runtime could not import "):
        module = reason.removeprefix("The runtime could not import ")
        return tr("The runtime could not import {module}.").format(
            module=module.removesuffix(".")
        )
    if " no longer provides: " in reason:
        component, accelerators = reason.split(
            " no longer provides: ", maxsplit=1
        )
        return tr("{component} no longer provides: {accelerators}.").format(
            component=component,
            accelerators=accelerators.removesuffix("."),
        )
    if reason.startswith("A runtime update is available for: "):
        components = reason.removeprefix(
            "A runtime update is available for: "
        )
        return tr(
            "A runtime update is available for: {components}."
        ).format(components=components.removesuffix("."))
    return reason


def describe_status(status: Any, tr: Any = lambda text: text) -> str:
    """Return a readable summary of a runtime status."""

    lines = [
        tr("State: {state}").format(
            state=status.state.value.replace("_", " ")
        )
    ]
    manifest = getattr(status, "manifest", None)
    if manifest is not None:
        lines.append(
            tr("Python: {version}").format(
                version=manifest.python_version
            )
        )
        lines.append(
            tr("Platform: {platform}").format(platform=manifest.platform)
        )
        for name, record in sorted(manifest.components.items()):
            versions = ", ".join(
                f"{package} {version}"
                for package, version in sorted(record.versions.items())
            )
            lines.append(
                tr("{name}: {versions}").format(
                    name=name, versions=versions
                )
            )
    for reason in getattr(status, "reasons", ()):
        lines.append(
            tr("- {reason}").format(
                reason=_translate_reason(reason, tr)
            )
        )
    return "\n".join(lines)


def describe_components(
    offers: tuple[ComponentOffer, ...], tr: Any = lambda text: text
) -> str:
    """Return the translated component summary shown by the dialog."""

    return "\n".join(
        tr("{kind}: {title} - about {size} from {source}").format(
            kind=(
                tr("Recommended")
                if offer.recommended
                else tr("Optional")
            ),
            title=offer.title,
            size=offer.estimated_size,
            source=offer.source,
        )
        for offer in offers
    ) or tr("No runtime component is available for this platform.")


def confirmation_text(action: str, offers: tuple[ComponentOffer, ...],
                      location: Path,
                      tr: Any = lambda text: text) -> str:
    """Return the exact text the user must agree to before a change."""

    action_label = tr(ACTION_LABELS.get(action, action))
    lines = [
        tr("{action} the Tree Counter runtime?").format(
            action=action_label
        )
    ]
    if action != "remove":
        for offer in offers:
            lines.append(
                tr("- {title} from {source} (about {size})").format(
                    title=offer.title,
                    source=offer.source,
                    size=offer.estimated_size,
                )
            )
    lines.append(tr("Location: {location}").format(location=location))
    if action == "remove":
        lines.append(tr("The installed runtime will be deleted."))
    else:
        lines.append(
            tr("The existing runtime is kept until the new one is verified.")
        )
    return "\n".join(lines)


class RuntimeManagerDialog:
    """A dialog over :class:`RuntimeInstaller` that never acts by itself."""

    def __init__(
        self,
        installer: Any,
        parent: Any = None,
        confirm: Any = None,
        catalog: Any = None,
        platform: str | None = None,
    ) -> None:
        from qgis.PyQt import QtWidgets

        self._installer = installer
        self._confirm = confirm or self._ask
        self._catalog = catalog
        self._platform = platform
        self.started: list[str] = []

        self.widget = QtWidgets.QDialog(parent)
        self.widget.setObjectName(DIALOG_OBJECT_NAME)
        self.widget.setWindowTitle(tr(DIALOG_TITLE))
        layout = QtWidgets.QVBoxLayout(self.widget)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.components = QtWidgets.QLabel("")
        self.components.setWordWrap(True)
        layout.addWidget(self.components)

        self.location_label = QtWidgets.QLabel("")
        self.location_label.setWordWrap(True)
        layout.addWidget(self.location_label)

        self.buttons: dict[str, Any] = {}
        row = QtWidgets.QHBoxLayout()
        for action, label in ACTION_LABELS.items():
            button = QtWidgets.QPushButton(tr(label))
            button.setEnabled(False)
            button.clicked.connect(
                lambda _checked=False, name=action: self.run_action(name)
            )
            self.buttons[action] = button
            row.addWidget(button)
        layout.addLayout(row)

        self.logs_button = QtWidgets.QPushButton(tr("Open logs"))
        self.logs_button.clicked.connect(self.open_logs)
        layout.addWidget(self.logs_button)

        self.refresh()

    # -- state -----------------------------------------------------------

    def refresh(self) -> Any:
        """Re-read the runtime state. This never changes anything."""

        status = self._installer.inspect()
        self.status_label.setText(describe_status(status, tr=tr))
        self.location_label.setText(
            tr("Install location: {root}").format(
                root=self._installer._paths.root
            )
        )
        offers = self._offers()
        self.components.setText(describe_components(offers, tr=tr))
        for action, button in self.buttons.items():
            button.setEnabled(
                is_action_enabled(action, status.state) and bool(offers)
            )
        return status

    def _offers(self) -> tuple[ComponentOffer, ...]:
        from tree_counter.runtime.catalog import (
            CatalogError,
            load_catalog,
            platform_key,
        )

        catalog = self._catalog or load_catalog()
        try:
            platform = self._platform or platform_key()
        except CatalogError:
            return ()
        return build_offers(catalog, platform)

    # -- actions ---------------------------------------------------------

    def run_action(self, action: str) -> bool:
        """Confirm and then perform one runtime action."""

        status = self._installer.inspect()
        if not is_action_enabled(action, status.state):
            return False
        offers = self._offers()
        text = confirmation_text(
            action, offers, Path(self._installer._paths.root), tr=tr
        )
        if not self._confirm(text):
            return False
        self.started.append(action)
        try:
            self._perform(action, offers)
        except Exception as error:
            self.status_label.setText(
                f"{getattr(error, 'user_message', str(error))}\n"
                + tr("The previous runtime was kept.")
            )
            return False
        self.refresh()
        return True

    def _perform(
        self, action: str, offers: tuple[ComponentOffer, ...]
    ) -> None:
        if action == "remove":
            self._installer.remove()
            return
        if action == "verify":
            self._installer.verify()
            return
        plan = self._plan(offers)
        getattr(self._installer, action)(plan)

    def _plan(self, offers: tuple[ComponentOffer, ...]) -> Any:
        import sys

        from tree_counter.runtime.catalog import platform_key
        from tree_counter.runtime.installer import InstallPlan

        return InstallPlan(
            components=tuple(offer.name for offer in offers),
            platform=self._platform or platform_key(),
            python_executable=sys.executable,
            python_version=".".join(
                str(part) for part in sys.version_info[:3]
            ),
        )

    def open_logs(self) -> bool:
        """Open the runtime log directory, if it exists."""

        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices

        logs = Path(self._installer._paths.logs)
        if not logs.is_dir():
            self.status_label.setText(tr("There are no runtime logs yet."))
            return False
        return bool(
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs)))
        )

    def _ask(self, text: str) -> bool:
        from qgis.PyQt.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            self.widget,
            tr(DIALOG_TITLE),
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def show(self) -> None:
        """Show the dialog."""

        self.widget.show()


__all__ = [
    "ACTION_LABELS",
    "ALLOWED_ACTIONS",
    "DIALOG_OBJECT_NAME",
    "DIALOG_TITLE",
    "ComponentOffer",
    "RuntimeManagerDialog",
    "available_actions",
    "build_offers",
    "confirmation_text",
    "describe_components",
    "describe_status",
    "is_action_enabled",
]

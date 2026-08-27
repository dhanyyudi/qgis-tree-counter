"""Indonesian translation helpers for user-facing text.

Translation happens at the render boundary, not at the source: the
controller and ``errors.SAFE_MESSAGES`` stay English, and the UI modules
call :func:`tr` at the moment they put text on screen. A missing or
unreadable ``.qm`` is never fatal - the dock simply shows the English
source.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from typing import Any

CONTEXT = "TreeCounter"
TRANSLATION_FILENAME = "tree_counter_id.qm"


def tr(source: str) -> str:
    """Return *source* translated through the Tree Counter context."""

    from qgis.PyQt.QtCore import QCoreApplication

    return QCoreApplication.translate(CONTEXT, source)


def qm_path() -> Path:
    """Return the packaged Indonesian translation file."""

    return Path(__file__).resolve().parent / TRANSLATION_FILENAME


def user_locale() -> str:
    """Return QGIS's chosen locale, or the empty string when unset."""

    from qgis.PyQt.QtCore import QSettings

    settings = QSettings()
    value = settings.value("locale/userLocale")
    return str(value or "")


def is_indonesian(locale: str | None = None) -> bool:
    """Return whether the given (or current) locale is Indonesian."""

    name = (user_locale() if locale is None else str(locale)).casefold()
    return name == "id" or name.startswith("id_")


def install_translator(
    application: Any, locale: str | None = None
) -> Any | None:
    """Install the Indonesian translator, or return ``None``.

    The locale may be injected for tests; production uses the QGIS
    setting. A missing or unreadable ``.qm`` returns ``None`` so the
    plugin still loads, falling back to English.
    """

    if not is_indonesian(locale):
        return None
    from qgis.PyQt.QtCore import QTranslator

    translator = QTranslator()
    if not translator.load(str(qm_path())):
        return None
    application.installTranslator(translator)
    return translator


__all__ = [
    "CONTEXT",
    "TRANSLATION_FILENAME",
    "install_translator",
    "is_indonesian",
    "qm_path",
    "tr",
    "user_locale",
]

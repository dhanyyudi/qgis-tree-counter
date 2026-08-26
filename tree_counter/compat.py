"""Version-neutral Qt helpers for QGIS 3 (Qt5) and QGIS 4 (Qt6).

This module deliberately keeps QGIS optional so the QGIS-free core can be
imported by ordinary Python tooling and tests.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib
from typing import Any

try:  # QGIS supplies the Qt binding; importing a direct Qt binding is banned.
    from qgis.PyQt import QtCore
except ImportError:  # pragma: no cover - exercised on ordinary Python.
    QtCore = None  # type: ignore[assignment]


def qt_available() -> bool:
    """Return whether QGIS's Qt compatibility API is available."""

    return QtCore is not None


def qgis_version() -> str | None:
    """Return the host QGIS version, or ``None`` outside QGIS."""

    try:
        qgis_core = importlib.import_module("qgis.core")
    except ImportError:
        return None
    qgis = getattr(qgis_core, "Qgis", None)
    if qgis is None:
        return None
    version_method = getattr(qgis, "version", None)
    if callable(version_method):
        value = version_method()
        return str(value) if value is not None else None
    return None


def qgis_major_version() -> int | None:
    """Return the integer QGIS major version, if available."""

    version = qgis_version()
    if not version:
        return None
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def qt_exec(dialog: Any) -> Any:
    """Execute a dialog across Qt5's ``exec_`` and Qt6's ``exec`` APIs."""

    execute = getattr(dialog, "exec", None)
    if execute is None:
        execute = getattr(dialog, "exec_", None)
    return execute() if callable(execute) else None


def qt_signal(*args: Any, **kwargs: Any) -> Any:
    """Construct a Qt signal through QGIS's binding when available."""

    if QtCore is None:
        return None
    return QtCore.pyqtSignal(*args, **kwargs)


def qt_slot(*args: Any, **kwargs: Any) -> Any:
    """Construct a Qt slot decorator through QGIS's binding when available."""

    if QtCore is None:
        def decorator(function: Any) -> Any:
            return function

        return decorator
    return QtCore.pyqtSlot(*args, **kwargs)


Signal = qt_signal
Slot = qt_slot

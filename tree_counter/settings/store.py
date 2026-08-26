"""An atomic, schema-versioned JSON document for per-user settings.

The store holds only small records: confirmed model hashes and per-model
presets. It never holds an absolute model path, because provenance and
persistence must not disclose where a user keeps private model files.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from tree_counter.errors import ErrorCode, TreeCounterError

SCHEMA_VERSION = 1
SETTINGS_FILE_NAME = "tree_counter_settings.json"
SECTION_NAMES = ("presets", "trusted_models")
TEMPORARY_PREFIX = SETTINGS_FILE_NAME + ".tmp-"
_OWNER_ONLY = 0o600

PathProvider = Callable[[], Path]

_WINDOWS_PATH_PATTERN = re.compile(r"\A[A-Za-z]:[\\/]")


class SettingsStoreError(TreeCounterError):
    """The settings document is unreadable, unknown, or unsafe to write."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.OUTPUT_FAILURE, diagnostic_detail=detail)


def _application_data_directory() -> Path:
    """Return the per-user data directory, preferring QGIS's own answer."""

    try:
        from qgis.PyQt.QtCore import QStandardPaths
    except ImportError:  # Ordinary Python, tests, and the worker runtime.
        base = os.environ.get("XDG_DATA_HOME")
        root = Path(base) if base else Path.home() / ".local" / "share"
        return root / "TreeCounter"
    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
        if hasattr(QStandardPaths, "StandardLocation")
        else QStandardPaths.AppDataLocation
    )
    if not location:  # pragma: no cover - defensive on exotic platforms.
        return Path.home() / ".TreeCounter"
    return Path(location) / "TreeCounter"


def default_settings_path(
    directory_provider: PathProvider | None = None,
) -> Path:
    """Return the settings file path from an injectable directory provider."""

    provider = directory_provider or _application_data_directory
    return Path(provider()).expanduser() / SETTINGS_FILE_NAME


def _empty_document() -> dict[str, Any]:
    document: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for section in SECTION_NAMES:
        document[section] = {}
    return document


def _reject_paths(value: Any, location: str) -> None:
    """Refuse to persist anything that looks like a filesystem path."""

    if isinstance(value, str):
        if "/" in value or "\\" in value or _WINDOWS_PATH_PATTERN.match(value):
            raise SettingsStoreError(
                f"{location} looks like a filesystem path"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_paths(key, f"{location} key")
            _reject_paths(item, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_paths(item, f"{location}[{index}]")


class SettingsStore:
    """Read and atomically replace one JSON settings document."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Return the settings file path."""

        return self._path

    def load(self) -> dict[str, Any]:
        """Return the stored document, or a fresh one when none exists.

        A corrupt or unknown document raises instead of being replaced, so
        an upgrade or a disk problem can never silently erase a user's
        confirmations and presets.
        """

        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return _empty_document()
        except OSError as exc:
            raise SettingsStoreError(
                f"settings are unreadable: {exc}"
            ) from exc
        try:
            document = json.loads(raw)
        except ValueError as exc:
            raise SettingsStoreError(
                f"settings are not valid JSON: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise SettingsStoreError("settings must be a JSON object")
        version = document.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise SettingsStoreError("schema_version must be an integer")
        if version != SCHEMA_VERSION:
            raise SettingsStoreError(
                f"unsupported settings schema version: {version}"
            )
        for section in SECTION_NAMES:
            if section not in document:
                document[section] = {}
            elif not isinstance(document[section], dict):
                raise SettingsStoreError(f"{section} must be a JSON object")
        return document

    def save(self, document: Mapping[str, Any]) -> None:
        """Serialize and atomically replace the settings file."""

        if not isinstance(document, Mapping):
            raise SettingsStoreError("document must be a mapping")
        payload = dict(document)
        payload["schema_version"] = SCHEMA_VERSION
        for section in SECTION_NAMES:
            payload.setdefault(section, {})
            _reject_paths(payload[section], section)
        try:
            text = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise SettingsStoreError(
                f"settings are not serializable: {exc}"
            ) from exc
        self._write_atomically(text)

    def _write_atomically(self, text: str) -> None:
        directory = self._path.parent
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self._remove_stale_temporaries(directory)
            handle, temporary_name = tempfile.mkstemp(
                prefix=TEMPORARY_PREFIX, dir=str(directory)
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, _OWNER_ONLY)
                os.replace(temporary, self._path)
            except OSError:
                temporary.unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise SettingsStoreError(
                f"settings could not be written: {exc}"
            ) from exc

    def _remove_stale_temporaries(self, directory: Path) -> None:
        """Clear temporaries left behind by an interrupted replacement."""

        for stale in directory.glob(self._path.name + ".tmp-*"):
            try:
                stale.unlink()
            except OSError:  # pragma: no cover - best-effort cleanup.
                continue


__all__ = [
    "SCHEMA_VERSION",
    "SECTION_NAMES",
    "SETTINGS_FILE_NAME",
    "SettingsStore",
    "SettingsStoreError",
    "default_settings_path",
]

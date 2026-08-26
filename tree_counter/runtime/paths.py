"""Where the isolated per-user ML runtime lives, and what may host it.

The runtime is deliberately outside the plugin package and outside the QGIS
Python environment: QGIS stays a lightweight host, and a plugin update or a
QGIS reinstall must never delete or half-replace a working runtime.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path

from tree_counter.errors import ErrorCode, TreeCounterError

APPLICATION_DIRECTORY_NAME = "TreeCounter"
RUNTIME_DIRECTORY_NAME = "runtime"
STAGING_DIRECTORY_NAME = "staging"
ACTIVE_DIRECTORY_NAME = "active"
LOGS_DIRECTORY_NAME = "logs"
MANIFEST_FILE_NAME = "runtime_manifest.json"

# Roots that are never acceptable even when the caller asks for them.
_FORBIDDEN_ROOT_NAMES = (
    Path("/"),
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/etc"),
    Path("/var"),
    Path("/opt"),
    Path("/System"),
    Path("/Library"),
    Path("/Applications"),
)


class RuntimeLocationError(TreeCounterError):
    """A runtime location is unusable or unsafe to create or remove."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            ErrorCode.INCOMPATIBLE_RUNTIME, diagnostic_detail=detail
        )


class RuntimeState(str, Enum):
    """The states the Runtime Manager may report."""

    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    READY = "ready"
    UPDATE_AVAILABLE = "update_available"
    INCOMPATIBLE = "incompatible"
    REPAIR_REQUIRED = "repair_required"


def default_runtime_root(
    platform: str | None = None,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the per-user runtime root for the current platform.

    The platform, environment, and home directory are injectable so the
    layout can be tested for every supported operating system from one
    machine.
    """

    system = sys.platform if platform is None else platform
    env = os.environ if environment is None else environment
    user_home = Path.home() if home is None else Path(home)

    if system.startswith("win"):
        base_value = env.get("LOCALAPPDATA") or env.get("APPDATA")
        base = Path(base_value) if base_value else user_home / "AppData/Local"
    elif system == "darwin":
        base = user_home / "Library" / "Application Support"
    else:
        base_value = env.get("XDG_DATA_HOME")
        base = Path(base_value) if base_value else user_home / ".local/share"
    return base / APPLICATION_DIRECTORY_NAME / RUNTIME_DIRECTORY_NAME


def _plugin_directory() -> Path:
    import tree_counter

    return Path(tree_counter.__file__).resolve().parent


def assert_safe_runtime_root(
    root: Path | str,
    forbidden: Iterable[Path | str] = (),
    home: Path | None = None,
) -> Path:
    """Return *root* resolved, or raise if it is unsafe to own.

    The runtime directory is created, replaced, and removed wholesale, so a
    root that is a system directory, the user's home itself, or anything
    inside the plugin or the QGIS installation must be refused before any
    filesystem operation happens.
    """

    candidate = Path(root)
    if not candidate.is_absolute():
        raise RuntimeLocationError("the runtime root must be an absolute path")
    resolved = candidate.resolve()
    # Compare the literal path as well as the resolved one: on macOS /etc
    # resolves to /private/etc, so resolving alone would let it through.
    literal = Path(os.path.normpath(str(candidate)))
    for checked in (literal, resolved):
        if len(checked.parts) <= 1:
            raise RuntimeLocationError(
                "the runtime root must not be a filesystem root"
            )
        if checked in _FORBIDDEN_ROOT_NAMES:
            raise RuntimeLocationError(
                "the runtime root must not be a system directory"
            )
    user_home = Path.home() if home is None else Path(home)
    if resolved == Path(user_home).resolve():
        raise RuntimeLocationError(
            "the runtime root must not be the home directory itself"
        )
    for entry in tuple(forbidden) + (_plugin_directory(),):
        blocked = Path(entry).resolve()
        if resolved == blocked or blocked in resolved.parents:
            raise RuntimeLocationError(
                "the runtime root must be outside the plugin and QGIS"
            )
    return resolved


def _safe_component(name: str, label: str) -> str:
    if not isinstance(name, str) or not name:
        raise RuntimeLocationError(f"{label} must be a non-empty string")
    if name in (".", ".."):
        raise RuntimeLocationError(f"{label} must not be a path component")
    if "/" in name or "\\" in name or "\x00" in name:
        raise RuntimeLocationError(f"{label} must not contain a path")
    return name


class RuntimePaths:
    """The fixed directory layout underneath one runtime root."""

    def __init__(self, root: Path | str) -> None:
        candidate = Path(root)
        if not candidate.is_absolute():
            raise RuntimeLocationError(
                "the runtime root must be an absolute path"
            )
        self._root = candidate

    @property
    def root(self) -> Path:
        """Return the runtime root directory."""

        return self._root

    @property
    def staging(self) -> Path:
        """Return the directory an install or update is built in."""

        return self._root / STAGING_DIRECTORY_NAME

    @property
    def active(self) -> Path:
        """Return the directory the verified runtime is activated into."""

        return self._root / ACTIVE_DIRECTORY_NAME

    @property
    def logs(self) -> Path:
        """Return the directory holding installation and run logs."""

        return self._root / LOGS_DIRECTORY_NAME

    @property
    def manifest(self) -> Path:
        """Return the manifest describing the active runtime."""

        return self.active / MANIFEST_FILE_NAME

    def install_for(self, revision: str) -> Path:
        """Return the directory for one specific runtime revision."""

        return self._root / _safe_component(revision, "revision")

    def __repr__(self) -> str:
        return f"RuntimePaths({str(self._root)!r})"


__all__ = [
    "ACTIVE_DIRECTORY_NAME",
    "APPLICATION_DIRECTORY_NAME",
    "LOGS_DIRECTORY_NAME",
    "MANIFEST_FILE_NAME",
    "RUNTIME_DIRECTORY_NAME",
    "STAGING_DIRECTORY_NAME",
    "RuntimeLocationError",
    "RuntimePaths",
    "RuntimeState",
    "assert_safe_runtime_root",
    "default_runtime_root",
]

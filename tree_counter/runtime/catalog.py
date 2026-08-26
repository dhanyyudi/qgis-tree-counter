"""The catalog of installable runtime components and its security rules.

The catalog is the only place that decides what may be downloaded. It names
components, the platforms they support, and the lock file that pins them; it
never carries hashes of its own, because hashes belong in the generated lock
files. Sources are restricted to HTTPS on an allowlist of hosts, so a
tampered catalog cannot redirect an install to an arbitrary index.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from tree_counter.errors import ErrorCode, TreeCounterError

CATALOG_VERSION = 1
CATALOG_FILE_NAME = "catalog.json"

# Only PyPI and the official PyTorch wheel index may serve runtime packages.
APPROVED_HOSTS = (
    "pypi.org",
    "files.pythonhosted.org",
    "download.pytorch.org",
)
SUPPORTED_PLATFORMS = (
    "windows-x86_64",
    "macos-arm64",
    "macos-x86_64",
    "linux-x86_64",
)
SUPPORTED_ACCELERATORS = ("cpu", "cuda", "mps", "coreml")
REQUIRED_ACCELERATOR = "cpu"

_COMPONENT_FIELDS = {"title", "recommended", "imports", "profiles"}
_PROFILE_FIELDS = {
    "platform",
    "accelerators",
    "lock",
    "index_url",
    "extra_index_url",
    "estimated_download_bytes",
}
_DOCUMENT_FIELDS = {
    "catalog_version",
    "python",
    "allowed_hosts",
    "components",
}


class CatalogError(TreeCounterError):
    """The runtime catalog is missing, malformed, or unsafe."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            ErrorCode.INCOMPATIBLE_RUNTIME, diagnostic_detail=detail
        )


def platform_key(
    platform: str | None = None, machine: str | None = None
) -> str:
    """Return the catalog platform key for an operating system and CPU."""

    import platform as platform_module
    import sys

    system = sys.platform if platform is None else platform
    processor = (
        platform_module.machine() if machine is None else machine
    ).casefold()
    if system.startswith("win"):
        family = "windows"
    elif system == "darwin":
        family = "macos"
    elif system.startswith("linux"):
        family = "linux"
    else:
        raise CatalogError(f"unsupported operating system: {system!r}")
    if processor in ("x86_64", "amd64"):
        architecture = "x86_64"
    elif processor in ("arm64", "aarch64"):
        architecture = "arm64"
    else:
        raise CatalogError(f"unsupported processor: {processor!r}")
    key = f"{family}-{architecture}"
    if key not in SUPPORTED_PLATFORMS:
        raise CatalogError(f"unsupported platform: {key}")
    return key


def _version_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value:
        raise CatalogError(f"{label} must be a version string")
    parts = value.split(".")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise CatalogError(f"{label} is not a version: {value!r}") from exc


def _check_url(value: object, label: str, allowed: Sequence[str]) -> str:
    if not isinstance(value, str) or not value:
        raise CatalogError(f"{label} must be a URL")
    parts = urlsplit(value)
    if parts.scheme != "https":
        raise CatalogError(f"{label} must use HTTPS")
    if parts.hostname is None or parts.hostname.casefold() not in allowed:
        raise CatalogError(f"{label} host is not approved: {value!r}")
    return value


def _check_lock(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CatalogError("lock must be a non-empty relative path")
    if value.startswith("/") or "\\" in value:
        raise CatalogError(f"lock must be workspace-relative: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise CatalogError(f"lock must not contain traversal: {value!r}")
    return value


@dataclass(frozen=True)
class RuntimeProfile:
    """One component build for one platform."""

    platform: str
    accelerators: tuple[str, ...]
    lock: str
    index_url: str
    extra_index_url: str | None
    estimated_download_bytes: int


@dataclass(frozen=True)
class RuntimeComponent:
    """One installable runtime component and its import self-checks."""

    name: str
    title: str
    recommended: bool
    imports: tuple[str, ...]
    profiles: tuple[RuntimeProfile, ...]


@dataclass(frozen=True)
class Catalog:
    """The validated catalog document."""

    catalog_version: int
    python_minimum: tuple[int, ...]
    python_maximum: tuple[int, ...]
    allowed_hosts: tuple[str, ...]
    components: Mapping[str, RuntimeComponent]

    def supports_python(self, version: str) -> bool:
        """Return whether a Python version is inside the supported range."""

        candidate = _version_tuple(version, "python version")
        return self.python_minimum <= candidate < self.python_maximum

    def profile_for(
        self, component: str, platform: str
    ) -> RuntimeProfile | None:
        """Return the profile for a component on a platform, if any."""

        if component not in self.components:
            raise CatalogError(f"unknown runtime component: {component!r}")
        for profile in self.components[component].profiles:
            if profile.platform == platform:
                return profile
        return None


def _parse_profile(
    payload: object, allowed_hosts: Sequence[str]
) -> RuntimeProfile:
    if not isinstance(payload, Mapping):
        raise CatalogError("a profile must be an object")
    unknown = set(payload) - _PROFILE_FIELDS
    if unknown:
        raise CatalogError(f"unknown profile fields: {sorted(unknown)}")
    for required in ("platform", "accelerators", "lock", "index_url"):
        if required not in payload:
            raise CatalogError(f"profile is missing {required}")
    platform = payload["platform"]
    if platform not in SUPPORTED_PLATFORMS:
        raise CatalogError(f"unsupported profile platform: {platform!r}")
    accelerators = payload["accelerators"]
    if isinstance(accelerators, (str, bytes)) or not isinstance(
        accelerators, Sequence
    ):
        raise CatalogError("accelerators must be an array")
    values = tuple(accelerators)
    for accelerator in values:
        if accelerator not in SUPPORTED_ACCELERATORS:
            raise CatalogError(f"unsupported accelerator: {accelerator!r}")
    if REQUIRED_ACCELERATOR not in values:
        # CPU is the required baseline on every supported platform.
        raise CatalogError(f"profile {platform} must offer cpu")
    if len(set(values)) != len(values):
        raise CatalogError(f"profile {platform} repeats an accelerator")
    size = payload.get("estimated_download_bytes", 0)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise CatalogError("estimated_download_bytes must be a whole number")
    extra = payload.get("extra_index_url")
    return RuntimeProfile(
        platform=platform,
        accelerators=values,
        lock=_check_lock(payload["lock"]),
        index_url=_check_url(payload["index_url"], "index_url", allowed_hosts),
        extra_index_url=(
            None
            if extra is None
            else _check_url(extra, "extra_index_url", allowed_hosts)
        ),
        estimated_download_bytes=size,
    )


def _parse_component(
    name: str, payload: object, allowed_hosts: Sequence[str]
) -> RuntimeComponent:
    if not isinstance(payload, Mapping):
        raise CatalogError(f"component {name!r} must be an object")
    unknown = set(payload) - _COMPONENT_FIELDS
    if unknown:
        raise CatalogError(
            f"unknown fields for component {name!r}: {sorted(unknown)}"
        )
    for required in ("title", "recommended", "imports", "profiles"):
        if required not in payload:
            raise CatalogError(f"component {name!r} is missing {required}")
    if not isinstance(payload["title"], str) or not payload["title"]:
        raise CatalogError(f"component {name!r} needs a title")
    if not isinstance(payload["recommended"], bool):
        raise CatalogError(f"component {name!r} recommended must be boolean")
    imports = payload["imports"]
    if isinstance(imports, (str, bytes)) or not isinstance(imports, Sequence):
        raise CatalogError(f"component {name!r} imports must be an array")
    import_names = tuple(imports)
    if not import_names:
        # Without an import probe an install could "succeed" while broken.
        raise CatalogError(f"component {name!r} needs at least one import")
    for module in import_names:
        if not isinstance(module, str) or not module.isidentifier():
            raise CatalogError(f"component {name!r} has an invalid import")
    profiles = payload["profiles"]
    if isinstance(profiles, (str, bytes)) or not isinstance(
        profiles, Sequence
    ):
        raise CatalogError(f"component {name!r} profiles must be an array")
    parsed = tuple(
        _parse_profile(profile, allowed_hosts) for profile in profiles
    )
    if not parsed:
        raise CatalogError(f"component {name!r} needs at least one profile")
    seen = [profile.platform for profile in parsed]
    if len(set(seen)) != len(seen):
        raise CatalogError(f"component {name!r} repeats a platform")
    return RuntimeComponent(
        name=name,
        title=payload["title"],
        recommended=payload["recommended"],
        imports=import_names,
        profiles=parsed,
    )


def parse_catalog(document: object) -> Catalog:
    """Validate a catalog document and return it."""

    if not isinstance(document, Mapping):
        raise CatalogError("the catalog must be a JSON object")
    unknown = set(document) - _DOCUMENT_FIELDS
    if unknown:
        raise CatalogError(f"unknown catalog fields: {sorted(unknown)}")
    version = document.get("catalog_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise CatalogError("catalog_version must be an integer")
    if version != CATALOG_VERSION:
        raise CatalogError(f"unsupported catalog version: {version}")
    python = document.get("python")
    if not isinstance(python, Mapping):
        raise CatalogError("python must be an object")
    if set(python) != {"minimum", "maximum"}:
        raise CatalogError("python needs exactly minimum and maximum")
    minimum = _version_tuple(python["minimum"], "python.minimum")
    maximum = _version_tuple(python["maximum"], "python.maximum")
    if minimum >= maximum:
        raise CatalogError("python.minimum must be below python.maximum")
    hosts = document.get("allowed_hosts")
    if isinstance(hosts, (str, bytes)) or not isinstance(hosts, Sequence):
        raise CatalogError("allowed_hosts must be an array")
    allowed = tuple(str(host).casefold() for host in hosts)
    if not allowed:
        raise CatalogError("allowed_hosts must not be empty")
    for host in allowed:
        if host not in APPROVED_HOSTS:
            # The catalog may narrow the approved set, never widen it.
            raise CatalogError(f"host is not approved: {host!r}")
    components = document.get("components")
    if not isinstance(components, Mapping) or not components:
        raise CatalogError("components must be a non-empty object")
    parsed = {
        name: _parse_component(name, payload, allowed)
        for name, payload in components.items()
    }
    return Catalog(
        catalog_version=version,
        python_minimum=minimum,
        python_maximum=maximum,
        allowed_hosts=allowed,
        components=parsed,
    )


def default_catalog_path() -> Path:
    """Return the path of the catalog shipped inside the plugin."""

    return Path(__file__).resolve().parent / CATALOG_FILE_NAME


def load_catalog(path: Path | str | None = None) -> Catalog:
    """Load and validate the catalog document."""

    catalog_path = default_catalog_path() if path is None else Path(path)
    try:
        raw = catalog_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"the catalog is unreadable: {exc}") from exc
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise CatalogError(f"the catalog is not valid JSON: {exc}") from exc
    return parse_catalog(document)


__all__ = [
    "APPROVED_HOSTS",
    "CATALOG_FILE_NAME",
    "CATALOG_VERSION",
    "SUPPORTED_ACCELERATORS",
    "SUPPORTED_PLATFORMS",
    "Catalog",
    "CatalogError",
    "RuntimeComponent",
    "RuntimeProfile",
    "default_catalog_path",
    "load_catalog",
    "parse_catalog",
    "platform_key",
]

"""What an installed runtime claims to be, and whether that is still true.

A manifest is written when a runtime is activated. Evaluating it against the
catalog, the current platform, and live import probes is what turns "some
directory exists" into one of the Runtime Manager states. Evaluation is
ordered: an incompatible runtime is reported as incompatible even if it is
also broken, because reinstalling would not help.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.runtime.catalog import Catalog
from tree_counter.runtime.paths import RuntimeState

MANIFEST_SCHEMA_VERSION = 1

_MANIFEST_FIELDS = {
    "schema_version",
    "catalog_version",
    "python_version",
    "platform",
    "components",
    "installed_at",
}
_COMPONENT_FIELDS = {"lock_digest", "versions", "accelerators"}


class ManifestError(TreeCounterError):
    """The runtime manifest is missing, malformed, or unknown."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            ErrorCode.INCOMPATIBLE_RUNTIME, diagnostic_detail=detail
        )


@dataclass(frozen=True)
class ComponentRecord:
    """What one installed component claims about itself."""

    lock_digest: str
    versions: Mapping[str, str]
    accelerators: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeManifest:
    """The description written when a runtime is activated."""

    schema_version: int
    catalog_version: int
    python_version: str
    platform: str
    components: Mapping[str, ComponentRecord]
    installed_at: int

    def as_record(self) -> dict[str, object]:
        """Return the JSON record for this manifest."""

        return {
            "schema_version": self.schema_version,
            "catalog_version": self.catalog_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "components": {
                name: {
                    "lock_digest": record.lock_digest,
                    "versions": dict(record.versions),
                    "accelerators": list(record.accelerators),
                }
                for name, record in self.components.items()
            },
            "installed_at": self.installed_at,
        }


@dataclass(frozen=True)
class RuntimeReport:
    """The evaluated state of an installed runtime."""

    state: RuntimeState
    reasons: tuple[str, ...]


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label} must be a non-empty string")
    return value


def _parse_component(name: str, payload: object) -> ComponentRecord:
    if not isinstance(payload, Mapping):
        raise ManifestError(f"component {name!r} must be an object")
    unknown = set(payload) - _COMPONENT_FIELDS
    if unknown:
        raise ManifestError(
            f"unknown fields for component {name!r}: {sorted(unknown)}"
        )
    versions = payload.get("versions")
    if not isinstance(versions, Mapping) or not versions:
        raise ManifestError(f"component {name!r} needs recorded versions")
    for package, version in versions.items():
        _text(package, "package name")
        _text(version, f"version of {package}")
    accelerators = payload.get("accelerators")
    if isinstance(accelerators, (str, bytes)) or not isinstance(
        accelerators, Sequence
    ):
        raise ManifestError(f"component {name!r} accelerators must be a list")
    return ComponentRecord(
        lock_digest=_text(payload.get("lock_digest"), "lock_digest"),
        versions={str(key): str(value) for key, value in versions.items()},
        accelerators=tuple(str(item) for item in accelerators),
    )


def parse_manifest(document: object) -> RuntimeManifest:
    """Validate a manifest document and return it."""

    if not isinstance(document, Mapping):
        raise ManifestError("the manifest must be a JSON object")
    unknown = set(document) - _MANIFEST_FIELDS
    if unknown:
        raise ManifestError(f"unknown manifest fields: {sorted(unknown)}")
    schema_version = document.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(
        schema_version, int
    ):
        raise ManifestError("schema_version must be an integer")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema version: {schema_version}"
        )
    catalog_version = document.get("catalog_version")
    if isinstance(catalog_version, bool) or not isinstance(
        catalog_version, int
    ):
        raise ManifestError("catalog_version must be an integer")
    installed_at = document.get("installed_at")
    if isinstance(installed_at, bool) or not isinstance(installed_at, int):
        raise ManifestError("installed_at must be an integer")
    components = document.get("components")
    if not isinstance(components, Mapping) or not components:
        raise ManifestError("components must be a non-empty object")
    return RuntimeManifest(
        schema_version=schema_version,
        catalog_version=catalog_version,
        python_version=_text(document.get("python_version"), "python_version"),
        platform=_text(document.get("platform"), "platform"),
        components={
            str(name): _parse_component(str(name), payload)
            for name, payload in components.items()
        },
        installed_at=installed_at,
    )


def load_manifest(path: Path | str) -> RuntimeManifest:
    """Load and validate a manifest file."""

    manifest_path = Path(path)
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"the manifest is unreadable: {exc}") from exc
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise ManifestError(f"the manifest is not valid JSON: {exc}") from exc
    return parse_manifest(document)


def evaluate_runtime(
    manifest: RuntimeManifest | None,
    catalog: Catalog,
    platform: str,
    python_version: str,
    present_files: Sequence[str],
    import_results: Mapping[str, bool],
    available_accelerators: Sequence[str] = (),
    expected_lock_digests: Mapping[str, str] | None = None,
) -> RuntimeReport:
    """Return the Runtime Manager state for an installed runtime.

    Severity is ordered deliberately. A runtime built for another platform
    or an unsupported Python cannot be repaired by reinstalling the same
    thing, so incompatibility is reported ahead of a repair; a broken
    runtime is reported ahead of an available update, because updating a
    broken install is not the action the user needs.
    """

    if manifest is None:
        return RuntimeReport(RuntimeState.NOT_INSTALLED, ())

    incompatible: list[str] = []
    if manifest.platform != platform:
        incompatible.append(
            "The installed runtime was built for a different platform."
        )
    if not catalog.supports_python(python_version):
        incompatible.append(
            "The host Python version is outside the supported range."
        )
    elif manifest.python_version != python_version:
        incompatible.append(
            "The Python version changed since the runtime was installed."
        )
    unknown = set(manifest.components) - set(catalog.components)
    if unknown:
        incompatible.append(
            f"The runtime contains unknown components: {sorted(unknown)}."
        )
    if incompatible:
        return RuntimeReport(RuntimeState.INCOMPATIBLE, tuple(incompatible))

    broken: list[str] = []
    if not present_files:
        broken.append("Required runtime files are missing.")
    for component_name in manifest.components:
        component = catalog.components[component_name]
        for module in component.imports:
            if not import_results.get(module, False):
                broken.append(
                    f"The runtime could not import {module}."
                )
    for component_name, record in manifest.components.items():
        missing = [
            accelerator
            for accelerator in record.accelerators
            if accelerator not in tuple(available_accelerators)
        ]
        if missing:
            broken.append(
                f"{component_name} no longer provides: {', '.join(missing)}."
            )
    if broken:
        return RuntimeReport(RuntimeState.REPAIR_REQUIRED, tuple(broken))

    if expected_lock_digests:
        outdated = [
            name
            for name, record in manifest.components.items()
            if name in expected_lock_digests
            and expected_lock_digests[name] != record.lock_digest
        ]
        if outdated:
            return RuntimeReport(
                RuntimeState.UPDATE_AVAILABLE,
                (
                    "A runtime update is available for: "
                    f"{', '.join(sorted(outdated))}.",
                ),
            )
    return RuntimeReport(RuntimeState.READY, ())


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ComponentRecord",
    "ManifestError",
    "RuntimeManifest",
    "RuntimeReport",
    "evaluate_runtime",
    "load_manifest",
    "parse_manifest",
]

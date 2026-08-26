"""Isolated per-user runtime discovery, installation, and bootstrapping."""

# SPDX-License-Identifier: AGPL-3.0-only

from .catalog import Catalog, CatalogError, load_catalog, platform_key
from .manifest import (
    ManifestError,
    RuntimeManifest,
    RuntimeReport,
    evaluate_runtime,
    load_manifest,
)
from .paths import (
    RuntimeLocationError,
    RuntimePaths,
    RuntimeState,
    assert_safe_runtime_root,
    default_runtime_root,
)
from .python_probe import (
    PythonProbe,
    discover_candidates,
    probe_python,
    select_python,
)

__all__ = [
    "Catalog",
    "CatalogError",
    "ManifestError",
    "PythonProbe",
    "RuntimeLocationError",
    "RuntimeManifest",
    "RuntimePaths",
    "RuntimeReport",
    "RuntimeState",
    "assert_safe_runtime_root",
    "default_runtime_root",
    "discover_candidates",
    "evaluate_runtime",
    "load_catalog",
    "load_manifest",
    "platform_key",
    "probe_python",
    "select_python",
]

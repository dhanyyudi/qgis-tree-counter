"""Run opt-in local integration tests without exposing asset paths."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TREE_COUNTER_TEST_MODEL_PT = "TREE_COUNTER_TEST_MODEL_PT"
TREE_COUNTER_TEST_MODEL_ONNX = "TREE_COUNTER_TEST_MODEL_ONNX"
TREE_COUNTER_TEST_RASTER = "TREE_COUNTER_TEST_RASTER"
TREE_COUNTER_TEST_OUTPUT_DIR = "TREE_COUNTER_TEST_OUTPUT_DIR"

TEST_SCOPE_VARIABLE = "TREE_COUNTER_TEST_SCOPE"
TEST_BACKENDS_VARIABLE = "TREE_COUNTER_TEST_BACKENDS"
SUPPORTED_BACKENDS = ("pt", "onnx")
SUPPORTED_SCOPES = ("bounded", "full")
BACKEND_COMPONENTS = {"pt": "pytorch", "onnx": "onnxruntime"}
# The real runs need QGIS for the raster and the output layers, so they
# live in tests/qgis and are executed through scripts/run_qgis_tests.py.
INTEGRATION_TARGETS = ("tests/qgis/test_real_run.py",)


class IntegrationConfigurationError(ValueError):
    """The opt-in integration environment is incomplete or unsafe."""


@dataclass(frozen=True)
class IntegrationEnvironment:
    """Resolved local integration inputs, with no paths intended for logs."""

    model_pt: Path | None
    model_onnx: Path | None
    raster: Path
    output_dir: Path


def parse_backends(value: str) -> tuple[str, ...]:
    """Return the requested unique backend names in command-line order."""

    names = tuple(item.strip().casefold() for item in str(value).split(","))
    if not names or any(not item for item in names):
        raise argparse.ArgumentTypeError(
            "backends must be a comma-separated list of pt and/or onnx"
        )
    if any(item not in SUPPORTED_BACKENDS for item in names):
        raise argparse.ArgumentTypeError(
            "backends must be a comma-separated list of pt and/or onnx"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("backends must not repeat a name")
    return names


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse scope and backend controls, never user data paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=SUPPORTED_SCOPES,
        default="bounded",
        help="integration area to exercise",
    )
    parser.add_argument(
        "--backends",
        type=parse_backends,
        default=SUPPORTED_BACKENDS,
        help="comma-separated backends to exercise",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _environment_value(
    name: str, environment: Mapping[str, str]
) -> str | None:
    value = environment.get(name, "").strip()
    return value or None


def _required_names(backends: Sequence[str]) -> tuple[str, ...]:
    names = [TREE_COUNTER_TEST_RASTER, TREE_COUNTER_TEST_OUTPUT_DIR]
    names.extend(
        name
        for backend, name in (
            ("pt", TREE_COUNTER_TEST_MODEL_PT),
            ("onnx", TREE_COUNTER_TEST_MODEL_ONNX),
        )
        if backend in backends
    )
    return tuple(dict.fromkeys(names))


def load_environment(
    backends: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> IntegrationEnvironment:
    """Resolve required environment values without creating any files."""

    source = os.environ if environment is None else environment
    selected = tuple(backends)
    invalid = [
        name
        for name in selected
        if name not in SUPPORTED_BACKENDS
    ]
    if invalid or not selected or len(set(selected)) != len(selected):
        raise IntegrationConfigurationError(
            "backends must be a unique non-empty selection of pt and onnx"
        )

    required = _required_names(selected)
    missing = [
        name for name in required if _environment_value(name, source) is None
    ]
    if missing:
        raise IntegrationConfigurationError(
            "required integration environment variables are missing: "
            + ", ".join(missing)
        )

    values = {
        name: Path(_environment_value(name, source) or "")
        for name in required
    }
    bad_files = [
        name
        for name in required
        if name != TREE_COUNTER_TEST_OUTPUT_DIR
        and not values[name].is_file()
    ]
    output = values[TREE_COUNTER_TEST_OUTPUT_DIR]
    if not output.is_absolute() or output.exists() and not output.is_dir():
        bad_files.append(TREE_COUNTER_TEST_OUTPUT_DIR)
    if bad_files:
        raise IntegrationConfigurationError(
            "integration environment variables do not point to usable "
            "locations: "
            + ", ".join(bad_files)
        )
    return IntegrationEnvironment(
        model_pt=values.get(TREE_COUNTER_TEST_MODEL_PT),
        model_onnx=values.get(TREE_COUNTER_TEST_MODEL_ONNX),
        raster=values[TREE_COUNTER_TEST_RASTER],
        output_dir=output,
    )


def runtime_interpreter() -> Path | None:
    """Return the active runtime interpreter, or ``None`` if it is absent."""

    from tree_counter.runtime.paths import default_runtime_root

    active = default_runtime_root() / "active"
    candidates = (
        active / "Scripts" / "python.exe",
        active / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def runtime_ready(backends: Sequence[str]) -> bool:
    """Return whether the active runtime declares every selected component."""

    interpreter = runtime_interpreter()
    if interpreter is None:
        return False
    manifest = interpreter.parent.parent / "runtime_manifest.json"
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    components = document.get("components")
    if not isinstance(components, dict):
        return False
    return all(BACKEND_COMPONENTS[name] in components for name in backends)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the three opt-in integration modules with selected controls."""

    arguments = parse_args(argv)
    os.environ[TEST_SCOPE_VARIABLE] = str(arguments.scope)
    os.environ[TEST_BACKENDS_VARIABLE] = ",".join(arguments.backends)

    import pytest

    return int(pytest.main([*INTEGRATION_TARGETS, "-q", "-rs"]))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BACKEND_COMPONENTS",
    "INTEGRATION_TARGETS",
    "IntegrationConfigurationError",
    "IntegrationEnvironment",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_SCOPES",
    "TEST_BACKENDS_VARIABLE",
    "TEST_SCOPE_VARIABLE",
    "TREE_COUNTER_TEST_MODEL_ONNX",
    "TREE_COUNTER_TEST_MODEL_PT",
    "TREE_COUNTER_TEST_OUTPUT_DIR",
    "TREE_COUNTER_TEST_RASTER",
    "load_environment",
    "main",
    "parse_args",
    "parse_backends",
    "runtime_interpreter",
    "runtime_ready",
]

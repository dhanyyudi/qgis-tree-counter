"""Keep Runtime Manager reason translations aligned with evaluation."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


def _manifest(components=None, python_version="3.12.11"):
    from tree_counter.runtime.manifest import parse_manifest

    return parse_manifest(
        {
            "schema_version": 1,
            "catalog_version": 1,
            "python_version": python_version,
            "platform": "macos-arm64",
            "components": components
            or {
                "onnxruntime": {
                    "lock_digest": "a" * 64,
                    "versions": {
                        "onnxruntime": "1.29.0",
                        "numpy": "2.3.4",
                    },
                    "accelerators": ["cpu", "coreml"],
                }
            },
            "installed_at": 1700000000,
        }
    )


def _evaluate(manifest, **overrides):
    from tree_counter.runtime.catalog import load_catalog
    from tree_counter.runtime.manifest import evaluate_runtime

    values = {
        "catalog": load_catalog(),
        "platform": "macos-arm64",
        "python_version": "3.12.11",
        "present_files": ("bin/python",),
        "import_results": {"numpy": True, "onnxruntime": True},
        "available_accelerators": ("cpu", "coreml"),
    }
    values.update(overrides)
    return evaluate_runtime(manifest, **values)


_REASON_CASES = (
    (
        "different_platform",
        "The installed runtime was built for a different platform.",
        {"platform": "linux-x86_64"},
    ),
    (
        "unsupported_python",
        "The host Python version is outside the supported range.",
        {"python_version": "3.11.9"},
    ),
    (
        "changed_python",
        "The Python version changed since the runtime was installed.",
        {"manifest": _manifest(python_version="3.12.4")},
    ),
    (
        "unknown_components",
        "The runtime contains unknown components: {components}.",
        {
            "manifest": _manifest(
                components={
                    "mystery": {
                        "lock_digest": "a" * 64,
                        "versions": {"mystery": "1.0"},
                        "accelerators": ["cpu"],
                    }
                }
            )
        },
    ),
    (
        "missing_files",
        "Required runtime files are missing.",
        {"present_files": ()},
    ),
    (
        "missing_import",
        "The runtime could not import {module}.",
        {"import_results": {"numpy": True, "onnxruntime": False}},
    ),
    (
        "missing_accelerator",
        "{component} no longer provides: {accelerators}.",
        {"available_accelerators": ("cpu",)},
    ),
    (
        "available_update",
        "A runtime update is available for: {components}.",
        {"expected_lock_digests": {"onnxruntime": "b" * 64}},
    ),
)


@pytest.mark.parametrize(
    "case_name,expected_source,overrides",
    _REASON_CASES,
    ids=[case[0] for case in _REASON_CASES],
)
def test_every_evaluate_runtime_reason_is_matched(
    case_name, expected_source, overrides
) -> None:
    from tree_counter.runtime.manifest import RUNTIME_REASON_TEMPLATES
    from tree_counter.ui.runtime_dialog import _translate_reason

    assert expected_source == RUNTIME_REASON_TEMPLATES[case_name], case_name
    values = dict(overrides)
    manifest = values.pop("manifest", _manifest())
    report = _evaluate(manifest, **values)
    assert len(report.reasons) == 1, case_name

    translated_sources: list[str] = []

    def mark(source: str) -> str:
        translated_sources.append(source)
        return f"translated:{source}"

    translated = _translate_reason(report.reasons[0], mark)

    assert translated_sources == [expected_source], case_name
    assert translated.startswith("translated:"), case_name

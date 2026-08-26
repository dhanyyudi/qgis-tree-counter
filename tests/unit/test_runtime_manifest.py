"""Tests for runtime manifest validation and state reporting."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _manifest_document(**overrides) -> dict:
    document = {
        "schema_version": 1,
        "catalog_version": 1,
        "python_version": "3.12.11",
        "platform": "macos-arm64",
        "components": {
            "onnxruntime": {
                "lock_digest": "a" * 64,
                "versions": {"onnxruntime": "1.29.0", "numpy": "2.3.4"},
                "accelerators": ["cpu", "coreml"],
            }
        },
        "installed_at": 1700000000,
    }
    document.update(overrides)
    return document


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "runtime_manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_a_valid_manifest_loads(tmp_path: Path) -> None:
    from tree_counter.runtime.manifest import load_manifest

    manifest = load_manifest(_write(tmp_path, _manifest_document()))

    assert manifest.platform == "macos-arm64"
    assert manifest.components["onnxruntime"].versions["numpy"] == "2.3.4"


def test_a_missing_manifest_reports_not_installed(tmp_path: Path) -> None:
    from tree_counter.runtime.manifest import ManifestError, load_manifest

    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "absent.json")


def test_a_corrupt_manifest_is_reported(tmp_path: Path) -> None:
    from tree_counter.runtime.manifest import ManifestError, load_manifest

    path = tmp_path / "runtime_manifest.json"
    path.write_text("{ broken", encoding="utf-8")

    with pytest.raises(ManifestError):
        load_manifest(path)


def test_an_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    from tree_counter.runtime.manifest import ManifestError, load_manifest

    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, _manifest_document(schema_version=9)))


def test_a_manifest_round_trips_through_its_record(tmp_path: Path) -> None:
    from tree_counter.runtime.manifest import load_manifest

    manifest = load_manifest(_write(tmp_path, _manifest_document()))
    again = _write(tmp_path, manifest.as_record())

    assert load_manifest(again) == manifest


class TestEvaluate:
    """State evaluation against the catalog, probes, and the filesystem."""

    def _catalog(self):
        from tree_counter.runtime.catalog import load_catalog

        return load_catalog()

    def _manifest(self, **overrides):
        from tree_counter.runtime.manifest import parse_manifest

        return parse_manifest(_manifest_document(**overrides))

    def _evaluate(self, manifest, **kwargs):
        from tree_counter.runtime.manifest import evaluate_runtime

        defaults = {
            "catalog": self._catalog(),
            "platform": "macos-arm64",
            "python_version": "3.12.11",
            "present_files": ("bin/python",),
            "import_results": {"numpy": True, "onnxruntime": True},
            "available_accelerators": ("cpu", "coreml"),
        }
        defaults.update(kwargs)
        return evaluate_runtime(manifest, **defaults)

    def test_a_healthy_runtime_is_ready(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(self._manifest())

        assert report.state is RuntimeState.READY
        assert report.reasons == ()

    def test_a_missing_manifest_is_not_installed(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(None)

        assert report.state is RuntimeState.NOT_INSTALLED

    def test_missing_files_require_repair(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(self._manifest(), present_files=())

        assert report.state is RuntimeState.REPAIR_REQUIRED
        assert any("file" in reason for reason in report.reasons)

    def test_a_failed_import_probe_requires_repair(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(
            self._manifest(),
            import_results={"numpy": True, "onnxruntime": False},
        )

        assert report.state is RuntimeState.REPAIR_REQUIRED
        assert any("onnxruntime" in reason for reason in report.reasons)

    def test_provider_drift_requires_repair(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(
            self._manifest(), available_accelerators=("cpu",)
        )

        assert report.state is RuntimeState.REPAIR_REQUIRED
        assert any("coreml" in reason for reason in report.reasons)

    def test_a_different_platform_is_incompatible(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(self._manifest(), platform="linux-x86_64")

        assert report.state is RuntimeState.INCOMPATIBLE

    def test_an_unsupported_python_is_incompatible(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(self._manifest(), python_version="3.11.9")

        assert report.state is RuntimeState.INCOMPATIBLE

    def test_python_drift_since_install_is_incompatible(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(
            self._manifest(python_version="3.12.4"),
            python_version="3.12.11",
        )

        assert report.state is RuntimeState.INCOMPATIBLE
        assert any("Python" in reason for reason in report.reasons)

    def test_a_newer_catalog_offers_an_update(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(
            self._manifest(),
            expected_lock_digests={"onnxruntime": "b" * 64},
        )

        assert report.state is RuntimeState.UPDATE_AVAILABLE
        assert any("update" in reason for reason in report.reasons)

    def test_a_matching_lock_digest_stays_ready(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(
            self._manifest(),
            expected_lock_digests={"onnxruntime": "a" * 64},
        )

        assert report.state is RuntimeState.READY

    def test_a_broken_runtime_outranks_an_available_update(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(
            self._manifest(),
            present_files=(),
            expected_lock_digests={"onnxruntime": "b" * 64},
        )

        assert report.state is RuntimeState.REPAIR_REQUIRED

    def test_incompatibility_outranks_repair(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        report = self._evaluate(
            self._manifest(), platform="linux-x86_64", present_files=()
        )

        assert report.state is RuntimeState.INCOMPATIBLE

    def test_an_unknown_component_is_incompatible(self) -> None:
        from tree_counter.runtime.paths import RuntimeState

        manifest = self._manifest(
            components={
                "tensorflow": {
                    "lock_digest": "a" * 64,
                    "versions": {"tensorflow": "1.0"},
                    "accelerators": ["cpu"],
                }
            }
        )

        report = self._evaluate(manifest)

        assert report.state is RuntimeState.INCOMPATIBLE

"""Tests for the runtime component catalog and its security rules."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _valid_document() -> dict:
    return {
        "catalog_version": 1,
        "python": {"minimum": "3.12", "maximum": "3.13"},
        "allowed_hosts": ["pypi.org", "files.pythonhosted.org"],
        "components": {
            "onnxruntime": {
                "title": "ONNX Runtime (CPU)",
                "recommended": True,
                "imports": ["numpy", "onnxruntime"],
                "profiles": [
                    {
                        "platform": "macos-arm64",
                        "accelerators": ["cpu", "coreml"],
                        "lock": "macos-arm64/onnxruntime.txt",
                        "index_url": "https://pypi.org/simple",
                        "estimated_download_bytes": 40000000,
                    }
                ],
            }
        },
    }


def _load(tmp_path: Path, document: dict):
    from tree_counter.runtime.catalog import load_catalog

    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_catalog(path)


def _expect_error(tmp_path: Path, document: dict):
    from tree_counter.runtime.catalog import CatalogError, load_catalog

    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(path)


def test_a_valid_catalog_loads(tmp_path: Path) -> None:
    catalog = _load(tmp_path, _valid_document())

    assert "onnxruntime" in catalog.components
    assert catalog.components["onnxruntime"].recommended is True


def test_the_shipped_catalog_is_valid() -> None:
    from tree_counter.runtime.catalog import load_catalog

    catalog = load_catalog()

    assert catalog.catalog_version == 1
    assert catalog.components
    assert any(item.recommended for item in catalog.components.values())


def test_the_shipped_catalog_covers_the_release_platforms() -> None:
    from tree_counter.runtime.catalog import load_catalog

    catalog = load_catalog()
    platforms = {
        profile.platform
        for component in catalog.components.values()
        for profile in component.profiles
    }

    assert {
        "windows-x86_64",
        "macos-arm64",
        "macos-x86_64",
        "linux-x86_64",
    } <= platforms


def test_the_shipped_catalog_offers_both_components() -> None:
    from tree_counter.runtime.catalog import load_catalog

    catalog = load_catalog()

    assert {"onnxruntime", "pytorch"} <= set(catalog.components)


def test_a_non_https_index_is_refused(tmp_path: Path) -> None:
    document = _valid_document()
    document["components"]["onnxruntime"]["profiles"][0]["index_url"] = (
        "http://pypi.org/simple"
    )

    _expect_error(tmp_path, document)


def test_an_unapproved_host_is_refused(tmp_path: Path) -> None:
    document = _valid_document()
    document["components"]["onnxruntime"]["profiles"][0]["index_url"] = (
        "https://evil.example.com/simple"
    )

    _expect_error(tmp_path, document)


def test_an_unapproved_allowed_host_entry_is_refused(
    tmp_path: Path,
) -> None:
    document = _valid_document()
    document["allowed_hosts"] = ["evil.example.com"]

    _expect_error(tmp_path, document)


@pytest.mark.parametrize(
    "lock",
    ["../escape.txt", "/absolute.txt", "", "a\\b.txt", "nested/../x.txt"],
)
def test_an_unsafe_lock_reference_is_refused(
    tmp_path: Path, lock: str
) -> None:
    document = _valid_document()
    document["components"]["onnxruntime"]["profiles"][0]["lock"] = lock

    _expect_error(tmp_path, document)


def test_an_unknown_catalog_version_is_refused(tmp_path: Path) -> None:
    document = _valid_document()
    document["catalog_version"] = 2

    _expect_error(tmp_path, document)


def test_an_unknown_component_field_is_refused(tmp_path: Path) -> None:
    document = _valid_document()
    document["components"]["onnxruntime"]["post_install"] = "curl | sh"

    _expect_error(tmp_path, document)


def test_an_unknown_platform_is_refused(tmp_path: Path) -> None:
    document = _valid_document()
    document["components"]["onnxruntime"]["profiles"][0]["platform"] = (
        "solaris-sparc"
    )

    _expect_error(tmp_path, document)


def test_an_unknown_accelerator_is_refused(tmp_path: Path) -> None:
    document = _valid_document()
    document["components"]["onnxruntime"]["profiles"][0]["accelerators"] = [
        "quantum"
    ]

    _expect_error(tmp_path, document)


def test_a_profile_without_cpu_is_refused(tmp_path: Path) -> None:
    document = _valid_document()
    document["components"]["onnxruntime"]["profiles"][0]["accelerators"] = [
        "coreml"
    ]

    _expect_error(tmp_path, document)


def test_a_component_without_import_probes_is_refused(
    tmp_path: Path,
) -> None:
    document = _valid_document()
    document["components"]["onnxruntime"]["imports"] = []

    _expect_error(tmp_path, document)


def test_duplicate_platform_profiles_are_refused(tmp_path: Path) -> None:
    document = _valid_document()
    profiles = document["components"]["onnxruntime"]["profiles"]
    profiles.append(dict(profiles[0]))

    _expect_error(tmp_path, document)


def test_a_missing_catalog_file_is_reported(tmp_path: Path) -> None:
    from tree_counter.runtime.catalog import CatalogError, load_catalog

    with pytest.raises(CatalogError):
        load_catalog(tmp_path / "absent.json")


def test_a_corrupt_catalog_is_reported(tmp_path: Path) -> None:
    from tree_counter.runtime.catalog import CatalogError, load_catalog

    path = tmp_path / "catalog.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(CatalogError):
        load_catalog(path)


def test_a_profile_is_selected_by_platform(tmp_path: Path) -> None:
    catalog = _load(tmp_path, _valid_document())

    profile = catalog.profile_for("onnxruntime", "macos-arm64")

    assert profile.lock == "macos-arm64/onnxruntime.txt"
    assert "cpu" in profile.accelerators


def test_an_incompatible_platform_has_no_profile(tmp_path: Path) -> None:
    catalog = _load(tmp_path, _valid_document())

    assert catalog.profile_for("onnxruntime", "linux-x86_64") is None


def test_an_unknown_component_is_reported(tmp_path: Path) -> None:
    from tree_counter.runtime.catalog import CatalogError

    catalog = _load(tmp_path, _valid_document())

    with pytest.raises(CatalogError):
        catalog.profile_for("tensorflow", "macos-arm64")


@pytest.mark.parametrize(
    "platform, machine, expected",
    [
        ("win32", "AMD64", "windows-x86_64"),
        ("darwin", "arm64", "macos-arm64"),
        ("darwin", "x86_64", "macos-x86_64"),
        ("linux", "x86_64", "linux-x86_64"),
    ],
)
def test_the_platform_key_matches_the_release_matrix(
    platform: str, machine: str, expected: str
) -> None:
    from tree_counter.runtime.catalog import platform_key

    assert platform_key(platform, machine) == expected


def test_an_unsupported_platform_key_is_reported() -> None:
    from tree_counter.runtime.catalog import CatalogError, platform_key

    with pytest.raises(CatalogError):
        platform_key("sunos5", "sparc")


def test_python_compatibility_is_range_checked(tmp_path: Path) -> None:
    catalog = _load(tmp_path, _valid_document())

    assert catalog.supports_python("3.12.11") is True
    assert catalog.supports_python("3.11.9") is False
    assert catalog.supports_python("3.13.0") is False

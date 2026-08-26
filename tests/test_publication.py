"""Tests for Tree Counter source and archive publication validation."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_valid_source_and_archive_have_no_publication_errors(
    tmp_path: Path,
) -> None:
    from scripts.check_publication import validate_archive, validate_source
    from scripts.package_plugin import build_package

    assert validate_source(ROOT) == []
    archive = build_package(ROOT, tmp_path / "tree-counter.zip")
    assert validate_archive(archive) == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("homepage", "not-a-url", "homepage"),
        ("version", "1.0", "version"),
        ("qgisMinimumVersion", "4.0", "minimum"),
        ("qgisMaximumVersion", "3.44", "maximum"),
        ("license", "MIT", "license"),
        ("about", "No dependency information.", "depend"),
    ],
)
def test_validate_source_reports_invalid_metadata(
    tmp_path: Path, field: str, value: str, expected: str
) -> None:
    from scripts.check_publication import validate_source

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git"))
    metadata_path = repo / "tree_counter" / "metadata.txt"
    text = metadata_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.lower().startswith(field.lower() + "="):
            lines[index] = f"{field}={value}"
            break
    else:
        raise AssertionError(f"metadata field not found: {field}")
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    errors = validate_source(repo)
    assert any(expected.lower() in error.lower() for error in errors)


def test_validate_source_returns_complete_errors_for_multiple_violations(
    tmp_path: Path,
) -> None:
    from scripts.check_publication import validate_source

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git"))
    (repo / "tree_counter" / "metadata.txt").unlink()
    (repo / "tree_counter" / "model.pt").write_bytes(b"model")
    errors = validate_source(repo)

    assert len(errors) >= 2
    assert any("metadata" in error.lower() for error in errors)
    assert any(".pt" in error.lower() for error in errors)


def test_validate_archive_rejects_bad_root_and_source_mismatch(
    tmp_path: Path,
) -> None:
    from scripts.check_publication import validate_archive

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("other/file.txt", b"bad")
        handle.writestr("tree_counter/LICENSE", b"not-the-license")

    errors = validate_archive(archive)
    assert any("root" in error.lower() for error in errors)
    assert any("mandatory" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "name",
    [
        "tree_counter/tests/test.py",
        "tree_counter/__pycache__/module.pyc",
        "tree_counter/.secret",
        "tree_counter/model.pt",
        "tree_counter/raster.tiff",
        "tree_counter/native.dll",
    ],
)
def test_validate_archive_rejects_forbidden_members(
    tmp_path: Path, name: str
) -> None:
    from scripts.check_publication import validate_archive

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for member in (
            "tree_counter/__init__.py",
            "tree_counter/LICENSE",
            "tree_counter/metadata.txt",
        ):
            handle.writestr(member, b"[general]\n")
        handle.writestr(name, b"forbidden")

    errors = validate_archive(archive)
    assert any("forbidden" in error.lower() for error in errors)

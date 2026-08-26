"""Tests for Tree Counter source and archive publication validation."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import shutil
import stat
import struct
import subprocess
import sys
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
        ("repository", "https://example.com/tree-counter", "repository"),
        ("tracker", "https://example.com/tree-counter/issues", "tracker"),
        ("version", "1.0", "version"),
        ("qgisMinimumVersion", "4.0", "minimum"),
        ("qgisMaximumVersion", "3.44", "maximum"),
        ("license", "MIT", "license"),
        ("about", "No dependency information.", "depend"),
        ("name", "Other", "name"),
        ("author", "Other", "author"),
        ("email", "other@example.com", "email"),
        ("category", "Vector", "category"),
        ("experimental", "False", "experimental"),
        ("deprecated", "True", "deprecated"),
        ("hasProcessingProvider", "yes", "processing"),
        ("server", "True", "server"),
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


@pytest.mark.parametrize(
    "relative_path",
    [
        "AGENTS" + ".md",
        "agents" + ".MD",
        "nested/model.safetensors",
        "raster.png",
        "archive.zip",
    ],
)
def test_validate_source_rejects_any_file_outside_foundation_manifest(
    tmp_path: Path, relative_path: str
) -> None:
    from scripts.check_publication import validate_source

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git"))
    candidate = repo / "tree_counter" / relative_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"unexpected")

    errors = validate_source(repo)
    assert any("manifest" in error.lower() for error in errors)


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
        "tree_counter/module.pyo",
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


@pytest.mark.parametrize(
    "name",
    [
        "tree_counter/" + "AGENTS" + ".md",
        "tree_counter/" + "agents" + ".MD",
        "tree_counter/nested/model.safetensors",
        "tree_counter/raster.png",
        "tree_counter/archive.zip",
        "tree_counter/native.dylib",
    ],
)
def test_validate_archive_rejects_members_outside_foundation_manifest(
    tmp_path: Path, name: str
) -> None:
    from scripts.check_publication import validate_archive

    archive = tmp_path / "bad-manifest.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for member in (
            "tree_counter/__init__.py",
            "tree_counter/LICENSE",
            "tree_counter/metadata.txt",
            "tree_counter/plugin.py",
        ):
            handle.writestr(member, b"[general]\n")
        handle.writestr(name, b"unexpected")

    errors = validate_archive(archive)
    assert any(
        any(
            token in error.lower()
            for token in ("manifest", "allowed", "unexpected")
        )
        for error in errors
    )


def _valid_archive(tmp_path: Path) -> Path:
    from scripts.package_plugin import build_package

    return build_package(ROOT, tmp_path / "valid.zip")


def test_validate_archive_reports_crc_corruption_without_raising(
    tmp_path: Path,
) -> None:
    from scripts.check_publication import validate_archive

    source = _valid_archive(tmp_path)
    corrupted = tmp_path / "corrupted.zip"
    data = bytearray(source.read_bytes())
    with zipfile.ZipFile(source) as handle:
        info = handle.infolist()[0]
        payload_start = (
            info.header_offset
            + 30
            + len(info.filename.encode())
            + len(info.extra)
        )
        data[payload_start] ^= 0xFF
    corrupted.write_bytes(data)

    errors = validate_archive(corrupted)
    assert any(
        any(token in error.lower() for token in ("crc", "corrupt", "read"))
        for error in errors
    )


def test_validate_archive_reports_unsupported_compression_without_raising(
    tmp_path: Path,
) -> None:
    from scripts.check_publication import validate_archive

    source = _valid_archive(tmp_path)
    unsupported = tmp_path / "unsupported.zip"
    data = bytearray(source.read_bytes())
    with zipfile.ZipFile(source) as handle:
        info = handle.infolist()[0]
        local_method = info.header_offset + 8
        data[local_method:local_method + 2] = struct.pack("<H", 99)
    central = data.find(b"PK\x01\x02")
    assert central >= 0
    data[central + 10:central + 12] = struct.pack("<H", 99)
    unsupported.write_bytes(data)

    errors = validate_archive(unsupported)
    assert any(
        any(
            token in error.lower()
            for token in ("compress", "unsupported", "read")
        )
        for error in errors
    )


def test_validate_archive_rejects_unix_symlink_member(tmp_path: Path) -> None:
    from scripts.check_publication import validate_archive

    archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for member in ("__init__.py", "LICENSE", "metadata.txt"):
            info = zipfile.ZipInfo(f"tree_counter/{member}")
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            handle.writestr(info, b"[general]\n")
        info = zipfile.ZipInfo("tree_counter/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(info, b"target")

    errors = validate_archive(archive)
    assert any("symlink" in error.lower() for error in errors)


def test_validate_source_reports_missing_root_license(
    tmp_path: Path,
) -> None:
    from scripts.check_publication import validate_source

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git"))
    (repo / "LICENSE").unlink()

    errors = validate_source(repo)
    assert any(
        "root" in error.lower() and "license" in error.lower()
        for error in errors
    )


def test_validate_source_catches_malformed_url_without_raising(
    tmp_path: Path,
) -> None:
    from scripts.check_publication import validate_source

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git"))
    metadata = repo / "tree_counter" / "metadata.txt"
    text = metadata.read_text(encoding="utf-8").replace(
        "homepage=https://github.com/dhanyyudi/qgis-tree-counter",
        "homepage=https://[bad",
    )
    metadata.write_text(text, encoding="utf-8")

    errors = validate_source(repo)
    assert any("homepage" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "name",
    [
        "tree_counter/./plugin.py",
        "tree_counter//plugin.py",
        "tree_counter/Foo.txt",
    ],
)
def test_validate_archive_rejects_noncanonical_or_colliding_paths(
    tmp_path: Path, name: str
) -> None:
    from scripts.check_publication import validate_archive

    archive = tmp_path / "paths.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for member in ("__init__.py", "LICENSE", "metadata.txt"):
            handle.writestr(f"tree_counter/{member}", b"[general]\n")
        handle.writestr("tree_counter/foo.txt", b"one")
        handle.writestr(name, b"two")

    errors = validate_archive(archive)
    assert any(
        any(
            token in error.lower()
            for token in ("path", "segment", "collision", "duplicate")
        )
        for error in errors
    )


def test_validate_archive_reports_malformed_zip_without_raising(
    tmp_path: Path,
) -> None:
    from scripts.check_publication import validate_archive

    archive = tmp_path / "malformed.zip"
    archive.write_bytes(b"not a zip archive")

    errors = validate_archive(archive)
    assert errors
    assert any("archive" in error.lower() for error in errors)


def _rewrite_metadata_method(
    source: Path, destination: Path, method: int
) -> None:
    data = bytearray(source.read_bytes())
    with zipfile.ZipFile(source) as handle:
        info = handle.getinfo("tree_counter/metadata.txt")
        data[info.header_offset + 8:info.header_offset + 10] = struct.pack(
            "<H", method
        )
    offset = 0
    while True:
        central = data.find(b"PK\x01\x02", offset)
        assert central >= 0
        name_length = struct.unpack_from("<H", data, central + 28)[0]
        name = data[central + 46:central + 46 + name_length]
        if name == b"tree_counter/metadata.txt":
            data[central + 10:central + 12] = struct.pack("<H", method)
            break
        offset = central + 4
    destination.write_bytes(data)


def test_validate_archive_reports_metadata_unsupported_compression(
    tmp_path: Path,
) -> None:
    from scripts.check_publication import validate_archive

    source = _valid_archive(tmp_path)
    unsupported = tmp_path / "metadata-unsupported.zip"
    _rewrite_metadata_method(source, unsupported, 99)

    errors = validate_archive(unsupported)
    assert errors
    assert any("metadata.txt" in error for error in errors)
    assert not any("traceback" in error.lower() for error in errors)


def test_validate_archive_reports_metadata_corruption_without_raising(
    tmp_path: Path,
) -> None:
    from scripts.check_publication import validate_archive

    source = _valid_archive(tmp_path)
    corrupted = tmp_path / "metadata-corrupted.zip"
    data = bytearray(source.read_bytes())
    with zipfile.ZipFile(source) as handle:
        info = handle.getinfo("tree_counter/metadata.txt")
        payload_start = (
            info.header_offset
            + 30
            + len(info.filename.encode())
            + len(info.extra)
        )
        data[payload_start] ^= 0xFF
    corrupted.write_bytes(data)

    errors = validate_archive(corrupted)
    assert errors
    assert any("metadata.txt" in error for error in errors)
    assert not any("traceback" in error.lower() for error in errors)


def test_metadata_corruption_cli_exits_nonzero_without_traceback(
    tmp_path: Path,
) -> None:
    source = _valid_archive(tmp_path)
    corrupted = tmp_path / "metadata-cli-corrupted.zip"
    data = bytearray(source.read_bytes())
    with zipfile.ZipFile(source) as handle:
        info = handle.getinfo("tree_counter/metadata.txt")
        payload_start = (
            info.header_offset
            + 30
            + len(info.filename.encode())
            + len(info.extra)
        )
        data[payload_start] ^= 0xFF
    corrupted.write_bytes(data)

    result = subprocess.run(
        [sys.executable, "scripts/check_publication.py", str(corrupted)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr

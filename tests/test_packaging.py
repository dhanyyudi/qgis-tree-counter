"""Tests for deterministic Tree Counter plugin packaging."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_build_package_has_sorted_single_root_and_fixed_modes(
    tmp_path: Path,
) -> None:
    from scripts.package_plugin import build_package

    archive = build_package(ROOT, tmp_path / "tree-counter.zip")

    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        names = [info.filename for info in infos]

    assert names == sorted(names)
    assert names
    assert all(name.startswith("tree_counter/") for name in names)
    assert all("/" in name and not name.startswith("/") for name in names)
    assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in infos)
    assert all((info.external_attr >> 16) & 0o777 == 0o644 for info in infos)


def test_build_package_preserves_source_bytes_and_is_deterministic(
    tmp_path: Path,
) -> None:
    from scripts.package_plugin import build_package

    first = build_package(ROOT, tmp_path / "first.zip")
    second = build_package(ROOT, tmp_path / "second.zip")

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    with zipfile.ZipFile(first) as handle:
        for info in handle.infolist():
            source = ROOT / info.filename
            assert handle.read(info.filename) == source.read_bytes()


@pytest.mark.parametrize(
    "relative_path",
    [
        "tree_counter/.hidden",
        "tree_counter/internal/notes.md",
        "tree_counter/model.pt",
        "tree_counter/model.pth",
        "tree_counter/model.onnx",
        "tree_counter/image.tif",
        "tree_counter/wheel.whl",
        "tree_counter/native.so",
        "tree_counter/native.dylib",
        "tree_counter/native.dll",
        "tree_counter/native.pyd",
        "tree_counter/tool.exe",
    ],
)
def test_build_package_rejects_forbidden_source_paths(
    tmp_path: Path, relative_path: str
) -> None:
    from scripts.package_plugin import build_package

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git"))
    candidate = repo / relative_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"forbidden")

    with pytest.raises(ValueError, match="forbidden|invalid"):
        build_package(repo, tmp_path / "bad.zip")


def test_build_package_rejects_missing_mandatory_files(tmp_path: Path) -> None:
    from scripts.package_plugin import build_package

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git"))
    (repo / "tree_counter" / "LICENSE").unlink()

    with pytest.raises(ValueError, match="LICENSE"):
        build_package(repo, tmp_path / "bad.zip")


def test_build_package_rejects_source_archive_over_20_mib(
    tmp_path: Path,
) -> None:
    from scripts.package_plugin import build_package

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git"))
    # The source file is deliberately just over the archive ceiling.  Stored
    # ZIP members make the resulting size deterministic for this policy test.
    (repo / "tree_counter" / "large.bin").write_bytes(
        b"x" * (20 * 1024 * 1024)
    )

    with pytest.raises(ValueError, match="20 MiB|20 MB|ceiling"):
        build_package(repo, tmp_path / "large.zip")


@pytest.mark.parametrize("relative_path", [".pyc", ".pyo"])
def test_build_package_excludes_python_cache_and_bytecode(
    tmp_path: Path, relative_path: str
) -> None:
    from scripts.package_plugin import build_package

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git"))
    cache = repo / "tree_counter" / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / f"cached{relative_path}").write_bytes(b"cache")

    archive = build_package(repo, tmp_path / "cache.zip")
    with zipfile.ZipFile(archive) as handle:
        assert all(
            "__pycache__" not in name and not name.endswith(relative_path)
            for name in handle.namelist()
        )


@pytest.mark.parametrize("kind", ["root", "directory", "file", "broken"])
def test_build_package_rejects_source_symlinks(
    tmp_path: Path, kind: str
) -> None:
    from scripts.package_plugin import build_package

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git"))
    package = repo / "tree_counter"
    if kind == "root":
        shutil.rmtree(package)
        os.symlink(repo / "README.md", package)
    elif kind == "directory":
        os.symlink(
            repo / "docs", package / "linked-dir", target_is_directory=True
        )
    elif kind == "file":
        os.symlink(repo / "README.md", package / "linked-file.md")
    else:
        os.symlink(repo / "missing-target", package / "broken-link")

    with pytest.raises(ValueError, match="symlink"):
        build_package(repo, tmp_path / "symlink.zip")


def test_builder_marks_archive_members_as_regular_files(
    tmp_path: Path,
) -> None:
    from scripts.package_plugin import build_package

    archive = build_package(ROOT, tmp_path / "regular.zip")
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            mode = (info.external_attr >> 16) & 0o170000
            assert mode == stat.S_IFREG

"""Tests for the committed runtime lock files and their validator."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LOCK_ROOT = ROOT / "tree_counter" / "runtime" / "locks"


def _locks() -> list[Path]:
    return sorted(LOCK_ROOT.rglob("*.txt"))


def test_lock_files_exist() -> None:
    assert _locks()


def test_the_committed_locks_validate() -> None:
    from scripts.check_runtime_locks import validate

    assert validate() == []


def test_the_validator_exits_zero_on_a_valid_tree() -> None:
    from scripts.check_runtime_locks import main

    assert main([]) == 0


@pytest.mark.parametrize("lock", _locks(), ids=lambda path: path.name)
def test_every_requirement_is_exact_and_hashed(lock: Path) -> None:
    from scripts.check_runtime_locks import parse_lock

    entries = parse_lock(lock.read_text(encoding="utf-8"))

    assert entries
    for name, (version, hashes) in entries.items():
        assert re.match(r"\A[0-9]", version), f"{name} is not a real version"
        assert hashes, f"{name} has no hash"
        for digest in hashes:
            assert re.match(r"\A--hash=sha256:[0-9a-f]{64}\Z", digest)


@pytest.mark.parametrize("lock", _locks(), ids=lambda path: path.name)
def test_no_lock_contains_a_range_or_marker(lock: Path) -> None:
    text = lock.read_text(encoding="utf-8")

    for symbol in (">=", "<=", "~=", "!=", ">", "<", ";"):
        assert symbol not in text, f"{lock.name} is not fully pinned"


@pytest.mark.parametrize("lock", _locks(), ids=lambda path: path.name)
def test_no_lock_names_an_unapproved_host(lock: Path) -> None:
    from tree_counter.runtime.catalog import APPROVED_HOSTS

    text = lock.read_text(encoding="utf-8")

    for match in re.findall(r"https?://([^/\s]+)", text):
        assert match.casefold() in APPROVED_HOSTS


def test_every_catalog_profile_has_a_lock() -> None:
    from tree_counter.runtime.catalog import load_catalog

    catalog = load_catalog()

    for component in catalog.components.values():
        for profile in component.profiles:
            assert (LOCK_ROOT / profile.lock).is_file(), profile.lock


def test_every_lock_is_referenced_by_the_catalog() -> None:
    from tree_counter.runtime.catalog import load_catalog

    catalog = load_catalog()
    referenced = {
        profile.lock
        for component in catalog.components.values()
        for profile in component.profiles
    }

    for lock in _locks():
        assert lock.relative_to(LOCK_ROOT).as_posix() in referenced


def test_the_pytorch_locks_pin_the_validated_baseline() -> None:
    from scripts.check_runtime_locks import (
        REQUIRED_TOP_LEVEL_VERSIONS,
        parse_lock,
    )

    baseline = REQUIRED_TOP_LEVEL_VERSIONS["pytorch"]

    for lock in LOCK_ROOT.rglob("pytorch.txt"):
        entries = parse_lock(lock.read_text(encoding="utf-8"))
        for name, version in baseline.items():
            assert entries[name][0] == version, lock.name


def test_the_onnxruntime_locks_pin_the_validated_baseline() -> None:
    from scripts.check_runtime_locks import (
        REQUIRED_TOP_LEVEL_VERSIONS,
        parse_lock,
    )

    baseline = REQUIRED_TOP_LEVEL_VERSIONS["onnxruntime"]

    for lock in LOCK_ROOT.rglob("onnxruntime.txt"):
        entries = parse_lock(lock.read_text(encoding="utf-8"))
        for name, version in baseline.items():
            assert entries[name][0] == version, lock.name


def test_every_lock_includes_numpy() -> None:
    from scripts.check_runtime_locks import parse_lock

    # Both backends decode raw model output through NumPy.
    for lock in _locks():
        entries = parse_lock(lock.read_text(encoding="utf-8"))
        assert "numpy" in entries, lock.name


class TestValidatorDetection:
    """The validator must actually reject the things it claims to."""

    def test_an_unpinned_requirement_is_rejected(self) -> None:
        from scripts.check_runtime_locks import parse_lock

        with pytest.raises(ValueError):
            parse_lock("numpy>=2.0\n")

    def test_an_environment_marker_is_rejected(self) -> None:
        from scripts.check_runtime_locks import parse_lock

        with pytest.raises(ValueError):
            parse_lock('numpy==2.0 ; python_version < "3.13"\n')

    def test_a_non_sha256_hash_is_rejected(self) -> None:
        from scripts.check_runtime_locks import parse_lock

        with pytest.raises(ValueError):
            parse_lock("numpy==2.0 \\\n    --hash=md5:abc\n")

    def test_a_hash_without_a_requirement_is_rejected(self) -> None:
        from scripts.check_runtime_locks import parse_lock

        with pytest.raises(ValueError):
            parse_lock(f"    --hash=sha256:{'a' * 64}\n")

    def test_a_valid_entry_parses(self) -> None:
        from scripts.check_runtime_locks import parse_lock

        entries = parse_lock(
            f"numpy==2.5.2 \\\n    --hash=sha256:{'a' * 64}\n"
        )

        assert entries == {"numpy": ("2.5.2", (f"--hash=sha256:{'a' * 64}",))}


def test_locks_are_shipped_in_the_package() -> None:
    from scripts.check_publication import PACKAGE_MANIFEST

    for lock in _locks():
        relative = (
            "runtime/locks/" + lock.relative_to(LOCK_ROOT).as_posix()
        )
        assert relative in PACKAGE_MANIFEST, relative


def test_locks_are_present_in_the_built_archive(tmp_path: Path) -> None:
    import zipfile

    from scripts.package_plugin import build_package

    archive = build_package(ROOT, tmp_path / "tree-counter.zip")

    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())

    for lock in _locks():
        relative = lock.relative_to(LOCK_ROOT).as_posix()
        assert f"tree_counter/runtime/locks/{relative}" in names

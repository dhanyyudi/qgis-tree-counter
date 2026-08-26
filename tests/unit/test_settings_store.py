"""Tests for the atomic per-user JSON settings store."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _store(path: Path):
    from tree_counter.settings.store import SettingsStore

    return SettingsStore(path)


def test_a_missing_file_loads_an_empty_document(tmp_path: Path) -> None:
    from tree_counter.settings.store import SCHEMA_VERSION

    document = _store(tmp_path / "settings.json").load()

    assert document["schema_version"] == SCHEMA_VERSION
    assert document["presets"] == {}
    assert document["trusted_models"] == {}


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    store = _store(tmp_path / "settings.json")

    document = store.load()
    document["presets"]["a" * 64] = {"filename": "best.onnx"}
    store.save(document)

    assert _store(tmp_path / "settings.json").load() == document


def test_save_creates_the_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "settings.json"
    store = _store(path)

    store.save(store.load())

    assert path.is_file()


def test_the_stored_file_is_owner_only_where_supported(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    store = _store(path)

    store.save(store.load())

    if os.name == "posix":
        assert path.stat().st_mode & 0o077 == 0


def test_saving_replaces_atomically_and_leaves_no_temporary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    store = _store(path)

    store.save(store.load())

    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "settings.json"
    ]


def test_an_interrupted_replacement_keeps_the_previous_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    store = _store(path)
    document = store.load()
    document["presets"]["a" * 64] = {"filename": "best.onnx"}
    store.save(document)

    # A crash between writing the temporary file and replacing the target
    # leaves a stray temporary; the committed document must be untouched.
    stray = path.with_name(path.name + ".tmp-crash")
    stray.write_text("{ truncated", encoding="utf-8")

    assert _store(path).load() == document


def test_a_stale_temporary_is_removed_by_the_next_save(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    store = _store(path)
    store.save(store.load())
    stray = path.with_name(path.name + ".tmp-crash")
    stray.write_text("{ truncated", encoding="utf-8")

    store.save(store.load())

    assert not stray.exists()


def test_corrupt_json_is_reported_without_erasing_the_file(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.store import SettingsStoreError

    path = tmp_path / "settings.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(SettingsStoreError):
        _store(path).load()

    assert path.read_text(encoding="utf-8") == "{ not json"


def test_an_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    from tree_counter.settings.store import SettingsStoreError

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"schema_version": 99, "presets": {}}), encoding="utf-8"
    )

    with pytest.raises(SettingsStoreError):
        _store(path).load()

    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 99


@pytest.mark.parametrize(
    "payload",
    ["[]", '"text"', "null", "42", '{"schema_version": "1"}'],
)
def test_a_non_document_payload_is_rejected(
    tmp_path: Path, payload: str
) -> None:
    from tree_counter.settings.store import SettingsStoreError

    path = tmp_path / "settings.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(SettingsStoreError):
        _store(path).load()


def test_missing_sections_are_restored_without_losing_others(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "presets": {"a" * 64: {"f": 1}}}
        ),
        encoding="utf-8",
    )

    document = _store(path).load()

    assert document["presets"] == {"a" * 64: {"f": 1}}
    assert document["trusted_models"] == {}


def test_a_non_mapping_section_is_rejected(tmp_path: Path) -> None:
    from tree_counter.settings.store import SettingsStoreError

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"schema_version": 1, "presets": []}), encoding="utf-8"
    )

    with pytest.raises(SettingsStoreError):
        _store(path).load()


def test_saving_a_non_serializable_document_is_rejected(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.store import SettingsStoreError

    path = tmp_path / "settings.json"
    store = _store(path)
    document = store.load()
    document["presets"]["a" * 64] = {"bad": object()}

    with pytest.raises(SettingsStoreError):
        store.save(document)

    assert not path.exists()


def test_saving_a_document_with_a_path_value_is_rejected(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.store import SettingsStoreError

    store = _store(tmp_path / "settings.json")
    document = store.load()
    document["presets"]["a" * 64] = {
        "filename": "/home/someone/models/best.pt"
    }

    with pytest.raises(SettingsStoreError):
        store.save(document)


def test_saving_a_windows_path_value_is_rejected(tmp_path: Path) -> None:
    from tree_counter.settings.store import SettingsStoreError

    store = _store(tmp_path / "settings.json")
    document = store.load()
    document["presets"]["a" * 64] = {
        "filename": "C:\\models\\someone\\best.pt"
    }

    with pytest.raises(SettingsStoreError):
        store.save(document)


def test_the_default_path_uses_the_injected_provider(tmp_path: Path) -> None:
    from tree_counter.settings.store import (
        SETTINGS_FILE_NAME,
        default_settings_path,
    )

    path = default_settings_path(lambda: tmp_path)

    assert path == tmp_path / SETTINGS_FILE_NAME


def test_the_default_path_provider_falls_back_outside_qgis() -> None:
    from tree_counter.settings.store import (
        SETTINGS_FILE_NAME,
        default_settings_path,
    )

    path = default_settings_path()

    assert path.name == SETTINGS_FILE_NAME
    assert path.is_absolute()


def test_stored_json_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = _store(path)
    document = store.load()
    document["presets"] = {"b" * 64: {"z": 1, "a": 2}}

    store.save(document)
    first = path.read_bytes()
    store.save(document)

    assert path.read_bytes() == first
    assert first.index(b'"a"') < first.index(b'"z"')

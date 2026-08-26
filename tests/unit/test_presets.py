"""Tests for per-model inference presets."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _identity(name: str = "best.onnx", digest: str = "a" * 64):
    from tree_counter.settings.trust import ModelIdentity

    return ModelIdentity(name, digest, Path(name).suffix)


def _presets(tmp_path: Path, warn=None):
    from tree_counter.settings.presets import PresetStore
    from tree_counter.settings.store import SettingsStore

    store = SettingsStore(tmp_path / "settings.json")
    if warn is None:
        return PresetStore(store)
    return PresetStore(store, warn=warn)


def _settings(**overrides):
    from tree_counter.core.types import InferenceSettings

    return InferenceSettings(**overrides)


def test_an_unknown_model_returns_the_defaults(tmp_path: Path) -> None:
    preset = _presets(tmp_path).load(_identity())

    assert preset.settings == _settings()
    assert preset.last_backend is None


def test_a_preset_round_trips_every_setting(tmp_path: Path) -> None:
    from tree_counter.settings.presets import ModelPreset

    identity = _identity()
    settings = _settings(
        confidence=0.4,
        nms_iou=0.55,
        duplicate_iou=0.35,
        tile_size=1024,
        overlap_percent=45,
        selected_class_ids=(2, 0),
        requested_device="cpu",
    )
    store = _presets(tmp_path)

    store.save(ModelPreset(identity, settings, last_backend="onnx"))
    restored = _presets(tmp_path).load(identity)

    assert restored.settings == settings
    assert restored.settings.selected_class_ids == (2, 0)
    assert restored.last_backend == "onnx"
    assert restored.identity.filename == identity.filename


def test_presets_are_keyed_by_hash_not_by_filename(tmp_path: Path) -> None:
    from tree_counter.settings.presets import ModelPreset

    first = _identity("best.onnx", "a" * 64)
    second = _identity("best.onnx", "b" * 64)
    store = _presets(tmp_path)

    store.save(ModelPreset(first, _settings(tile_size=1024)))
    store.save(ModelPreset(second, _settings(tile_size=512)))

    assert _presets(tmp_path).load(first).settings.tile_size == 1024
    assert _presets(tmp_path).load(second).settings.tile_size == 512


def test_saving_twice_replaces_the_record(tmp_path: Path) -> None:
    from tree_counter.settings.presets import ModelPreset

    identity = _identity()
    store = _presets(tmp_path)

    store.save(ModelPreset(identity, _settings(tile_size=1024)))
    store.save(ModelPreset(identity, _settings(tile_size=320)))

    assert _presets(tmp_path).load(identity).settings.tile_size == 320
    document = json.loads(
        (tmp_path / "settings.json").read_text(encoding="utf-8")
    )
    assert len(document["presets"]) == 1


def test_a_renamed_model_keeps_its_preset_and_updates_the_filename(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.presets import ModelPreset

    original = _identity("best.onnx")
    store = _presets(tmp_path)
    store.save(ModelPreset(original, _settings(tile_size=1024)))

    renamed = _identity("renamed.onnx")
    loaded = _presets(tmp_path).load(renamed)

    assert loaded.settings.tile_size == 1024
    assert loaded.identity.filename == "renamed.onnx"


def test_an_invalid_stored_value_falls_back_with_a_warning(
    tmp_path: Path,
) -> None:
    identity = _identity()
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presets": {
                    identity.sha256: {
                        "filename": "best.onnx",
                        "settings": {"tile_size": 641, "confidence": 0.4},
                    }
                },
                "trusted_models": {},
            }
        ),
        encoding="utf-8",
    )
    warnings: list[str] = []

    preset = _presets(tmp_path, warn=warnings.append).load(identity)

    assert preset.settings == _settings()
    assert len(warnings) == 1
    assert identity.sha256 not in warnings[0]


def test_an_unknown_stored_field_falls_back_with_a_warning(
    tmp_path: Path,
) -> None:
    identity = _identity()
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presets": {
                    identity.sha256: {
                        "filename": "best.onnx",
                        "settings": {"min_distance": 5},
                    }
                },
                "trusted_models": {},
            }
        ),
        encoding="utf-8",
    )
    warnings: list[str] = []

    preset = _presets(tmp_path, warn=warnings.append).load(identity)

    assert preset.settings == _settings()
    assert warnings


@pytest.mark.parametrize(
    "record", ["text", 5, [], {"settings": []}, {"settings": None}]
)
def test_a_malformed_record_falls_back_with_a_warning(
    tmp_path: Path, record: object
) -> None:
    identity = _identity()
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presets": {identity.sha256: record},
                "trusted_models": {},
            }
        ),
        encoding="utf-8",
    )
    warnings: list[str] = []

    preset = _presets(tmp_path, warn=warnings.append).load(identity)

    assert preset.settings == _settings()
    assert warnings


def test_a_fallback_does_not_erase_the_stored_record(tmp_path: Path) -> None:
    identity = _identity()
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presets": {
                    identity.sha256: {"settings": {"tile_size": 641}}
                },
                "trusted_models": {},
            }
        ),
        encoding="utf-8",
    )

    _presets(tmp_path).load(identity)

    assert json.loads(path.read_text(encoding="utf-8"))["presets"]


def test_invalid_selected_class_ids_fall_back(tmp_path: Path) -> None:
    identity = _identity()
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presets": {
                    identity.sha256: {
                        "settings": {"selected_class_ids": [0, 0]}
                    }
                },
                "trusted_models": {},
            }
        ),
        encoding="utf-8",
    )
    warnings: list[str] = []

    preset = _presets(tmp_path, warn=warnings.append).load(identity)

    assert preset.settings.selected_class_ids == ()
    assert warnings


def test_a_stored_preset_never_contains_a_model_path(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.presets import ModelPreset

    store = _presets(tmp_path)
    store.save(ModelPreset(_identity(), _settings()))

    raw = (tmp_path / "settings.json").read_text(encoding="utf-8")

    assert str(tmp_path) not in raw
    assert "/" not in raw.replace("\\/", "")


def test_saving_rejects_a_non_preset(tmp_path: Path) -> None:
    from tree_counter.errors import TreeCounterError

    with pytest.raises(TreeCounterError):
        _presets(tmp_path).save({"settings": {}})


def test_removing_a_preset_restores_the_defaults(tmp_path: Path) -> None:
    from tree_counter.settings.presets import ModelPreset

    identity = _identity()
    store = _presets(tmp_path)
    store.save(ModelPreset(identity, _settings(tile_size=1024)))

    store.remove(identity)

    assert _presets(tmp_path).load(identity).settings == _settings()


def test_removing_an_absent_preset_is_a_no_op(tmp_path: Path) -> None:
    _presets(tmp_path).remove(_identity())

    assert _presets(tmp_path).load(_identity()).settings == _settings()


def test_the_preset_provenance_carries_filename_hash_and_classes(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.presets import ModelPreset

    preset = ModelPreset(
        _identity(),
        _settings(selected_class_ids=(1,)),
        last_backend="onnx",
    )

    provenance = preset.as_provenance()

    assert provenance["model_filename"] == "best.onnx"
    assert provenance["model_sha256"] == "a" * 64
    assert provenance["selected_class_ids"] == [1]
    assert provenance["tile_size"] == 640
    assert "model_path" not in provenance

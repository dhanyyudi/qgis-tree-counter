"""Per-model inference presets keyed by model SHA-256.

A stored preset that no longer validates is replaced by the defaults and
reported through a warning callback; it is never silently applied and never
deletes the user's stored record.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tree_counter.core.types import InferenceSettings
from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.settings.store import SettingsStore
from tree_counter.settings.trust import ModelIdentity

PRESET_SECTION = "presets"
SETTING_FIELDS = (
    "confidence",
    "nms_iou",
    "duplicate_iou",
    "tile_size",
    "overlap_percent",
    "selected_class_ids",
    "requested_device",
)

Warn = Callable[[str], None]


class PresetError(TreeCounterError):
    """A preset argument is not a usable preset record."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.INVALID_SETTINGS, diagnostic_detail=detail)


@dataclass(frozen=True)
class ModelPreset:
    """The remembered inference settings for one exact model."""

    identity: ModelIdentity
    settings: InferenceSettings
    last_backend: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ModelIdentity):
            raise PresetError("identity must be a ModelIdentity")
        if not isinstance(self.settings, InferenceSettings):
            raise PresetError("settings must be an InferenceSettings")
        if self.last_backend is not None and (
            not isinstance(self.last_backend, str) or not self.last_backend
        ):
            raise PresetError("last_backend must be a non-empty string")

    def as_provenance(self) -> dict[str, Any]:
        """Return the run-provenance fields for this preset."""

        settings = self.settings
        provenance: dict[str, Any] = dict(self.identity.as_provenance())
        provenance.update(
            {
                "confidence": settings.confidence,
                "nms_iou": settings.nms_iou,
                "duplicate_iou": settings.duplicate_iou,
                "tile_size": settings.tile_size,
                "overlap_percent": settings.overlap_percent,
                "selected_class_ids": list(settings.selected_class_ids),
                "requested_device": settings.requested_device,
            }
        )
        if self.last_backend is not None:
            provenance["backend"] = self.last_backend
        return provenance

    def as_record(self) -> dict[str, Any]:
        """Return the persisted record, without any model path."""

        settings = self.settings
        record: dict[str, Any] = {
            "filename": self.identity.filename,
            "settings": {
                "confidence": settings.confidence,
                "nms_iou": settings.nms_iou,
                "duplicate_iou": settings.duplicate_iou,
                "tile_size": settings.tile_size,
                "overlap_percent": settings.overlap_percent,
                "selected_class_ids": list(settings.selected_class_ids),
                "requested_device": settings.requested_device,
            },
        }
        if self.last_backend is not None:
            record["last_backend"] = self.last_backend
        return record


def _ignore(message: str) -> None:
    """Discard a warning when the caller supplied no handler."""


class PresetStore:
    """Load and save per-model presets in the shared settings document."""

    def __init__(self, store: SettingsStore, warn: Warn = _ignore) -> None:
        self._store = store
        self._warn = warn

    def load(self, identity: Any) -> ModelPreset:
        """Return the stored preset for *identity*, or safe defaults."""

        model = self._require_identity(identity)
        record = self._store.load()[PRESET_SECTION].get(model.sha256)
        if record is None:
            return ModelPreset(model, InferenceSettings())
        if not isinstance(record, Mapping):
            self._warn(
                f"Stored settings for {model.filename} were unreadable and "
                "the defaults were restored."
            )
            return ModelPreset(model, InferenceSettings())
        settings = self._settings(record.get("settings"), model.filename)
        last_backend = record.get("last_backend")
        if not isinstance(last_backend, str) or not last_backend:
            last_backend = None
        # The identity always comes from the file on disk, so a renamed but
        # identical model keeps its preset under its current filename.
        return ModelPreset(model, settings, last_backend)

    def save(self, preset: Any) -> None:
        """Persist *preset* under its model hash."""

        if not isinstance(preset, ModelPreset):
            raise PresetError("preset must be a ModelPreset")
        document = self._store.load()
        document[PRESET_SECTION][preset.identity.sha256] = preset.as_record()
        self._store.save(document)

    def remove(self, identity: Any) -> None:
        """Forget the preset for *identity*, if one is stored."""

        model = self._require_identity(identity)
        document = self._store.load()
        if document[PRESET_SECTION].pop(model.sha256, None) is not None:
            self._store.save(document)

    @staticmethod
    def _require_identity(identity: Any) -> ModelIdentity:
        if not isinstance(identity, ModelIdentity):
            raise PresetError("identity must be a ModelIdentity")
        return identity

    def _settings(self, payload: Any, filename: str) -> InferenceSettings:
        # A stored record always carries its settings; a missing or null
        # block means the record was damaged, not that defaults were meant.
        if not isinstance(payload, Mapping):
            self._warn(
                f"Stored settings for {filename} were unreadable and the "
                "defaults were restored."
            )
            return InferenceSettings()
        unknown = set(payload) - set(SETTING_FIELDS)
        if unknown:
            self._warn(
                f"Stored settings for {filename} contained unsupported "
                "values and the defaults were restored."
            )
            return InferenceSettings()
        values = dict(payload)
        class_ids = values.get("selected_class_ids")
        if class_ids is not None:
            if isinstance(class_ids, (str, bytes)) or not isinstance(
                class_ids, Iterable
            ):
                self._warn(
                    f"Stored classes for {filename} were invalid and the "
                    "defaults were restored."
                )
                return InferenceSettings()
            values["selected_class_ids"] = tuple(class_ids)
        try:
            return InferenceSettings(**values)
        except (TreeCounterError, TypeError, ValueError):
            self._warn(
                f"Stored settings for {filename} were invalid and the "
                "defaults were restored."
            )
            return InferenceSettings()


__all__ = ["ModelPreset", "PresetError", "PresetStore"]

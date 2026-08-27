"""Model identity, streamed hashing, and PyTorch checkpoint trust.

A ``.pt`` checkpoint is only handed to the worker after the user has
confirmed its exact SHA-256, because loading a PyTorch checkpoint executes
code from the file. ``.onnx`` graphs carry no such marker. Nothing here ever
records the absolute path of a user's model.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.settings.store import SettingsStore

HASH_CHUNK_BYTES = 1024 * 1024
SUPPORTED_MODEL_SUFFIXES = (".onnx", ".pt")
TRUST_REQUIRED_SUFFIXES = (".pt",)
TRUST_SECTION = "trusted_models"

_SHA256_PATTERN = re.compile(r"\A[0-9a-fA-F]{64}\Z")


class ModelError(TreeCounterError):
    """The selected model file cannot be identified or is unsupported."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.INVALID_MODEL, diagnostic_detail=detail)


def hash_file(path: Path | str, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    """Return the SHA-256 of a file read in bounded chunks.

    Model checkpoints can be hundreds of megabytes, so the file is streamed
    rather than read into memory.
    """

    if chunk_bytes <= 0:
        raise ModelError("chunk_bytes must be positive")
    candidate = Path(path)
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as stream:
            while True:
                chunk = stream.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
    except IsADirectoryError as exc:
        raise ModelError("the model path is a directory") from exc
    except FileNotFoundError as exc:
        raise ModelError("the model file does not exist") from exc
    except (OSError, PermissionError) as exc:
        raise ModelError(f"the model file is unreadable: {exc}") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelIdentity:
    """The only model facts that may be persisted or published."""

    filename: str
    sha256: str
    suffix: str

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not self.filename:
            raise ModelError("filename must be a non-empty string")
        if (
            "/" in self.filename
            or "\\" in self.filename
            or self.filename in (".", "..")
        ):
            raise ModelError("filename must not contain a path")
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.match(
            self.sha256
        ):
            raise ModelError("sha256 must be a hexadecimal SHA-256 digest")
        if not isinstance(self.suffix, str):
            raise ModelError("suffix must be a string")
        suffix = self.suffix.casefold()
        if suffix not in SUPPORTED_MODEL_SUFFIXES:
            raise ModelError(f"unsupported model type: {suffix!r}")
        object.__setattr__(self, "sha256", self.sha256.casefold())
        object.__setattr__(self, "suffix", suffix)

    @property
    def requires_trust_confirmation(self) -> bool:
        """Return whether this model type needs an explicit confirmation."""

        return self.suffix in TRUST_REQUIRED_SUFFIXES

    def as_provenance(self) -> dict[str, str]:
        """Return the run-provenance fields for this model."""

        return {
            "model_filename": self.filename,
            "model_sha256": self.sha256,
        }


def identify_model(path: Path | str) -> ModelIdentity:
    """Return the identity of a model file, streaming its hash."""

    candidate = Path(path)
    suffix = candidate.suffix.casefold()
    if suffix not in SUPPORTED_MODEL_SUFFIXES:
        raise ModelError(f"unsupported model type: {suffix!r}")
    return ModelIdentity(candidate.name, hash_file(candidate), suffix)


class TrustStore:
    """Records which PyTorch checkpoint hashes the user has confirmed."""

    def __init__(
        self,
        store: SettingsStore,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._clock = clock

    @staticmethod
    def _identity(identity: Any) -> ModelIdentity:
        if not isinstance(identity, ModelIdentity):
            raise ModelError("identity must be a ModelIdentity")
        return identity

    def _record(self, identity: ModelIdentity) -> Mapping[str, Any] | None:
        document = self._store.load()
        record = document[TRUST_SECTION].get(identity.sha256)
        # A record damaged by hand-editing must not grant trust.
        if not isinstance(record, Mapping):
            return None
        if set(record) != {"filename", "confirmed_at"}:
            return None
        filename = record["filename"]
        confirmed_at = record["confirmed_at"]
        if (
            not isinstance(filename, str)
            or not filename
            or "/" in filename
            or "\\" in filename
            or filename in (".", "..")
        ):
            return None
        if (
            isinstance(confirmed_at, bool)
            or not isinstance(confirmed_at, int)
            or confirmed_at < 0
        ):
            return None
        return record

    def is_trusted(self, identity: Any) -> bool:
        """Return whether the model may be sent to the worker."""

        model = self._identity(identity)
        if not model.requires_trust_confirmation:
            return True
        return self._record(model) is not None

    def requires_confirmation(self, identity: Any) -> bool:
        """Return whether the user must confirm this exact model content."""

        return not self.is_trusted(identity)

    def confirm(self, identity: Any) -> None:
        """Record the user's confirmation of this exact model content.

        Confirming a model type that needs no marker is a no-op, so an
        ONNX graph never gains a PyTorch trust record.
        """

        model = self._identity(identity)
        if not model.requires_trust_confirmation:
            return
        document = self._store.load()
        document[TRUST_SECTION][model.sha256] = {
            "filename": model.filename,
            "confirmed_at": int(self._clock()),
        }
        self._store.save(document)

    def revoke(self, identity: Any) -> None:
        """Remove a confirmation so the model is challenged again."""

        model = self._identity(identity)
        document = self._store.load()
        if document[TRUST_SECTION].pop(model.sha256, None) is not None:
            self._store.save(document)


__all__ = [
    "HASH_CHUNK_BYTES",
    "SUPPORTED_MODEL_SUFFIXES",
    "TRUST_REQUIRED_SUFFIXES",
    "ModelError",
    "ModelIdentity",
    "TrustStore",
    "hash_file",
    "identify_model",
]

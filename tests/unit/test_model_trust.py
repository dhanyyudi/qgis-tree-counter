"""Tests for model identity, streamed hashing, and PyTorch trust."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _model(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def _trust(tmp_path: Path, clock=None):
    from tree_counter.settings.store import SettingsStore
    from tree_counter.settings.trust import TrustStore

    store = SettingsStore(tmp_path / "settings.json")
    if clock is None:
        return TrustStore(store)
    return TrustStore(store, clock=clock)


def test_hash_matches_hashlib(tmp_path: Path) -> None:
    from tree_counter.settings.trust import hash_file

    payload = b"model bytes" * 1000
    path = _model(tmp_path, "best.onnx", payload)

    assert hash_file(path) == hashlib.sha256(payload).hexdigest()


def test_hash_streams_the_file_in_bounded_chunks(tmp_path: Path) -> None:
    from tree_counter.settings.trust import HASH_CHUNK_BYTES, hash_file

    payload = b"x" * (HASH_CHUNK_BYTES * 3 + 7)
    path = _model(tmp_path, "best.pt", payload)
    sizes: list[int] = []
    real_open = Path.open

    class _Recording:
        def __init__(self, handle) -> None:
            self._handle = handle

        def read(self, size: int = -1) -> bytes:
            sizes.append(size)
            return self._handle.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            self._handle.close()

    def _open(self, *args: object, **kwargs: object):
        return _Recording(real_open(self, *args, **kwargs))

    original = Path.open
    Path.open = _open  # type: ignore[method-assign]
    try:
        hash_file(path)
    finally:
        Path.open = original  # type: ignore[method-assign]

    assert sizes
    assert max(sizes) <= HASH_CHUNK_BYTES
    assert len(sizes) >= 4


def test_hash_of_a_large_sparse_file_is_bounded(tmp_path: Path) -> None:
    from tree_counter.settings.trust import HASH_CHUNK_BYTES, hash_file

    path = tmp_path / "big.onnx"
    with path.open("wb") as handle:
        handle.truncate(HASH_CHUNK_BYTES * 8)

    assert len(hash_file(path)) == 64


def test_hash_rejects_a_missing_file(tmp_path: Path) -> None:
    from tree_counter.errors import TreeCounterError
    from tree_counter.settings.trust import hash_file

    with pytest.raises(TreeCounterError):
        hash_file(tmp_path / "absent.onnx")


def test_hash_rejects_a_directory(tmp_path: Path) -> None:
    from tree_counter.errors import TreeCounterError
    from tree_counter.settings.trust import hash_file

    with pytest.raises(TreeCounterError):
        hash_file(tmp_path)


def test_identify_model_returns_filename_hash_and_suffix(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.trust import identify_model

    path = _model(tmp_path, "Best.PT", b"weights")

    identity = identify_model(path)

    assert identity.filename == "Best.PT"
    assert identity.suffix == ".pt"
    assert identity.sha256 == hashlib.sha256(b"weights").hexdigest()


def test_identify_model_rejects_an_unsupported_suffix(
    tmp_path: Path,
) -> None:
    from tree_counter.errors import TreeCounterError
    from tree_counter.settings.trust import identify_model

    with pytest.raises(TreeCounterError):
        identify_model(_model(tmp_path, "best.pkl", b"weights"))


def test_model_identity_never_exposes_the_absolute_path(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.trust import identify_model

    path = _model(tmp_path, "best.pt", b"weights")

    identity = identify_model(path)
    rendered = repr(identity) + str(identity.as_provenance())

    assert str(tmp_path) not in rendered
    assert "/" not in identity.filename


def test_provenance_contains_only_filename_and_hash(tmp_path: Path) -> None:
    from tree_counter.settings.trust import identify_model

    identity = identify_model(_model(tmp_path, "best.onnx", b"weights"))

    assert set(identity.as_provenance()) == {
        "model_filename",
        "model_sha256",
    }


def test_identity_rejects_a_filename_with_a_separator() -> None:
    from tree_counter.errors import TreeCounterError
    from tree_counter.settings.trust import ModelIdentity

    with pytest.raises(TreeCounterError):
        ModelIdentity("dir/best.pt", "a" * 64, ".pt")


def test_identity_normalizes_the_hash_case() -> None:
    from tree_counter.settings.trust import ModelIdentity

    identity = ModelIdentity("best.pt", "A" * 64, ".PT")

    assert identity.sha256 == "a" * 64
    assert identity.suffix == ".pt"


def test_identity_rejects_a_malformed_hash() -> None:
    from tree_counter.errors import TreeCounterError
    from tree_counter.settings.trust import ModelIdentity

    with pytest.raises(TreeCounterError):
        ModelIdentity("best.pt", "zz", ".pt")


def test_a_pt_model_requires_confirmation_once(tmp_path: Path) -> None:
    from tree_counter.settings.trust import identify_model

    trust = _trust(tmp_path)
    identity = identify_model(_model(tmp_path, "best.pt", b"weights"))

    assert trust.requires_confirmation(identity) is True
    trust.confirm(identity)
    assert trust.requires_confirmation(identity) is False
    assert trust.is_trusted(identity) is True


def test_confirmation_survives_a_new_store_instance(tmp_path: Path) -> None:
    from tree_counter.settings.trust import identify_model

    identity = identify_model(_model(tmp_path, "best.pt", b"weights"))
    _trust(tmp_path).confirm(identity)

    assert _trust(tmp_path).requires_confirmation(identity) is False


def test_changed_content_requires_a_new_confirmation(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.trust import identify_model

    trust = _trust(tmp_path)
    path = _model(tmp_path, "best.pt", b"weights")
    trust.confirm(identify_model(path))

    path.write_bytes(b"different weights")

    assert trust.requires_confirmation(identify_model(path)) is True


def test_a_renamed_but_identical_model_stays_trusted(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.trust import identify_model

    trust = _trust(tmp_path)
    trust.confirm(identify_model(_model(tmp_path, "best.pt", b"weights")))

    renamed = identify_model(_model(tmp_path, "copy.pt", b"weights"))

    assert trust.requires_confirmation(renamed) is False


def test_an_onnx_model_never_requires_confirmation(tmp_path: Path) -> None:
    from tree_counter.settings.trust import identify_model

    trust = _trust(tmp_path)
    identity = identify_model(_model(tmp_path, "best.onnx", b"graph"))

    assert trust.requires_confirmation(identity) is False
    assert trust.is_trusted(identity) is True


def test_confirming_an_onnx_model_writes_no_trust_marker(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.store import SettingsStore
    from tree_counter.settings.trust import identify_model

    trust = _trust(tmp_path)
    identity = identify_model(_model(tmp_path, "best.onnx", b"graph"))

    trust.confirm(identity)

    document = SettingsStore(tmp_path / "settings.json").load()
    assert document["trusted_models"] == {}


def test_a_trust_record_stores_only_the_filename_and_timestamp(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.store import SettingsStore
    from tree_counter.settings.trust import identify_model

    trust = _trust(tmp_path, clock=lambda: 1_700_000_000.0)
    identity = identify_model(_model(tmp_path, "best.pt", b"weights"))

    trust.confirm(identity)

    record = SettingsStore(tmp_path / "settings.json").load()[
        "trusted_models"
    ][identity.sha256]
    assert set(record) == {"filename", "confirmed_at"}
    assert record["filename"] == "best.pt"
    assert record["confirmed_at"] == 1_700_000_000


def test_the_stored_file_never_contains_the_model_directory(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.trust import identify_model

    trust = _trust(tmp_path)
    trust.confirm(identify_model(_model(tmp_path, "best.pt", b"weights")))

    raw = (tmp_path / "settings.json").read_text(encoding="utf-8")

    assert str(tmp_path) not in raw


def test_revoking_a_model_requires_confirmation_again(
    tmp_path: Path,
) -> None:
    from tree_counter.settings.trust import identify_model

    trust = _trust(tmp_path)
    identity = identify_model(_model(tmp_path, "best.pt", b"weights"))
    trust.confirm(identity)

    trust.revoke(identity)

    assert trust.requires_confirmation(identity) is True


def test_a_corrupt_trust_record_is_treated_as_untrusted(
    tmp_path: Path,
) -> None:
    import json

    from tree_counter.settings.trust import identify_model

    identity = identify_model(_model(tmp_path, "best.pt", b"weights"))
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trusted_models": {identity.sha256: "yes"},
                "presets": {},
            }
        ),
        encoding="utf-8",
    )

    assert _trust(tmp_path).requires_confirmation(identity) is True


@pytest.mark.parametrize(
    "record",
    [
        {"filename": "best.pt"},
        {"filename": "best.pt", "confirmed_at": "yesterday"},
        {"filename": "best.pt", "confirmed_at": True},
        {"filename": "best.pt", "confirmed_at": 1, "extra": "grant"},
    ],
)
def test_partial_or_extra_trust_records_are_treated_as_untrusted(
    tmp_path: Path, record: dict[str, object]
) -> None:
    import json

    from tree_counter.settings.trust import identify_model

    identity = identify_model(_model(tmp_path, "best.pt", b"weights"))
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trusted_models": {identity.sha256: record},
                "presets": {},
            }
        ),
        encoding="utf-8",
    )

    assert _trust(tmp_path).requires_confirmation(identity) is True

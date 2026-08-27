"""Contract-level tests for backend selection and strict tile handling."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


def _description(suffix: str, backend: str):
    from tree_counter.worker.backend_base import ModelDescription

    return ModelDescription(
        filename=f"model{suffix}",
        sha256="a" * 64,
        model_format=suffix[1:],
        task="detect",
        family="yolo11",
        class_names=("tree",),
        input_width=640,
        input_height=640,
        dynamic_shape=False,
        backend=backend,
        provider=backend,
        device="cpu",
    )


class _Backend:
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix
        self.closed = False

    def capabilities(self):
        return ()

    def inspect(self, model_path: str, model_sha256: str):
        return _description(self.suffix, self.suffix[1:])

    def close(self):
        self.closed = True


def test_selecting_backend_switches_when_model_format_changes(
    monkeypatch,
) -> None:
    from tree_counter.worker import runner

    created: list[_Backend] = []

    def build(name: str):
        backend = _Backend(".onnx" if name == "onnx" else ".pt")
        created.append(backend)
        return backend

    monkeypatch.setattr(runner, "_build_backend", build)
    selected = runner.SelectingBackend()

    selected.inspect("model.onnx", "a" * 64)
    selected.inspect("model.pt", "a" * 64)

    assert len(created) == 2
    assert created[0].closed is True
    assert created[1].closed is False


def test_selecting_backend_switches_back_to_onnx(monkeypatch) -> None:
    from tree_counter.worker import runner

    created: list[_Backend] = []

    def build(name: str):
        backend = _Backend(".onnx" if name == "onnx" else ".pt")
        created.append(backend)
        return backend

    monkeypatch.setattr(runner, "_build_backend", build)
    selected = runner.SelectingBackend()

    selected.inspect("model.pt", "a" * 64)
    selected.inspect("model.onnx", "a" * 64)

    assert [item.suffix for item in created] == [
        ".pt", ".onnx"
    ]
    assert created[0].closed is True


def test_worker_rejects_tile_with_unsupported_encoding() -> None:
    from tree_counter.core.protocol import ProtocolError, validate_host_message

    message = {
        "type": "tile",
        "protocol_version": 1,
        "request_id": "tile",
        "run_id": "run",
        "tile_id": "tile",
        "tile_path": "tile.raw",
        "tile_encoding": "png",
        "x_offset": 0,
        "y_offset": 0,
        "valid_width": 1,
        "valid_height": 1,
        "model_width": 1,
        "model_height": 1,
    }

    with pytest.raises(ProtocolError):
        validate_host_message(message)

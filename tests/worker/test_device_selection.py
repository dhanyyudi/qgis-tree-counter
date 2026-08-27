"""Device selection as the worker actually applies it during a run."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


def _settings(**overrides):
    from tree_counter.core.types import InferenceSettings

    return InferenceSettings(**overrides)


def test_the_requested_device_reaches_selection() -> None:
    from tree_counter.worker.capabilities import select_device

    settings = _settings(requested_device="cpu")

    selection = select_device(
        requested=settings.requested_device,
        model_format="onnx",
        backend_accelerators=("coreml",),
        platform="darwin",
        machine="arm64",
    )

    assert selection.device == "cpu"


def test_the_default_device_is_auto() -> None:
    assert _settings().requested_device == "auto"


def test_auto_on_a_cpu_only_machine_warns_once() -> None:
    from tree_counter.worker.capabilities import select_device

    selection = select_device(
        requested="auto",
        model_format="onnx",
        backend_accelerators=(),
        platform="linux",
        machine="x86_64",
    )

    assert selection.device == "cpu"
    assert len(selection.warnings) == 1


def test_an_explicit_device_never_falls_back_silently() -> None:
    from tree_counter.worker.capabilities import (
        DeviceUnavailable,
        select_device,
    )

    # Falling back here would hide that the user's explicit choice was
    # ignored, and the run would be slower than they asked for with no
    # indication why.
    with pytest.raises(DeviceUnavailable):
        select_device(
            requested="cuda",
            model_format="onnx",
            backend_accelerators=(),
            platform="linux",
            machine="x86_64",
        )


def test_every_supported_device_name_is_selectable_somewhere() -> None:
    from tree_counter.constants import SUPPORTED_DEVICES
    from tree_counter.worker.capabilities import select_device

    cases = {
        "auto": ("onnx", (), "linux", "x86_64"),
        "cpu": ("onnx", (), "linux", "x86_64"),
        "cuda": ("onnx", ("cuda",), "linux", "x86_64"),
        "mps": ("pt", ("mps",), "darwin", "arm64"),
        "coreml": ("onnx", ("coreml",), "darwin", "arm64"),
    }

    assert set(cases) == set(SUPPORTED_DEVICES)
    for device, (fmt, backend, platform, machine) in cases.items():
        selection = select_device(
            requested=device,
            model_format=fmt,
            backend_accelerators=backend,
            platform=platform,
            machine=machine,
        )
        assert selection.device in (device, "cpu")


def test_device_selection_is_case_insensitive() -> None:
    from tree_counter.worker.capabilities import select_device

    selection = select_device(
        requested="CPU",
        model_format="onnx",
        backend_accelerators=(),
        platform="linux",
        machine="x86_64",
    )

    assert selection.device == "cpu"


def test_a_run_records_the_resolved_device_not_the_request() -> None:
    from tree_counter.worker.backend_base import ModelDescription
    from tree_counter.worker.capabilities import select_device

    selection = select_device(
        requested="auto",
        model_format="onnx",
        backend_accelerators=("coreml",),
        platform="darwin",
        machine="arm64",
    )
    description = ModelDescription(
        filename="best.onnx",
        sha256="a" * 64,
        model_format="onnx",
        task="detect",
        family="yolo11",
        class_names=("oil_palm",),
        input_width=640,
        input_height=640,
        dynamic_shape=False,
        backend="onnxruntime",
        provider="CoreMLExecutionProvider",
        device=selection.device,
    )

    assert description.as_message()["device"] == "coreml"

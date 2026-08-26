"""Tests for accelerator availability and device selection."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


def _select(requested, model_format="onnx", backend=("cpu",),
            platform="darwin", machine="arm64"):
    from tree_counter.worker.capabilities import select_device

    return select_device(
        requested=requested,
        model_format=model_format,
        backend_accelerators=backend,
        platform=platform,
        machine=machine,
    )


def test_cpu_is_always_selectable() -> None:
    for platform, machine in (
        ("darwin", "arm64"),
        ("win32", "AMD64"),
        ("linux", "x86_64"),
    ):
        assert _select("cpu", platform=platform, machine=machine).device == (
            "cpu"
        )


def test_cpu_is_always_offered() -> None:
    from tree_counter.worker.capabilities import available_devices

    devices = available_devices("onnx", (), "linux", "x86_64")

    assert devices[0] == "auto"
    assert "cpu" in devices


def test_auto_falls_back_to_cpu_with_a_warning() -> None:
    selection = _select("auto", backend=())

    assert selection.device == "cpu"
    assert selection.warnings
    assert "CPU" in selection.warnings[0]


def test_auto_prefers_an_available_accelerator() -> None:
    selection = _select(
        "auto", model_format="pt", backend=("cpu", "mps")
    )

    assert selection.device == "mps"
    assert selection.warnings == ()


def test_auto_prefers_cuda_over_other_accelerators() -> None:
    selection = _select(
        "auto",
        model_format="pt",
        backend=("cuda", "mps"),
        platform="linux",
        machine="x86_64",
    )

    assert selection.device == "cuda"


def test_an_unavailable_explicit_device_fails() -> None:
    from tree_counter.worker.capabilities import DeviceUnavailable

    with pytest.raises(DeviceUnavailable):
        _select("coreml", backend=("cpu",))


def test_mps_requires_apple_silicon() -> None:
    from tree_counter.worker.capabilities import DeviceUnavailable

    with pytest.raises(DeviceUnavailable):
        _select(
            "mps",
            model_format="pt",
            backend=("mps",),
            platform="linux",
            machine="x86_64",
        )


def test_mps_requires_a_pytorch_model() -> None:
    from tree_counter.worker.capabilities import DeviceUnavailable

    with pytest.raises(DeviceUnavailable):
        _select("mps", model_format="onnx", backend=("mps",))


def test_mps_is_accepted_for_pt_on_apple_silicon() -> None:
    assert _select(
        "mps", model_format="pt", backend=("mps",)
    ).device == "mps"


def test_coreml_requires_an_onnx_model() -> None:
    from tree_counter.worker.capabilities import DeviceUnavailable

    with pytest.raises(DeviceUnavailable):
        _select("coreml", model_format="pt", backend=("coreml",))


def test_coreml_requires_macos() -> None:
    from tree_counter.worker.capabilities import DeviceUnavailable

    with pytest.raises(DeviceUnavailable):
        _select(
            "coreml",
            backend=("coreml",),
            platform="win32",
            machine="AMD64",
        )


def test_coreml_is_accepted_for_onnx_on_macos() -> None:
    assert _select("coreml", backend=("coreml",)).device == "coreml"


def test_cuda_requires_the_backend_to_report_it() -> None:
    from tree_counter.worker.capabilities import DeviceUnavailable

    with pytest.raises(DeviceUnavailable):
        _select(
            "cuda", backend=("cpu",), platform="linux", machine="x86_64"
        )


def test_cuda_is_never_available_on_macos() -> None:
    from tree_counter.worker.capabilities import DeviceUnavailable

    with pytest.raises(DeviceUnavailable):
        _select("cuda", backend=("cuda",))


def test_cuda_is_accepted_when_reported() -> None:
    assert _select(
        "cuda", backend=("cuda",), platform="linux", machine="x86_64"
    ).device == "cuda"


def test_an_unknown_device_is_rejected() -> None:
    from tree_counter.worker.capabilities import DeviceUnavailable

    with pytest.raises(DeviceUnavailable):
        _select("quantum", backend=("cpu",))


def test_an_empty_device_is_rejected() -> None:
    from tree_counter.worker.capabilities import DeviceUnavailable

    with pytest.raises(DeviceUnavailable):
        _select("")


def test_an_unsupported_model_format_is_rejected() -> None:
    from tree_counter.worker.capabilities import DeviceUnavailable

    with pytest.raises(DeviceUnavailable):
        _select("auto", model_format="tflite")


def test_platform_accelerators_are_format_aware() -> None:
    from tree_counter.worker.capabilities import platform_accelerators

    assert platform_accelerators("darwin", "arm64", "pt") == ("mps",)
    assert platform_accelerators("darwin", "arm64", "onnx") == ("coreml",)
    assert platform_accelerators("linux", "x86_64", "pt") == ("cuda",)
    assert platform_accelerators("win32", "AMD64", "onnx") == ("cuda",)


def test_intel_macs_offer_no_pytorch_accelerator() -> None:
    from tree_counter.worker.capabilities import platform_accelerators

    assert platform_accelerators("darwin", "x86_64", "pt") == ()


def test_available_devices_lists_only_usable_accelerators() -> None:
    from tree_counter.worker.capabilities import available_devices

    devices = available_devices("onnx", ("coreml",), "darwin", "arm64")

    assert devices == ("auto", "cpu", "coreml")


def test_available_devices_omits_unreported_accelerators() -> None:
    from tree_counter.worker.capabilities import available_devices

    assert available_devices("onnx", ("cpu",), "darwin", "arm64") == (
        "auto",
        "cpu",
    )

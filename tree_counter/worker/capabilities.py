"""Which device a run may actually use.

CPU is always available. Every accelerator is conditional on three separate
things agreeing: the machine offers it, the installed backend reports it,
and the model format can use it. An explicit request that cannot be honoured
is an error; ``auto`` silently prefers the best accelerator and falls back
to CPU with a warning, so a run never fails just because a GPU is busy or
absent.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tree_counter.errors import ErrorCode, TreeCounterError

CPU = "cpu"
AUTO = "auto"
CUDA = "cuda"
MPS = "mps"
COREML = "coreml"

# Which accelerators each model format can ever use, regardless of machine.
FORMAT_ACCELERATORS = {
    "onnx": (CUDA, COREML),
    "pt": (CUDA, MPS),
}
# Preference order when the user asked for auto.
AUTO_PREFERENCE = (CUDA, MPS, COREML)


class DeviceUnavailable(TreeCounterError):
    """An explicitly requested device cannot be used for this run."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.INVALID_SETTINGS, diagnostic_detail=detail)


@dataclass(frozen=True)
class DeviceSelection:
    """The device a run will use and anything the user should be told."""

    device: str
    warnings: tuple[str, ...] = ()


def platform_accelerators(
    platform: str, machine: str, model_format: str
) -> tuple[str, ...]:
    """Return accelerators this platform could offer for a model format.

    MPS exists only on Apple Silicon and only for PyTorch; CoreML only on
    macOS and only for ONNX. CUDA is possible anywhere except macOS, but
    whether it is usable is still up to the installed backend.
    """

    if model_format not in FORMAT_ACCELERATORS:
        raise DeviceUnavailable(f"unsupported model format: {model_format!r}")
    allowed = set(FORMAT_ACCELERATORS[model_format])
    offered: list[str] = []
    is_macos = platform == "darwin"
    is_apple_silicon = is_macos and machine.casefold() in ("arm64", "aarch64")
    if CUDA in allowed and not is_macos:
        offered.append(CUDA)
    if MPS in allowed and is_apple_silicon:
        offered.append(MPS)
    if COREML in allowed and is_macos:
        offered.append(COREML)
    return tuple(offered)


def select_device(
    requested: str,
    model_format: str,
    backend_accelerators: Sequence[str],
    platform: str,
    machine: str,
) -> DeviceSelection:
    """Return the device to use, or explain why the request cannot be met."""

    if not isinstance(requested, str) or not requested:
        raise DeviceUnavailable("a device must be requested")
    choice = requested.casefold()
    possible = platform_accelerators(platform, machine, model_format)
    reported = {
        name.casefold() for name in backend_accelerators
    }
    usable = tuple(name for name in possible if name in reported)

    if choice == CPU:
        return DeviceSelection(CPU)
    if choice == AUTO:
        for candidate in AUTO_PREFERENCE:
            if candidate in usable:
                return DeviceSelection(candidate)
        return DeviceSelection(
            CPU,
            (
                "No compatible hardware acceleration was available, so the "
                "run will use the CPU.",
            ),
        )
    if choice not in FORMAT_ACCELERATORS[model_format]:
        raise DeviceUnavailable(
            f"{choice} cannot be used with a {model_format} model"
        )
    if choice not in possible:
        raise DeviceUnavailable(
            f"{choice} is not available on this platform"
        )
    if choice not in reported:
        raise DeviceUnavailable(
            f"the installed runtime does not provide {choice}"
        )
    return DeviceSelection(choice)


def available_devices(
    model_format: str,
    backend_accelerators: Sequence[str],
    platform: str,
    machine: str,
) -> tuple[str, ...]:
    """Return every device the UI may offer, CPU first and always present."""

    possible = platform_accelerators(platform, machine, model_format)
    reported = {name.casefold() for name in backend_accelerators}
    return (AUTO, CPU) + tuple(
        name for name in possible if name in reported
    )


__all__ = [
    "AUTO",
    "AUTO_PREFERENCE",
    "COREML",
    "CPU",
    "CUDA",
    "FORMAT_ACCELERATORS",
    "MPS",
    "DeviceSelection",
    "DeviceUnavailable",
    "available_devices",
    "platform_accelerators",
    "select_device",
]

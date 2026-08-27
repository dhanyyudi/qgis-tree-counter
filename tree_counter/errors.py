"""Stable error codes and safe messages for Tree Counter failures."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Machine-readable codes that remain stable across releases."""

    INVALID_RASTER = "invalid_raster"
    INVALID_MODEL = "invalid_model"
    INVALID_SCOPE = "invalid_scope"
    INVALID_SETTINGS = "invalid_settings"
    MISSING_RUNTIME = "missing_runtime"
    INCOMPATIBLE_RUNTIME = "incompatible_runtime"
    RUNTIME_INSTALL_FAILURE = "runtime_install_failure"
    NO_SUPPORTED_PYTHON = "no_supported_python"
    WORKER_PROTOCOL_FAILURE = "worker_protocol_failure"
    WORKER_PROCESS_FAILURE = "worker_process_failure"
    CANCELLATION = "cancellation"
    OUTPUT_FAILURE = "output_failure"

    # Short aliases retained for callers that use the noun form.
    WORKER_PROTOCOL = "worker_protocol_failure"
    WORKER_PROCESS = "worker_process_failure"
    CANCELLED = "cancellation"


SAFE_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INVALID_RASTER: "The selected raster is not supported.",
    ErrorCode.INVALID_MODEL: "The selected model is not supported.",
    ErrorCode.INVALID_SCOPE: "The processing scope is not valid.",
    ErrorCode.INVALID_SETTINGS: "One or more counting settings are invalid.",
    ErrorCode.MISSING_RUNTIME: "The Tree Counter runtime is not installed.",
    ErrorCode.INCOMPATIBLE_RUNTIME: "The installed runtime is not compatible.",
    ErrorCode.RUNTIME_INSTALL_FAILURE: (
        "The Tree Counter runtime could not be installed."
    ),
    ErrorCode.NO_SUPPORTED_PYTHON: (
        "No supported Python 3.12 interpreter was found."
    ),
    ErrorCode.WORKER_PROTOCOL_FAILURE: (
        "The counting worker sent an invalid response."
    ),
    ErrorCode.WORKER_PROCESS_FAILURE: (
        "The counting worker could not complete."
    ),
    ErrorCode.CANCELLATION: "Counting was cancelled.",
    ErrorCode.OUTPUT_FAILURE: "The counting output could not be published.",
}


class TreeCounterError(Exception):
    """Base exception carrying a stable code and safe user-facing message."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        diagnostic_detail: str | None = None,
        user_message: str | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.user_message = user_message or SAFE_MESSAGES[self.code]
        self.diagnostic_detail = diagnostic_detail or ""
        super().__init__(self.user_message)


class ValidationError(TreeCounterError, ValueError):
    """Invalid user or model settings with a safe public explanation."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            ErrorCode.INVALID_SETTINGS,
            diagnostic_detail=detail,
        )


def safe_message(code: ErrorCode) -> str:
    """Return the stable user-facing message for *code*."""

    return SAFE_MESSAGES[ErrorCode(code)]

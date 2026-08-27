"""Opt-in checks for real local model assets and the isolated runtime."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os

import pytest

from scripts.run_local_integration import (
    IntegrationConfigurationError,
    TEST_BACKENDS_VARIABLE,
    TREE_COUNTER_TEST_MODEL_ONNX,
    TREE_COUNTER_TEST_MODEL_PT,
    load_environment,
    parse_backends,
    runtime_ready,
)


def _require_real_run(backend: str, variable: str) -> None:
    """Skip with a safe reason until this backend's real run is configured."""

    configured = os.environ.get(TEST_BACKENDS_VARIABLE)
    if configured:
        try:
            selected = parse_backends(configured)
        except ValueError as error:
            pytest.skip(str(error))
        if backend not in selected:
            pytest.skip(f"{variable} is not selected by --backends")
    try:
        load_environment((backend,))
    except IntegrationConfigurationError as error:
        pytest.skip(str(error))
    if not runtime_ready((backend,)):
        pytest.skip("Tree Counter ML runtime is not ready")
    pytest.skip("real model assertions are enabled in Part 2")


def test_real_pt_model_inspection_is_opt_in() -> None:
    """Reserve the real PT inspection for the runtime-ready phase."""

    _require_real_run("pt", TREE_COUNTER_TEST_MODEL_PT)


def test_real_onnx_model_inspection_is_opt_in() -> None:
    """Reserve the real ONNX inspection for the runtime-ready phase."""

    _require_real_run("onnx", TREE_COUNTER_TEST_MODEL_ONNX)

"""Opt-in checks for PT/ONNX parity on the trusted local data."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os

import pytest

from scripts.run_local_integration import (
    IntegrationConfigurationError,
    TEST_BACKENDS_VARIABLE,
    TREE_COUNTER_TEST_MODEL_ONNX,
    load_environment,
    parse_backends,
    runtime_ready,
)


def _require_real_run() -> None:
    """Skip with a safe reason until parity inputs are configured."""

    configured = os.environ.get(TEST_BACKENDS_VARIABLE)
    if configured:
        try:
            selected = parse_backends(configured)
        except ValueError as error:
            pytest.skip(str(error))
        if set(selected) != {"pt", "onnx"}:
            pytest.skip(
                f"{TREE_COUNTER_TEST_MODEL_ONNX} is not selected by "
                "--backends"
            )
    try:
        load_environment(("pt", "onnx"))
    except IntegrationConfigurationError as error:
        pytest.skip(str(error))
    if not runtime_ready(("pt", "onnx")):
        pytest.skip("Tree Counter ML runtime is not ready")
    pytest.skip("backend parity assertions are enabled in Part 2")


def test_pt_and_onnx_parity_is_opt_in() -> None:
    """Reserve parity assertions for the runtime-ready phase."""

    _require_real_run()

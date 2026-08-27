"""Opt-in checks for real aerial-raster counting runs."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os

import pytest

from scripts.run_local_integration import (
    IntegrationConfigurationError,
    TEST_BACKENDS_VARIABLE,
    load_environment,
    parse_backends,
    runtime_ready,
)


def _require_real_run() -> None:
    """Skip with a safe reason until both real backends are configured."""

    configured = os.environ.get(TEST_BACKENDS_VARIABLE)
    if configured:
        try:
            selected = parse_backends(configured)
        except ValueError as error:
            pytest.skip(str(error))
        if set(selected) != {"pt", "onnx"}:
            pytest.skip(
                "TREE_COUNTER_TEST_RASTER is not selected for both backends"
            )
    try:
        load_environment(("pt", "onnx"))
    except IntegrationConfigurationError as error:
        pytest.skip(str(error))
    if not runtime_ready(("pt", "onnx")):
        pytest.skip("Tree Counter ML runtime is not ready")
    pytest.skip("real raster assertions are enabled in Part 2")


def test_real_bounded_raster_run_is_opt_in() -> None:
    """Reserve the bounded raster run for the runtime-ready phase."""

    _require_real_run()

"""QGIS-dependent tests are skipped where QGIS is unavailable.

These run under a QGIS Python via ``scripts/run_qgis_tests.py``. The rules
and arithmetic they build on are QGIS-free and covered in ``tests/unit``,
so an ordinary CI run still exercises the decisions; only the thin layer
that touches a provider needs the real application.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os

import pytest

pytest.importorskip("qgis.core", reason="QGIS is not available")


@pytest.fixture(scope="session", autouse=True)
def qgis_application():
    """Initialise QGIS once for the session, headless.

    Without an initialised application QGIS cannot find proj.db, so every
    coordinate reference system resolves to nothing and raster validation
    would fail for a reason that has nothing to do with the code.
    """

    from qgis.core import QgsApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QgsApplication.instance()
    if existing is not None:
        yield existing
        return

    application = QgsApplication([], False)
    prefix = os.environ.get("QGIS_PREFIX_PATH")
    if prefix:
        QgsApplication.setPrefixPath(prefix, True)
    application.initQgis()
    try:
        yield application
    finally:
        application.exitQgis()

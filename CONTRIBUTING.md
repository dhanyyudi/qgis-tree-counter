# Contributing to Tree Counter

Tree Counter is under active development. Contributions are welcome, but the public interfaces and geospatial counting decisions are still evolving.

Please open an issue before substantial changes so the scope and compatibility impact can be discussed. Keep changes focused, use English for public technical documentation and code comments, and add tests for behavior that you introduce.

Before submitting a change, run the complete local gate used by CI:

```bash
python3 -m pytest -q
python3 -m flake8 tree_counter scripts tests
python3 -m bandit --ini .bandit -r tree_counter scripts
git ls-files -z | xargs -0 detect-secrets-hook --baseline .secrets.baseline
python3 scripts/check_runtime_locks.py
python3 scripts/check_publication.py
python3 scripts/package_plugin.py
python3 scripts/check_publication.py dist/tree_counter-0.1.0.zip
```

QGIS-facing changes also need the tests in `tests/qgis` under both a supported
QGIS 3/Qt5 installation and QGIS 4/Qt6 installation. Do not commit models,
raster imagery, generated GeoPackages, runtime binaries, wheels, credentials,
or maintainer-specific local paths.

By contributing, you agree that your work is provided under the repository's AGPL-3.0-only license.

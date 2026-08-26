# Tree Counter release checklist

This checklist is a publication gate for every tag and GitHub Release.

## Blocking automated gates

- [ ] Unit, integration, import-smoke, and packaging tests pass.
- [ ] Flake8 passes with the repository configuration.
- [ ] Bandit passes with the repository configuration.
- [ ] The tracked-file detect-secrets hook passes.
- [ ] Source validation and deterministic package creation pass.
- [ ] The packaged archive validates, contains only the foundation manifest,
      and is within the 20 MiB limit.
- [ ] The official PyQGIS 4 checker passes both its dry run and blocking pass
      in the `ghcr.io/qgis/pyqgis4-checker:main-ubuntu` container.

## Owner-required manual pre-tag smoke matrix

Record the date, tester, QGIS version, operating system, and result for each
environment before creating a tag:

- [ ] Windows QGIS 3.44 — passed.
- [ ] macOS Apple Silicon QGIS 3.44.13 with Qt5 — passed.
- [ ] macOS Apple Silicon QGIS 4.2.1 with Qt6 — passed.

A tag or GitHub Release must not be created until this matrix is recorded as
passed.
The GitHub runner does not replace these manual operating-system smoke tests,
and the project does not claim QGIS 3 Docker coverage. Uploading to the official
QGIS plugin repository (`plugins.qgis.org`) remains a separate manual action
and is absent from the release workflow.

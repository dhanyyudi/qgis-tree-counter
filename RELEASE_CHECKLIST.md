# Tree Counter release checklist

This checklist is a publication gate for every tag, GitHub Release, and QGIS
Plugin Repository upload. Do not mark an item complete without retaining the
command output or manual evidence for the exact commit and ZIP.

## Blocking automated gates

- [ ] Unit, integration, import-smoke, and packaging tests pass.
- [ ] Flake8 passes with the repository configuration.
- [ ] Bandit passes with the repository configuration.
- [ ] The tracked-file detect-secrets hook passes.
- [ ] Source validation and deterministic package creation pass.
- [ ] The packaged archive validates, contains only the explicit release
      manifest, and is within the 20 MiB limit.
- [ ] The exact extracted ZIP passes the QGIS Plugin Repository security scan
      equivalents: all enabled Bandit rules, detect-secrets, Flake8 at 120
      columns, and suspicious/hidden/executable file analysis.
- [ ] Root and packaged `LICENSE` files are byte-identical; root and packaged
      `THIRD_PARTY_NOTICES.md` files are byte-identical.
- [ ] Repository source and packaged source are byte-identical for every ZIP
      member.
- [ ] No model, raster, GeoPackage, wheel, native library, bytecode, test,
      internal document, hidden file, or credential is present in the ZIP.
- [ ] `metadata.txt` has a unique SemVer version, complete English About text,
      correct QGIS range, public HTTPS links, external-runtime disclosure,
      trusted-`.pt` warning, privacy statement, and experimental status.
- [ ] The official PyQGIS 4 checker passes both its dry run and blocking pass
      in the `ghcr.io/qgis/pyqgis4-checker:main-ubuntu` container.

## Owner-required manual pre-tag smoke matrix

Record the date, tester, QGIS version, operating system, and result for each
environment before creating a tag:

- [ ] Windows QGIS 3.44 — passed.
- [ ] macOS Apple Silicon QGIS 3.44.13 with Qt5 — passed.
- [ ] macOS Apple Silicon QGIS 4.2.1 with Qt6 — passed.

For each environment, install the **exact release ZIP** and record:

- plugin installation, toolbar/menu entry, dock opening, and Indonesian UI;
- Runtime Manager install, verify, repair, remove, and reinstall behavior;
- ONNX CPU bounded run and output-layer autoload;
- trusted `.pt` CPU run and first-use hash confirmation;
- available accelerator path (MPS/CoreML on Apple Silicon);
- cancellation, failure recovery, GeoPackage layers/provenance, and logs;
- tester, date, OS/QGIS/Qt/Python versions, ZIP SHA-256, and result.

A tag or GitHub Release must not be created until this matrix is recorded as
passed. The GitHub runner does not replace the manual operating-system smoke
tests, and the project does not claim QGIS 3 Docker coverage.

## Publication sequence

- [ ] Merge the reviewed release PR into `main` with all required checks green.
- [ ] Confirm `metadata.txt` and `CHANGELOG.md` both describe the same version.
- [ ] Tag the audited commit as `vX.Y.Z`; never reuse or move a release tag.
- [ ] Let the release workflow rebuild and validate the ZIP, then compare its
      SHA-256 and inventory with the audited release candidate.
- [ ] Publish the GitHub Release first and verify its source/repository links.
- [ ] Upload the exact validated ZIP manually to `plugins.qgis.org`.
- [ ] Keep every mandatory QGIS security rule enabled. Disable a skippable
      rule only for a documented false positive reviewed by the maintainer.
- [ ] Review the Security tab after the server scan. A critical finding blocks
      approval and requires a new version; do not suppress a real issue.
- [ ] Verify the approved plugin can be discovered and installed from a clean
      QGIS profile.

The release workflow never uploads to `plugins.qgis.org`; repository upload is
always a separate, explicit maintainer action.

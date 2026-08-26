"""Build a deterministic QGIS plugin ZIP archive."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import configparser
import sys
import zipfile
from pathlib import Path

try:
    from scripts.check_publication import (
        PACKAGE_MANIFEST,
        PACKAGE_NAME,
        validate_archive,
        validate_source,
    )
except ModuleNotFoundError:  # Direct ``python3 scripts/package_plugin.py``.
    from check_publication import (  # type: ignore[no-redef]
        PACKAGE_MANIFEST,
        PACKAGE_NAME,
        validate_archive,
        validate_source,
    )


def _package_files(package: Path) -> list[Path]:
    files: list[Path] = []
    for path in package.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed: {path}")
        relative = path.relative_to(package)
        if (
            "__pycache__" in {part.casefold() for part in relative.parts}
            or path.suffix.casefold() in {".pyc", ".pyo"}
        ):
            raise ValueError(f"forbidden cache or bytecode file: {path}")
        if not path.is_file():
            continue
        if relative.as_posix() not in PACKAGE_MANIFEST:
            raise ValueError(
                "package file is not allowed by the foundation manifest: "
                f"{path}"
            )
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(package).as_posix())


def build_package(repo_root: Path, output_path: Path) -> Path:
    """Validate and write a reproducible plugin archive at ``output_path``."""

    root = Path(repo_root)
    errors = validate_source(root)
    if errors:
        raise ValueError(
            "cannot package invalid source:\n" + "\n".join(errors)
        )

    package = root / PACKAGE_NAME
    files = _package_files(package)
    names = [
        f"{PACKAGE_NAME}/{path.relative_to(package).as_posix()}"
        for path in files
    ]
    missing = [
        f"{PACKAGE_NAME}/{name}"
        for name in PACKAGE_MANIFEST
        if f"{PACKAGE_NAME}/{name}" not in names
    ]
    if missing:
        raise ValueError(
            "missing mandatory package files: " + ", ".join(missing)
        )

    metadata = configparser.ConfigParser(interpolation=None)
    metadata.read(package / "metadata.txt", encoding="utf-8")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_STORED
    ) as handle:
        for path, name in zip(files, names):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100000 | 0o644) << 16
            handle.writestr(info, path.read_bytes())

    archive_errors = validate_archive(destination)
    if archive_errors:
        raise ValueError(
            "built archive failed validation:\n" + "\n".join(archive_errors)
        )
    with zipfile.ZipFile(destination) as handle:
        for path, name in zip(files, names):
            if handle.read(name) != path.read_bytes():
                raise ValueError(f"archive bytes differ from source: {name}")
    return destination


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    metadata = configparser.ConfigParser(interpolation=None)
    metadata.read(root / PACKAGE_NAME / "metadata.txt", encoding="utf-8")
    version = metadata["general"]["version"]
    output = root / "dist" / f"{PACKAGE_NAME}-{version}.zip"
    try:
        archive = build_package(root, output)
    except (OSError, ValueError, KeyError, configparser.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Built and validated: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

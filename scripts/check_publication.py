"""Validate Tree Counter source trees and plugin archives.

The checks intentionally use only the Python standard library so they can run
before QGIS or the optional machine-learning runtime is installed.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import configparser
import re
import stat
import sys
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
PACKAGE_NAME = "tree_counter"
# This inventory is intentionally explicit. Feature work must expand it in the
# same change that adds a distributable package file.
PACKAGE_MANIFEST = (
    "LICENSE",
    "__init__.py",
    "compat.py",
    "constants.py",
    "core/__init__.py",
    "core/dedup.py",
    "core/geometry.py",
    "core/nms.py",
    "core/protocol.py",
    "core/tiling.py",
    "core/types.py",
    "core/validation.py",
    "errors.py",
    "metadata.txt",
    "plugin.py",
    "qgis_adapter/__init__.py",
    "qgis_adapter/raster.py",
    "qgis_adapter/scope.py",
    "qgis_adapter/workspace.py",
    "runtime/__init__.py",
    "runtime/catalog.json",
    "runtime/catalog.py",
    "runtime/installer.py",
    "runtime/locks/linux-x86_64/onnxruntime.txt",
    "runtime/locks/linux-x86_64/pytorch.txt",
    "runtime/locks/macos-arm64/onnxruntime.txt",
    "runtime/locks/macos-arm64/pytorch.txt",
    "runtime/locks/windows-x86_64/onnxruntime.txt",
    "runtime/locks/windows-x86_64/pytorch.txt",
    "runtime/manifest.py",
    "runtime/paths.py",
    "runtime/python_probe.py",
    "runtime/worker_bootstrap.py",
    "settings/__init__.py",
    "settings/presets.py",
    "settings/store.py",
    "settings/trust.py",
    "worker/__init__.py",
    "worker/__main__.py",
    "worker/backend_base.py",
    "worker/backend_onnx.py",
    "worker/backend_ultralytics.py",
    "worker/capabilities.py",
    "worker/image.py",
    "worker/model_info.py",
    "worker/runner.py",
    "worker/yolo_postprocess.py",
)
MANDATORY_FILES = PACKAGE_MANIFEST
EXPECTED_METADATA = {
    "name": "Tree Counter",
    "description": (
        "Count trees in georeferenced aerial imagery with user-provided "
        "YOLO models."
    ),
    "about": (
        "Open-source QGIS plugin foundation for local tree counting in "
        "georeferenced aerial imagery. Under active development; contains "
        "no inference runtime; contains no external ML dependencies; "
        "not ready for production use."
    ),
    "version": "0.1.0",
    "author": "Dhany Yudi Prasetyo",
    "email": "dhanyyudi.prasetyo@gmail.com",
    "homepage": "https://github.com/dhanyyudi/qgis-tree-counter",
    "repository": "https://github.com/dhanyyudi/qgis-tree-counter",
    "tracker": "https://github.com/dhanyyudi/qgis-tree-counter/issues",
    "license": "AGPL-3.0-only",
    "qgisminimumversion": "3.44",
    "qgismaximumversion": "4.99",
    "category": "Raster",
    "experimental": "True",
    "deprecated": "False",
    "tags": "tree counting,YOLO,aerial imagery,geospatial",
    "hasprocessingprovider": "no",
    "server": "False",
}
FORBIDDEN_EXTENSIONS = {
    ".pt",
    ".pth",
    ".onnx",
    ".tif",
    ".tiff",
    ".ecw",
    ".vrt",
    ".gpkg",
    ".whl",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".pyd",
    ".exe",
}
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
QGIS_VERSION = re.compile(r"^[0-9]+\.[0-9]+$")


def _metadata_errors(values: configparser.SectionProxy) -> list[str]:
    errors: list[str] = []
    required = (
        "name",
        "description",
        "about",
        "version",
        "author",
        "email",
        "homepage",
        "repository",
        "tracker",
        "license",
        "qgisminimumversion",
        "qgismaximumversion",
    )
    for key in required:
        if not values.get(key, "").strip():
            errors.append(f"metadata is missing required field: {key}")

    for key in ("homepage", "repository", "tracker"):
        value = values.get(key, "").strip()
        try:
            parsed = urlparse(value)
        except ValueError:
            parsed = None
        if (
            not value
            or parsed is None
            or parsed.scheme != "https"
            or not parsed.netloc
            or any(character.isspace() for character in value)
        ):
            errors.append(f"metadata {key} must be a valid HTTPS URL")

    version = values.get("version", "").strip()
    if version and not SEMVER.fullmatch(version):
        errors.append("metadata version must use MAJOR.MINOR.PATCH format")

    minimum = values.get("qgisminimumversion", "").strip()
    maximum = values.get("qgismaximumversion", "").strip()
    if minimum and not QGIS_VERSION.fullmatch(minimum):
        errors.append("metadata minimum QGIS version is invalid")
    if maximum and not QGIS_VERSION.fullmatch(maximum):
        errors.append("metadata maximum QGIS version is invalid")
    if QGIS_VERSION.fullmatch(minimum) and QGIS_VERSION.fullmatch(maximum):
        if tuple(map(int, minimum.split("."))) > tuple(
            map(int, maximum.split("."))
        ):
            errors.append("metadata minimum QGIS version exceeds maximum")
    if minimum and minimum != "3.44":
        errors.append("metadata minimum QGIS version must be 3.44")
    if maximum and maximum != "4.99":
        errors.append("metadata maximum QGIS version must be 4.99")

    about = values.get("about", "").lower()
    for key, expected in EXPECTED_METADATA.items():
        actual = values.get(key, "").strip()
        if actual != expected:
            errors.append(
                f"metadata {key} must equal the locked foundation value"
            )
    if "active development" not in about:
        errors.append("metadata about must disclose active development status")
    if "no inference runtime" not in about:
        errors.append(
            "metadata about must disclose that this build has no inference "
            "runtime"
        )
    if "no external ml dependencies" not in about:
        errors.append(
            "metadata about must disclose that this build has no external "
            "ML dependencies"
        )
    if "not ready for production use" not in about:
        errors.append(
            "metadata about must disclose that this build is not ready for "
            "production use"
        )
    return errors


def _forbidden_path_error(relative: str) -> str | None:
    raw_parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        return f"forbidden empty or non-canonical path segment: {relative}"
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        return f"forbidden path traversal or absolute path: {relative}"
    if "\\" in relative:
        return f"forbidden non-POSIX path separator: {relative}"
    if any(part.startswith(".") for part in path.parts):
        return f"forbidden hidden path: {relative}"
    if any(
        part in {"tests", "__pycache__", "internal"} for part in path.parts
    ):
        return f"forbidden internal path: {relative}"
    if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        return f"forbidden packaged file type: {relative}"
    return None


def _is_cache_path(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return "__pycache__" in {part.casefold() for part in parts} or (
        PurePosixPath(relative).suffix.casefold() in {".pyc", ".pyo"}
    )


def _content_error(relative: str, data: bytes) -> str | None:
    """Reject maintainer-only paths accidentally copied into source text."""

    try:
        text = data.decode("utf-8").casefold()
    except UnicodeDecodeError:
        return None
    markers = (
        "/" + "users/",
        "agents" + ".md",
        "docs/" + "internal",
        "." + "superpowers",
        "graphify-" + "out",
    )
    if any(marker in text for marker in markers):
        return f"forbidden maintainer or internal path in file: {relative}"
    return None


def _manifest_error(relative: str) -> str | None:
    expected = {name.casefold() for name in PACKAGE_MANIFEST}
    if relative.casefold() not in expected or relative not in PACKAGE_MANIFEST:
        return (
            "package file is not allowed by the foundation manifest: "
            f"{PACKAGE_NAME}/{relative}"
        )
    return None


def _read_metadata(
    path: Path, errors: list[str]
) -> configparser.SectionProxy | None:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(path, encoding="utf-8")
    except (OSError, UnicodeError, configparser.Error) as exc:
        errors.append(f"metadata cannot be read: {exc}")
        return None
    if not parser.has_section("general"):
        errors.append("metadata is missing the [general] section")
        return None
    return parser["general"]


def validate_source(repo_root: Path) -> list[str]:
    """Return all publication errors found in a source checkout."""

    root = Path(repo_root)
    package = root / PACKAGE_NAME
    errors: list[str] = []
    if package.is_symlink():
        return [f"package directory must not be a symlink: {PACKAGE_NAME}"]
    if not package.is_dir():
        return [f"missing package directory: {PACKAGE_NAME}"]

    for relative in MANDATORY_FILES:
        path = package / relative
        if not path.is_file():
            errors.append(
                f"missing mandatory package file: {PACKAGE_NAME}/{relative}"
            )

    root_license = root / "LICENSE"
    package_license = package / "LICENSE"
    if not root_license.is_file():
        errors.append("missing repository-root LICENSE")
    elif package_license.is_file():
        try:
            if root_license.read_bytes() != package_license.read_bytes():
                errors.append("root LICENSE and packaged LICENSE differ")
        except OSError as exc:
            errors.append(f"licenses cannot be read: {exc}")

    metadata = package / "metadata.txt"
    values = _read_metadata(metadata, errors) if metadata.is_file() else None
    if values is not None:
        errors.extend(_metadata_errors(values))

    notices = root / "THIRD_PARTY_NOTICES.md"
    if not notices.is_file():
        errors.append("missing dependency disclosure: THIRD_PARTY_NOTICES.md")
    else:
        try:
            notice_text = notices.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeError) as exc:
            errors.append(f"dependency disclosure cannot be read: {exc}")
            notice_text = ""
        for dependency in ("onnx runtime", "pytorch", "ultralytics"):
            if dependency not in notice_text:
                errors.append(
                    f"dependency disclosure is missing: {dependency}"
                )

    seen_paths: dict[str, str] = {}
    for path in sorted(package.rglob("*")):
        relative_path = path.relative_to(package).as_posix()
        cache_path = _is_cache_path(relative_path)
        if cache_path:
            errors.append(
                f"forbidden cache or bytecode path: {relative_path}"
            )
        if path.is_symlink():
            errors.append(
                "symlinks are not allowed in package files: "
                f"{relative_path}"
            )
            continue
        if not path.is_file():
            continue
        forbidden = _forbidden_path_error(relative_path)
        if forbidden:
            errors.append(forbidden)
        manifest_error = _manifest_error(relative_path)
        if manifest_error:
            errors.append(manifest_error)
        try:
            content_error = _content_error(relative_path, path.read_bytes())
        except OSError as exc:
            errors.append(
                f"package file cannot be read: {relative_path}: {exc}"
            )
        else:
            if content_error:
                errors.append(content_error)
        canonical = "/".join(
            part.casefold() for part in relative_path.split("/")
        )
        previous = seen_paths.setdefault(canonical, relative_path)
        if previous != relative_path:
            errors.append(
                "case-folded package path collision: "
                f"{previous} and {relative_path}"
            )

    return errors


def validate_archive(archive_path: Path) -> list[str]:
    """Return all publication errors found in a plugin ZIP archive."""

    archive = Path(archive_path)
    errors: list[str] = []
    if not archive.is_file():
        return [f"archive does not exist: {archive}"]
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        errors.append("archive exceeds the 20 MiB publication ceiling")

    try:
        with zipfile.ZipFile(archive) as handle:
            infos = handle.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                errors.append("archive contains duplicate member names")
            if names != sorted(names):
                errors.append(
                    "archive members are not sorted lexicographically"
                )
            if not names:
                errors.append("archive is empty")

            package_names: list[str] = []
            seen_paths: dict[str, str] = {}
            member_bytes: dict[str, bytes | None] = {}
            for info in infos:
                name = info.filename
                if not name.endswith("/"):
                    try:
                        member_bytes[name] = handle.read(name)
                    except (
                        EOFError,
                        NotImplementedError,
                        OSError,
                        RuntimeError,
                        zipfile.BadZipFile,
                        zlib.error,
                    ) as exc:
                        errors.append(
                            f"archive member cannot be read: {name}: {exc}"
                        )
                        member_bytes[name] = None
                if not name.startswith(f"{PACKAGE_NAME}/"):
                    errors.append(
                        f"archive must have one {PACKAGE_NAME}/ root: {name}"
                    )
                    continue
                relative = name.removeprefix(f"{PACKAGE_NAME}/")
                if not relative or name.endswith("/"):
                    errors.append(
                        f"archive contains a directory member: {name}"
                    )
                    continue
                package_names.append(relative)
                canonical = "/".join(
                    part.casefold() for part in relative.split("/")
                )
                previous = seen_paths.setdefault(canonical, relative)
                if previous != relative:
                    errors.append(
                        "canonical archive path collision: "
                        f"{previous} and {relative}"
                    )
                forbidden = _forbidden_path_error(relative)
                if forbidden:
                    errors.append(forbidden)
                manifest_error = _manifest_error(relative)
                if manifest_error:
                    errors.append(manifest_error)
                content = member_bytes.get(name)
                if content is not None:
                    content_error = _content_error(relative, content)
                    if content_error:
                        errors.append(content_error)
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type == stat.S_IFLNK:
                    errors.append(
                        f"archive symlink member is forbidden: {name}"
                    )
                elif file_type not in (0, stat.S_IFREG):
                    errors.append(
                        f"archive non-regular member is forbidden: {name}"
                    )
                if info.date_time != (1980, 1, 1, 0, 0, 0):
                    errors.append(
                        "archive member has a non-deterministic timestamp: "
                        f"{name}"
                    )
                if ((info.external_attr >> 16) & 0o777) != 0o644:
                    errors.append(f"archive member mode must be 0644: {name}")

            canonical_package_names = {
                relative.casefold() for relative in package_names
            }
            for required in PACKAGE_MANIFEST:
                if required not in package_names:
                    errors.append(
                        "archive missing mandatory file: "
                        f"{PACKAGE_NAME}/{required}"
                    )
                if required.casefold() not in canonical_package_names:
                    errors.append(
                        "archive missing foundation manifest file: "
                        f"{PACKAGE_NAME}/{required}"
                    )

            metadata_name = f"{PACKAGE_NAME}/metadata.txt"
            if metadata_name in names:
                metadata_bytes = member_bytes.get(metadata_name)
                if metadata_bytes is not None:
                    parser = configparser.ConfigParser(interpolation=None)
                    try:
                        parser.read_string(metadata_bytes.decode("utf-8"))
                        if parser.has_section("general"):
                            errors.extend(_metadata_errors(parser["general"]))
                        else:
                            errors.append(
                                "metadata is missing the [general] section"
                            )
                    except (UnicodeError, configparser.Error) as exc:
                        errors.append(
                            f"metadata cannot be read from archive: {exc}"
                        )
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"archive cannot be read: {exc}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)
    errors = (
        validate_archive(args.path)
        if args.path.suffix.lower() == ".zip"
        else validate_source(args.path)
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Valid: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

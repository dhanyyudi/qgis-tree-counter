"""Validate that every runtime lock is exact, hashed, and reachable.

Maintainer and CI tooling. This script is not shipped in the plugin archive
and never touches the network: it checks the committed lock files against
the committed catalog, so a hand-edited or unpinned requirement fails the
build rather than reaching a user's machine.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

# Importing the package must not leave bytecode in a checkout that
# the publication scanner then rejects.
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tree_counter.runtime.catalog import (  # noqa: E402
    APPROVED_HOSTS,
    load_catalog,
)

LOCK_ROOT = REPO_ROOT / "tree_counter" / "runtime" / "locks"

# The validated baseline from the product decisions. A lock that drifts
# from these top-level versions must be an explicit, reviewed change.
REQUIRED_TOP_LEVEL_VERSIONS = {
    "pytorch": {
        "torch": "2.13.0",
        "torchvision": "0.28.0",
        "ultralytics": "8.4.120",
    },
    "onnxruntime": {"onnxruntime": "1.29.0"},
}

_REQUIREMENT_PATTERN = re.compile(
    r"\A(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s\\]+)\s*\\?\Z"
)
_HASH_PATTERN = re.compile(r"\A--hash=sha256:[0-9a-f]{64}\Z")


def parse_lock(text: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Return {name: (version, hashes)} for one lock file's contents."""

    entries: dict[str, tuple[str, tuple[str, ...]]] = {}
    current: str | None = None
    version = ""
    hashes: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--hash="):
            if current is None:
                raise ValueError(f"hash without a requirement: {line}")
            if not _HASH_PATTERN.match(line):
                raise ValueError(f"unsupported hash form: {line}")
            hashes.append(line)
            continue
        match = _REQUIREMENT_PATTERN.match(line)
        if match is None:
            raise ValueError(f"requirement is not exactly pinned: {line}")
        if current is not None:
            entries[current] = (version, tuple(hashes))
        current = match.group("name").casefold()
        version = match.group("version")
        hashes = []
    if current is not None:
        entries[current] = (version, tuple(hashes))
    return entries


def validate() -> list[str]:
    """Return every problem found in the catalog and its lock files."""

    errors: list[str] = []
    catalog = load_catalog()

    for host in catalog.allowed_hosts:
        if host not in APPROVED_HOSTS:
            errors.append(f"catalog allows an unapproved host: {host}")

    referenced: set[str] = set()
    for component_name, component in sorted(catalog.components.items()):
        expected = REQUIRED_TOP_LEVEL_VERSIONS.get(component_name)
        if expected is None:
            errors.append(
                f"no validated baseline is declared for {component_name}"
            )
        for profile in component.profiles:
            for url in (profile.index_url, profile.extra_index_url):
                if url is None:
                    continue
                parts = urlsplit(url)
                if parts.scheme != "https":
                    errors.append(f"{profile.lock}: index is not HTTPS")
                if (parts.hostname or "").casefold() not in APPROVED_HOSTS:
                    errors.append(
                        f"{profile.lock}: index host is not approved"
                    )
            referenced.add(profile.lock)
            path = LOCK_ROOT / profile.lock
            if not path.is_file():
                errors.append(
                    f"{component_name}/{profile.platform}: missing lock "
                    f"{profile.lock}"
                )
                continue
            try:
                entries = parse_lock(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                errors.append(f"{profile.lock}: {exc}")
                continue
            if not entries:
                errors.append(f"{profile.lock}: contains no requirements")
            for name, (version, hashes) in sorted(entries.items()):
                if not hashes:
                    errors.append(
                        f"{profile.lock}: {name}=={version} has no hash"
                    )
            for name, wanted in (expected or {}).items():
                if name not in entries:
                    errors.append(
                        f"{profile.lock}: missing baseline package {name}"
                    )
                elif entries[name][0] != wanted:
                    errors.append(
                        f"{profile.lock}: {name} is {entries[name][0]}, "
                        f"the validated baseline is {wanted}"
                    )

    if LOCK_ROOT.is_dir():
        for path in sorted(LOCK_ROOT.rglob("*.txt")):
            relative = path.relative_to(LOCK_ROOT).as_posix()
            if relative not in referenced:
                errors.append(
                    f"{relative}: lock file is not referenced by the catalog"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Print every lock problem and return a non-zero status if any."""

    problems = validate()
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print("Runtime locks are exact, hashed, and referenced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

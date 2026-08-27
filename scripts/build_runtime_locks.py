"""Generate hashed runtime lock files from the real package index.

Maintainer tooling. This script is not shipped in the plugin archive.

Wheel hashes are never written by hand. Each lock is produced by asking pip
to resolve the component for one target platform and Python version and to
report exactly what it would install, including the sha256 published by the
index for every artifact. Re-running with the same inputs and an unchanged
index produces byte-identical output.

Usage:

    python scripts/build_runtime_locks.py                # every profile
    python scripts/build_runtime_locks.py --platform macos-arm64
    python scripts/build_runtime_locks.py --check        # report drift only

Network access is required. Run it from a checkout, not from an install.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

# Importing the package must not leave bytecode in a checkout that
# the publication scanner then rejects.
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tree_counter.runtime.catalog import load_catalog  # noqa: E402

PYTHON_VERSION = "3.12"
RESOLVE_TIMEOUT_SECONDS = 900

# The wheel tags each supported platform may consume, most specific first.
PLATFORM_TAGS = {
    "windows-x86_64": ("win_amd64", "any"),
    "macos-arm64": (
        "macosx_14_0_arm64",
        "macosx_12_0_arm64",
        "macosx_11_0_arm64",
        "any",
    ),
    "linux-x86_64": (
        "manylinux_2_28_x86_64",
        "manylinux_2_17_x86_64",
        "manylinux2014_x86_64",
        "any",
    ),
}

# The top-level requirements each component pins. Transitive versions and
# hashes are resolved from the index, never invented here.
COMPONENT_REQUIREMENTS = {
    "onnxruntime": ("onnxruntime==1.29.0",),
    "pytorch": (
        "torch==2.13.0",
        "torchvision==0.28.0",
        "ultralytics==8.4.120",
    ),
}

LOCK_ROOT = REPO_ROOT / "tree_counter" / "runtime" / "locks"


async def _run_fixed_argv(argv: list[str]) -> tuple[int, str, str]:
    """Run pip with an explicit argument vector and timeout bound."""

    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=RESOLVE_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise SystemExit("pip resolution timed out") from exc
    return (
        process.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def resolve(
    requirements: tuple[str, ...], platform: str, index_url: str
) -> list[tuple[str, str, str]]:
    """Return sorted (name, version, sha256) for a component and platform."""

    tags = PLATFORM_TAGS[platform]
    with tempfile.TemporaryDirectory() as work:
        report_path = Path(work) / "report.json"
        argv = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--quiet",
            "--disable-pip-version-check",
            "--no-input",
            "--only-binary=:all:",
            "--index-url",
            index_url,
            "--python-version",
            PYTHON_VERSION,
            "--target",
            str(Path(work) / "target"),
            "--report",
            str(report_path),
        ]
        for tag in tags:
            argv.extend(["--platform", tag])
        argv.extend(requirements)
        return_code, stdout, stderr = asyncio.run(_run_fixed_argv(argv))
        if return_code != 0 or not report_path.is_file():
            raise SystemExit(
                f"could not resolve {platform}: "
                f"{stderr.strip() or stdout.strip()}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved: list[tuple[str, str, str]] = []
    for item in report.get("install", []):
        metadata = item["metadata"]
        hashes = item["download_info"]["archive_info"]["hashes"]
        digest = hashes.get("sha256")
        if not digest:
            raise SystemExit(
                f"{metadata['name']} has no sha256 on the index; refusing "
                "to write a lock without a published hash"
            )
        resolved.append(
            (metadata["name"].casefold(), metadata["version"], digest)
        )
    return sorted(resolved)


def render(
    component: str, platform: str, entries: list[tuple[str, str, str]]
) -> str:
    """Return the deterministic text of one lock file."""

    lines = [
        f"# Tree Counter runtime lock: {component} / {platform}",
        f"# Python {PYTHON_VERSION}, generated by "
        "scripts/build_runtime_locks.py",
        "# Every requirement is exact and hash-pinned. Do not edit by hand.",
        "",
    ]
    for name, version, digest in entries:
        lines.append(f"{name}=={version} \\")
        lines.append(f"    --hash=sha256:{digest}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Generate or check every lock file the catalog references."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", action="append", dest="platforms")
    parser.add_argument("--component", action="append", dest="components")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without rewriting any lock file",
    )
    arguments = parser.parse_args(argv)

    catalog = load_catalog()
    drifted: list[str] = []
    for component_name, component in sorted(catalog.components.items()):
        if arguments.components and component_name not in arguments.components:
            continue
        if component_name not in COMPONENT_REQUIREMENTS:
            raise SystemExit(f"no requirements defined for {component_name}")
        for profile in component.profiles:
            if arguments.platforms and profile.platform not in (
                arguments.platforms
            ):
                continue
            print(f"resolving {component_name} / {profile.platform} ...")
            entries = resolve(
                COMPONENT_REQUIREMENTS[component_name],
                profile.platform,
                profile.index_url,
            )
            text = render(component_name, profile.platform, entries)
            target = LOCK_ROOT / profile.lock
            if arguments.check:
                current = (
                    target.read_text(encoding="utf-8")
                    if target.is_file()
                    else ""
                )
                if current != text:
                    drifted.append(profile.lock)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            print(f"  wrote {profile.lock} ({len(entries)} requirements)")
    if drifted:
        print("lock files differ from the index:", ", ".join(sorted(drifted)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

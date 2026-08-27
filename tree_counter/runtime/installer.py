"""Install, verify, update, repair, and remove the isolated ML runtime.

Every mutation is transactional. Work happens in a staging directory, is
verified there, and only then replaces the active runtime; a failure at any
point leaves the previously working runtime exactly as it was. Nothing is
ever run through a shell: every child process is an argument vector, and
process execution itself is injected so the plugin uses QProcess while tests
use a deterministic fake.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import hashlib
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.runtime.catalog import (
    Catalog,
    CatalogError,
    load_catalog,
    platform_key,
)
from tree_counter.runtime.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    RuntimeManifest,
    evaluate_runtime,
    load_manifest,
    parse_manifest,
)
from tree_counter.runtime.paths import (
    ACTIVATION_JOURNAL_FILE_NAME,
    ACTIVE_DIRECTORY_NAME,
    LOGS_DIRECTORY_NAME,
    OWNERSHIP_MARKER_FILE_NAME,
    STAGING_DIRECTORY_NAME,
    RuntimePaths,
    RuntimeLocationError,
    RuntimeState,
    assert_safe_runtime_root,
    default_runtime_root,
)

PREVIOUS_DIRECTORY_NAME = "previous"
ACTIVATION_JOURNAL_NAME = ACTIVATION_JOURNAL_FILE_NAME
OWNERSHIP_MARKER = {
    "format": 1,
    "owner": "qgis-tree-counter",
    "runtime_directory": "runtime",
}
PROCESS_TIMEOUT_SECONDS = 3600.0
VENV_TIMEOUT_SECONDS = 300.0

# A fixed literal. Nothing from the catalog or the user is interpolated.
SELF_CHECK_SOURCE = (
    "import json,importlib,sys;"
    "names=sys.argv[1].split(',') if len(sys.argv)>1 else [];"
    "versions={};"
    "accelerators=['cpu'];"
    "\n"
    "for name in names:\n"
    "    module=importlib.import_module(name)\n"
    "    versions[name]=getattr(module,'__version__','unknown')\n"
    "\n"
    "try:\n"
    "    import onnxruntime\n"
    "    providers=onnxruntime.get_available_providers()\n"
    "    if 'CoreMLExecutionProvider' in providers:\n"
    "        accelerators.append('coreml')\n"
    "    if 'CUDAExecutionProvider' in providers:\n"
    "        accelerators.append('cuda')\n"
    "except Exception:\n"
    "    pass\n"
    "\n"
    "try:\n"
    "    import torch\n"
    "    if torch.backends.mps.is_available():\n"
    "        accelerators.append('mps')\n"
    "    if torch.cuda.is_available():\n"
    "        accelerators.append('cuda')\n"
    "except Exception:\n"
    "    pass\n"
    "\n"
    "print(json.dumps({'python_version':'.'.join(str(item) for item in "
    "sys.version_info[:3]),'versions':versions,'accelerators':accelerators}))"
)

_SECRET_PATTERNS = (
    re.compile(r"(?<=//)[^/\s:@]+:[^/\s@]+(?=@)"),
    re.compile(r"(?i)(--?(?:token|password|secret|api[-_]?key)[= ])\S+"),
)


class InstallError(TreeCounterError):
    """A runtime mutation could not be completed."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            ErrorCode.INCOMPATIBLE_RUNTIME, diagnostic_detail=detail
        )


class InstallCancelled(TreeCounterError):
    """The user cancelled a runtime mutation."""

    def __init__(self, detail: str = "cancelled by the user") -> None:
        super().__init__(ErrorCode.CANCELLATION, diagnostic_detail=detail)


class ProcessResult(NamedTuple):
    """The outcome of one child process."""

    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str], float], ProcessResult]
Progress = Callable[[str, int], None]
ShouldCancel = Callable[[], bool]
PlatformDetector = Callable[[], str]


@dataclass(frozen=True)
class InstallPlan:
    """What the user asked to install, already resolved to specifics."""

    components: tuple[str, ...]
    platform: str
    python_executable: str
    python_version: str

    def __post_init__(self) -> None:
        if not self.components:
            raise InstallError("an install plan needs at least one component")
        if len(set(self.components)) != len(self.components):
            raise InstallError("an install plan repeats a component")


@dataclass(frozen=True)
class RuntimeStatus:
    """What the Runtime Manager should display."""

    state: RuntimeState
    reasons: tuple[str, ...] = ()
    manifest: RuntimeManifest | None = None
    accelerators: tuple[str, ...] = field(default=())


def redact(text: str, home: Path | None = None) -> str:
    """Return *text* with credentials and the user's home path removed.

    Installation logs are kept for diagnostics, so they must not become a
    place where a proxy password or a private directory layout is stored.
    """

    result = str(text)
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(
            lambda match: (
                match.group(1) + "[redacted]"
                if match.re.groups
                else "[redacted]"
            ),
            result,
        )
    user_home = Path.home() if home is None else Path(home)
    home_text = str(user_home)
    if home_text and home_text != "/":
        result = result.replace(home_text, "~")
    return result


def _noop_progress(message: str, percent: int) -> None:
    """Discard progress when the caller supplied no handler."""


def _never_cancel() -> bool:
    """Report that no cancellation was requested."""

    return False


class RuntimeInstaller:
    """Transactional lifecycle for the isolated per-user runtime."""

    def __init__(
        self,
        paths: RuntimePaths,
        runner: Runner,
        lock_root: Path | str,
        catalog: Catalog | None = None,
        home: Path | None = None,
        clock: Callable[[], float] | None = None,
        platform_detector: PlatformDetector | None = None,
        expected_root: Path | str | None = None,
    ) -> None:
        self._paths = paths
        self._runner = runner
        self._lock_root = Path(lock_root)
        self._catalog = load_catalog() if catalog is None else catalog
        self._home = home
        self._expected_root = (
            default_runtime_root(home=home)
            if expected_root is None
            else Path(expected_root)
        ).resolve()
        self._platform_detector = (
            platform_key if platform_detector is None else platform_detector
        )
        if clock is None:
            import time

            clock = time.time
        self._clock = clock
        self._log: list[str] = []

    # -- public lifecycle ------------------------------------------------

    def inspect(self) -> RuntimeStatus:
        """Return state, recovering an interrupted activation if needed."""

        self._assert_expected_root()
        try:
            current_platform = self._platform_detector()
        except CatalogError as exc:
            return RuntimeStatus(
                RuntimeState.INCOMPATIBLE,
                (str(exc.diagnostic_detail or exc),),
            )
        if not any(
            profile.platform == current_platform
            for component in self._catalog.components.values()
            for profile in component.profiles
        ):
            return RuntimeStatus(
                RuntimeState.INCOMPATIBLE,
                (
                    "No runtime component is available for the current "
                    f"platform: {current_platform}.",
                ),
            )
        if self._paths.root.exists():
            try:
                self._assert_owned_root(self._paths.root)
            except RuntimeLocationError:
                return RuntimeStatus(
                    RuntimeState.REPAIR_REQUIRED,
                    ("The runtime directory is not owned by Tree Counter.",),
                )
        self._recover_activation()
        if self._paths.activation_journal.exists():
            return RuntimeStatus(
                RuntimeState.REPAIR_REQUIRED,
                (
                    "A previous runtime activation could not be recovered; "
                    "repair is required.",
                ),
            )
        root = self._paths.root
        try:
            manifest = load_manifest(self._paths.manifest)
        except ManifestError:
            if not root.exists() or not self._has_runtime_artifacts():
                return RuntimeStatus(RuntimeState.NOT_INSTALLED)
            reasons = (
                "The runtime manifest is missing or invalid; repair is "
                "required.",
            )
            return RuntimeStatus(RuntimeState.REPAIR_REQUIRED, reasons)
        present = self._present_files()
        imports: dict[str, bool] = {}
        live_versions: Mapping[str, object] = {}
        accelerators: set[str] = set()
        live_python_version: str | None = None
        probe_error: str | None = None
        if present:
            try:
                modules = self._modules_for(tuple(manifest.components))
                probe = self._self_check(self._paths.active, modules)
                versions = probe.get("versions")
                if isinstance(versions, Mapping):
                    live_versions = versions
                    imports = {
                        module: module in versions for module in modules
                    }
                live_version = probe.get("python_version")
                if isinstance(live_version, str) and live_version:
                    live_python_version = live_version
                reported_accelerators = probe.get("accelerators", ())
                is_accelerator_list = isinstance(
                    reported_accelerators, Sequence
                ) and not isinstance(reported_accelerators, (str, bytes))
                if is_accelerator_list:
                    accelerators.update(
                        item
                        for item in reported_accelerators
                        if isinstance(item, str)
                    )
                if live_python_version is None:
                    probe_error = (
                        "The runtime self-check did not report its Python "
                        "version."
                    )
            except TreeCounterError as exc:
                probe_error = str(exc.diagnostic_detail or exc)
        else:
            probe_error = "The active runtime Python executable is missing."
        if not imports:
            imports = {
                module: False
                for name in manifest.components
                if name in self._catalog.components
                for module in self._catalog.components[name].imports
            }
        expected_locks, lock_errors = self._current_lock_digests(
            manifest, current_platform
        )
        report = evaluate_runtime(
            manifest=manifest,
            catalog=self._catalog,
            platform=current_platform,
            python_version=live_python_version or manifest.python_version,
            present_files=present,
            import_results=imports,
            available_accelerators=tuple(sorted(accelerators)),
            expected_lock_digests=expected_locks or None,
        )
        reasons = list(report.reasons)
        version_drift = self._version_drift(manifest, live_versions)
        if report.state is RuntimeState.INCOMPATIBLE:
            report_state = report.state
        elif version_drift:
            report_state = RuntimeState.REPAIR_REQUIRED
            reasons.extend(version_drift)
        elif lock_errors:
            report_state = RuntimeState.REPAIR_REQUIRED
            reasons.extend(lock_errors)
        elif probe_error:
            report_state = RuntimeState.REPAIR_REQUIRED
            reasons.append(probe_error)
        else:
            report_state = report.state
        return RuntimeStatus(
            state=report_state,
            reasons=tuple(reasons),
            manifest=manifest,
            accelerators=tuple(sorted(accelerators)),
        )

    def install(
        self,
        plan: InstallPlan,
        progress: Progress = _noop_progress,
        should_cancel: ShouldCancel = _never_cancel,
    ) -> RuntimeManifest:
        """Build a runtime in staging and activate it once verified."""

        return self._transaction(plan, progress, should_cancel, "install")

    def update(
        self,
        plan: InstallPlan,
        progress: Progress = _noop_progress,
        should_cancel: ShouldCancel = _never_cancel,
    ) -> RuntimeManifest:
        """Replace the active runtime, keeping it until the new one works."""

        return self._transaction(plan, progress, should_cancel, "update")

    def repair(
        self,
        plan: InstallPlan,
        progress: Progress = _noop_progress,
        should_cancel: ShouldCancel = _never_cancel,
    ) -> RuntimeManifest:
        """Rebuild a broken runtime, replacing it only after verification."""

        return self._transaction(plan, progress, should_cancel, "repair")

    def verify(self) -> RuntimeManifest:
        """Re-run the self-checks against the active runtime."""

        self._assert_expected_root()
        try:
            manifest = load_manifest(self._paths.manifest)
        except ManifestError as exc:
            raise InstallError(f"no runtime to verify: {exc}") from exc
        modules = self._modules_for(tuple(manifest.components))
        self._self_check(self._paths.active, modules)
        return manifest

    def remove(self) -> None:
        """Delete the runtime root, after proving it is safe to delete."""

        self._assert_expected_root()
        root = assert_safe_runtime_root(self._paths.root, home=self._home)
        if not root.exists():
            return
        self._assert_owned_root(root)
        allowed = {
            ACTIVE_DIRECTORY_NAME,
            STAGING_DIRECTORY_NAME,
            PREVIOUS_DIRECTORY_NAME,
            LOGS_DIRECTORY_NAME,
            OWNERSHIP_MARKER_FILE_NAME,
            ACTIVATION_JOURNAL_FILE_NAME,
        }
        unknown = [
            item.name for item in root.iterdir() if item.name not in allowed
        ]
        if unknown:
            raise RuntimeLocationError(
                "the runtime root contains unexpected files: "
                + ", ".join(sorted(unknown))
            )
        for name in allowed - {OWNERSHIP_MARKER_FILE_NAME}:
            self._clear(root / name)
        self._clear(self._paths.ownership_marker)
        try:
            root.rmdir()
        except OSError as exc:
            raise RuntimeLocationError(
                f"the owned runtime root could not be removed: {exc}"
            ) from exc

    # -- transaction -----------------------------------------------------

    def _transaction(
        self,
        plan: InstallPlan,
        progress: Progress,
        should_cancel: ShouldCancel,
        operation: str,
    ) -> RuntimeManifest:
        self._assert_expected_root()
        root = assert_safe_runtime_root(self._paths.root, home=self._home)
        self._log = [f"{operation} started"]
        locks = self._resolve_locks(plan)
        self._check_python(plan)

        self._prepare_owned_root(root)
        self._recover_activation()
        if self._paths.activation_journal.exists():
            raise InstallError(
                "a previous runtime activation could not be recovered; "
                "repair is required"
            )
        staging = self._paths.staging
        self._clear(staging)
        total_steps = len(plan.components) + 3
        step = 0
        try:
            self._guard(should_cancel)
            step += 1
            progress(
                "Creating the runtime environment",
                self._percent(step, total_steps),
            )
            self._create_venv(plan, staging)

            for component in plan.components:
                self._guard(should_cancel)
                step += 1
                progress(
                    f"Installing {self._catalog.components[component].title}",
                    self._percent(step, total_steps),
                )
                self._pip_install(
                    staging, plan.platform, component, locks[component]
                )

            self._guard(should_cancel)
            step += 1
            progress("Verifying the runtime", self._percent(step, total_steps))
            modules = self._modules_for(plan.components)
            report = self._self_check(staging, modules)

            manifest = self._build_manifest(plan, locks, report)
            (staging / self._paths.manifest.name).write_text(
                json.dumps(manifest.as_record(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

            self._guard(should_cancel)
            step += 1
            progress("Activating the runtime", 100)
            self._activate(staging)
        except InstallCancelled:
            self._log.append(f"{operation} cancelled")
            self._clear(staging)
            self._write_log(operation, root)
            raise
        except TreeCounterError as exc:
            self._log.append(
                f"{operation} failed: {exc.diagnostic_detail or exc}"
            )
            self._clear(staging)
            self._write_log(operation, root)
            raise
        self._log.append(f"{operation} completed")
        self._write_log(operation, root)
        return manifest

    # -- steps -----------------------------------------------------------

    def _resolve_locks(self, plan: InstallPlan) -> dict[str, Path]:
        locks: dict[str, Path] = {}
        for component in plan.components:
            try:
                profile = self._catalog.profile_for(component, plan.platform)
            except CatalogError as exc:
                raise InstallError(str(exc.diagnostic_detail)) from exc
            if profile is None:
                raise InstallError(
                    f"{component} has no build for {plan.platform}"
                )
            lock = self._lock_root / profile.lock
            if not lock.is_file():
                raise InstallError(f"missing lock file: {profile.lock}")
            locks[component] = lock
        return locks

    def _check_python(self, plan: InstallPlan) -> None:
        if not self._catalog.supports_python(plan.python_version):
            raise InstallError(
                f"Python {plan.python_version} is outside the supported range"
            )

    def _create_venv(self, plan: InstallPlan, staging: Path) -> None:
        staging.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                plan.python_executable,
                "-I",
                "-m",
                "venv",
                "--copies",
                str(staging),
            ],
            VENV_TIMEOUT_SECONDS,
            "create the runtime environment",
        )

    def _pip_install(
        self, staging: Path, platform: str, component: str, lock: Path
    ) -> None:
        profile = self._catalog.profile_for(component, platform)
        argv = [
            str(self._venv_python(staging)),
            "-I",
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary=:all:",
            "--no-input",
            "--no-cache-dir",
            "--disable-pip-version-check",
            "--index-url",
            profile.index_url if profile else "https://pypi.org/simple",
            "--requirement",
            str(lock),
        ]
        if profile is not None and profile.extra_index_url:
            argv.extend(["--extra-index-url", profile.extra_index_url])
        self._run(argv, PROCESS_TIMEOUT_SECONDS, f"install {component}")

    def _self_check(
        self, environment: Path, modules: Sequence[str]
    ) -> Mapping[str, object]:
        result = self._run(
            [
                str(self._venv_python(environment)),
                "-I",
                "-c",
                SELF_CHECK_SOURCE,
                ",".join(modules),
                "--self-check",
            ],
            VENV_TIMEOUT_SECONDS,
            "verify the runtime",
        )
        try:
            report = json.loads(result.stdout)
        except ValueError as exc:
            raise InstallError(
                f"the runtime self-check returned unreadable output: {exc}"
            ) from exc
        if not isinstance(report, dict):
            raise InstallError("the runtime self-check returned no report")
        versions = report.get("versions")
        if not isinstance(versions, dict):
            raise InstallError("the runtime self-check reported no versions")
        missing = [name for name in modules if name not in versions]
        if missing:
            raise InstallError(
                f"the runtime could not import: {', '.join(sorted(missing))}"
            )
        return report

    def _activate(self, staging: Path) -> None:
        active = self._paths.active
        previous = self._paths.root / PREVIOUS_DIRECTORY_NAME
        journal = self._paths.activation_journal
        if previous.exists() and not journal.exists():
            raise InstallError(
                "a stale previous runtime exists without a recovery journal"
            )
        self._atomic_write_json(
            journal,
            {"schema_version": 1, "had_active": active.exists()},
        )
        moved = False
        try:
            if active.exists():
                active.replace(previous)
                moved = True
            staging.replace(active)
        except OSError as exc:
            if moved:
                try:
                    previous.replace(active)
                except OSError:
                    # Keep the journal so a later inspect can recover.
                    raise InstallError(
                        "the runtime activation failed and rollback also "
                        "failed"
                    ) from exc
            self._clear(journal)
            raise InstallError(
                f"the runtime could not be activated: {exc}"
            ) from exc
        self._clear(previous)
        self._clear(journal)

    def _recover_activation(self) -> None:
        """Recover a swap interrupted between the two directory renames."""

        journal = self._paths.activation_journal
        if not journal.is_file():
            return
        try:
            self._assert_owned_root(self._paths.root)
        except RuntimeLocationError:
            return
        try:
            document = json.loads(journal.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(document, Mapping) or document.get(
            "schema_version"
        ) != 1:
            return
        active = self._paths.active
        previous = self._paths.root / PREVIOUS_DIRECTORY_NAME
        staging = self._paths.staging
        try:
            if not active.exists():
                if (
                    staging.exists()
                    and (staging / self._paths.manifest.name).is_file()
                ):
                    staging.replace(active)
                elif previous.exists():
                    previous.replace(active)
            if active.exists():
                self._clear(previous)
                self._clear(journal)
        except OSError:
            # Leave the journal for the next inspect/repair attempt.
            return

    # -- helpers ---------------------------------------------------------

    def _modules_for(self, components: Sequence[str]) -> tuple[str, ...]:
        modules: list[str] = []
        for component in components:
            if component not in self._catalog.components:
                raise InstallError(f"unknown runtime component: {component}")
            for module in self._catalog.components[component].imports:
                if module not in modules:
                    modules.append(module)
        return tuple(modules)

    @staticmethod
    def _venv_python(environment: Path) -> Path:
        windows = environment / "Scripts" / "python.exe"
        if windows.exists():
            return windows
        return environment / "bin" / "python"

    def _present_files(self) -> tuple[str, ...]:
        candidates = (
            self._paths.active / "bin" / "python",
            self._paths.active / "Scripts" / "python.exe",
        )
        return tuple(
            str(path.relative_to(self._paths.active))
            for path in candidates
            if path.exists()
        )

    def _has_runtime_artifacts(self) -> bool:
        return any(
            path.exists()
            for path in (
                self._paths.active,
                self._paths.staging,
                self._paths.root / PREVIOUS_DIRECTORY_NAME,
            )
        )

    def _current_lock_digests(
        self, manifest: RuntimeManifest, platform: str
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        digests: dict[str, str] = {}
        errors: list[str] = []
        for component in manifest.components:
            try:
                profile = self._catalog.profile_for(component, platform)
            except CatalogError as exc:
                errors.append(str(exc.diagnostic_detail or exc))
                continue
            if profile is None:
                errors.append(
                    f"No lock profile is available for {component} on "
                    f"{platform}."
                )
                continue
            lock = self._lock_root / profile.lock
            try:
                if not lock.is_file():
                    raise OSError("file does not exist")
                digests[component] = hashlib.sha256(
                    lock.read_bytes()
                ).hexdigest()
            except OSError as exc:
                errors.append(
                    f"The runtime lock for {component} is unavailable: {exc}."
                )
        return digests, tuple(errors)

    @staticmethod
    def _version_drift(
        manifest: RuntimeManifest, live_versions: Mapping[str, object]
    ) -> tuple[str, ...]:
        drift: list[str] = []
        for component, record in manifest.components.items():
            for module, expected in record.versions.items():
                observed = live_versions.get(module)
                if observed is not None and str(observed) != expected:
                    drift.append(
                        f"The installed version of {module} changed from "
                        f"{expected} to {observed}."
                    )
        return tuple(drift)

    def _prepare_owned_root(self, root: Path) -> None:
        existed = root.exists()
        root.mkdir(parents=True, exist_ok=True)
        marker = self._paths.ownership_marker
        if marker.exists():
            self._assert_owned_root(root)
            return
        if existed and any(root.iterdir()):
            raise RuntimeLocationError(
                "the runtime root is not owned by Tree Counter"
            )
        self._atomic_write_json(marker, OWNERSHIP_MARKER)

    def _assert_expected_root(self) -> None:
        actual = self._paths.root.resolve()
        if actual != self._expected_root:
            raise RuntimeLocationError(
                "the runtime root does not match the managed Tree Counter "
                "runtime location"
            )

    def _assert_owned_root(self, root: Path) -> None:
        marker = root / OWNERSHIP_MARKER_FILE_NAME
        try:
            document = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeLocationError(
                "the runtime root is not owned by Tree Counter"
            ) from exc
        if document != OWNERSHIP_MARKER:
            raise RuntimeLocationError(
                "the runtime root is not owned by Tree Counter"
            )

    @staticmethod
    def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(document, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)

    @staticmethod
    def _percent(step: int, total: int) -> int:
        return max(0, min(100, round(step * 100 / max(total, 1))))

    @staticmethod
    def _guard(should_cancel: ShouldCancel) -> None:
        if should_cancel():
            raise InstallCancelled()

    def _run(
        self, argv: Sequence[str], timeout: float, what: str
    ) -> ProcessResult:
        self._log.append("$ " + " ".join(str(part) for part in argv))
        try:
            result = self._runner(list(argv), timeout)
        except KeyboardInterrupt as exc:
            raise InstallCancelled(
                f"cancelled while trying to {what}"
            ) from exc
        except TreeCounterError:
            raise
        except Exception as exc:
            raise InstallError(
                f"could not {what}: {type(exc).__name__}: {exc}"
            ) from exc
        if result.stdout:
            self._log.append(result.stdout)
        if result.stderr:
            self._log.append(result.stderr)
        if result.returncode != 0:
            raise InstallError(
                f"could not {what} (exit {result.returncode})"
            )
        return result

    def _build_manifest(
        self,
        plan: InstallPlan,
        locks: Mapping[str, Path],
        report: Mapping[str, object],
    ) -> RuntimeManifest:
        import hashlib

        versions = report.get("versions", {})
        accelerators = report.get("accelerators", ["cpu"])
        components: dict[str, object] = {}
        for component in plan.components:
            digest = hashlib.sha256(
                locks[component].read_bytes()
            ).hexdigest()
            profile = self._catalog.profile_for(component, plan.platform)
            offered = () if profile is None else profile.accelerators
            components[component] = {
                "lock_digest": digest,
                "versions": {
                    module: str(versions.get(module, "unknown"))
                    for module in self._catalog.components[component].imports
                },
                "accelerators": [
                    name
                    for name in offered
                    if name in tuple(accelerators)  # type: ignore[arg-type]
                ]
                or ["cpu"],
            }
        return parse_manifest(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "catalog_version": self._catalog.catalog_version,
                "python_version": plan.python_version,
                "platform": plan.platform,
                "components": components,
                "installed_at": int(self._clock()),
            }
        )

    @staticmethod
    def _clear(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _write_log(self, operation: str, root: Path) -> None:
        try:
            self._paths.logs.mkdir(parents=True, exist_ok=True)
            stamp = int(self._clock())
            target = self._paths.logs / f"{operation}-{stamp}.log"
            target.write_text(
                redact("\n".join(self._log) + "\n", home=self._home),
                encoding="utf-8",
            )
        except OSError:  # A missing log must never fail an install.
            return


__all__ = [
    "InstallCancelled",
    "InstallError",
    "InstallPlan",
    "ProcessResult",
    "RuntimeInstaller",
    "RuntimeStatus",
    "redact",
]

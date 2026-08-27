"""The private per-run workspace that holds temporary tile files.

Tiles are raster pixels from the user's data, so the workspace is created
with owner-only permissions, kept outside the project directory, and
removed on success, failure, and cancellation alike. Only the diagnostic
log survives a failure. Every path handed out is checked to be inside the
workspace, so a malformed tile name can never write elsewhere.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from tree_counter.errors import ErrorCode, TreeCounterError

WORKSPACE_PREFIX = "tree_counter_run_"
TILES_DIRECTORY_NAME = "tiles"
LOG_FILE_NAME = "run.log"
OWNER_ONLY_DIRECTORY = 0o700
OWNER_ONLY_FILE = 0o600
# QGIS sends one tile at a time, so only a small number may ever exist.
MAX_RESIDENT_TILES = 4


class WorkspaceError(TreeCounterError):
    """The private run workspace could not be created or used safely."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            ErrorCode.WORKER_PROCESS_FAILURE, diagnostic_detail=detail
        )


def _safe_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise WorkspaceError("a tile name must be a non-empty string")
    if "\x00" in name:
        raise WorkspaceError("a tile name must not contain a null byte")
    if name.startswith("/") or name.startswith("\\"):
        raise WorkspaceError("a tile name must be relative")
    if "\\" in name:
        raise WorkspaceError("a tile name must use forward slashes")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise WorkspaceError("a tile name must not contain traversal")
    return name


class RunWorkspace:
    """A private directory for one counting run."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()
        self._closed = False

    @classmethod
    def create(cls, parent: Path | str | None = None) -> "RunWorkspace":
        """Create a new owner-only workspace and return it."""

        try:
            root = Path(
                tempfile.mkdtemp(
                    prefix=WORKSPACE_PREFIX,
                    dir=str(parent) if parent is not None else None,
                )
            )
        except OSError as exc:
            raise WorkspaceError(
                f"the run workspace could not be created: {exc}"
            ) from exc
        workspace = cls(root)
        workspace._prepare()
        return workspace

    def _prepare(self) -> None:
        try:
            self._root.chmod(OWNER_ONLY_DIRECTORY)
            self.tiles.mkdir(parents=True, exist_ok=True)
            self.tiles.chmod(OWNER_ONLY_DIRECTORY)
        except OSError as exc:  # Windows may not honour every mode bit.
            if not self.tiles.is_dir():
                raise WorkspaceError(
                    f"the run workspace could not be prepared: {exc}"
                ) from exc

    @property
    def root(self) -> Path:
        """Return the workspace root."""

        return self._root

    @property
    def tiles(self) -> Path:
        """Return the directory holding temporary tile files."""

        return self._root / TILES_DIRECTORY_NAME

    @property
    def log(self) -> Path:
        """Return the diagnostic log path, which survives a failure."""

        return self._root / LOG_FILE_NAME

    @property
    def closed(self) -> bool:
        """Return whether the workspace has been removed."""

        return self._closed

    def tile_path(self, name: str) -> Path:
        """Return a validated absolute path for a tile inside the tiles dir."""

        self._require_open()
        candidate = (self.tiles / _safe_name(name)).resolve()
        if candidate != self.tiles and self.tiles not in candidate.parents:
            raise WorkspaceError("the tile path escapes the run workspace")
        return candidate

    def write_tile(self, name: str, data: bytes) -> Path:
        """Write one tile and return its path, keeping it owner-only."""

        if not isinstance(data, (bytes, bytearray)):
            raise WorkspaceError("tile data must be bytes")
        path = self.tile_path(name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(data))
            path.chmod(OWNER_ONLY_FILE)
        except OSError as exc:
            raise WorkspaceError(
                f"the tile could not be written: {exc}"
            ) from exc
        return path

    def resident_tiles(self) -> tuple[Path, ...]:
        """Return the tile files currently on disk."""

        if not self.tiles.is_dir():
            return ()
        return tuple(
            sorted(path for path in self.tiles.rglob("*") if path.is_file())
        )

    def discard_tile(self, name: str) -> None:
        """Remove one tile once the worker has acknowledged it."""

        path = self.tile_path(name)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise WorkspaceError(
                f"the tile could not be removed: {exc}"
            ) from exc

    def clear_tiles(self) -> None:
        """Remove every tile, leaving the workspace and log in place."""

        self._require_open()
        if self.tiles.is_dir():
            shutil.rmtree(self.tiles, ignore_errors=True)
        self.tiles.mkdir(parents=True, exist_ok=True)

    def close(self, keep_log: bool = False) -> Path | None:
        """Remove the workspace, optionally preserving the log elsewhere.

        The log is copied out before removal because a failed run must
        leave a diagnostic behind without leaving raster pixels behind.
        """

        if self._closed:
            return None
        preserved: Path | None = None
        if keep_log and self.log.is_file():
            try:
                target = Path(tempfile.gettempdir()) / (
                    f"{WORKSPACE_PREFIX}{self._root.name}.log"
                )
                shutil.copyfile(self.log, target)
                preserved = target
            except OSError:  # A missing diagnostic must not mask the failure.
                preserved = None
        shutil.rmtree(self._root, ignore_errors=True)
        self._closed = True
        return preserved

    def _require_open(self) -> None:
        if self._closed:
            raise WorkspaceError("the run workspace has been closed")

    def __enter__(self) -> "RunWorkspace":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(keep_log=exc_type is not None)


__all__ = [
    "LOG_FILE_NAME",
    "MAX_RESIDENT_TILES",
    "TILES_DIRECTORY_NAME",
    "WORKSPACE_PREFIX",
    "RunWorkspace",
    "WorkspaceError",
]

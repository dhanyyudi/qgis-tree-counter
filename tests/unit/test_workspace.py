"""Tests for the private per-run tile workspace."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _workspace(tmp_path: Path):
    from tree_counter.qgis_adapter.workspace import RunWorkspace

    return RunWorkspace.create(parent=tmp_path)


def test_a_workspace_is_created_with_a_tiles_directory(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)

    assert workspace.root.is_dir()
    assert workspace.tiles.is_dir()
    assert workspace.root.name.startswith("tree_counter_run_")


def test_the_workspace_is_owner_only_where_supported(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)

    if os.name == "posix":
        assert workspace.root.stat().st_mode & 0o077 == 0
        assert workspace.tiles.stat().st_mode & 0o077 == 0


def test_a_tile_is_written_and_readable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    path = workspace.write_tile("tile_r00000_c00000.raw", b"\x01\x02\x03")

    assert path.read_bytes() == b"\x01\x02\x03"
    assert workspace.tiles in path.parents


def test_a_written_tile_is_owner_only(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    path = workspace.write_tile("tile.raw", b"data")

    if os.name == "posix":
        assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    "name",
    [
        "../escape.raw",
        "nested/../../escape.raw",
        "/absolute.raw",
        "..\\escape.raw",
        "sub\\dir.raw",
        "tile\x00.raw",
        "",
        ".",
        "..",
    ],
)
def test_an_unsafe_tile_name_is_refused(tmp_path: Path, name: str) -> None:
    from tree_counter.qgis_adapter.workspace import WorkspaceError

    workspace = _workspace(tmp_path)

    with pytest.raises(WorkspaceError):
        workspace.tile_path(name)


def test_a_nested_tile_name_is_allowed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    path = workspace.write_tile("row0/tile.raw", b"data")

    assert path.is_file()
    assert workspace.tiles in path.parents


def test_non_bytes_tile_data_is_refused(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.workspace import WorkspaceError

    workspace = _workspace(tmp_path)

    with pytest.raises(WorkspaceError):
        workspace.write_tile("tile.raw", "not bytes")


def test_resident_tiles_are_reported(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.write_tile("a.raw", b"a")
    workspace.write_tile("b.raw", b"b")

    assert len(workspace.resident_tiles()) == 2


def test_a_discarded_tile_is_removed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.write_tile("a.raw", b"a")

    workspace.discard_tile("a.raw")

    assert workspace.resident_tiles() == ()


def test_discarding_an_absent_tile_is_a_no_op(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    workspace.discard_tile("absent.raw")


def test_sequential_use_keeps_the_tile_count_bounded(
    tmp_path: Path,
) -> None:
    from tree_counter.qgis_adapter.workspace import MAX_RESIDENT_TILES

    workspace = _workspace(tmp_path)

    # The host writes one tile, waits for acknowledgement, then replaces
    # it. Disk use must not grow with the size of the raster.
    for index in range(50):
        name = f"tile_{index:05d}.raw"
        workspace.write_tile(name, b"x" * 16)
        assert len(workspace.resident_tiles()) <= MAX_RESIDENT_TILES
        workspace.discard_tile(name)

    assert workspace.resident_tiles() == ()


def test_clearing_tiles_keeps_the_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.write_tile("a.raw", b"a")
    workspace.log.write_text("diagnostic", encoding="utf-8")

    workspace.clear_tiles()

    assert workspace.resident_tiles() == ()
    assert workspace.tiles.is_dir()
    assert workspace.log.read_text(encoding="utf-8") == "diagnostic"


def test_closing_removes_everything(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.write_tile("a.raw", b"a")
    root = workspace.root

    workspace.close()

    assert not root.exists()
    assert workspace.closed is True


def test_closing_twice_is_safe(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    workspace.close()
    workspace.close()


def test_using_a_closed_workspace_is_refused(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.workspace import WorkspaceError

    workspace = _workspace(tmp_path)
    workspace.close()

    with pytest.raises(WorkspaceError):
        workspace.tile_path("a.raw")


def test_closing_with_keep_log_preserves_the_diagnostic(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    workspace.write_tile("a.raw", b"pixels")
    workspace.log.write_text("what went wrong", encoding="utf-8")
    root = workspace.root

    preserved = workspace.close(keep_log=True)

    # The diagnostic survives a failure; the raster pixels do not.
    assert not root.exists()
    assert preserved is not None
    assert preserved.read_text(encoding="utf-8") == "what went wrong"
    preserved.unlink()


def test_the_context_manager_cleans_up_on_success(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.workspace import RunWorkspace

    with RunWorkspace.create(parent=tmp_path) as workspace:
        workspace.write_tile("a.raw", b"a")
        root = workspace.root

    assert not root.exists()


def test_the_context_manager_cleans_up_on_failure(tmp_path: Path) -> None:
    from tree_counter.qgis_adapter.workspace import RunWorkspace

    root = None
    with pytest.raises(RuntimeError):
        with RunWorkspace.create(parent=tmp_path) as workspace:
            root = workspace.root
            workspace.write_tile("a.raw", b"a")
            raise RuntimeError("the run failed")

    assert root is not None and not root.exists()

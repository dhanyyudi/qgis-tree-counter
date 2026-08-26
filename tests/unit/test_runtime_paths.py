"""Tests for isolated per-user runtime locations."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path

import pytest


def _root(platform: str, environment: dict[str, str], home: str) -> Path:
    from tree_counter.runtime.paths import default_runtime_root

    return default_runtime_root(
        platform=platform, environment=environment, home=Path(home)
    )


def test_windows_uses_local_app_data() -> None:
    root = _root(
        "win32",
        {"LOCALAPPDATA": "C:\\Data\\AppData\\Local"},
        "C:\\Data",
    )

    assert "AppData" in str(root)
    assert root.name == "runtime"


def test_windows_falls_back_to_the_home_directory() -> None:
    root = _root("win32", {}, "C:\\Data")

    assert str(root).startswith("C:")


def test_macos_uses_application_support() -> None:
    root = _root("darwin", {}, "/testhome/u")

    assert "Application Support" in str(root)
    assert "TreeCounter" in str(root)


def test_linux_honours_xdg_data_home() -> None:
    root = _root("linux", {"XDG_DATA_HOME": "/data/xdg"}, "/home/u")

    assert str(root).startswith("/data/xdg")


def test_linux_falls_back_to_local_share() -> None:
    root = _root("linux", {}, "/home/u")

    assert str(root).startswith("/home/u/.local/share")


def test_the_root_is_never_the_home_directory_itself() -> None:
    for platform, home in (
        ("darwin", "/testhome/u"),
        ("linux", "/home/u"),
    ):
        assert _root(platform, {}, home) != Path(home)


def test_runtime_paths_are_derived_from_one_root(tmp_path: Path) -> None:
    from tree_counter.runtime.paths import RuntimePaths

    paths = RuntimePaths(tmp_path / "runtime")

    assert paths.staging.parent == paths.root
    assert paths.active.parent == paths.root
    assert paths.logs.parent == paths.root
    assert paths.manifest.parent == paths.active
    for path in (paths.staging, paths.active, paths.logs):
        assert paths.root in path.parents


def test_runtime_paths_reject_a_relative_root() -> None:
    from tree_counter.runtime.paths import RuntimeLocationError, RuntimePaths

    with pytest.raises(RuntimeLocationError):
        RuntimePaths(Path("relative/runtime"))


@pytest.mark.parametrize("name", ["/", "/usr", "/etc"])
def test_a_system_directory_is_refused(name: str) -> None:
    from tree_counter.runtime.paths import (
        RuntimeLocationError,
        assert_safe_runtime_root,
    )

    with pytest.raises(RuntimeLocationError):
        assert_safe_runtime_root(Path(name))


def test_the_home_directory_itself_is_refused(tmp_path: Path) -> None:
    from tree_counter.runtime.paths import (
        RuntimeLocationError,
        assert_safe_runtime_root,
    )

    with pytest.raises(RuntimeLocationError):
        assert_safe_runtime_root(tmp_path, home=tmp_path)


def test_a_root_inside_the_plugin_is_refused(tmp_path: Path) -> None:
    from tree_counter.runtime.paths import (
        RuntimeLocationError,
        assert_safe_runtime_root,
    )

    plugin = tmp_path / "plugins" / "tree_counter"
    plugin.mkdir(parents=True)

    with pytest.raises(RuntimeLocationError):
        assert_safe_runtime_root(plugin / "runtime", forbidden=(plugin,))


def test_a_root_inside_the_qgis_prefix_is_refused(tmp_path: Path) -> None:
    from tree_counter.runtime.paths import (
        RuntimeLocationError,
        assert_safe_runtime_root,
    )

    qgis = tmp_path / "QGIS.app"
    qgis.mkdir()

    with pytest.raises(RuntimeLocationError):
        assert_safe_runtime_root(
            qgis / "Contents" / "runtime", forbidden=(qgis,)
        )


def test_a_root_equal_to_a_forbidden_directory_is_refused(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.paths import (
        RuntimeLocationError,
        assert_safe_runtime_root,
    )

    with pytest.raises(RuntimeLocationError):
        assert_safe_runtime_root(tmp_path, forbidden=(tmp_path,))


def test_a_normal_per_user_root_is_accepted(tmp_path: Path) -> None:
    from tree_counter.runtime.paths import assert_safe_runtime_root

    root = tmp_path / "TreeCounter" / "runtime"

    assert assert_safe_runtime_root(root) == root.resolve()


def test_the_default_root_is_outside_the_installed_plugin() -> None:
    from tree_counter.runtime.paths import default_runtime_root

    import tree_counter

    plugin = Path(tree_counter.__file__).resolve().parent
    root = default_runtime_root()

    assert plugin not in root.parents
    assert root != plugin


def test_runtime_states_cover_the_locked_set() -> None:
    from tree_counter.runtime.paths import RuntimeState

    assert {state.value for state in RuntimeState} == {
        "not_installed",
        "installing",
        "ready",
        "update_available",
        "incompatible",
        "repair_required",
    }


def test_an_install_directory_is_named_for_its_revision(
    tmp_path: Path,
) -> None:
    from tree_counter.runtime.paths import RuntimePaths

    paths = RuntimePaths(tmp_path / "runtime")

    install = paths.install_for("abc123")

    assert install.name == "abc123"
    assert paths.root in install.parents


@pytest.mark.parametrize("revision", ["", "..", "a/b", "a\\b", "."])
def test_an_unsafe_revision_is_refused(
    tmp_path: Path, revision: str
) -> None:
    from tree_counter.runtime.paths import RuntimeLocationError, RuntimePaths

    paths = RuntimePaths(tmp_path / "runtime")

    with pytest.raises(RuntimeLocationError):
        paths.install_for(revision)

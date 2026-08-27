"""Tests for Runtime Manager decisions that need no Qt."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations


def test_a_missing_runtime_offers_only_install() -> None:
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import available_actions

    assert available_actions(RuntimeState.NOT_INSTALLED) == ("install",)


def test_a_ready_runtime_is_not_offered_a_reinstall() -> None:
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import is_action_enabled

    assert is_action_enabled("install", RuntimeState.READY) is False
    assert is_action_enabled("verify", RuntimeState.READY) is True
    assert is_action_enabled("remove", RuntimeState.READY) is True


def test_an_update_is_offered_only_when_one_exists() -> None:
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import is_action_enabled

    assert is_action_enabled("update", RuntimeState.READY) is False
    assert is_action_enabled("update", RuntimeState.UPDATE_AVAILABLE) is True


def test_a_broken_runtime_is_offered_repair() -> None:
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import is_action_enabled

    assert is_action_enabled("repair", RuntimeState.REPAIR_REQUIRED) is True
    assert is_action_enabled("repair", RuntimeState.READY) is False


def test_an_incompatible_runtime_can_only_be_removed() -> None:
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import available_actions

    # Reinstalling the same thing would not help.
    assert available_actions(RuntimeState.INCOMPATIBLE) == ("remove",)


def test_nothing_may_start_while_installing() -> None:
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import available_actions

    assert available_actions(RuntimeState.INSTALLING) == ()


def test_every_state_is_covered() -> None:
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import ALLOWED_ACTIONS

    assert set(ALLOWED_ACTIONS) == set(RuntimeState)


def test_every_runtime_state_has_its_own_display_label() -> None:
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import STATE_LABELS

    assert STATE_LABELS == {
        RuntimeState.NOT_INSTALLED: "not installed",
        RuntimeState.INSTALLING: "installing",
        RuntimeState.READY: "ready",
        RuntimeState.UPDATE_AVAILABLE: "update available",
        RuntimeState.INCOMPATIBLE: "incompatible",
        RuntimeState.REPAIR_REQUIRED: "repair required",
    }


def _catalog():
    from tree_counter.runtime.catalog import load_catalog

    return load_catalog()


def test_offers_put_the_recommended_component_first() -> None:
    from tree_counter.ui.runtime_dialog import build_offers

    offers = build_offers(_catalog(), "macos-arm64")

    assert offers[0].name == "onnxruntime"
    assert offers[0].recommended is True
    assert any(not offer.recommended for offer in offers)


def test_offers_describe_source_and_size() -> None:
    from tree_counter.ui.runtime_dialog import build_offers

    offer = build_offers(_catalog(), "macos-arm64")[0]

    assert offer.source.startswith("https://")
    assert offer.estimated_bytes > 0
    assert offer.estimated_size.endswith(("MB", "GB"))


def test_offers_report_the_accelerators_a_platform_provides() -> None:
    from tree_counter.ui.runtime_dialog import build_offers

    macos = {o.name: o for o in build_offers(_catalog(), "macos-arm64")}
    linux = {o.name: o for o in build_offers(_catalog(), "linux-x86_64")}

    assert "coreml" in macos["onnxruntime"].accelerators
    assert "mps" in macos["pytorch"].accelerators
    assert "coreml" not in linux["onnxruntime"].accelerators
    assert "mps" not in linux["pytorch"].accelerators


def test_a_platform_without_a_runtime_offers_nothing() -> None:
    from tree_counter.ui.runtime_dialog import build_offers

    # macOS Intel has no runtime profile in v1.
    assert build_offers(_catalog(), "macos-x86_64") == ()


def test_the_confirmation_names_what_will_happen() -> None:
    from pathlib import Path

    from tree_counter.ui.runtime_dialog import build_offers, confirmation_text

    offers = build_offers(_catalog(), "macos-arm64")

    text = confirmation_text("install", offers, Path("/data/TreeCounter"))

    assert "Install" in text
    assert "https://pypi.org/simple" in text
    assert "/data/TreeCounter" in text
    assert "kept until the new one is verified" in text


def test_the_remove_confirmation_says_it_deletes() -> None:
    from pathlib import Path

    from tree_counter.ui.runtime_dialog import confirmation_text

    text = confirmation_text("remove", (), Path("/data/TreeCounter"))

    assert "deleted" in text


def test_a_status_summary_lists_versions_and_reasons() -> None:
    from tree_counter.runtime.installer import RuntimeStatus
    from tree_counter.runtime.manifest import parse_manifest
    from tree_counter.runtime.paths import RuntimeState
    from tree_counter.ui.runtime_dialog import describe_status

    manifest = parse_manifest(
        {
            "schema_version": 1,
            "catalog_version": 1,
            "python_version": "3.12.11",
            "platform": "macos-arm64",
            "components": {
                "onnxruntime": {
                    "lock_digest": "a" * 64,
                    "versions": {"onnxruntime": "1.29.0"},
                    "accelerators": ["cpu"],
                }
            },
            "installed_at": 1700000000,
        }
    )
    status = RuntimeStatus(
        RuntimeState.REPAIR_REQUIRED,
        ("Required runtime files are missing.",),
        manifest,
    )

    text = describe_status(status)

    assert "repair required" in text
    assert "3.12.11" in text
    assert "onnxruntime 1.29.0" in text
    assert "Required runtime files are missing." in text


def test_a_size_is_shown_in_readable_units() -> None:
    from tree_counter.ui.runtime_dialog import ComponentOffer

    offer = ComponentOffer(
        name="pytorch",
        title="PyTorch",
        recommended=False,
        version_summary="torch",
        source="https://pypi.org/simple",
        estimated_bytes=900_000_000,
        accelerators=("cpu",),
    )

    assert offer.estimated_size == "858 MB"


def test_the_dialog_module_performs_no_direct_download() -> None:
    from pathlib import Path

    from tree_counter.ui import runtime_dialog

    source = Path(runtime_dialog.__file__).read_text(encoding="utf-8")

    # Network access belongs to the installer, behind a confirmation.
    for marker in ("urllib", "requests", "socket", "QNetwork"):
        assert marker not in source, marker

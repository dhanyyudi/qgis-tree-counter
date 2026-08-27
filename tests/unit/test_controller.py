"""Tests for the dock's state machine, without any Qt."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


def FakeIdentity(filename="best", suffix=".onnx"):
    """Return a real ModelIdentity; the preset store validates its type."""

    from tree_counter.settings.trust import ModelIdentity

    return ModelIdentity(f"{filename}{suffix}", "a" * 64, suffix)


class FakeTrustStore:
    def __init__(self, trusted=False) -> None:
        self._trusted = trusted
        self.confirmed: list[object] = []

    def is_trusted(self, identity) -> bool:
        return self._trusted

    def confirm(self, identity) -> None:
        self._trusted = True
        self.confirmed.append(identity)


class FakePresetStore:
    def __init__(self, preset=None) -> None:
        self.preset = preset
        self.saved: list[object] = []

    def load(self, identity):
        from tree_counter.core.types import InferenceSettings
        from tree_counter.settings.presets import ModelPreset

        return self.preset or ModelPreset(identity, InferenceSettings())

    def save(self, preset) -> None:
        self.saved.append(preset)


def _controller(
    class_names=("oil_palm",),
    suffix=".onnx",
    trusted=False,
    runtime="ready",
    preset_store=None,
    inspect_error=None,
    identify_error=None,
    **kwargs,
):
    from tree_counter.ui.controller import CountingController

    identity = FakeIdentity(suffix=suffix)

    def identify(path):
        if identify_error is not None:
            raise identify_error
        return identity

    def inspect(chosen):
        if inspect_error is not None:
            raise inspect_error
        return {"class_names": list(class_names), "backend": "onnxruntime"}

    return CountingController(
        identify_model=identify,
        inspect_model=inspect,
        trust_store=FakeTrustStore(trusted),
        preset_store=preset_store,
        runtime_status=lambda: runtime,
        **kwargs,
    )


def _ready(controller, tmp_path=None):
    from tree_counter.qgis_adapter.scope import ScopeKind

    controller.set_raster("aerial")
    controller.set_scope(ScopeKind.WHOLE_RASTER)
    controller.set_output_path("/tmp/out.gpkg")
    return controller.select_model("/models/best.onnx")


def test_a_fresh_controller_cannot_start() -> None:
    from tree_counter.ui.controller import Phase

    controller = _controller()

    assert controller.state.phase is Phase.IDLE
    assert controller.state.can_start is False
    assert controller.state.primary_action == "start"


def test_a_complete_selection_enables_start() -> None:
    controller = _controller()

    state = _ready(controller)

    assert state.can_start is True
    assert state.selected_class_ids == (0,)


def test_a_single_class_model_selects_its_class() -> None:
    controller = _controller(class_names=("oil_palm",))

    state = _ready(controller)

    assert state.model.class_names == ("oil_palm",)
    assert state.selected_class_ids == (0,)


def test_a_multi_class_model_selects_nothing() -> None:
    controller = _controller(class_names=("oil_palm", "shade_tree"))

    state = _ready(controller)

    # Choosing silently would change the count without being asked.
    assert state.selected_class_ids == ()
    assert state.can_start is False


def test_choosing_classes_enables_start() -> None:
    controller = _controller(class_names=("oil_palm", "shade_tree"))
    _ready(controller)

    state = controller.set_selected_classes([1])

    assert state.selected_class_ids == (1,)
    assert state.can_start is True


def test_out_of_range_classes_are_dropped() -> None:
    controller = _controller(class_names=("oil_palm",))
    _ready(controller)

    state = controller.set_selected_classes([0, 5, -1, 0])

    assert state.selected_class_ids == (0,)


class TestTrust:
    """A PyTorch checkpoint is not inspected before it is confirmed."""

    def test_an_unconfirmed_checkpoint_is_not_inspected(self) -> None:
        controller = _controller(suffix=".pt", trusted=False)

        state = _ready(controller)

        assert state.model.needs_trust is True
        assert state.model.trusted is False
        assert state.model.inspected is False
        assert state.can_start is False
        assert "confirmed" in state.message

    def test_confirming_inspects_and_enables_start(self) -> None:
        controller = _controller(suffix=".pt", trusted=False)
        _ready(controller)

        state = controller.confirm_trust()

        assert state.model.trusted is True
        assert state.model.inspected is True
        assert state.can_start is True

    def test_an_already_trusted_checkpoint_needs_no_confirmation(
        self,
    ) -> None:
        controller = _controller(suffix=".pt", trusted=True)

        state = _ready(controller)

        assert state.model.inspected is True
        assert state.can_start is True

    def test_an_onnx_model_never_asks_for_confirmation(self) -> None:
        controller = _controller(suffix=".onnx")

        state = _ready(controller)

        assert state.model.needs_trust is False


class TestPreconditions:
    """Start stays disabled until every section is complete."""

    def test_without_a_raster(self) -> None:
        controller = _controller()
        controller.set_output_path("/tmp/out.gpkg")
        controller.select_model("/models/best.onnx")

        assert controller.state.can_start is False

    def test_without_an_output_path(self) -> None:
        controller = _controller()
        controller.set_raster("aerial")
        controller.select_model("/models/best.onnx")

        assert controller.state.can_start is False

    def test_without_a_ready_runtime(self) -> None:
        controller = _controller(runtime="not_installed")

        state = _ready(controller)

        assert state.can_start is False

    def test_a_polygon_scope_needs_a_layer(self) -> None:
        from tree_counter.qgis_adapter.scope import ScopeKind

        controller = _controller()
        _ready(controller)

        state = controller.set_scope(ScopeKind.POLYGON)
        assert state.can_start is False

        state = controller.set_scope(ScopeKind.POLYGON, "blocks")
        assert state.can_start is True

    def test_pressing_start_too_early_explains_why(self) -> None:
        controller = _controller()

        state = controller.start()

        assert state.message
        assert state.phase.is_busy is False


class TestRunLifecycle:
    """Controls restore after every terminal state."""

    def _running(self, **kwargs):
        started: list[object] = []
        cancelled: list[int] = []
        controller = _controller(
            start_run=lambda state: started.append(state),
            cancel_run=lambda: cancelled.append(1),
            **kwargs,
        )
        _ready(controller)
        controller.start()
        return controller, started, cancelled

    def test_starting_switches_the_button_to_cancel(self) -> None:
        from tree_counter.ui.controller import Phase

        controller, started, _ = self._running()

        assert controller.state.phase is Phase.RUNNING
        assert controller.state.primary_action == "cancel"
        assert controller.state.controls_enabled is False
        assert len(started) == 1

    def test_progress_is_reported(self) -> None:
        controller, _, _ = self._running()

        controller.on_event(
            {"type": "progress", "completed_tiles": 3, "total_tiles": 12}
        )

        assert controller.state.completed_tiles == 3
        assert controller.state.progress_percent == 25

    def test_warnings_accumulate(self) -> None:
        controller, _, _ = self._running()

        controller.on_event({"type": "warning", "message": "CPU fallback."})
        controller.on_event({"type": "warning", "message": "Slow disk."})

        assert controller.state.warnings == ("CPU fallback.", "Slow disk.")

    def test_completion_reports_counts_and_restores_controls(self) -> None:
        from tree_counter.qgis_adapter.task import RunResult
        from tree_counter.ui.controller import Phase

        controller, _, _ = self._running()
        result = RunResult(run_id="run-1", tile_count=4)
        result.detections = ()

        state = controller.on_completed(result, "/tmp/out.gpkg")

        assert state.phase is Phase.COMPLETED
        assert state.controls_enabled is True
        assert state.primary_action == "start"
        assert state.output_path == "/tmp/out.gpkg"

    def test_cancelling_asks_the_task_to_stop(self) -> None:
        controller, _, cancelled = self._running()

        controller.cancel()

        assert cancelled == [1]

    def test_cancellation_restores_controls_and_says_no_output(
        self,
    ) -> None:
        from tree_counter.ui.controller import Phase

        controller, _, _ = self._running()
        controller.cancel()

        state = controller.on_cancelled()

        assert state.phase is Phase.CANCELLED
        assert state.controls_enabled is True
        assert "No output" in state.message

    def test_failure_restores_controls_with_a_safe_message(self) -> None:
        from tree_counter.errors import ErrorCode, TreeCounterError
        from tree_counter.ui.controller import Phase

        controller, _, _ = self._running()

        state = controller.on_failed(
            TreeCounterError(
                ErrorCode.INVALID_MODEL,
                diagnostic_detail="/home/u/models/best.pt is broken",
            )
        )

        assert state.phase is Phase.FAILED
        assert state.controls_enabled is True
        # The user message must not carry the private diagnostic.
        assert "/home/u" not in state.message

    def test_cancelling_when_not_running_does_nothing(self) -> None:
        controller = _controller()

        assert controller.cancel().phase.is_busy is False


class TestErrors:
    """A failing service leaves the dock usable."""

    def test_an_unreadable_model_is_reported(self) -> None:
        from tree_counter.errors import ErrorCode, TreeCounterError
        from tree_counter.ui.controller import Phase

        controller = _controller(
            identify_error=TreeCounterError(ErrorCode.INVALID_MODEL)
        )

        state = controller.select_model("/models/broken.onnx")

        assert state.phase is Phase.FAILED
        assert state.controls_enabled is True

    def test_a_rejected_model_is_reported(self) -> None:
        from tree_counter.errors import ErrorCode, TreeCounterError
        from tree_counter.ui.controller import Phase

        controller = _controller(
            inspect_error=TreeCounterError(ErrorCode.INVALID_MODEL)
        )

        state = controller.select_model("/models/segment.onnx")

        assert state.phase is Phase.FAILED

    def test_a_model_without_classes_is_reported(self) -> None:
        from tree_counter.ui.controller import Phase

        controller = _controller(class_names=())

        state = _ready(controller)

        assert state.phase is Phase.FAILED


class TestPresets:
    """Per-model settings are remembered and restored."""

    def test_a_stored_preset_is_applied_on_selection(self) -> None:
        from tree_counter.core.types import InferenceSettings
        from tree_counter.settings.presets import ModelPreset

        identity = FakeIdentity()
        store = FakePresetStore(
            ModelPreset(
                identity,
                InferenceSettings(confidence=0.4, tile_size=1024),
            )
        )
        controller = _controller(preset_store=store)

        state = _ready(controller)

        assert state.settings.confidence == pytest.approx(0.4)
        assert state.settings.tile_size == 1024

    def test_settings_are_saved_when_a_run_completes(self) -> None:
        from tree_counter.qgis_adapter.task import RunResult

        store = FakePresetStore()
        controller = _controller(preset_store=store, start_run=lambda s: None)
        _ready(controller)
        controller.start()

        controller.on_completed(RunResult(run_id="run-1"), "/tmp/out.gpkg")

        assert len(store.saved) == 1
        assert store.saved[0].settings.selected_class_ids == (0,)


def test_listeners_render_every_change() -> None:
    seen: list[object] = []
    controller = _controller()

    controller.subscribe(seen.append)
    controller.set_raster("aerial")

    # Subscribing renders once immediately, then once per change.
    assert len(seen) == 2
    assert seen[-1].raster_name == "aerial"


class TestOutputLayers:
    """At least one detection layer must be requested."""

    def test_the_flags_default_to_centres_only(self) -> None:
        controller = _controller()

        assert controller.state.write_centers is True
        assert controller.state.write_boxes is False

    def test_the_flags_update_together(self) -> None:
        controller = _controller()

        controller.set_output_layers(False, True)

        assert controller.state.write_centers is False
        assert controller.state.write_boxes is True

    def test_neither_layer_blocks_start(self) -> None:
        controller = _controller()
        _ready(controller)

        state = controller.set_output_layers(False, False)

        assert state.can_start is False

    def test_boxes_alone_keeps_start_enabled(self) -> None:
        controller = _controller()
        _ready(controller)

        state = controller.set_output_layers(False, True)

        assert state.can_start is True


def test_the_controller_imports_no_qt() -> None:
    from pathlib import Path

    from tree_counter.ui import controller

    source = Path(controller.__file__).read_text(encoding="utf-8")

    assert "PyQt" not in source
    assert "QtWidgets" not in source


def test_the_controller_performs_no_network_operation() -> None:
    from pathlib import Path

    from tree_counter.ui import controller

    source = Path(controller.__file__).read_text(encoding="utf-8")

    for marker in ("urllib", "requests", "http", "socket", "QNetwork"):
        assert marker not in source, marker


def test_a_failed_selection_does_not_keep_the_previous_model() -> None:
    """A stale choice would pair an old hash with a new, unusable file.

    The previous ModelChoice stayed on the state after a failed
    selection, so its hash could still reach a run while the plugin had
    already moved on to the file the user just picked.
    """

    from tree_counter.errors import ErrorCode, TreeCounterError
    from tree_counter.qgis_adapter.scope import ScopeKind
    from tree_counter.ui.controller import CountingController

    attempts: list[str] = []
    identity = FakeIdentity(suffix=".onnx")

    def identify(path):
        attempts.append(str(path))
        if len(attempts) > 1:
            raise TreeCounterError(ErrorCode.INVALID_MODEL)
        return identity

    controller = CountingController(
        identify_model=identify,
        inspect_model=lambda chosen: {
            "class_names": ["oil_palm"],
            "backend": "onnxruntime",
        },
        trust_store=FakeTrustStore(True),
        runtime_status=lambda: "ready",
    )
    controller.set_raster("aerial")
    controller.set_scope(ScopeKind.WHOLE_RASTER)
    controller.set_output_path("/tmp/out.gpkg")
    controller.select_model("/models/good.onnx")
    assert controller.state.model is not None

    state = controller.select_model("/models/missing.onnx")

    assert state.model is None
    assert state.can_start is False


def test_a_complete_selection_has_nothing_blocking_it() -> None:
    controller = _controller()

    state = _ready(controller)

    assert state.can_start is True
    assert state.blocking_reason == ""


def test_the_missing_output_path_is_named() -> None:
    """A disabled Start button must say what it is waiting for.

    Everything else can be set correctly and Start still stays grey; with
    no explanation the user has no way to discover which field is empty.
    """

    from tree_counter.qgis_adapter.scope import ScopeKind
    from tree_counter.ui.controller import BLOCKING_REASONS

    controller = _controller()
    controller.set_raster("aerial")
    controller.set_scope(ScopeKind.WHOLE_RASTER)
    controller.select_model("/models/best.onnx")

    state = controller.state

    assert state.can_start is False
    assert state.blocking_reason == BLOCKING_REASONS["output"]


def test_the_missing_raster_is_named_before_anything_else() -> None:
    """The reason follows the order the user fills the panel in."""

    from tree_counter.ui.controller import BLOCKING_REASONS

    controller = _controller()

    assert controller.state.blocking_reason == BLOCKING_REASONS["raster"]

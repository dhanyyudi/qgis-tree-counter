"""The state behind the dock.

The controller owns every transition and every service call; widgets only
render what it reports and forward what the user did. It imports no Qt, so
the whole workflow - eligibility, model trust, class selection, when Start
may be pressed, what happens on cancel or failure - is testable without an
application, which is where the behaviour actually lives.

Nothing here reaches the network. The only component that may is the
Runtime Manager, and only after an explicit action there.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from tree_counter.core.types import InferenceSettings
from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.qgis_adapter.scope import ScopeKind

Listener = Callable[["ViewState"], None]


class Phase(str, Enum):
    """What the dock is doing right now."""

    IDLE = "idle"
    INSPECTING = "inspecting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_busy(self) -> bool:
        """Return whether a long operation is in progress."""

        return self in (Phase.INSPECTING, Phase.RUNNING)


# Why Start is not available yet, in the order the panel is filled in. A
# disabled button with no explanation is a dead end: everything else can
# be set correctly and Start still stays grey.
BLOCKING_REASONS = {
    "raster": "Choose a raster layer to count trees in.",
    "model": "Choose a detection model.",
    "inspecting": "Waiting for the model to be inspected.",
    "trust": "Confirm this PyTorch checkpoint before it can be used.",
    "classes": "Select at least one class to count.",
    "runtime": "Install the Tree Counter runtime first.",
    "polygon": "Choose the polygon layer that defines the scope.",
    "layers": "Select at least one output layer.",
    "output": "Choose where to write the results.",
}


@dataclass(frozen=True)
class ModelChoice:
    """A chosen model, once it has been hashed and inspected."""

    filename: str
    sha256: str
    suffix: str
    class_names: tuple[str, ...] = ()
    backend: str = ""
    needs_trust: bool = False
    trusted: bool = False
    inspected: bool = False


@dataclass(frozen=True)
class ViewState:
    """Everything the dock needs to render itself."""

    phase: Phase = Phase.IDLE
    raster_name: str = ""
    scope: ScopeKind = ScopeKind.WHOLE_RASTER
    polygon_layer_name: str = ""
    model: ModelChoice | None = None
    selected_class_ids: tuple[int, ...] = ()
    settings: InferenceSettings = field(default_factory=InferenceSettings)
    runtime_state: str = "not_installed"
    output_path: str = ""
    write_centers: bool = True
    write_boxes: bool = False
    completed_tiles: int = 0
    total_tiles: int = 0
    warnings: tuple[str, ...] = ()
    message: str = ""
    total_count: int = 0
    counts_by_class: Mapping[str, int] = field(default_factory=dict)

    @property
    def can_start(self) -> bool:
        """Return whether Start may be pressed.

        Every precondition must hold: a raster, an inspected model, a
        confirmed hash for a PyTorch checkpoint, at least one class, a
        ready runtime, and somewhere to write.
        """

        if self.phase.is_busy:
            return False
        if not self.raster_name or self.model is None:
            return False
        if not self.model.inspected:
            return False
        if self.model.needs_trust and not self.model.trusted:
            return False
        if not self.selected_class_ids:
            return False
        if self.runtime_state != "ready":
            return False
        if self.scope is ScopeKind.POLYGON and not self.polygon_layer_name:
            return False
        if not (self.write_centers or self.write_boxes):
            return False
        return bool(self.output_path)

    @property
    def blocking_reason(self) -> str:
        """Return why Start is unavailable, or an empty string."""

        if not self.raster_name:
            return BLOCKING_REASONS["raster"]
        if self.model is None:
            return BLOCKING_REASONS["model"]
        if self.model.needs_trust and not self.model.trusted:
            return BLOCKING_REASONS["trust"]
        if not self.model.inspected:
            return BLOCKING_REASONS["inspecting"]
        if not self.selected_class_ids:
            return BLOCKING_REASONS["classes"]
        if self.runtime_state != "ready":
            return BLOCKING_REASONS["runtime"]
        if self.scope is ScopeKind.POLYGON and not self.polygon_layer_name:
            return BLOCKING_REASONS["polygon"]
        if not (self.write_centers or self.write_boxes):
            return BLOCKING_REASONS["layers"]
        if not self.output_path:
            return BLOCKING_REASONS["output"]
        return ""

    @property
    def primary_action(self) -> str:
        """Return whether the main button starts or cancels."""

        return "cancel" if self.phase is Phase.RUNNING else "start"

    @property
    def controls_enabled(self) -> bool:
        """Return whether the settings controls accept input."""

        return not self.phase.is_busy

    @property
    def progress_percent(self) -> int:
        """Return tile progress as a percentage."""

        if self.total_tiles <= 0:
            return 0
        return min(
            100, round(self.completed_tiles * 100 / self.total_tiles)
        )


class CountingController:
    """Owns dock state and the services the dock is not allowed to call."""

    def __init__(
        self,
        identify_model: Callable[[str], Any],
        inspect_model: Callable[[Any], Mapping[str, Any]],
        trust_store: Any | None = None,
        preset_store: Any | None = None,
        runtime_status: Callable[[], str] | None = None,
        start_run: Callable[[ViewState], None] | None = None,
        cancel_run: Callable[[], None] | None = None,
        start_model_inspection: Callable[[Any], None] | None = None,
    ) -> None:
        self._identify_model = identify_model
        self._inspect_model = inspect_model
        self._trust_store = trust_store
        self._preset_store = preset_store
        self._runtime_status = runtime_status or (lambda: "not_installed")
        self._start_run = start_run
        self._cancel_run = cancel_run
        self._start_model_inspection = start_model_inspection
        self._listeners: list[Listener] = []
        self._identity: Any = None
        self._state = ViewState(runtime_state=self._runtime_status())

    # -- observation -----------------------------------------------------

    @property
    def state(self) -> ViewState:
        """Return the current view state."""

        return self._state

    def subscribe(self, listener: Listener) -> None:
        """Register a listener that renders every state change."""

        self._listeners.append(listener)
        listener(self._state)

    def _update(self, **changes: Any) -> ViewState:
        self._state = replace(self._state, **changes)
        for listener in list(self._listeners):
            listener(self._state)
        return self._state

    # -- data ------------------------------------------------------------

    def set_raster(self, name: str) -> ViewState:
        """Choose the raster to count."""

        return self._update(raster_name=str(name or ""), message="")

    def set_scope(
        self, scope: ScopeKind, polygon_layer_name: str = ""
    ) -> ViewState:
        """Choose the processing scope."""

        return self._update(
            scope=scope,
            polygon_layer_name=str(polygon_layer_name or ""),
            message="",
        )

    def set_output_path(self, path: str | Path) -> ViewState:
        """Choose where results are written."""

        return self._update(output_path=str(path or ""))

    def set_output_layers(
        self, write_centers: bool, write_boxes: bool
    ) -> ViewState:
        """Choose which detection layers are written."""

        return self._update(
            write_centers=bool(write_centers),
            write_boxes=bool(write_boxes),
        )

    def refresh_runtime(self) -> ViewState:
        """Re-read the runtime state without changing it."""

        return self._update(runtime_state=self._runtime_status())

    # -- model -----------------------------------------------------------

    def select_model(self, path: str | Path) -> ViewState:
        """Hash, trust-check and inspect a model file.

        A PyTorch checkpoint is never inspected before the user has
        confirmed its exact hash, because inspecting it means loading it.
        """

        self._update(phase=Phase.INSPECTING, message="")
        try:
            identity = self._identify_model(str(path))
        except TreeCounterError as error:
            # Keeping the previous choice would leave its hash on screen
            # and let Start pair it with a file that never identified.
            self._identity = None
            self._update(model=None, selected_class_ids=())
            return self._fail(error)
        self._identity = identity
        needs_trust = bool(
            getattr(identity, "requires_trust_confirmation", False)
        )
        trusted = True
        if needs_trust and self._trust_store is not None:
            trusted = bool(self._trust_store.is_trusted(identity))
        choice = ModelChoice(
            filename=identity.filename,
            sha256=identity.sha256,
            suffix=identity.suffix,
            needs_trust=needs_trust,
            trusted=trusted,
        )
        self._update(model=choice, selected_class_ids=())
        if needs_trust and not trusted:
            return self._update(
                phase=Phase.IDLE,
                message=(
                    "This PyTorch checkpoint must be confirmed before it "
                    "can be used."
                ),
            )
        return self._inspect()

    def confirm_trust(self) -> ViewState:
        """Record the user's confirmation and continue inspecting."""

        if self._identity is None or self._state.model is None:
            return self._state
        if self._trust_store is not None:
            self._trust_store.confirm(self._identity)
        self._update(
            model=replace(self._state.model, trusted=True),
            phase=Phase.INSPECTING,
        )
        return self._inspect()

    def _inspect(self) -> ViewState:
        if self._start_model_inspection is not None:
            try:
                self._start_model_inspection(self._identity)
            except TreeCounterError as error:
                return self._fail(error)
            return self._state
        try:
            info = self._inspect_model(self._identity)
        except TreeCounterError as error:
            return self._fail(error)
        return self.on_model_inspected(info)

    def on_model_inspected(self, info: Mapping[str, Any]) -> ViewState:
        """Apply model information delivered by a background task."""

        names = tuple(str(name) for name in info.get("class_names", ()))
        if not names:
            return self._fail(
                TreeCounterError(
                    ErrorCode.INVALID_MODEL,
                    diagnostic_detail="the model declares no classes",
                )
            )
        model = replace(
            self._state.model,
            class_names=names,
            backend=str(info.get("backend", "")),
            inspected=True,
        )
        # One class is not a choice, so it is made for the user. Several
        # are a choice, and making it silently would change the count.
        selected = (0,) if len(names) == 1 else ()
        state = self._update(
            model=model,
            selected_class_ids=selected,
            phase=Phase.READY,
            message="",
        )
        return self._apply_preset(state)

    def on_model_inspection_failed(self, error: Any) -> ViewState:
        """Apply a background model-inspection failure."""

        return self._fail(error)

    def _apply_preset(self, state: ViewState) -> ViewState:
        if self._preset_store is None or self._identity is None:
            return state
        try:
            preset = self._preset_store.load(self._identity)
        except TreeCounterError:
            return state
        selected = tuple(preset.settings.selected_class_ids)
        model = state.model
        if selected and model is not None:
            selected = tuple(
                index for index in selected if index < len(model.class_names)
            )
        return self._update(
            settings=preset.settings,
            selected_class_ids=selected or state.selected_class_ids,
        )

    def set_selected_classes(
        self, class_ids: Sequence[int]
    ) -> ViewState:
        """Choose which classes are counted."""

        model = self._state.model
        allowed = len(model.class_names) if model else 0
        chosen = tuple(
            int(value)
            for value in dict.fromkeys(class_ids)
            if 0 <= int(value) < allowed
        )
        return self._update(selected_class_ids=chosen)

    def set_settings(self, settings: InferenceSettings) -> ViewState:
        """Replace the inference settings."""

        return self._update(settings=settings)

    # -- run -------------------------------------------------------------

    def start(self) -> ViewState:
        """Begin a counting run, if every precondition holds."""

        if not self._state.can_start:
            return self._update(
                message="Complete every section before counting."
            )
        state = self._update(
            phase=Phase.RUNNING,
            completed_tiles=0,
            total_tiles=0,
            warnings=(),
            message="",
            total_count=0,
            counts_by_class={},
        )
        if self._start_run is not None:
            self._start_run(state)
        return self._state

    def cancel(self) -> ViewState:
        """Ask the running task to stop."""

        if self._state.phase is not Phase.RUNNING:
            return self._state
        if self._cancel_run is not None:
            self._cancel_run()
        return self._update(message="Cancelling...")

    def on_event(self, event: Mapping[str, Any]) -> ViewState:
        """Apply a progress or warning event from the running task."""

        kind = str(event.get("type", ""))
        if kind == "progress":
            return self._update(
                completed_tiles=int(event.get("completed_tiles", 0)),
                total_tiles=int(event.get("total_tiles", 0)),
            )
        if kind == "warning":
            text = str(event.get("message", ""))
            return self._update(warnings=self._state.warnings + (text,))
        return self._state

    def on_completed(self, result: Any, output_path: str) -> ViewState:
        """Record a finished run and restore the controls."""

        counts = (
            result.counts_by_class()
            if hasattr(result, "counts_by_class")
            else {}
        )
        if self._preset_store is not None and self._identity is not None:
            self._remember_preset()
        return self._update(
            phase=Phase.COMPLETED,
            total_count=int(getattr(result, "total_count", 0)),
            counts_by_class=counts,
            warnings=tuple(getattr(result, "warnings", ())),
            output_path=str(output_path),
            message="Counting finished.",
        )

    def on_cancelled(self) -> ViewState:
        """Record a cancelled run and restore the controls."""

        return self._update(
            phase=Phase.CANCELLED,
            message="Counting was cancelled. No output was written.",
        )

    def on_failed(self, error: Any) -> ViewState:
        """Record a failed run and restore the controls."""

        return self._fail(error)

    def _fail(self, error: Any) -> ViewState:
        message = getattr(error, "user_message", None) or str(error)
        return self._update(phase=Phase.FAILED, message=str(message))

    def _remember_preset(self) -> None:
        from tree_counter.settings.presets import ModelPreset

        settings = replace(
            self._state.settings,
            selected_class_ids=self._state.selected_class_ids,
        )
        try:
            self._preset_store.save(
                ModelPreset(
                    self._identity,
                    settings,
                    last_backend=(
                        self._state.model.backend
                        if self._state.model
                        else None
                    )
                    or None,
                )
            )
        except TreeCounterError:  # A preset is a convenience, not a result.
            return


__all__ = [
    "BLOCKING_REASONS",
    "CountingController",
    "ModelChoice",
    "Phase",
    "ViewState",
]

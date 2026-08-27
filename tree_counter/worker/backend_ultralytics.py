"""The trusted PyTorch/Ultralytics backend for ``.pt`` checkpoints.

Loading a PyTorch checkpoint executes code from the file, so this backend
runs only against content the user has explicitly trusted. The host sends
the SHA-256 it obtained that confirmation for, and the worker hashes the
file again before loading it: without that second check, a file swapped
between the confirmation and the run would load unchallenged.

Predictions come from the detection module directly, never from the
high-level ``predict`` path, because that path applies Ultralytics' own NMS
before the plugin's NMS IoU setting could take effect.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tree_counter.errors import ErrorCode, TreeCounterError
from tree_counter.settings.trust import hash_file
from tree_counter.worker.backend_base import ModelDescription, ModelRejected
from tree_counter.worker.capabilities import select_device
from tree_counter.worker.image import (
    letterbox,
    read_rgb_tile,
    to_model_input,
)
from tree_counter.worker.model_info import (
    describe_output_layout,
    normalize_family,
    parse_class_map,
    require_detection_task,
    validate_selected_class_ids,
)
from tree_counter.worker.yolo_postprocess import decode_predictions

BACKEND_NAME = "ultralytics"
DEFAULT_INPUT_SIZE = 640


class UltralyticsMissing(TreeCounterError):
    """The runtime does not provide PyTorch and Ultralytics."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.MISSING_RUNTIME, diagnostic_detail=detail)


class ModelNotTrusted(TreeCounterError):
    """The checkpoint content does not match the confirmed hash."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.INVALID_MODEL, diagnostic_detail=detail)


def _import_runtime() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise UltralyticsMissing(
            f"PyTorch is not installed in this runtime: {exc}"
        ) from exc
    try:
        import ultralytics
    except ImportError as exc:
        raise UltralyticsMissing(
            f"Ultralytics is not installed in this runtime: {exc}"
        ) from exc
    return torch, ultralytics


def verify_trusted_checkpoint(model_path: str, declared_sha256: object) -> str:
    """Return the verified hash, or refuse to touch the checkpoint.

    The hash is recomputed here rather than trusted from the message: the
    host confirmed a specific content, and only that content may be loaded.
    """

    if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
        raise ModelNotTrusted(
            "a PyTorch checkpoint needs a confirmed SHA-256 before it can "
            "be loaded"
        )
    declared = declared_sha256.casefold()
    try:
        int(declared, 16)
    except ValueError as exc:
        raise ModelNotTrusted(
            "the confirmed SHA-256 is not a hexadecimal digest"
        ) from exc
    actual = hash_file(model_path)
    if actual != declared:
        # The file changed between the user's confirmation and this load.
        raise ModelNotTrusted(
            "the checkpoint content does not match the confirmed hash; "
            "confirm the model again before running it"
        )
    return actual


def _reports_available(probe: Any) -> bool:
    """Return whether a device probe says yes, treating any error as no.

    A PyTorch build without a given backend raises rather than returning
    False, and a failed probe must never stop the other devices from being
    listed.
    """

    try:
        return bool(probe())
    except Exception:
        return False


def _raw_predictions(output: Any) -> Any:
    """Return the raw detection tensor from a detection module's output.

    Ultralytics returns either the prediction tensor itself or a tuple whose
    first element is the prediction and whose remainder are feature maps.
    Anything else means the pinned version changed shape, and guessing would
    silently misread every box.
    """

    candidate = output
    if isinstance(candidate, (tuple, list)):
        if not candidate:
            raise ModelRejected(
                "the detection module returned no predictions"
            )
        candidate = candidate[0]
    if isinstance(candidate, (tuple, list)):
        raise ModelRejected(
            "the installed Ultralytics version returns an unsupported "
            "prediction structure"
        )
    if not hasattr(candidate, "shape"):
        raise ModelRejected(
            "the installed Ultralytics version returns an unsupported "
            "prediction type"
        )
    return candidate


class UltralyticsBackend:
    """Runs trusted Ultralytics YOLO11 ``.pt`` detection checkpoints."""

    name = BACKEND_NAME

    def __init__(self) -> None:
        self._torch: Any = None
        self._module: Any = None
        self._layout: Any = None
        self._description: ModelDescription | None = None
        self._selected: tuple[int, ...] = ()
        self._confidence = 0.0
        self._device = "cpu"

    def capabilities(self) -> tuple[str, ...]:
        """Return the accelerators PyTorch reports on this machine."""

        try:
            torch, _ = _import_runtime()
        except UltralyticsMissing:
            return ()
        accelerators: list[str] = []
        if _reports_available(lambda: torch.cuda.is_available()):
            accelerators.append("cuda")
        if _reports_available(lambda: torch.backends.mps.is_available()):
            accelerators.append("mps")
        return tuple(accelerators)

    def inspect(
        self, model_path: str, model_sha256: str
    ) -> ModelDescription:
        """Describe a trusted checkpoint without preparing it for a run."""

        verified = verify_trusted_checkpoint(model_path, model_sha256)
        model = self._load(model_path)
        return self._describe(model, model_path, verified, "cpu")

    def _describe(
        self,
        model: Any,
        model_path: str,
        sha256: str,
        device: str,
        warnings: Sequence[str] = (),
    ) -> ModelDescription:
        metadata: dict[str, object] = {}
        task = getattr(model, "task", None)
        if task is not None:
            metadata["task"] = task
        require_detection_task(metadata)
        metadata["model"] = self._family_hint(model)
        family = normalize_family(metadata)
        class_names = parse_class_map(getattr(model, "names", None))
        input_size = int(
            getattr(model, "imgsz", None) or DEFAULT_INPUT_SIZE
        )
        return ModelDescription(
            filename=Path(model_path).name,
            sha256=sha256,
            model_format="pt",
            task="detect",
            family=family,
            class_names=class_names,
            input_width=input_size,
            input_height=input_size,
            dynamic_shape=True,
            backend=self.name,
            provider="torch",
            device=device,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _family_hint(model: Any) -> str:
        """Return the architecture this checkpoint was built from.

        The architecture decides, never the file name. Ultralytics puts
        the path the user loaded in ``model_name`` and ``ckpt_path``, so
        trusting those rejects every trained checkpoint that is not still
        called ``yolo11n.pt`` - and would accept a foreign architecture
        that merely happens to be named like one.
        """

        inner = getattr(model, "model", None)
        yaml = getattr(inner, "yaml", None)
        if isinstance(yaml, Mapping):
            architecture = str(
                yaml.get("yaml_file") or yaml.get("model") or ""
            )
            if architecture:
                return Path(architecture).name
        for attribute in ("cfg", "model_name", "ckpt_path"):
            value = getattr(model, attribute, None)
            if value:
                return Path(str(value)).name
        return ""

    def start_run(
        self,
        model_path: str,
        model_sha256: str,
        settings: Any,
        platform: str | None = None,
        machine: str | None = None,
    ) -> Mapping[str, Any]:
        """Verify, load, and place the checkpoint on the chosen device."""

        import sys
        import platform as platform_module

        verified = verify_trusted_checkpoint(model_path, model_sha256)
        torch, _ = _import_runtime()
        selection = select_device(
            requested=getattr(settings, "requested_device", "auto"),
            model_format="pt",
            backend_accelerators=self.capabilities(),
            platform=sys.platform if platform is None else platform,
            machine=(
                platform_module.machine() if machine is None else machine
            ),
        )
        model = self._load(model_path)
        description = self._describe(
            model, model_path, verified, selection.device, selection.warnings
        )
        module = getattr(model, "model", None)
        if module is None:
            raise ModelRejected(
                "the checkpoint does not expose a detection module"
            )
        try:
            module = module.to(selection.device)
            module.eval()
        except Exception as exc:
            raise ModelRejected(
                f"the model could not be placed on {selection.device}: "
                f"{type(exc).__name__}"
            ) from exc

        self._torch = torch
        self._module = module
        self._description = description
        self._device = selection.device
        self._selected = validate_selected_class_ids(
            tuple(getattr(settings, "selected_class_ids", ()) or ()),
            description.class_names,
        )
        self._confidence = float(getattr(settings, "confidence", 0.25))
        self._layout = None
        return {
            "backend": self.name,
            "provider": description.provider,
            "device": selection.device,
            "warnings": list(selection.warnings),
        }

    def infer_tile(
        self, tile_path: str, tile: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Return raw tile-local predictions for one tile."""

        if self._module is None or self._description is None:
            raise ModelRejected("the model has not been loaded")
        torch = self._torch
        image = read_rgb_tile(
            tile_path, int(tile["valid_width"]), int(tile["valid_height"])
        )
        padded, transform = letterbox(
            image, int(tile["model_width"]), int(tile["model_height"])
        )
        batch = to_model_input(padded)
        tensor = torch.from_numpy(batch).to(self._device)
        with torch.no_grad():
            # The detection module directly: predict() would apply
            # Ultralytics' own NMS before ours could.
            output = self._module(tensor)
        raw = _raw_predictions(output)
        array = self._to_numpy(raw)
        if self._layout is None:
            self._layout = describe_output_layout(
                list(array.shape), len(self._description.class_names)
            )
        detections = decode_predictions(
            array,
            self._layout,
            transform,
            self._confidence,
            self._selected,
        )
        names = self._description.class_names
        return [detection.as_payload(names) for detection in detections]

    def close(self) -> None:
        """Release the model and any device memory."""

        self._module = None
        self._description = None
        self._layout = None
        self._torch = None

    # -- helpers ---------------------------------------------------------

    def _load(self, model_path: str) -> Any:
        _, ultralytics = _import_runtime()
        path = Path(model_path)
        if not path.is_file():
            raise ModelRejected("the model file does not exist")
        try:
            return ultralytics.YOLO(str(path))
        except ModelRejected:
            raise
        except Exception as exc:
            # Never surface the absolute checkpoint path to the user.
            raise ModelRejected(
                f"the checkpoint could not be loaded: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _to_numpy(tensor: Any) -> Any:
        detached = getattr(tensor, "detach", None)
        if callable(detached):
            tensor = detached()
        moved = getattr(tensor, "cpu", None)
        if callable(moved):
            tensor = moved()
        converter = getattr(tensor, "numpy", None)
        if callable(converter):
            return converter()
        import numpy

        return numpy.asarray(tensor)


def create_backend() -> UltralyticsBackend:
    """Return a new trusted Ultralytics backend."""

    return UltralyticsBackend()


__all__ = [
    "BACKEND_NAME",
    "ModelNotTrusted",
    "UltralyticsBackend",
    "UltralyticsMissing",
    "create_backend",
    "verify_trusted_checkpoint",
]

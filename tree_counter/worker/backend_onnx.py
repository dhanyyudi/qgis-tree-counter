"""The ONNX Runtime detection backend.

ONNX Runtime and NumPy are imported here and nowhere else, so the QGIS
process never loads them. Everything about the model is validated before a
run starts: the metadata, the class map, the number and names of outputs,
the input shape, and the provider. An export that already applies NMS, or
whose output layout could be read two ways, is refused rather than guessed
at, because either would silently produce a wrong count.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tree_counter.errors import ErrorCode, TreeCounterError
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
    reject_embedded_nms,
    require_detection_task,
    validate_selected_class_ids,
)
from tree_counter.worker.yolo_postprocess import decode_predictions

BACKEND_NAME = "onnxruntime"
DEVICE_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}
PROVIDER_DEVICES = {value: key for key, value in DEVICE_PROVIDERS.items()}


class OnnxRuntimeMissing(TreeCounterError):
    """The runtime does not provide ONNX Runtime."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.MISSING_RUNTIME, diagnostic_detail=detail)


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime
    except ImportError as exc:
        raise OnnxRuntimeMissing(
            f"ONNX Runtime is not installed in this runtime: {exc}"
        ) from exc
    return onnxruntime


def _metadata_mapping(session: Any) -> dict[str, object]:
    raw = session.get_modelmeta()
    custom = dict(getattr(raw, "custom_metadata_map", {}) or {})
    metadata: dict[str, object] = {
        str(key): value for key, value in custom.items()
    }
    for attribute in ("description", "graph_name", "producer_name"):
        value = getattr(raw, attribute, None)
        if value and attribute not in metadata:
            metadata[attribute] = value
    return metadata


def _decode_names(raw: object) -> object:
    """Ultralytics stores the class map as a repr string in ONNX metadata."""

    if isinstance(raw, str):
        import ast

        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise ModelRejected(
                "the class map in the model metadata is unreadable"
            ) from exc
    return raw


def _input_shape(session: Any) -> tuple[int | None, int | None, bool]:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ModelRejected(
            f"expected one model input, found {len(inputs)}"
        )
    _require_float_tensor(inputs[0].type, "input")
    shape = list(inputs[0].shape)
    if len(shape) != 4:
        raise ModelRejected(f"unsupported input shape: {shape}")
    batch, channels = shape[0], shape[1]
    if isinstance(batch, int) and batch != 1:
        raise ModelRejected(
            f"unsupported batch dimension: expected 1 or dynamic, found "
            f"{batch}"
        )
    if channels != 3:
        raise ModelRejected(
            "unsupported channel dimension: expected RGB with 3 channels"
        )
    height, width = shape[2], shape[3]
    dynamic = not isinstance(height, int) or not isinstance(width, int)
    if dynamic:
        return (None, None, True)
    return (int(width), int(height), False)


def _require_float_tensor(data_type: object, role: str) -> None:
    if data_type != "tensor(float)":
        raise ModelRejected(
            f"unsupported {role} dtype: expected tensor(float), found "
            f"{data_type!r}"
        )


class OnnxBackend:
    """Runs Ultralytics YOLO11 ONNX detection exports."""

    name = BACKEND_NAME

    def __init__(self) -> None:
        self._session: Any = None
        self._layout: Any = None
        self._description: ModelDescription | None = None
        self._selected: tuple[int, ...] = ()
        self._confidence = 0.0
        self._input_name = ""

    # -- inspection ------------------------------------------------------

    def capabilities(self) -> tuple[str, ...]:
        """Return the accelerators ONNX Runtime reports on this machine."""

        try:
            onnxruntime = _import_onnxruntime()
        except OnnxRuntimeMissing:
            return ()
        providers = tuple(onnxruntime.get_available_providers())
        return tuple(
            PROVIDER_DEVICES[name]
            for name in providers
            if name in PROVIDER_DEVICES and name != "CPUExecutionProvider"
        )

    def inspect(
        self, model_path: str, model_sha256: str, device: str = "auto"
    ) -> ModelDescription:
        """Describe an ONNX model, rejecting anything unsupported."""

        session = self._open(model_path, ("CPUExecutionProvider",))
        return self._describe(
            session, model_path, model_sha256, device, "cpu"
        )

    def _describe(
        self,
        session: Any,
        model_path: str,
        model_sha256: str,
        requested_device: str,
        resolved_device: str,
        warnings: Sequence[str] = (),
    ) -> ModelDescription:
        metadata = _metadata_mapping(session)
        require_detection_task(metadata)
        family = normalize_family(metadata)
        class_names = parse_class_map(_decode_names(metadata.get("names")))

        outputs = session.get_outputs()
        if len(outputs) != 1:
            raise ModelRejected(
                f"expected one raw detection output, found {len(outputs)}"
            )
        _require_float_tensor(outputs[0].type, "output")
        reject_embedded_nms([item.name for item in outputs])
        layout = describe_output_layout(
            list(outputs[0].shape), len(class_names)
        )
        width, height, dynamic = _input_shape(session)
        self._layout = layout
        return ModelDescription(
            filename=Path(model_path).name,
            sha256=model_sha256.casefold(),
            model_format="onnx",
            task="detect",
            family=family,
            class_names=class_names,
            input_width=width or 0,
            input_height=height or 0,
            dynamic_shape=dynamic,
            backend=self.name,
            provider=DEVICE_PROVIDERS.get(resolved_device, resolved_device),
            device=resolved_device,
            warnings=tuple(warnings),
        )

    # -- run -------------------------------------------------------------

    def start_run(
        self,
        model_path: str,
        model_sha256: str,
        settings: Any,
        platform: str | None = None,
        machine: str | None = None,
    ) -> Mapping[str, Any]:
        """Load the model once and fix the provider for the whole run."""

        import sys
        import platform as platform_module

        selection = select_device(
            requested=getattr(settings, "requested_device", "auto"),
            model_format="onnx",
            backend_accelerators=self.capabilities(),
            platform=sys.platform if platform is None else platform,
            machine=(
                platform_module.machine() if machine is None else machine
            ),
        )
        providers = [DEVICE_PROVIDERS[selection.device]]
        if selection.device != "cpu":
            providers.append("CPUExecutionProvider")

        session = self._open(model_path, providers)
        description = self._describe(
            session,
            model_path,
            model_sha256,
            getattr(settings, "requested_device", "auto"),
            selection.device,
            selection.warnings,
        )
        tile_size = int(getattr(settings, "tile_size", 640))
        if not description.dynamic_shape and (
            description.input_width != tile_size
            or description.input_height != tile_size
        ):
            raise ModelRejected(
                "this export has a fixed input of "
                f"{description.input_width}x{description.input_height}; set "
                f"the tile size to {description.input_width}"
            )
        self._session = session
        self._description = description
        self._input_name = session.get_inputs()[0].name
        self._selected = validate_selected_class_ids(
            tuple(getattr(settings, "selected_class_ids", ()) or ()),
            description.class_names,
        )
        self._confidence = float(getattr(settings, "confidence", 0.25))
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

        if self._session is None or self._description is None:
            raise ModelRejected("the model has not been loaded")
        image = read_rgb_tile(
            tile_path, int(tile["valid_width"]), int(tile["valid_height"])
        )
        target_width = int(tile["model_width"])
        target_height = int(tile["model_height"])
        padded, transform = letterbox(image, target_width, target_height)
        batch = to_model_input(padded)
        outputs = self._session.run(None, {self._input_name: batch})
        detections = decode_predictions(
            outputs[0],
            self._layout,
            transform,
            self._confidence,
            self._selected,
        )
        names = self._description.class_names
        return [detection.as_payload(names) for detection in detections]

    def close(self) -> None:
        """Release the session and any provider resources."""

        self._session = None
        self._description = None
        self._layout = None

    # -- helpers ---------------------------------------------------------

    def _open(self, model_path: str, providers: Sequence[str]) -> Any:
        onnxruntime = _import_onnxruntime()
        path = Path(model_path)
        if not path.is_file():
            raise ModelRejected("the model file does not exist")
        available = set(onnxruntime.get_available_providers())
        usable = [name for name in providers if name in available]
        if not usable:
            raise ModelRejected(
                "no usable ONNX Runtime execution provider is installed"
            )
        try:
            return onnxruntime.InferenceSession(
                str(path), providers=usable
            )
        except ModelRejected:
            raise
        except Exception as exc:
            # Never surface the absolute model path in a user-facing error.
            raise ModelRejected(
                f"the model could not be opened: {type(exc).__name__}"
            ) from exc


def create_backend() -> OnnxBackend:
    """Return a new ONNX Runtime backend."""

    return OnnxBackend()


__all__ = [
    "BACKEND_NAME",
    "DEVICE_PROVIDERS",
    "OnnxBackend",
    "OnnxRuntimeMissing",
    "create_backend",
]

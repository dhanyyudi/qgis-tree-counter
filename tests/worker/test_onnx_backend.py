"""Tests for the ONNX Runtime backend against a fake session.

The real library is not installed in the QGIS host environment and is not
needed here: these tests exercise the plugin's own validation, provider
selection, and tile pipeline, which is where the risk actually is.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy
import pytest


class FakeMeta:
    def __init__(self, metadata: dict) -> None:
        self.custom_metadata_map = metadata
        self.description = metadata.get("description", "")


class FakeIO:
    def __init__(self, name: str, shape: list) -> None:
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(
        self,
        path,
        providers=(),
        metadata=None,
        input_shape=None,
        output_shape=None,
        output_names=("output0",),
        result=None,
    ) -> None:
        self.path = path
        self.providers = list(providers)
        self._metadata = metadata or {
            "task": "detect",
            "names": "{0: 'oil_palm'}",
            "description": "Ultralytics YOLO11n model",
        }
        self._inputs = [
            FakeIO("images", input_shape or [1, 3, 640, 640])
        ]
        self._outputs = [
            FakeIO(name, output_shape or [1, 5, 8400])
            for name in output_names
        ]
        self._result = result
        self.last_feed = None

    def get_modelmeta(self):
        return FakeMeta(self._metadata)

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        self.last_feed = feed
        if self._result is not None:
            return [self._result]
        return [numpy.zeros((1, 5, 8400), dtype=numpy.float32)]


@pytest.fixture
def fake_onnxruntime(monkeypatch):
    """Install a fake onnxruntime module for the duration of one test."""

    state = {"providers": ["CPUExecutionProvider"], "kwargs": {}}

    module = types.ModuleType("onnxruntime")

    def get_available_providers():
        return list(state["providers"])

    def inference_session(path, providers=()):
        session = FakeSession(path, providers, **state["kwargs"])
        state["session"] = session
        return session

    module.get_available_providers = get_available_providers
    module.InferenceSession = inference_session
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    return state


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "best.onnx"
    path.write_bytes(b"fake onnx graph")
    return path


def _settings(**overrides):
    from tree_counter.core.types import InferenceSettings

    return InferenceSettings(**overrides)


def _backend():
    from tree_counter.worker.backend_onnx import OnnxBackend

    return OnnxBackend()


def test_a_valid_model_is_described(fake_onnxruntime, tmp_path: Path) -> None:
    description = _backend().inspect(str(_model(tmp_path)), "a" * 64)

    assert description.class_names == ("oil_palm",)
    assert description.task == "detect"
    assert description.family == "yolo11"
    assert description.input_width == 640
    assert description.dynamic_shape is False
    assert description.backend == "onnxruntime"


def test_the_description_carries_no_model_path(
    fake_onnxruntime, tmp_path: Path
) -> None:
    description = _backend().inspect(str(_model(tmp_path)), "a" * 64)

    assert description.filename == "best.onnx"
    assert str(tmp_path) not in repr(description)


def test_a_missing_model_is_reported(
    fake_onnxruntime, tmp_path: Path
) -> None:
    from tree_counter.worker.backend_base import ModelRejected

    with pytest.raises(ModelRejected):
        _backend().inspect(str(tmp_path / "absent.onnx"), "a" * 64)


def test_a_segmentation_model_is_rejected(
    fake_onnxruntime, tmp_path: Path
) -> None:
    from tree_counter.worker.backend_base import ModelRejected

    fake_onnxruntime["kwargs"]["metadata"] = {
        "task": "segment",
        "names": "{0: 'tree'}",
        "description": "Ultralytics YOLO11n-seg",
    }

    with pytest.raises(ModelRejected):
        _backend().inspect(str(_model(tmp_path)), "a" * 64)


def test_a_non_yolo11_family_is_rejected(
    fake_onnxruntime, tmp_path: Path
) -> None:
    from tree_counter.worker.backend_base import ModelRejected

    fake_onnxruntime["kwargs"]["metadata"] = {
        "task": "detect",
        "names": "{0: 'tree'}",
        "description": "Ultralytics YOLOv8n model",
    }

    with pytest.raises(ModelRejected):
        _backend().inspect(str(_model(tmp_path)), "a" * 64)


def test_an_embedded_nms_export_is_rejected(
    fake_onnxruntime, tmp_path: Path
) -> None:
    from tree_counter.worker.backend_base import ModelRejected

    fake_onnxruntime["kwargs"]["output_names"] = ("num_dets", "boxes")

    with pytest.raises(ModelRejected):
        _backend().inspect(str(_model(tmp_path)), "a" * 64)


def test_an_ambiguous_output_layout_is_rejected(
    fake_onnxruntime, tmp_path: Path
) -> None:
    from tree_counter.worker.backend_base import ModelRejected

    fake_onnxruntime["kwargs"]["output_shape"] = [1, 5, 5]

    with pytest.raises(ModelRejected):
        _backend().inspect(str(_model(tmp_path)), "a" * 64)


def test_an_unreadable_class_map_is_rejected(
    fake_onnxruntime, tmp_path: Path
) -> None:
    from tree_counter.worker.backend_base import ModelRejected

    fake_onnxruntime["kwargs"]["metadata"] = {
        "task": "detect",
        "names": "{0: oil_palm",
        "description": "Ultralytics YOLO11n model",
    }

    with pytest.raises(ModelRejected):
        _backend().inspect(str(_model(tmp_path)), "a" * 64)


def test_a_dynamic_input_is_reported_as_dynamic(
    fake_onnxruntime, tmp_path: Path
) -> None:
    fake_onnxruntime["kwargs"]["input_shape"] = [1, 3, "height", "width"]

    description = _backend().inspect(str(_model(tmp_path)), "a" * 64)

    assert description.dynamic_shape is True


def test_a_multi_input_model_is_rejected(
    fake_onnxruntime, tmp_path: Path
) -> None:
    from tree_counter.worker.backend_base import ModelRejected

    fake_onnxruntime["kwargs"]["input_shape"] = [1, 3, 640]

    with pytest.raises(ModelRejected):
        _backend().inspect(str(_model(tmp_path)), "a" * 64)


class TestProviders:
    """Provider selection follows the device rules exactly."""

    def test_cpu_only_machines_use_the_cpu_provider(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        backend = _backend()

        result = backend.start_run(
            str(_model(tmp_path)),
            "a" * 64,
            _settings(),
            platform="linux",
            machine="x86_64",
        )

        assert result["device"] == "cpu"
        assert fake_onnxruntime["session"].providers == [
            "CPUExecutionProvider"
        ]

    def test_auto_warns_when_falling_back_to_cpu(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        result = _backend().start_run(
            str(_model(tmp_path)),
            "a" * 64,
            _settings(requested_device="auto"),
            platform="linux",
            machine="x86_64",
        )

        assert result["warnings"]

    def test_coreml_is_used_when_available(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        fake_onnxruntime["providers"] = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        result = _backend().start_run(
            str(_model(tmp_path)),
            "a" * 64,
            _settings(requested_device="auto"),
            platform="darwin",
            machine="arm64",
        )

        assert result["device"] == "coreml"
        assert fake_onnxruntime["session"].providers == [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert result["warnings"] == []

    def test_an_unavailable_explicit_provider_fails(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.capabilities import DeviceUnavailable

        with pytest.raises(DeviceUnavailable):
            _backend().start_run(
                str(_model(tmp_path)),
                "a" * 64,
                _settings(requested_device="coreml"),
                platform="darwin",
                machine="arm64",
            )

    def test_capabilities_exclude_the_cpu_provider(
        self, fake_onnxruntime
    ) -> None:
        fake_onnxruntime["providers"] = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        assert _backend().capabilities() == ("coreml",)

    def test_capabilities_are_empty_without_the_library(
        self, monkeypatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "onnxruntime", None)

        assert _backend().capabilities() == ()


class TestStaticShape:
    """A fixed-input export must be matched by the tile size."""

    def test_a_matching_tile_size_is_accepted(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        result = _backend().start_run(
            str(_model(tmp_path)),
            "a" * 64,
            _settings(tile_size=640),
            platform="linux",
            machine="x86_64",
        )

        assert result["device"] == "cpu"

    def test_a_mismatched_tile_size_is_rejected(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        with pytest.raises(ModelRejected) as error:
            _backend().start_run(
                str(_model(tmp_path)),
                "a" * 64,
                _settings(tile_size=1024),
                platform="linux",
                machine="x86_64",
            )

        assert "640" in str(error.value.diagnostic_detail)

    def test_a_dynamic_input_accepts_any_valid_tile_size(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        fake_onnxruntime["kwargs"]["input_shape"] = [1, 3, "height", "width"]

        result = _backend().start_run(
            str(_model(tmp_path)),
            "a" * 64,
            _settings(tile_size=1024),
            platform="linux",
            machine="x86_64",
        )

        assert result["backend"] == "onnxruntime"


class TestInference:
    """The tile pipeline feeds the model and returns canonical payloads."""

    def _tile(self, tmp_path: Path, width=32, height=32) -> dict:
        path = tmp_path / "tile.raw"
        path.write_bytes(bytes([120]) * (width * height * 3))
        return {
            "tile_id": "r00000_c00000",
            "tile_path": str(path),
            "x_offset": 0,
            "y_offset": 0,
            "valid_width": width,
            "valid_height": height,
            "model_width": 64,
            "model_height": 64,
        }

    def _prediction(self, rows, class_count=1):
        array = numpy.asarray(rows, dtype=numpy.float32)
        return array.T[None, ...]

    def test_inference_before_loading_is_rejected(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        tile = self._tile(tmp_path)

        with pytest.raises(ModelRejected):
            _backend().infer_tile(tile["tile_path"], tile)

    def test_a_detection_is_returned_in_tile_coordinates(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        fake_onnxruntime["kwargs"]["result"] = self._prediction(
            [[20, 20, 10, 10, 0.9]]
        )
        backend = _backend()
        backend.start_run(
            str(_model(tmp_path)),
            "a" * 64,
            _settings(tile_size=640),
            platform="linux",
            machine="x86_64",
        )
        tile = self._tile(tmp_path)

        [payload] = backend.infer_tile(tile["tile_path"], tile)

        assert payload["box"] == [15.0, 15.0, 25.0, 25.0]
        assert payload["class_name"] == "oil_palm"
        assert payload["confidence"] == pytest.approx(0.9)

    def test_the_model_receives_a_normalized_batch(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        backend = _backend()
        backend.start_run(
            str(_model(tmp_path)),
            "a" * 64,
            _settings(tile_size=640),
            platform="linux",
            machine="x86_64",
        )
        tile = self._tile(tmp_path)

        backend.infer_tile(tile["tile_path"], tile)

        batch = fake_onnxruntime["session"].last_feed["images"]
        assert batch.shape == (1, 3, 64, 64)
        assert batch.dtype == numpy.float32
        assert 0.0 <= float(batch.min()) and float(batch.max()) <= 1.0

    def test_low_confidence_predictions_are_dropped(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        fake_onnxruntime["kwargs"]["result"] = self._prediction(
            [[20, 20, 10, 10, 0.1]]
        )
        backend = _backend()
        backend.start_run(
            str(_model(tmp_path)),
            "a" * 64,
            _settings(confidence=0.5, tile_size=640),
            platform="linux",
            machine="x86_64",
        )
        tile = self._tile(tmp_path)

        assert backend.infer_tile(tile["tile_path"], tile) == []

    def test_closing_releases_the_session(
        self, fake_onnxruntime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        backend = _backend()
        backend.start_run(
            str(_model(tmp_path)),
            "a" * 64,
            _settings(tile_size=640),
            platform="linux",
            machine="x86_64",
        )
        backend.close()
        tile = self._tile(tmp_path)

        with pytest.raises(ModelRejected):
            backend.infer_tile(tile["tile_path"], tile)


def test_a_missing_library_reports_a_runtime_error(
    monkeypatch, tmp_path
) -> None:
    from tree_counter.worker.backend_onnx import OnnxRuntimeMissing

    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(OnnxRuntimeMissing):
        _backend().inspect(str(_model(tmp_path)), "a" * 64)

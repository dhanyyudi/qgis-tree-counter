"""Tests for the trusted PyTorch/Ultralytics backend.

torch and ultralytics are faked: what matters here is the plugin's own
trust check, its refusal to use the NMS-applying predict path, and its
normalization of raw output to the same shape the ONNX backend produces.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy
import pytest

CHECKPOINT_BYTES = b"fake ultralytics checkpoint"
CHECKPOINT_SHA = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()


class FakeTensor:
    """Just enough of a torch tensor for the backend's own code paths."""

    def __init__(self, array) -> None:
        self.array = numpy.asarray(array)
        self.device = "cpu"

    @property
    def shape(self):
        return self.array.shape

    def to(self, device):
        self.device = device
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeModule:
    """Stands in for a DetectionModel."""

    def __init__(self, result, yaml=None) -> None:
        self._result = result
        self.yaml = yaml or {"yaml_file": "yolo11n.yaml"}
        self.device = "cpu"
        self.eval_called = False
        self.seen_input = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, tensor):
        self.seen_input = tensor
        return self._result


class FakeYOLO:
    """Stands in for ultralytics.YOLO."""

    predict_calls = 0

    def __init__(self, path) -> None:
        state = _STATE
        if state.get("load_error"):
            raise RuntimeError(state["load_error"])
        self.path = path
        self.task = state.get("task", "detect")
        self.names = state.get("names", {0: "oil_palm"})
        self.model_name = state.get("model_name", "yolo11n.pt")
        self.model = FakeModule(state["result"])

    def predict(self, *args, **kwargs):
        FakeYOLO.predict_calls += 1
        raise AssertionError("predict applies Ultralytics NMS; never call it")


_STATE: dict = {}


@pytest.fixture
def fake_runtime(monkeypatch):
    """Install fake torch and ultralytics modules for one test."""

    _STATE.clear()
    _STATE.update(
        {
            "result": FakeTensor(
                numpy.asarray(
                    [[20.0, 20.0, 10.0, 10.0, 0.9]], dtype=numpy.float32
                ).T[None, ...]
            ),
            "cuda": False,
            "mps": False,
        }
    )
    FakeYOLO.predict_calls = 0

    torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    torch.no_grad = _NoGrad
    torch.from_numpy = FakeTensor
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: _STATE.get("cuda", False)
    )
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(
            is_available=lambda: _STATE.get("mps", False)
        )
    )

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    return _STATE


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "best.pt"
    path.write_bytes(CHECKPOINT_BYTES)
    return path


def _settings(**overrides):
    from tree_counter.core.types import InferenceSettings

    return InferenceSettings(**overrides)


def _backend():
    from tree_counter.worker.backend_ultralytics import UltralyticsBackend

    return UltralyticsBackend()


def _tile(tmp_path: Path, width=32, height=32) -> dict:
    path = tmp_path / "tile.raw"
    path.write_bytes(bytes([100]) * (width * height * 3))
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


class TestTrust:
    """A checkpoint is only loaded when its content matches."""

    def test_a_matching_hash_is_accepted(self, tmp_path: Path) -> None:
        from tree_counter.worker.backend_ultralytics import (
            verify_trusted_checkpoint,
        )

        assert (
            verify_trusted_checkpoint(_checkpoint(tmp_path), CHECKPOINT_SHA)
            == CHECKPOINT_SHA
        )

    def test_an_uppercase_hash_is_accepted(self, tmp_path: Path) -> None:
        from tree_counter.worker.backend_ultralytics import (
            verify_trusted_checkpoint,
        )

        assert verify_trusted_checkpoint(
            _checkpoint(tmp_path), CHECKPOINT_SHA.upper()
        ) == CHECKPOINT_SHA

    def test_swapped_content_is_refused(self, tmp_path: Path) -> None:
        from tree_counter.worker.backend_ultralytics import (
            ModelNotTrusted,
            verify_trusted_checkpoint,
        )

        path = _checkpoint(tmp_path)
        # The file changed after the user confirmed its hash.
        path.write_bytes(b"something else entirely")

        with pytest.raises(ModelNotTrusted):
            verify_trusted_checkpoint(path, CHECKPOINT_SHA)

    @pytest.mark.parametrize("declared", [None, "", "abc", 5, "z" * 64])
    def test_a_missing_or_malformed_declaration_is_refused(
        self, tmp_path: Path, declared: object
    ) -> None:
        from tree_counter.worker.backend_ultralytics import (
            ModelNotTrusted,
            verify_trusted_checkpoint,
        )

        with pytest.raises(ModelNotTrusted):
            verify_trusted_checkpoint(_checkpoint(tmp_path), declared)

    def test_inspection_requires_the_declaration(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_ultralytics import ModelNotTrusted

        with pytest.raises(ModelNotTrusted):
            _backend().inspect(str(_checkpoint(tmp_path)), "b" * 64)

    def test_a_run_requires_the_declaration(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_ultralytics import ModelNotTrusted

        with pytest.raises(ModelNotTrusted):
            _backend().start_run(
                str(_checkpoint(tmp_path)),
                "b" * 64,
                _settings(),
                platform="linux",
                machine="x86_64",
            )


class TestInspection:
    """Only validated YOLO11 detection checkpoints are accepted."""

    def test_a_valid_checkpoint_is_described(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        description = _backend().inspect(
            str(_checkpoint(tmp_path)), CHECKPOINT_SHA
        )

        assert description.model_format == "pt"
        assert description.class_names == ("oil_palm",)
        assert description.family == "yolo11"
        assert description.backend == "ultralytics"

    def test_the_description_carries_no_checkpoint_path(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        description = _backend().inspect(
            str(_checkpoint(tmp_path)), CHECKPOINT_SHA
        )

        assert description.filename == "best.pt"
        assert str(tmp_path) not in repr(description)

    def test_a_segmentation_checkpoint_is_rejected(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        fake_runtime["task"] = "segment"

        with pytest.raises(ModelRejected):
            _backend().inspect(str(_checkpoint(tmp_path)), CHECKPOINT_SHA)

    def test_a_non_yolo11_checkpoint_is_rejected(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        fake_runtime["model_name"] = "yolov8n.pt"

        with pytest.raises(ModelRejected):
            _backend().inspect(str(_checkpoint(tmp_path)), CHECKPOINT_SHA)

    def test_a_checkpoint_without_classes_is_rejected(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        fake_runtime["names"] = {}

        with pytest.raises(ModelRejected):
            _backend().inspect(str(_checkpoint(tmp_path)), CHECKPOINT_SHA)

    def test_a_corrupt_checkpoint_is_reported_without_its_path(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        fake_runtime["load_error"] = "unpickling failed"

        with pytest.raises(ModelRejected) as error:
            _backend().inspect(str(_checkpoint(tmp_path)), CHECKPOINT_SHA)

        assert str(tmp_path) not in str(error.value.diagnostic_detail)

    def test_a_missing_checkpoint_is_reported(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.settings.trust import ModelError

        with pytest.raises(ModelError):
            _backend().inspect(str(tmp_path / "absent.pt"), CHECKPOINT_SHA)

    def test_a_missing_runtime_is_reported(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_ultralytics import UltralyticsMissing

        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(UltralyticsMissing):
            _backend().inspect(str(_checkpoint(tmp_path)), CHECKPOINT_SHA)

    def test_capabilities_are_empty_without_the_runtime(
        self, monkeypatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)

        assert _backend().capabilities() == ()

    def test_capabilities_report_mps_when_available(
        self, fake_runtime
    ) -> None:
        fake_runtime["mps"] = True

        assert _backend().capabilities() == ("mps",)


class TestRun:
    """A run loads once, on the chosen device, and never calls predict."""

    def _started(self, tmp_path: Path, **overrides):
        backend = _backend()
        result = backend.start_run(
            str(_checkpoint(tmp_path)),
            CHECKPOINT_SHA,
            _settings(**overrides),
            platform=overrides.pop("platform", None) or "linux",
            machine="x86_64",
        )
        return backend, result

    def test_a_cpu_run_starts(self, fake_runtime, tmp_path: Path) -> None:
        _, result = self._started(tmp_path)

        assert result["device"] == "cpu"
        assert result["backend"] == "ultralytics"

    def test_the_module_is_moved_and_set_to_eval(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        backend, _ = self._started(tmp_path)

        module = backend._module
        assert module.eval_called is True
        assert module.device == "cpu"

    def test_mps_is_used_on_apple_silicon_when_available(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        fake_runtime["mps"] = True
        backend = _backend()

        result = backend.start_run(
            str(_checkpoint(tmp_path)),
            CHECKPOINT_SHA,
            _settings(requested_device="auto"),
            platform="darwin",
            machine="arm64",
        )

        assert result["device"] == "mps"
        assert backend._module.device == "mps"

    def test_an_unavailable_device_fails(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.capabilities import DeviceUnavailable

        with pytest.raises(DeviceUnavailable):
            _backend().start_run(
                str(_checkpoint(tmp_path)),
                CHECKPOINT_SHA,
                _settings(requested_device="cuda"),
                platform="linux",
                machine="x86_64",
            )

    def test_inference_never_uses_the_predict_path(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        backend, _ = self._started(tmp_path)
        tile = _tile(tmp_path)

        backend.infer_tile(tile["tile_path"], tile)

        # predict() would apply Ultralytics NMS before ours.
        assert FakeYOLO.predict_calls == 0

    def test_a_detection_matches_the_onnx_payload_shape(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        backend, _ = self._started(tmp_path)
        tile = _tile(tmp_path)

        [payload] = backend.infer_tile(tile["tile_path"], tile)

        assert set(payload) == {
            "box",
            "confidence",
            "class_id",
            "class_name",
        }
        assert payload["box"] == [15.0, 15.0, 25.0, 25.0]
        assert payload["class_name"] == "oil_palm"

    def test_selected_classes_are_respected(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        fake_runtime["names"] = {0: "oil_palm", 1: "shade"}
        fake_runtime["result"] = FakeTensor(
            numpy.asarray(
                [
                    [20.0, 20.0, 10.0, 10.0, 0.9, 0.1],
                    [40.0, 40.0, 10.0, 10.0, 0.1, 0.8],
                ],
                dtype=numpy.float32,
            ).T[None, ...]
        )
        backend, _ = self._started(tmp_path, selected_class_ids=(1,))
        tile = _tile(tmp_path)

        payloads = backend.infer_tile(tile["tile_path"], tile)

        assert [item["class_id"] for item in payloads] == [1]

    def test_a_tuple_output_is_unwrapped(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        prediction = fake_runtime["result"]
        fake_runtime["result"] = (prediction, ["feature maps"])
        backend, _ = self._started(tmp_path)
        tile = _tile(tmp_path)

        assert len(backend.infer_tile(tile["tile_path"], tile)) == 1

    def test_an_unexpected_output_structure_fails_closed(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        fake_runtime["result"] = (["nested", "lists"],)
        backend, _ = self._started(tmp_path)
        tile = _tile(tmp_path)

        with pytest.raises(ModelRejected):
            backend.infer_tile(tile["tile_path"], tile)

    def test_an_empty_output_fails_closed(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        fake_runtime["result"] = ()
        backend, _ = self._started(tmp_path)
        tile = _tile(tmp_path)

        with pytest.raises(ModelRejected):
            backend.infer_tile(tile["tile_path"], tile)

    def test_a_class_count_mismatch_fails_closed(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        # The head reports two classes but the class map declares one.
        fake_runtime["result"] = FakeTensor(
            numpy.asarray(
                [[20.0, 20.0, 10.0, 10.0, 0.9, 0.2]], dtype=numpy.float32
            ).T[None, ...]
        )
        backend, _ = self._started(tmp_path)
        tile = _tile(tmp_path)

        with pytest.raises(ModelRejected):
            backend.infer_tile(tile["tile_path"], tile)

    def test_inference_before_loading_is_rejected(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        tile = _tile(tmp_path)

        with pytest.raises(ModelRejected):
            _backend().infer_tile(tile["tile_path"], tile)

    def test_closing_releases_the_model(
        self, fake_runtime, tmp_path: Path
    ) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        backend, _ = self._started(tmp_path)
        backend.close()
        tile = _tile(tmp_path)

        with pytest.raises(ModelRejected):
            backend.infer_tile(tile["tile_path"], tile)

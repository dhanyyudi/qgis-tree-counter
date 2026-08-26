"""Tests for model metadata validation and output-layout detection."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest


def test_a_mapping_class_map_is_ordered_by_index() -> None:
    from tree_counter.worker.model_info import parse_class_map

    assert parse_class_map({1: "shade", 0: "oil_palm"}) == (
        "oil_palm",
        "shade",
    )


def test_string_indices_are_accepted() -> None:
    from tree_counter.worker.model_info import parse_class_map

    assert parse_class_map({"0": "oil_palm"}) == ("oil_palm",)


def test_a_list_class_map_is_accepted() -> None:
    from tree_counter.worker.model_info import parse_class_map

    assert parse_class_map(["oil_palm", "shade"]) == ("oil_palm", "shade")


@pytest.mark.parametrize("raw", [{}, [], "oil_palm", None, 5])
def test_an_unusable_class_map_is_rejected(raw: object) -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import parse_class_map

    with pytest.raises(ModelRejected):
        parse_class_map(raw)


def test_a_gap_in_class_indices_is_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import parse_class_map

    # A gap would shift every later name onto the wrong detections.
    with pytest.raises(ModelRejected):
        parse_class_map({0: "oil_palm", 2: "shade"})


def test_a_negative_class_index_is_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import parse_class_map

    with pytest.raises(ModelRejected):
        parse_class_map({-1: "oil_palm"})


def test_a_non_integer_class_index_is_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import parse_class_map

    with pytest.raises(ModelRejected):
        parse_class_map({"first": "oil_palm"})


def test_an_empty_class_name_is_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import parse_class_map

    with pytest.raises(ModelRejected):
        parse_class_map({0: "  "})


def test_a_detection_task_is_accepted() -> None:
    from tree_counter.worker.model_info import require_detection_task

    assert require_detection_task({"task": "detect"}) == "detect"


@pytest.mark.parametrize("task", ["segment", "classify", "pose", "obb"])
def test_a_non_detection_task_is_rejected(task: str) -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import require_detection_task

    with pytest.raises(ModelRejected):
        require_detection_task({"task": task})


def test_a_missing_task_is_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import require_detection_task

    with pytest.raises(ModelRejected):
        require_detection_task({})


@pytest.mark.parametrize(
    "metadata",
    [
        {"family": "YOLO11"},
        {"model": "yolo11n"},
        {"description": "Ultralytics YOLO11n model"},
    ],
)
def test_a_yolo11_family_is_recognized(metadata: dict) -> None:
    from tree_counter.worker.model_info import normalize_family

    assert normalize_family(metadata) == "yolo11"


@pytest.mark.parametrize(
    "metadata",
    [{"family": "yolov8n"}, {"model": "rtdetr"}, {}, {"description": ""}],
)
def test_an_unsupported_family_is_rejected(metadata: dict) -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import normalize_family

    with pytest.raises(ModelRejected):
        normalize_family(metadata)


def test_a_single_raw_output_is_accepted() -> None:
    from tree_counter.worker.model_info import reject_embedded_nms

    reject_embedded_nms(["output0"])


@pytest.mark.parametrize(
    "names",
    [
        ["num_dets", "boxes"],
        ["nmsed_boxes"],
        ["output0", "selected_indices"],
        ["detection_boxes"],
    ],
)
def test_an_embedded_nms_export_is_rejected(names: list) -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import reject_embedded_nms

    with pytest.raises(ModelRejected):
        reject_embedded_nms(names)


def test_multiple_raw_outputs_are_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import reject_embedded_nms

    with pytest.raises(ModelRejected):
        reject_embedded_nms(["output0", "output1"])


def test_a_transposed_layout_is_detected() -> None:
    from tree_counter.worker.model_info import describe_output_layout

    layout = describe_output_layout((1, 5, 8400), class_count=1)

    assert layout.transposed is True
    assert layout.predictions == 8400


def test_a_row_major_layout_is_detected() -> None:
    from tree_counter.worker.model_info import describe_output_layout

    layout = describe_output_layout((1, 8400, 5), class_count=1)

    assert layout.transposed is False
    assert layout.predictions == 8400


def test_a_multi_class_layout_is_detected() -> None:
    from tree_counter.worker.model_info import describe_output_layout

    layout = describe_output_layout((1, 84, 8400), class_count=80)

    assert layout.transposed is True
    assert layout.class_count == 80


def test_an_ambiguous_layout_is_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import describe_output_layout

    # Both axes equal 4 + nc, so either reading is defensible and a guess
    # would silently transpose every box.
    with pytest.raises(ModelRejected):
        describe_output_layout((1, 5, 5), class_count=1)


def test_a_layout_matching_no_axis_is_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import describe_output_layout

    with pytest.raises(ModelRejected):
        describe_output_layout((1, 7, 8400), class_count=1)


def test_a_dynamic_output_axis_is_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import describe_output_layout

    with pytest.raises(ModelRejected):
        describe_output_layout((1, "classes", 8400), class_count=1)


def test_a_non_three_dimensional_output_is_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import describe_output_layout

    with pytest.raises(ModelRejected):
        describe_output_layout((1, 5), class_count=1)


def test_a_single_class_model_auto_selects_its_class() -> None:
    from tree_counter.worker.model_info import auto_selected_class_ids

    assert auto_selected_class_ids(("oil_palm",)) == (0,)


def test_a_multi_class_model_selects_nothing() -> None:
    from tree_counter.worker.model_info import auto_selected_class_ids

    # Choosing for the user would change the count without being asked.
    assert auto_selected_class_ids(("oil_palm", "shade")) == ()


def test_selected_classes_are_validated() -> None:
    from tree_counter.worker.model_info import validate_selected_class_ids

    assert validate_selected_class_ids([1, 0], ("a", "b")) == (1, 0)


@pytest.mark.parametrize("selected", [[2], [-1], [0, 0], [True], ["0"]])
def test_invalid_class_selections_are_rejected(selected: list) -> None:
    from tree_counter.worker.backend_base import ModelRejected
    from tree_counter.worker.model_info import validate_selected_class_ids

    with pytest.raises(ModelRejected):
        validate_selected_class_ids(selected, ("a", "b"))


class TestModelDescription:
    """The description is the only model fact the host ever receives."""

    def _description(self, **overrides):
        from tree_counter.worker.backend_base import ModelDescription

        values = {
            "filename": "best.onnx",
            "sha256": "a" * 64,
            "model_format": "onnx",
            "task": "detect",
            "family": "yolo11",
            "class_names": ("oil_palm",),
            "input_width": 640,
            "input_height": 640,
            "dynamic_shape": False,
            "backend": "onnxruntime",
            "device": "cpu",
        }
        values.update(overrides)
        return ModelDescription(**values)

    def test_a_valid_description_is_accepted(self) -> None:
        assert self._description().is_single_class is True

    def test_a_multi_class_model_is_not_single_class(self) -> None:
        assert (
            self._description(class_names=("a", "b")).is_single_class is False
        )

    def test_a_non_detection_task_is_rejected(self) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        with pytest.raises(ModelRejected):
            self._description(task="segment")

    def test_a_model_without_classes_is_rejected(self) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        with pytest.raises(ModelRejected):
            self._description(class_names=())

    def test_an_unsupported_format_is_rejected(self) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        with pytest.raises(ModelRejected):
            self._description(model_format="tflite")

    def test_a_filename_with_a_path_is_rejected(self) -> None:
        from tree_counter.worker.backend_base import ModelRejected

        with pytest.raises(ModelRejected):
            self._description(filename="models/best.onnx")

    def test_the_message_payload_carries_no_path(self) -> None:
        payload = self._description().as_message()

        assert set(payload) == {
            "class_names",
            "backend",
            "device",
            "input_size",
        }
        assert payload["input_size"] == 640

    def test_a_dynamic_model_reports_no_input_size(self) -> None:
        payload = self._description(dynamic_shape=True).as_message()

        assert "input_size" not in payload


def test_a_raw_detection_maps_to_its_class_name() -> None:
    from tree_counter.worker.backend_base import RawDetection

    payload = RawDetection(1.0, 2.0, 3.0, 4.0, 0.9, 1).as_payload(
        ("oil_palm", "shade")
    )

    assert payload["class_name"] == "shade"
    assert payload["box"] == [1.0, 2.0, 3.0, 4.0]


def test_a_raw_detection_outside_the_class_map_is_rejected() -> None:
    from tree_counter.worker.backend_base import ModelRejected, RawDetection

    with pytest.raises(ModelRejected):
        RawDetection(1.0, 2.0, 3.0, 4.0, 0.9, 5).as_payload(("oil_palm",))

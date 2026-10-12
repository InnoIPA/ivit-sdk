"""Per-call inference options exposed to Python (needs the built bindings)."""

import inspect

import pytest

import ivit

pytestmark = pytest.mark.skipif(
    not ivit.is_cpp_available(), reason="C++ bindings are not built"
)


def test_infer_config_exposes_agnostic_nms_defaulting_to_per_class():
    config = ivit._ivit_core.InferConfig()
    assert config.agnostic_nms is False
    config.agnostic_nms = True
    assert config.agnostic_nms is True


def test_detector_predict_accepts_agnostic_nms():
    from ivit.vision import Detector

    assert "agnostic_nms" in inspect.signature(Detector.predict).parameters
    assert inspect.signature(Detector.predict).parameters["agnostic_nms"].default is False


# ------------------------------------------------------------ confidence: task default


def classifier(tmp_path):
    from test_model_contract import model_file

    path = model_file(tmp_path / "three.onnx", [1, 3, 2, 4], [[0.1, 0.1, 0.1]])  # 3 equal scores
    return ivit.load(
        str(path), device="cpu", backend="openvino", task="classification",
        decoder="logits", use_cache=False,
    )


IMAGE = __import__("numpy").zeros((2, 4, 3), "uint8")


def test_unset_confidence_is_the_task_default_marker():
    assert ivit._ivit_core.InferConfig().conf_threshold == -1.0


def test_classification_returns_every_class_by_default(tmp_path):
    model = classifier(tmp_path)
    # Three equal classes score ~0.33 each; the old 0.5 default returned an empty list here.
    assert len(model.predict(IMAGE).classifications) == 3
    assert len(model.predict(IMAGE, ivit._ivit_core.InferConfig()).classifications) == 3


def test_an_explicit_threshold_still_filters_classification(tmp_path):
    model = classifier(tmp_path)
    config = ivit._ivit_core.InferConfig()
    config.conf_threshold = 0.5
    assert model.predict(IMAGE, config).classifications == []
    config.conf_threshold = 0.3
    assert len(model.predict(IMAGE, config).classifications) == 3


@pytest.mark.parametrize("bad", [-0.3, 1.5, float("nan")])
def test_invalid_confidence_is_rejected_not_treated_as_unset(tmp_path, bad):
    model = classifier(tmp_path)
    config = ivit._ivit_core.InferConfig()
    config.conf_threshold = bad
    with pytest.raises(Exception, match="thresholds"):
        model.predict(IMAGE, config)


def test_async_snapshot_keeps_every_inference_option():
    seen = {}

    class Recorder:
        def predict(self, image, config):
            seen.update(
                conf=config.conf_threshold, iou=config.iou_threshold,
                max=config.max_detections, classes=list(config.classes),
                agnostic=config.agnostic_nms,
            )
            return "done"

    config = ivit._ivit_core.InferConfig()
    config.conf_threshold, config.iou_threshold, config.max_detections = 0.3, 0.6, 7
    config.classes, config.agnostic_nms = [2, 5], True
    from ivit import _async

    assert _async.submit(Recorder(), IMAGE, config).result(timeout=10) == "done"
    assert seen == {"conf": pytest.approx(0.3), "iou": pytest.approx(0.6), "max": 7,
                    "classes": [2, 5], "agnostic": True}


# ------------------------------------------------------------- NMS through a real model


def nms_model(tmp_path):
    import numpy as np
    from test_model_contract import model_file

    # YOLOv8 layout [1, 4 + 2 classes, 3 anchors]: anchors 0 and 1 are almost the same box with
    # different classes, anchor 2 is far away.
    output = np.array(
        [[[20, 21, 50], [20, 20, 50], [10, 10, 8], [10, 10, 8],
          [0.9, 0.1, 0.8], [0.1, 0.8, 0.1]]], np.float32)
    path = model_file(tmp_path / "nms.onnx", [1, 3, 64, 64], output)
    return ivit.load(
        path, device="cpu", backend="openvino", task="detection", decoder="yolov8",
        resize_mode="stretch", use_cache=False,
    )


def nms_config(agnostic):
    config = ivit._ivit_core.InferConfig()
    config.conf_threshold, config.agnostic_nms = 0.5, agnostic
    return config


def test_nms_is_per_class_by_default_and_suppresses_across_classes_when_agnostic(tmp_path):
    model = nms_model(tmp_path)
    image = __import__("numpy").zeros((64, 64, 3), "uint8")
    per_class = model.predict(image, nms_config(False))
    assert sorted(d.class_id for d in per_class.detections) == [0, 0, 1]  # near-duplicate kept
    agnostic = model.predict(image, nms_config(True))
    assert sorted(d.class_id for d in agnostic.detections) == [0, 0]  # the weaker one is gone


def test_async_predictions_honour_agnostic_nms(tmp_path):
    model = nms_model(tmp_path)
    image = __import__("numpy").zeros((64, 64, 3), "uint8")
    assert len(model.predict_async(image, nms_config(True)).result(timeout=30).detections) == 2
    assert len(model.predict_async(image, nms_config(False)).result(timeout=30).detections) == 3

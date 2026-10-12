import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def sdk():
    import ivit

    if not ivit.is_cpp_available():
        pytest.skip("Compiled bindings required")
    return ivit


def model_file(path, shape, constant=None, nhwc=False):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    nodes = []
    if constant is None:
        nodes.append(
            helper.make_node(
                "ReduceMean",
                ["input"],
                ["output"],
                axes=[1, 2] if nhwc else [2, 3],
                keepdims=0,
            )
        )
        output_shape = [1, 3]
    else:
        nodes = [
            helper.make_node(
                "ReduceMean", ["input"], ["mean"], axes=[1, 2, 3], keepdims=0
            ),
            helper.make_node(
                "Constant",
                [],
                ["zero"],
                value=numpy_helper.from_array(np.array(0, np.float32)),
            ),
            helper.make_node("Mul", ["mean", "zero"], ["offset"]),
            helper.make_node(
                "Constant",
                [],
                ["constant"],
                value=numpy_helper.from_array(np.asarray(constant, np.float32)),
            ),
            helper.make_node("Add", ["constant", "offset"], ["output"]),
        ]
        output_shape = np.shape(constant)
    graph = helper.make_graph(
        nodes,
        "contract",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)
    return str(path)


def infer_config(sdk):
    config = sdk.InferConfig()
    config.conf_threshold = 0
    return config


def test_classifier_and_generic_model_share_contract(sdk, tmp_path):
    from ivit.vision import Classifier

    path = model_file(tmp_path / "arbitrary.onnx", [1, 3, 2, 4])
    options = dict(
        backend="openvino",
        use_cache=False,
        input_scale=0.1,
        mean=[1, 2, 3],
        std=[2, 2, 2],
        labels=['a"b', "line\nbreak", "中文"],
    )
    model = sdk.load(path, device="cpu", task="classification", **options)
    classifier = Classifier(path, device="cpu", **options)
    image = np.full((2, 4, 3), [10, 20, 30], np.uint8)
    generic = model.predict(image, infer_config(sdk))
    specialized = classifier.predict(image, top_k=3)
    expected = np.exp([1, 0, -1])
    expected /= expected.sum()
    for result in (generic, specialized):
        received = sorted(result.classifications, key=lambda item: item.class_id)
        np.testing.assert_allclose([c.score for c in received], expected, atol=1e-5)
        data = json.loads(result.to_json())
        assert {c["label"] for c in data["classifications"]} == set(options["labels"])


def test_nhwc_preprocessing(sdk, tmp_path):
    path = model_file(tmp_path / "nhwc.onnx", [1, 2, 4, 3], nhwc=True)
    model = sdk.load(
        path,
        device="cpu",
        backend="openvino",
        task="classification",
        use_cache=False,
        input_layout="nhwc",
        input_scale=0.1,
        mean=[1, 2, 3],
        std=[2, 2, 2],
    )
    result = model.predict(
        np.full((2, 4, 3), [10, 20, 30], np.uint8), infer_config(sdk)
    )
    assert result.top1().class_id == 0
    assert result.top1().score == pytest.approx(0.66524, abs=1e-4)


def test_explicit_probabilities(sdk, tmp_path):
    path = model_file(tmp_path / "probabilities.onnx", [1, 3, 2, 2], [[0.2, 0.8]])
    model = sdk.load(
        path,
        device="cpu",
        task="classification",
        backend="openvino",
        use_cache=False,
        decoder="probabilities",
    )
    assert model.predict(
        np.zeros((2, 2, 3), np.uint8), infer_config(sdk)
    ).top1().score == pytest.approx(0.8)


def test_detection_decoder_filters_and_letterbox(sdk, tmp_path):
    from ivit.vision import Detector

    output = np.array(
        [[[32, 32], [32, 32], [32, 32], [16, 16], [0.9, 0.1], [0.1, 0.8]]], np.float32
    )
    path = model_file(tmp_path / "arbitrary.onnx", [1, 3, 64, 64], output)
    options = dict(
        backend="openvino",
        use_cache=False,
        decoder="yolov8",
        labels=["first", "second"],
    )
    model = sdk.load(path, device="cpu", task="detection", **options)
    detector = Detector(path, device="cpu", **options)
    image = np.zeros((20, 40, 3), np.uint8)
    config = infer_config(sdk)
    config.classes = [1]
    config.max_detections = 1
    generic = model.predict(image, config)
    specialized = detector.predict(
        image, conf_threshold=0, classes=[1], max_detections=1
    )
    for result in (generic, specialized):
        assert len(result.detections) == 1
        detection = result.detections[0]
        assert detection.label == "second"
        np.testing.assert_allclose(
            [
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2,
            ],
            [10, 5, 30, 15],
        )


def test_per_model_labels_sidecar_wins(sdk, tmp_path):
    path = model_file(tmp_path / "classifier.onnx", [1, 3, 2, 2])
    (tmp_path / "classifier.labels.txt").write_text("red\ngreen\nblue\n")
    (tmp_path / "labels.txt").write_text("wrong\n")
    model = sdk.load(
        path, device="cpu", task="classification", backend="openvino", use_cache=False
    )
    assert list(model.labels) == ["red", "green", "blue"]


@pytest.mark.parametrize(
    "options", [{"std": [0]}, {"input_color": "unknown"}, {"decoder": "unknown"}]
)
def test_invalid_contract_fails_before_loading(sdk, options):
    with pytest.raises(sdk.IVITError):
        sdk.load("missing.onnx", **options)


# ------------------------------------------------ labels are checked when the model loads


def load_with(sdk, path, task="classification", **kwargs):
    return sdk.load(
        path, device="cpu", task=task, backend="openvino", use_cache=False, **kwargs
    )


def test_wrong_label_count_fails_at_load_with_both_numbers(sdk, tmp_path):
    path = model_file(tmp_path / "two.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    with pytest.raises(Exception, match=r"labels has 3 entries.*outputs 2 classes"):
        load_with(sdk, path, decoder="logits", labels=["a", "b", "c"])
    # The failed load leaves nothing behind: the same model loads fine afterwards.
    model = load_with(sdk, path, decoder="logits", labels=["a", "b"])
    assert model.predict(np.zeros((2, 4, 3), np.uint8), infer_config(sdk)).top1().label == "b"


def test_wrong_sidecar_label_count_fails_at_load(sdk, tmp_path):
    path = model_file(tmp_path / "classifier.onnx", [1, 3, 2, 2])  # 3 classes
    (tmp_path / "classifier.labels.txt").write_text("red\ngreen\n")
    with pytest.raises(Exception, match="labels has 2 entries"):
        load_with(sdk, path)


def test_label_count_is_not_checked_when_the_class_count_is_unknown(sdk, tmp_path):
    path = model_file(tmp_path / "two.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    # decoder="raw" hands tensors back untouched, so labels are not interpreted at all.
    assert load_with(sdk, path, decoder="raw", labels=["a", "b", "c"]) is not None
    # No labels at all is always fine: results use class_<id>.
    assert load_with(sdk, path, decoder="logits") is not None


def test_detection_label_count_is_checked_for_each_decoder_layout(sdk, tmp_path):
    # YOLOv8 layout [1, 4 + classes, anchors]: 2 classes -> 6 features.
    v8 = model_file(tmp_path / "v8.onnx", [1, 3, 64, 64], np.zeros((1, 6, 8), np.float32))
    with pytest.raises(Exception, match="outputs 2 classes"):
        load_with(sdk, v8, task="detection", decoder="yolov8", labels=["a", "b", "c"])
    load_with(sdk, v8, task="detection", decoder="yolov8", labels=["a", "b"])
    # YOLOv5 layout [1, anchors, 5 + classes]: 3 classes -> 8 features.
    v5 = model_file(tmp_path / "v5.onnx", [1, 3, 64, 64], np.zeros((1, 6, 8), np.float32))
    with pytest.raises(Exception, match="outputs 3 classes"):
        load_with(sdk, v5, task="detection", decoder="yolov5", labels=["a", "b"])
    load_with(sdk, v5, task="detection", decoder="yolov5", labels=["a", "b", "c"])
    # auto with an ambiguous (square) head cannot tell the layouts apart: no load-time check.
    square = model_file(tmp_path / "sq.onnx", [1, 3, 64, 64], np.zeros((1, 8, 8), np.float32))
    load_with(sdk, square, task="detection", decoder="auto", labels=["x"])

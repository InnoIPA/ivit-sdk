import json

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


def test_paddle_models_are_rejected(sdk, tmp_path):
    # PaddlePaddle (.pdmodel) was never tested or packaged (the deb has no Paddle frontend).
    path = tmp_path / "model.pdmodel"
    path.write_bytes(b"\0")
    with pytest.raises(sdk.IVITError, match=r"\.pdmodel"):
        sdk.load(str(path), device="cpu", backend="openvino")


STAGES = ("pre_process", "infer_start", "infer_end", "post_process")


def test_stage_callbacks_fire_in_order_for_single_and_batch_predict(sdk, tmp_path):
    # Output [N, 3] follows the batch, so predict_batch runs as one batched inference.
    path = model_file(tmp_path / "dynamic.onnx", ["N", 3, 2, 4])
    model = load_with(sdk, path, decoder="logits")
    seen = []
    for stage in STAGES:
        model.on(stage, lambda ctx, stage=stage: seen.append((stage, ctx.batch_size, ctx.latency_ms)))
    image = np.zeros((2, 4, 3), np.uint8)

    model.predict(image, infer_config(sdk))
    assert [s for s, _, _ in seen] == list(STAGES)
    assert all(batch == 1 for _, batch, _ in seen)
    assert all(latency >= 0 for _, _, latency in seen)

    seen.clear()
    model.predict_batch([image, image], infer_config(sdk))  # dynamic batch: one batched call
    assert [s for s, _, _ in seen] == list(STAGES)
    assert all(batch == 2 for _, batch, _ in seen)

    for stage in STAGES:
        model.remove_all_callbacks(stage)
    seen.clear()
    model.predict(image, infer_config(sdk))
    assert not seen


def segmentation_model_file(path, size):
    """[1,3,S,S] -> [1,2,S,S]: class 1 where the channel mean exceeds 0.7, else class 0.

    A white image gives class 1 everywhere; letterbox padding (gray 114/255) gives class 0.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    zero = numpy_helper.from_array(np.zeros((1, 1, size, size), np.float32), "zero")
    thresh = numpy_helper.from_array(np.full((1, 1, size, size), 0.7, np.float32), "thresh")
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["mean"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["mean", "thresh"], ["class1"]),  # > 0 where bright
        helper.make_node("Concat", ["zero", "class1"], ["output"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "seg",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, size, size])],
        initializer=[zero, thresh],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)
    return str(path)


@pytest.mark.parametrize("mode", ["stretch", "letterbox", "letterbox_top_left"])
@pytest.mark.parametrize("shape", [(40, 100), (101, 33), (64, 64)])  # wide, tall (odd pad), square
def test_segmentation_mask_covers_only_the_image_in_every_resize_mode(sdk, tmp_path, mode, shape):
    path = segmentation_model_file(tmp_path / "seg.onnx", 64)
    model = load_with(sdk, path, task="segmentation", resize_mode=mode, input_scale=1.0 / 255.0)
    white = np.full((shape[0], shape[1], 3), 255, np.uint8)
    mask = np.asarray(model.predict(white).segmentation_mask)
    assert mask.shape == shape
    # Every output pixel maps to a pixel of the white image, never to letterbox padding.
    assert (mask == 1).mean() > 0.98, f"{mode} {shape}: only {(mask == 1).mean():.0%} class 1"


def tta_maps():
    """(augmentation, undo) per name, both built from OpenCV so they do not share code with the SDK."""
    cv2 = pytest.importorskip("cv2")
    forward = {
        "hflip": lambda image: cv2.flip(image, 1),
        "vflip": lambda image: cv2.flip(image, 0),
        "rotate180": lambda image: cv2.rotate(image, cv2.ROTATE_180),
        "rotate90": lambda image: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        "rotate270": lambda image: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        "scale_up": lambda image: cv2.resize(image, None, fx=1.2, fy=1.2),
        "scale_down": lambda image: cv2.resize(image, None, fx=0.8, fy=0.8),
    }
    undo = {
        "hflip": lambda canvas, size: cv2.flip(canvas, 1),
        "vflip": lambda canvas, size: cv2.flip(canvas, 0),
        "rotate180": lambda canvas, size: cv2.rotate(canvas, cv2.ROTATE_180),
        "rotate90": lambda canvas, size: cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE),
        "rotate270": lambda canvas, size: cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE),
        "scale_up": lambda canvas, size: cv2.resize(canvas, size, interpolation=cv2.INTER_NEAREST),
        "scale_down": lambda canvas, size: cv2.resize(canvas, size, interpolation=cv2.INTER_NEAREST),
    }
    return forward, undo


TTA_NAMES = ["hflip", "vflip", "rotate180", "rotate90", "rotate270", "scale_up", "scale_down"]


def tta_detector(sdk, tmp_path):
    output = np.array(
        [[[32, 32], [32, 32], [32, 32], [16, 16], [0.9, 0.1], [0.1, 0.8]]], np.float32
    )
    path = model_file(tmp_path / "tta.onnx", [1, 3, 64, 64], output)
    model = sdk.load(
        path, device="cpu", task="detection", backend="openvino", use_cache=False,
        decoder="yolov8", labels=["first", "second"],
    )
    config = infer_config(sdk)
    config.classes = [1]
    return model, config


@pytest.mark.parametrize("aug", TTA_NAMES)
def test_predict_tta_maps_boxes_back_to_the_original_image(sdk, tmp_path, aug):
    forward, undo = tta_maps()
    model, config = tta_detector(sdk, tmp_path)
    image = np.zeros((20, 40, 3), np.uint8)  # wide, so rotations change the aspect ratio
    augmented = forward[aug](image)
    found = model.predict(augmented, config).detections
    assert len(found) == 1
    box = found[0].bbox
    # Independent reference: paint the box on a canvas of the augmented size and undo the
    # augmentation with OpenCV; its bounding rectangle is where the box belongs.
    canvas = np.zeros(augmented.shape[:2], np.uint8)
    canvas[int(round(box.y1)):int(round(box.y2)), int(round(box.x1)):int(round(box.x2))] = 255
    ys, xs = np.nonzero(undo[aug](canvas, (40, 20)))
    expected = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]

    merged = model.predict_tta(image, [aug], config).detections
    assert len(merged) == 1
    got = [merged[0].bbox.x1, merged[0].bbox.y1, merged[0].bbox.x2, merged[0].bbox.y2]
    np.testing.assert_allclose(got, expected, atol=1.6)
    assert 0 <= min(got) and got[2] <= 40 and got[3] <= 20  # inside the original image


def test_predict_tta_rejects_what_it_cannot_do(sdk, tmp_path):
    model, config = tta_detector(sdk, tmp_path)
    image = np.zeros((20, 40, 3), np.uint8)
    with pytest.raises(sdk.IVITError, match="bogus"):
        model.predict_tta(image, ["original", "bogus"], config)
    with pytest.raises(sdk.IVITError, match="at least one"):
        model.predict_tta(image, [], config)
    segmenter = load_with(sdk, segmentation_model_file(tmp_path / "seg.onnx", 64), task="segmentation")
    with pytest.raises(sdk.IVITError, match="segmentation"):
        segmenter.predict_tta(np.full((20, 40, 3), 255, np.uint8), ["original", "hflip"])


def test_predict_tta_merge_is_per_class_and_honours_max_detections(sdk, tmp_path):
    model, _ = tta_detector(sdk, tmp_path)  # two classes, same box
    image = np.zeros((20, 40, 3), np.uint8)
    both = infer_config(sdk)
    both.max_detections = 100
    assert {d.label for d in model.predict_tta(image, ["original", "hflip"], both).detections} == {
        "first", "second"
    }  # class-agnostic merging used to drop one of them
    one = infer_config(sdk)
    one.max_detections = 1
    assert len(model.predict_tta(image, ["original", "hflip"], one).detections) == 1


def test_results_save_follows_format_not_extension_and_reports_failures(sdk, tmp_path, monkeypatch):
    model, config = tta_detector(sdk, tmp_path)
    result = model.predict(np.zeros((20, 40, 3), np.uint8), config)
    assert len(result.detections) == 1

    result.save(str(tmp_path / "a.txt"), "json")  # the extension does not change the format
    assert (tmp_path / "a.txt").read_text().lstrip().startswith("{")
    result.save(str(tmp_path / "b.json"), "txt")
    fields = (tmp_path / "b.json").read_text().split()
    assert len(fields) == 5 and fields[0] == "1"  # class cx cy w h
    result.save(str(tmp_path / "c.json"))  # default format
    assert (tmp_path / "c.json").read_text().lstrip().startswith("{")

    monkeypatch.chdir(tmp_path)
    result.save("x", "txt")  # a 1-character relative path
    assert (tmp_path / "x").exists()
    with pytest.raises(sdk.IVITError, match="Unsupported save format"):
        result.save(str(tmp_path / "d.json"), "xml")
    with pytest.raises(sdk.IVITError, match="Cannot open"):
        result.save(str(tmp_path / "no_such_dir" / "e.json"))

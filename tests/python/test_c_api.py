"""Exercise the C ABI through ctypes; set IVIT_C_LIBRARY to an SDK shared library."""

import ctypes as c
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest


class LoadOptions(c.Structure):
    _fields_ = (
        [("struct_size", c.c_uint32), ("abi_version", c.c_uint32)]
        + [
            (name, c.c_char_p)
            for name in (
                "device",
                "backend",
                "task",
                "precision",
                "decoder",
                "input_color",
                "input_layout",
                "resize_mode",
                "labels_path",
                "output_name",
            )
        ]
        + [
            ("input_scale", c.c_float),
            ("mean", c.c_float * 3),
            ("stddev", c.c_float * 3),
            ("use_cache", c.c_int32),
            ("batch_size", c.c_int32),
        ]
    )


def test_extended_load_contract_and_struct_validation(native, classifier):
    native.ivit_load_options_init.argtypes = [c.POINTER(LoadOptions)]
    native.ivit_model_load_ex.argtypes = [
        c.c_char_p,
        c.POINTER(LoadOptions),
        c.POINTER(c.c_uint64),
    ]
    options = LoadOptions()
    options.struct_size = c.sizeof(options)
    assert native.ivit_load_options_init(c.byref(options)) == 0
    assert options.abi_version == 1
    options.device = b"cpu"
    options.backend = b"openvino"
    options.task = b"classification"
    options.input_color = b"bgr"
    options.input_scale = 0.1
    options.mean[:] = [1, 2, 3]
    options.stddev[:] = [2, 2, 2]
    model = c.c_uint64()
    assert (
        native.ivit_model_load_ex(
            str(classifier).encode(), c.byref(options), c.byref(model)
        )
        == 0
    )
    image = c.create_string_buffer(bytes([10, 20, 30]) * (32 * 32))
    result = c.c_uint64()
    assert (
        native.ivit_model_predict_bgr(
            model, image, 32 * 32 * 3, 32, 32, 32 * 3, 0, 0.45, 100, c.byref(result)
        )
        == 0
    )
    classifications = json.loads(text(native.ivit_result_json, result.value))[
        "classifications"
    ]
    assert all(abs(item["score"] - 1 / 3) < 1e-5 for item in classifications)
    assert native.ivit_result_release(result) == 0
    assert native.ivit_model_release(model) == 0
    options.abi_version = 99
    assert (
        native.ivit_model_load_ex(
            str(classifier).encode(), c.byref(options), c.byref(model)
        )
        == 1
    )
    options.struct_size = 0
    assert native.ivit_load_options_init(c.byref(options)) == 1


@pytest.fixture(scope="module")
def native():
    path = os.environ.get("IVIT_C_LIBRARY")
    if not path:
        pytest.skip("Set IVIT_C_LIBRARY to test a built C ABI")
    lib = c.CDLL(path)
    u64 = c.c_uint64
    ptr = c.POINTER(u64)
    lib.ivit_abi_version.restype = c.c_uint32
    lib.ivit_model_load.argtypes = [c.c_char_p] * 5 + [c.c_int32, ptr]
    lib.ivit_model_release.argtypes = [u64]
    lib.ivit_result_release.argtypes = [u64]
    lib.ivit_model_predict_bgr.argtypes = [
        u64,
        c.c_void_p,
        u64,
        c.c_int32,
        c.c_int32,
        u64,
        c.c_float,
        c.c_float,
        c.c_int32,
        ptr,
    ]
    lib.ivit_result_json.argtypes = [u64, c.c_void_p, u64, ptr]
    lib.ivit_last_error.argtypes = [c.c_void_p, u64, ptr]
    lib.ivit_devices_json.argtypes = [c.c_void_p, u64, ptr]
    return lib


@pytest.fixture(scope="module")
def classifier(tmp_path_factory):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    path = tmp_path_factory.mktemp("c-api") / "classifier.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pool"]),
            helper.make_node("Flatten", ["pool"], ["output"], axis=1),
        ],
        "classifier",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def text(reader, *args):
    size = c.c_uint64()
    assert reader(*args, None, 0, c.byref(size)) == 3
    data = c.create_string_buffer(size.value)
    assert reader(*args, data, size.value, c.byref(size)) == 0
    return data.value.decode()


def load(native, classifier):
    model = c.c_uint64()
    assert (
        native.ivit_model_load(
            str(classifier).encode(),
            b"cpu",
            b"openvino",
            b"classification",
            b"fp32",
            0,
            c.byref(model),
        )
        == 0
    )
    return model.value


def predict(native, model, length=32 * 100):
    image = c.create_string_buffer(length)
    result = c.c_uint64(999)
    status = native.ivit_model_predict_bgr(
        model, image, length, 32, 32, 100, 0, 0.45, 100, c.byref(result)
    )
    return status, result.value


def test_c_api_padded_input_and_result_outlives_model(native, classifier):
    assert native.ivit_abi_version() == 1
    model = load(native, classifier)
    status, result = predict(native, model)
    assert status == 0
    assert native.ivit_model_release(model) == 0
    data = json.loads(text(native.ivit_result_json, result))
    assert len(data["classifications"]) == 3
    assert sum(item["score"] for item in data["classifications"]) == pytest.approx(1)
    assert native.ivit_result_release(result) == 0
    assert native.ivit_result_release(result) == 2
    assert native.ivit_model_release(model) == 2


def test_c_api_short_buffer_and_error_recovery(native, classifier):
    model = load(native, classifier)
    try:
        status, result = predict(native, model, 1)
        assert (status, result) == (1, 0)
        assert "buffer" in text(native.ivit_last_error)
        status, result = predict(native, model)
        assert status == 0
        assert native.ivit_result_release(result) == 0
    finally:
        assert native.ivit_model_release(model) == 0


def test_c_api_invalid_handle_and_thread_local_errors(native):
    def worker(handle):
        assert native.ivit_model_release(handle) == 2
        return text(native.ivit_last_error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(worker, [0, 999999])) == ["Invalid or released handle"] * 2


def test_c_api_shared_model_concurrency(native, classifier):
    model = load(native, classifier)

    def worker(_):
        status, result = predict(native, model)
        assert status == 0
        try:
            return json.loads(text(native.ivit_result_json, result))["classifications"]
        finally:
            assert native.ivit_result_release(result) == 0

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(worker, range(20)))
        assert all(value == values[0] for value in values)
    finally:
        assert native.ivit_model_release(model) == 0


def test_c_api_load_errors_do_not_cross_abi(native):
    model = c.c_uint64(999)
    assert (
        native.ivit_model_load(
            b"missing.onnx",
            b"cpu",
            b"bogus",
            b"classification",
            b"fp32",
            0,
            c.byref(model),
        )
        == 4
    )
    assert model.value == 0
    assert "backend" in text(native.ivit_last_error).lower()


def test_c_api_reports_sdk_version(native):
    native.ivit_version_string.restype = c.c_char_p
    version = native.ivit_version_string().decode()
    parts = version.split(".")
    assert len(parts) == 3 and all(part.isdigit() for part in parts)


def test_c_api_log_callback_routes_messages_and_can_be_cleared(native):
    callback_type = c.CFUNCTYPE(None, c.c_int32, c.c_char_p, c.c_void_p)
    native.ivit_set_log_callback.argtypes = [callback_type, c.c_void_p]
    native.ivit_set_log_level.argtypes = [c.c_char_p]
    messages = []
    callback = callback_type(lambda level, text, user: messages.append((level, text.decode())))
    assert native.ivit_set_log_callback(callback, None) == 0
    try:
        assert native.ivit_set_log_level(b"info") == 0
        assert (1, "Log level set to: info") in messages
        assert native.ivit_set_log_level(b"bogus") == 1
        assert "log level" in text(native.ivit_last_error).lower()
    finally:
        assert native.ivit_set_log_callback(callback_type(), None) == 0
        native.ivit_set_log_level(b"info")


def test_c_api_default_log_sink_uses_stderr_not_stdout(native, capfd):
    native.ivit_set_log_level.argtypes = [c.c_char_p]
    assert native.ivit_set_log_level(b"info") == 0
    captured = capfd.readouterr()
    assert "Log level set to: info" in captured.err
    assert captured.out == ""


# ------------------------------------------------------------- additions in SDK 1.1


class ImageV1(c.Structure):
    _fields_ = [
        ("pixels", c.c_void_p),
        ("bytes", c.c_uint64),
        ("stride", c.c_uint64),
        ("width", c.c_int32),
        ("height", c.c_int32),
        ("channels", c.c_int32),
        ("reserved", c.c_int32),
    ]


class InferOptionsV1(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("abi_version", c.c_uint32),
        ("confidence", c.c_float),
        ("iou", c.c_float),
        ("max_detections", c.c_int32),
        ("agnostic_nms", c.c_int32),
        ("classes", c.POINTER(c.c_int32)),
        ("class_count", c.c_uint32),
        ("reserved", c.c_uint32),
    ]


class LoadOptionsV2(c.Structure):
    _fields_ = (
        [("struct_size", c.c_uint32), ("abi_version", c.c_uint32)]
        + [
            (name, c.c_char_p)
            for name in (
                "device",
                "backend",
                "task",
                "precision",
                "decoder",
                "input_color",
                "input_layout",
                "resize_mode",
                "labels_path",
                "output_name",
                "cache_dir",
                "performance_mode",
            )
        ]
        + [
            ("labels", c.POINTER(c.c_char_p)),
            ("workspace_bytes", c.c_uint64),
            ("label_count", c.c_uint32),
            ("use_cache", c.c_int32),
            ("batch_size", c.c_int32),
            ("max_batch_size", c.c_int32),
            ("num_threads", c.c_int32),
            ("enable_profiling", c.c_int32),
            ("input_scale", c.c_float),
            ("mean", c.c_float * 3),
            ("stddev", c.c_float * 3),
            ("num_instances", c.c_int32),
        ]
    )


def image_struct(buffer, width=32, height=32, channels=3, stride=None):
    stride = stride or width * channels
    return ImageV1(c.cast(buffer, c.c_void_p), len(buffer), stride, width, height, channels, 0)


def infer_options(native, **values):
    options = InferOptionsV1()
    options.struct_size = c.sizeof(options)
    native.ivit_infer_options_init.argtypes = [c.POINTER(InferOptionsV1)]
    assert native.ivit_infer_options_init(c.byref(options)) == 0
    for key, value in values.items():
        setattr(options, key, value)
    return options


def test_struct_layouts_have_no_hidden_padding():
    # The C header relies on these sizes; a mismatch would silently shift every field.
    assert c.sizeof(ImageV1) == 40
    assert c.sizeof(InferOptionsV1) == 40
    # Pinned by static_assert in src/core/c_api.cpp; this keeps the ctypes mirror honest.
    assert c.sizeof(LoadOptionsV2) == 176


def test_predict_ex_accepts_defaults_gray_images_and_options(native, classifier):
    model = load(native, classifier)
    native.ivit_model_predict_ex.argtypes = [
        c.c_uint64,
        c.POINTER(ImageV1),
        c.POINTER(InferOptionsV1),
        c.POINTER(c.c_uint64),
    ]
    try:
        color = c.create_string_buffer(32 * 32 * 3)
        result = c.c_uint64()
        assert (
            native.ivit_model_predict_ex(
                model, c.byref(image_struct(color)), None, c.byref(result)
            )
            == 0
        )
        assert native.ivit_result_release(result.value) == 0

        gray = c.create_string_buffer(32 * 32)
        options = infer_options(native, confidence=0.0)
        assert (
            native.ivit_model_predict_ex(
                model, c.byref(image_struct(gray, channels=1)), c.byref(options), c.byref(result)
            )
            == 0
        )
        assert len(json.loads(text(native.ivit_result_json, result.value))["classifications"]) == 3
        assert native.ivit_result_release(result.value) == 0
    finally:
        assert native.ivit_model_release(model) == 0


def test_predict_ex_rejects_bad_options_and_images_without_poisoning_the_model(native, classifier):
    model = load(native, classifier)
    native.ivit_model_predict_ex.argtypes = [
        c.c_uint64,
        c.POINTER(ImageV1),
        c.POINTER(InferOptionsV1),
        c.POINTER(c.c_uint64),
    ]
    buffer = c.create_string_buffer(32 * 32 * 3)
    result = c.c_uint64(7)
    try:
        for bad in (
            {"confidence": 1.5},
            {"confidence": -0.3},  # only exactly -1 means "task default"
            {"iou": float("nan")},
            {"max_detections": 0},
            {"agnostic_nms": 2},
            {"reserved": 1},
            {"class_count": 3},  # count without a pointer
        ):
            options = infer_options(native, **bad)
            assert (
                native.ivit_model_predict_ex(
                    model, c.byref(image_struct(buffer)), c.byref(options), c.byref(result)
                )
                == 1
            ), bad
            assert result.value == 0
        for image in (image_struct(buffer, channels=2), image_struct(buffer, width=0)):
            assert (
                native.ivit_model_predict_ex(model, c.byref(image), None, c.byref(result)) == 1
            )
        wrong_size = ImageV1(c.cast(buffer, c.c_void_p), 10, 96, 32, 32, 3, 0)
        assert native.ivit_model_predict_ex(model, c.byref(wrong_size), None, c.byref(result)) == 1
        stale = infer_options(native)
        stale.struct_size = 8
        assert (
            native.ivit_model_predict_ex(
                model, c.byref(image_struct(buffer)), c.byref(stale), c.byref(result)
            )
            == 1
        )
        good = c.c_uint64()
        assert (
            native.ivit_model_predict_ex(model, c.byref(image_struct(buffer)), None, c.byref(good))
            == 0
        )
        assert native.ivit_result_release(good.value) == 0
    finally:
        assert native.ivit_model_release(model) == 0


def test_predict_batch_is_all_or_nothing(native, classifier):
    model = load(native, classifier)
    native.ivit_model_predict_batch.argtypes = [
        c.c_uint64,
        c.POINTER(ImageV1),
        c.c_uint32,
        c.POINTER(InferOptionsV1),
        c.POINTER(c.c_uint64),
    ]
    buffer = c.create_string_buffer(32 * 32 * 3)
    try:
        images = (ImageV1 * 3)(*[image_struct(buffer) for _ in range(3)])
        handles = (c.c_uint64 * 3)(9, 9, 9)
        # Default options: confidence is the task default, so a classifier returns every class.
        assert native.ivit_model_predict_batch(model, images, 3, None, handles) == 0
        assert len({*handles}) == 3 and 0 not in handles
        for handle in handles:
            assert len(json.loads(text(native.ivit_result_json, handle))["classifications"]) == 3
            assert native.ivit_result_release(handle) == 0

        broken = (ImageV1 * 3)(
            image_struct(buffer), ImageV1(None, 0, 0, 32, 32, 3, 0), image_struct(buffer)
        )
        handles = (c.c_uint64 * 3)(9, 9, 9)
        assert native.ivit_model_predict_batch(model, broken, 3, None, handles) == 1
        assert list(handles) == [0, 0, 0]
        for count in (0, 1025):
            assert native.ivit_model_predict_batch(model, images, count, None, handles) == 1
    finally:
        assert native.ivit_model_release(model) == 0


def test_model_info_json_describes_the_model_and_contract(native, classifier):
    model = load(native, classifier)
    native.ivit_model_info_json.argtypes = [c.c_uint64, c.c_void_p, c.c_uint64, c.POINTER(c.c_uint64)]
    try:
        info = json.loads(text(native.ivit_model_info_json, model))
        assert info["task"] == "classification"
        assert info["backend"] == "openvino" and info["device"] == "cpu"
        assert info["inputs"][0]["shape"] == [1, 3, 32, 32]
        assert info["inputs"][0]["layout"] in ("nchw", "unknown")
        assert info["outputs"][0]["shape"] == [1, 3]
        assert info["contract"]["input_color"] == "rgb"
        assert info["contract"]["input_scale"] == pytest.approx(1 / 255)
        assert info["contract"]["std"] == [1, 1, 1]
    finally:
        assert native.ivit_model_release(model) == 0
    assert native.ivit_model_info_json(model, None, 0, c.byref(c.c_uint64())) == 2


def test_load_v2_applies_labels_and_contract_and_validates_input(native, classifier):
    native.ivit_load_options_v2_init.argtypes = [c.POINTER(LoadOptionsV2)]
    native.ivit_model_load_v2.argtypes = [c.c_char_p, c.POINTER(LoadOptionsV2), c.POINTER(c.c_uint64)]
    native.ivit_model_info_json.argtypes = [c.c_uint64, c.c_void_p, c.c_uint64, c.POINTER(c.c_uint64)]
    options = LoadOptionsV2()
    options.struct_size = c.sizeof(options)
    assert native.ivit_load_options_v2_init(c.byref(options)) == 0
    assert (options.abi_version, options.max_batch_size, options.workspace_bytes) == (2, 8, 1 << 30)
    options.device, options.backend, options.task = b"cpu", b"openvino", b"classification"
    options.use_cache = 0
    names = (c.c_char_p * 3)(b"cat", b"dog", b"bird")
    options.labels, options.label_count = names, 3
    model = c.c_uint64()
    assert native.ivit_model_load_v2(str(classifier).encode(), c.byref(options), c.byref(model)) == 0
    try:
        info = json.loads(text(native.ivit_model_info_json, model.value))
        assert info["labels"] == ["cat", "dog", "bird"]
        status, result = predict(native, model.value)
        assert status == 0
        labels = [i["label"] for i in json.loads(text(native.ivit_result_json, result))["classifications"]]
        assert sorted(labels) == ["bird", "cat", "dog"]
        assert native.ivit_result_release(result) == 0
    finally:
        assert native.ivit_model_release(model.value) == 0

    # A wrong label count is rejected while loading, with a model-load error category.
    options.label_count = 2
    assert native.ivit_model_load_v2(str(classifier).encode(), c.byref(options), c.byref(model)) == 4
    assert model.value == 0
    assert "labels has 2 entries" in text(native.ivit_last_error)
    native.ivit_last_error_category.restype = c.c_int32
    assert native.ivit_last_error_category() == 2  # IVIT_C_ERR_MODEL_LOAD
    options.label_count = 3
    options.num_instances = 0  # outside 1..16
    assert native.ivit_model_load_v2(str(classifier).encode(), c.byref(options), c.byref(model)) == 1
    options.num_instances = 1
    options.abi_version = 1
    assert native.ivit_model_load_v2(str(classifier).encode(), c.byref(options), c.byref(model)) == 1
    options.abi_version = 2
    options.label_count = 1
    options.labels = None
    assert native.ivit_model_load_v2(str(classifier).encode(), c.byref(options), c.byref(model)) == 1


def test_error_categories_distinguish_failures_without_changing_status(native, classifier, tmp_path):
    native.ivit_last_error_category.restype = c.c_int32
    model = c.c_uint64()
    # Unknown backend name -> generic runtime error status as before.
    assert native.ivit_model_load(
        str(classifier).encode(), b"cpu", b"bogus", b"classification", b"fp32", 0, c.byref(model)
    ) == 4
    first = native.ivit_last_error_category()
    assert first in (1, 2, 3, 4)
    # Corrupt model file -> model-load category.
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"not an onnx model")
    assert native.ivit_model_load(
        str(broken).encode(), b"cpu", b"openvino", b"classification", b"fp32", 0, c.byref(model)
    ) == 4
    assert native.ivit_last_error_category() in (2, 3)
    # A successful call resets the category.
    good = load(native, classifier)
    assert native.ivit_last_error_category() == 0
    # Bad arguments are INVALID_ARGUMENT with no category; handles likewise.
    assert native.ivit_model_release(good) == 0
    assert native.ivit_model_release(good) == 2
    assert native.ivit_last_error_category() == 0


def test_default_confidence_returns_all_classes_but_an_explicit_one_filters(native, classifier):
    model = load(native, classifier)
    native.ivit_model_predict_ex.argtypes = [
        c.c_uint64, c.POINTER(ImageV1), c.POINTER(InferOptionsV1), c.POINTER(c.c_uint64),
    ]
    buffer = c.create_string_buffer(32 * 32 * 3)
    try:
        defaults = infer_options(native)
        assert defaults.confidence == -1.0
        for options, expected in ((defaults, 3), (infer_options(native, confidence=0.5), 0)):
            result = c.c_uint64()
            assert native.ivit_model_predict_ex(
                model, c.byref(image_struct(buffer)), c.byref(options), c.byref(result)
            ) == 0
            found = json.loads(text(native.ivit_result_json, result.value))["classifications"]
            assert len(found) == expected
            assert native.ivit_result_release(result.value) == 0
    finally:
        assert native.ivit_model_release(model) == 0

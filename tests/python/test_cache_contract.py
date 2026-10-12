import os

import numpy as np
import pytest

from test_model_contract import infer_config, model_file, sdk


def test_memory_cache_includes_normalization_and_labels(sdk, tmp_path):
    path = model_file(tmp_path / "cache.onnx", [1, 3, 2, 4])
    first = sdk.load(
        path,
        device="cpu",
        backend="openvino",
        task="classification",
        input_color="bgr",
        labels=["a", "b", "c"],
    )
    second = sdk.load(
        path,
        device="cpu",
        backend="openvino",
        task="classification",
        input_color="rgb",
        labels=["x", "y", "z"],
    )
    image = np.full((2, 4, 3), [10, 20, 30], np.uint8)
    assert first.predict(image, infer_config(sdk)).top1().label == "c"
    assert second.predict(image, infer_config(sdk)).top1().label == "x"
    assert first is sdk.load(
        path,
        device="cpu",
        backend="openvino",
        task="classification",
        input_color="bgr",
        labels=["a", "b", "c"],
    )


def test_public_clear_cache_releases_memory_model(sdk, tmp_path):
    path = model_file(tmp_path / "release.onnx", [1, 3, 2, 4])
    options = dict(device="cpu", backend="openvino", task="classification")
    original = sdk.load(path, **options)
    sdk.clear_cache()
    reloaded = sdk.load(path, **options)
    assert original is not reloaded
    image = np.zeros((2, 4, 3), np.uint8)
    assert original.predict(image, infer_config(sdk)).classifications
    assert reloaded.predict(image, infer_config(sdk)).classifications


def test_auto_device_respects_explicit_backend(sdk, tmp_path):
    path = model_file(tmp_path / "auto.onnx", [1, 3, 2, 4])
    model = sdk.load(path, backend="openvino", task="classification", use_cache=False)
    assert model.backend == sdk.BackendType.OpenVINO
    assert any(
        d.id == model.device and d.backend == "openvino" for d in sdk.list_devices()
    )


@pytest.mark.parametrize("backend", ["hailort", "qnn"])
def test_removed_backends_are_rejected_by_name(sdk, tmp_path, backend):
    path = model_file(tmp_path / "model.onnx", [1, 3, 2, 4])
    with pytest.raises(Exception, match="Unknown backend"):
        sdk.load(str(path), backend=backend, use_cache=False)


def test_replaced_model_content_invalidates_memory_cache(sdk, tmp_path):
    path = model_file(tmp_path / "replace.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    stamp = os.stat(path)
    first = sdk.load(
        path,
        device="cpu",
        backend="openvino",
        task="classification",
        decoder="probabilities",
    )
    model_file(path, [1, 3, 2, 4], [[0.9, 0.1]])
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    second = sdk.load(
        path,
        device="cpu",
        backend="openvino",
        task="classification",
        decoder="probabilities",
    )
    image = np.zeros((2, 4, 3), np.uint8)
    assert first.predict(image, infer_config(sdk)).top1().class_id == 1
    assert second.predict(image, infer_config(sdk)).top1().class_id == 0


@pytest.mark.parametrize(
    "options",
    [
        {"precision": "typo"},
        {"num_threads": -1},
        {"batch_size": 0},
        {"performance_mode": "wrong"},
        {"workspace_size": 0},
        {"task": "pose"},
        {"num_threads": 2, "device": "gpu:0"},
        {"max_batch_size": 2},
        {"precision": "int8"},
    ],
)
def test_load_rejects_unsupported_configuration(sdk, tmp_path, options):
    path = model_file(tmp_path / "invalid.onnx", [1, 3, 2, 4])
    kwargs = dict(
        device="cpu", backend="openvino", task="classification", use_cache=False
    )
    kwargs.update(options)
    with pytest.raises(Exception):
        sdk.load(path, **kwargs)


def test_cpu_runtime_settings_apply_at_load(sdk, tmp_path):
    path = model_file(tmp_path / "threads.onnx", [1, 3, 2, 4])
    model = sdk.load(
        path,
        device="cpu",
        backend="openvino",
        task="classification",
        num_threads=2,
        performance_mode="THROUGHPUT",
        enable_profiling=True,
        use_cache=False,
    )
    image = np.zeros((2, 4, 3), np.uint8)
    assert model.predict(image, infer_config(sdk)).classifications

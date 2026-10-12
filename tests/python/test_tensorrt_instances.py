"""num_instances: concurrent TensorRT execution contexts behind one model."""

import gc
import threading

import numpy as np
import pytest

from test_model_contract import infer_config, model_file, sdk  # noqa: F401


@pytest.fixture
def trt_sdk(sdk):  # noqa: F811
    if not any(d.backend == "tensorrt" for d in sdk.list_devices()):
        pytest.skip("TensorRT GPU required")
    return sdk


def load(sdk, path, cache, **kwargs):
    return sdk.load(
        str(path), device="cuda:0", backend="tensorrt", task="classification",
        decoder="logits", cache_dir=str(cache), use_cache=True, **kwargs,
    )


def test_concurrent_callers_get_the_same_answers_as_a_single_caller(trt_sdk, tmp_path):
    sdk = trt_sdk
    path = model_file(tmp_path / "m.onnx", [1, 3, 2, 4], [[0.2, 0.7, 0.1]])
    model = load(sdk, path, tmp_path / "engines", num_instances=3)
    image = np.zeros((2, 4, 3), np.uint8)
    config = infer_config(sdk)
    expected = [(c.class_id, c.score) for c in model.predict(image, config).classifications]

    failures, lock = [], threading.Lock()

    def worker():
        try:
            for _ in range(25):
                got = [(c.class_id, c.score) for c in model.predict(image, config).classifications]
                if got != expected:
                    raise AssertionError(f"{got} != {expected}")
        except Exception as error:  # collected, asserted on the main thread
            with lock:
                failures.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]  # more callers than slots
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads), "callers deadlocked waiting for a slot"
    assert not failures, failures[:1]
    del model
    gc.collect()
    sdk.clear_cache()


def test_instance_counts_are_separate_models_and_the_default_is_one(trt_sdk, tmp_path):
    sdk = trt_sdk
    path = model_file(tmp_path / "m.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    cache = tmp_path / "engines"
    one = load(sdk, path, cache)
    four = load(sdk, path, cache, num_instances=4)
    assert one is not four  # the cache key includes num_instances
    assert len(list(cache.glob("*.engine"))) == 1  # ...but both reuse one built engine
    image = np.zeros((2, 4, 3), np.uint8)
    assert one.predict(image, infer_config(sdk)).top1().class_id == 1
    assert four.predict(image, infer_config(sdk)).top1().class_id == 1
    del one, four
    gc.collect()
    sdk.clear_cache()


@pytest.mark.parametrize("bad", [0, -1, 17])
def test_out_of_range_instance_counts_are_rejected(sdk, tmp_path, bad):  # noqa: F811
    path = model_file(tmp_path / "m.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    with pytest.raises(Exception, match="num_instances"):
        sdk.load(str(path), device="cpu", backend="openvino", task="classification",
                 decoder="logits", num_instances=bad, use_cache=False)


def test_openvino_rejects_instances_instead_of_ignoring_them(sdk, tmp_path):  # noqa: F811
    path = model_file(tmp_path / "m.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    with pytest.raises(Exception, match="TensorRT"):
        sdk.load(str(path), device="cpu", backend="openvino", task="classification",
                 decoder="logits", num_instances=2, use_cache=False)

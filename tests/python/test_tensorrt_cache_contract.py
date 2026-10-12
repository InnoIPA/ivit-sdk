import ctypes
import gc

import numpy as np
import pytest

from test_model_contract import infer_config, model_file, sdk


@pytest.fixture
def trt_sdk(sdk):
    if not any(d.backend == "tensorrt" for d in sdk.list_devices()):
        pytest.skip("TensorRT GPU required")
    return sdk


def load(sdk, path, cache, **kwargs):
    decoder = kwargs.pop("decoder", "logits")
    return sdk.load(
        str(path),
        device="cuda:0",
        backend="tensorrt",
        task="classification",
        decoder=decoder,
        cache_dir=str(cache),
        use_cache=True,
        **kwargs,
    )


def clear(sdk):
    sdk.clear_cache()
    gc.collect()


def test_same_stem_different_content_and_corrupt_cache(trt_sdk, tmp_path):
    sdk = trt_sdk
    first_dir, second_dir = tmp_path / "one", tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = model_file(first_dir / "same.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    second_path = model_file(second_dir / "same.onnx", [1, 3, 2, 4], [[0.9, 0.1]])
    cache = tmp_path / "engines"
    image = np.zeros((2, 4, 3), np.uint8)
    first = load(sdk, first_path, cache)
    second = load(sdk, second_path, cache)
    assert first.predict(image, infer_config(sdk)).top1().class_id == 1
    assert second.predict(image, infer_config(sdk)).top1().class_id == 0
    assert len(list(cache.glob("*.engine"))) == 2
    del first, second
    clear(sdk)
    # A failed deserialization must rebuild rather than poison the load path.
    for engine in cache.glob("*.engine"):
        engine.write_bytes(b"invalid engine")
    recovered = load(sdk, first_path, cache)
    assert recovered.predict(image, infer_config(sdk)).top1().class_id == 1
    del recovered
    clear(sdk)
    assert not list(cache.glob("*.tmp-*"))


def test_batch_profile_changes_cache_and_enforces_bound(trt_sdk, tmp_path):
    sdk = trt_sdk
    path = model_file(tmp_path / "dynamic.onnx", ["N", 3, 2, 4])
    cache = tmp_path / "profiles"
    model = load(sdk, path, cache, max_batch_size=2)
    image = np.zeros((2, 4, 3), np.uint8)
    assert len(model.predict_batch([image, image], infer_config(sdk))) == 2
    with pytest.raises(sdk.IVITError, match="optimization profile"):
        model.predict_batch([image] * 3, infer_config(sdk))
    assert model.predict(image, infer_config(sdk)).classifications
    other = load(sdk, path, cache, batch_size=2, max_batch_size=4)
    assert len(other.predict_batch([image] * 4, infer_config(sdk))) == 4
    assert len(list(cache.glob("*.engine"))) == 2
    del model, other
    clear(sdk)


def test_repeated_engine_release_has_bounded_device_memory(trt_sdk, tmp_path):
    sdk = trt_sdk
    path = model_file(tmp_path / "lifecycle.onnx", [1, 3, 2, 4])
    cache = tmp_path / "lifecycle-cache"
    image = np.zeros((2, 4, 3), np.uint8)
    # Warm up global CUDA/TRT allocations before measuring engine ownership.
    model = load(sdk, path, cache)
    model.predict(image, infer_config(sdk))
    del model
    clear(sdk)
    cuda = ctypes.CDLL("libcudart.so.12")
    cuda.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2

    def free_bytes():
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        assert cuda.cudaDeviceSynchronize() == 0
        assert cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0
        return free.value

    before = free_bytes()
    for _ in range(30):
        model = load(sdk, path, cache)
        model.predict(image, infer_config(sdk))
        del model
        clear(sdk)
    after = free_bytes()
    assert before - after < 16 * 1024 * 1024, (before, after)


def test_default_engine_cache_is_the_user_cache_not_the_model_directory(trt_sdk, tmp_path):
    import os
    import subprocess
    import sys

    models = tmp_path / "models"
    models.mkdir()
    path = model_file(models / "tiny.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    home = tmp_path / "home"
    home.mkdir()
    # A fresh process so no set_cache_dir() from this session applies.
    script = (
        "import ivit\n"
        f"m = ivit.load({str(path)!r}, device='cuda:0', backend='tensorrt', task='classification',"
        " decoder='logits')\n"
    )
    env = {**os.environ, "HOME": str(home)}
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert len(list((home / ".cache" / "ivit" / "engines").glob("tiny_trt_*.engine"))) == 1
    assert not list(models.glob("*.engine"))  # nothing is written next to the model


def test_unwritable_engine_cache_still_loads_the_model(trt_sdk, tmp_path):
    import os

    path = model_file(tmp_path / "tiny.onnx", [1, 3, 2, 4], [[0.1, 0.9]])
    cache = tmp_path / "readonly"
    cache.mkdir()
    cache.chmod(0o555)
    try:
        if os.access(cache, os.W_OK):
            pytest.skip("running with permissions that ignore the read-only mode")
        model = load(trt_sdk, path, cache)
        image = np.zeros((2, 4, 3), np.uint8)
        assert model.predict(image, infer_config(trt_sdk)).top1().class_id == 1
        assert not list(cache.iterdir())
        del model
        clear(trt_sdk)
    finally:
        cache.chmod(0o755)

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import threading

import numpy as np
import pytest

from test_model_contract import infer_config, model_file, sdk


def load_model(sdk, tmp_path):
    path = model_file(tmp_path / "async.onnx", [1, 3, 2, 4])
    return sdk.load(
        path, device="cpu", backend="openvino", task="classification", use_cache=False
    )


def test_native_future_survives_model_reference(sdk, tmp_path):
    model = load_model(sdk, tmp_path)
    future = model.predict_async(np.zeros((2, 4, 3), np.uint8), infer_config(sdk))
    assert isinstance(future, Future)
    del model
    assert future.result(timeout=10).classifications


def test_batch_future_and_asyncio(sdk, tmp_path):
    model = load_model(sdk, tmp_path)
    image = np.zeros((2, 4, 3), np.uint8)
    future = model.predict_batch_async([image, image], infer_config(sdk))
    assert len(future.result(timeout=10)) == 2

    async def run():
        return await asyncio.wrap_future(model.predict_async(image, infer_config(sdk)))

    assert asyncio.run(run()).classifications


def test_future_propagates_native_exception_and_recovers(sdk, tmp_path):
    model = load_model(sdk, tmp_path)
    future = model.predict_async(np.zeros((2, 4, 3), np.float32), infer_config(sdk))
    with pytest.raises(Exception, match="uint8"):
        future.result(timeout=10)
    assert (
        model.predict_async(np.zeros((2, 4, 3), np.uint8), infer_config(sdk))
        .result(timeout=10)
        .classifications
    )


def test_queued_input_snapshot_and_cancel(sdk, tmp_path, monkeypatch):
    import ivit._async as worker

    model = load_model(sdk, tmp_path)
    gate = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(worker, "_executor", executor)
        blocker = executor.submit(gate.wait)
        try:
            image = np.full((2, 4, 3), [10, 20, 30], np.uint8)
            config = infer_config(sdk)
            future = model.predict_async(image, config)
            assert not future.done()
            image[:] = [30, 20, 10]
            config.conf_threshold = 1
            cancelled = model.predict_async(image, config)
            assert cancelled.cancel()
        finally:
            gate.set()
        blocker.result(timeout=10)
        assert future.result(timeout=10).top1().class_id == 0
        assert cancelled.cancelled()


def test_async_queue_limit_recovers_after_cancellation(sdk, tmp_path, monkeypatch):
    import ivit._async as worker

    model = load_model(sdk, tmp_path)
    gate = threading.Event()
    pending = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(worker, "_executor", executor)
        blocker = executor.submit(gate.wait)
        image = np.zeros((2, 4, 3), np.uint8)
        try:
            pending = [model.predict_async(image, infer_config(sdk)) for _ in range(32)]
            with pytest.raises(RuntimeError, match="queue is full"):
                model.predict_async(image, infer_config(sdk))
            assert all(future.cancel() for future in pending)
            recovered = model.predict_async(image, infer_config(sdk))
        finally:
            for future in pending:
                future.cancel()
            gate.set()
        blocker.result(timeout=10)
        assert recovered.result(timeout=10).classifications

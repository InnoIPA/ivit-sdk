import gc

import numpy as np
import pytest

from test_model_contract import infer_config, model_file, sdk


def test_stream_keeps_model_and_each_source_alive(sdk, tmp_path):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    if not writer.isOpened():
        pytest.skip("MJPG video writer unavailable")
    for _ in range(3):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()
    path = model_file(tmp_path / "stream.onnx", [1, 3, 2, 4])
    model = sdk.load(
        path, device="cpu", backend="openvino", task="classification", use_cache=False
    )
    first = model.stream(str(video), infer_config(sdk))
    second = model.stream(str(video), infer_config(sdk))
    del model
    gc.collect()
    assert len(list(first)) == 3
    assert len(list(second)) == 3

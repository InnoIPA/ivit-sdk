"""Edge cases that used to crash, misreport or fail without a clear error."""

import numpy as np
import pytest

from test_model_contract import infer_config, model_file, sdk, tta_detector  # noqa: F401


# ---------------------------------------------------------------- arguments


def test_empty_model_path_is_a_model_load_error(sdk):
    with pytest.raises(sdk.ModelLoadError, match="empty"):
        sdk.load("")


def test_unknown_log_level_is_rejected_and_known_ones_are_case_insensitive(sdk):
    with pytest.raises(ValueError, match="bogus"):
        sdk.set_log_level("bogus")
    for level in ("WARNING", "Warn", "off", "info"):
        sdk.set_log_level(level)
    sdk.set_log_level("info")


def test_topk_rejects_a_negative_k(sdk, tmp_path):
    model = sdk.load(
        model_file(tmp_path / "c.onnx", [1, 3, 2, 4], [[0.1, 0.9]]), device="cpu",
        backend="openvino", task="classification", decoder="logits", use_cache=False,
    )
    result = model.predict(np.zeros((2, 4, 3), np.uint8), infer_config(sdk))
    assert len(result.topk(1)) == 1 and result.topk(0) == []
    with pytest.raises(ValueError, match="negative"):
        result.topk(-1)


# ---------------------------------------------------------------- devices


class FakeDevice:
    def __init__(self, id, type="gpu", backend="openvino", vendor="intel", discrete=False):
        self.id, self.type, self.backend, self.vendor = id, type, backend, vendor
        self.available, self.is_discrete = True, discrete


def test_devices_gpu_matches_the_requested_index_only(sdk, monkeypatch):
    manager = sdk.devices  # ivit.devices is the DeviceManager instance
    listing = [FakeDevice("cpu", "cpu"), FakeDevice("gpu:0"), FakeDevice("cuda:0", "cuda", "tensorrt", "nvidia")]
    monkeypatch.setattr(type(manager), "__call__", lambda self, refresh=False: listing)
    assert manager.gpu(0).id == "gpu:0"
    assert manager.gpu(7) is None  # used to return gpu:0
    assert manager.gpu(1) is None
    listing[1].id = "gpu"  # a lone GPU may be listed without an index
    assert manager.gpu(0).id == "gpu" and manager.gpu(1) is None


def test_devices_best_rejects_an_unknown_strategy(sdk):
    with pytest.raises(ValueError, match="bogus"):
        sdk.devices.best("bogus")
    for strategy in ("latency", "throughput", "efficiency", "balanced"):
        assert sdk.devices.best(strategy) is not None


# ---------------------------------------------------------------- images


@pytest.mark.parametrize("channels", [1, 4])
def test_visualize_accepts_grayscale_and_bgra_images(sdk, tmp_path, channels):
    from test_model_contract import load_with, segmentation_model_file

    model = load_with(sdk, segmentation_model_file(tmp_path / "s.onnx", 64), task="segmentation")
    bgr = np.full((30, 50, 3), 255, np.uint8)
    result = model.predict(bgr)
    # The picture to draw on need not be the 3-channel one that was inferred (predict() itself
    # accepts grayscale; BGRA is a common display format).
    canvas = bgr[:, :, 0] if channels == 1 else np.dstack([bgr, np.full((30, 50), 255, np.uint8)])
    assert canvas.shape[2:] == ((channels,) if channels == 4 else ())
    drawn = result.visualize(canvas)  # an OpenCV exception for 1 or 4 channels before
    assert drawn.shape == (30, 50, 3)


def test_detector_predict_video_reports_an_unopenable_source(sdk, tmp_path):
    from ivit.vision import Detector

    output = np.array([[[32, 32], [32, 32], [32, 32], [16, 16], [0.9, 0.1], [0.1, 0.8]]], np.float32)
    path = model_file(tmp_path / "d.onnx", [1, 3, 64, 64], output)
    detector = Detector(path, device="cpu", backend="openvino", use_cache=False, decoder="yolov8")
    with pytest.raises(RuntimeError, match="Failed to open video source"):
        detector.predict_video(str(tmp_path / "missing.mp4"), lambda results, frame: None)


# ---------------------------------------------------------------- events


def test_batch_end_fires_even_when_an_image_fails(sdk, tmp_path):
    model = sdk.load(
        model_file(tmp_path / "c.onnx", [1, 3, 2, 4], [[0.1, 0.9]]), device="cpu",
        backend="openvino", task="classification", decoder="logits", use_cache=False,
    )
    events = []
    model.on("batch_start", lambda ctx: events.append("start"))
    model.on("batch_end", lambda ctx: events.append("end"))
    good = np.zeros((2, 4, 3), np.uint8)
    bad = np.zeros((0, 4, 3), np.uint8)
    with pytest.raises(sdk.IVITError):
        model.predict_concurrent([good, bad], 2, infer_config(sdk))
    assert events == ["start", "end"]  # "end" used to be skipped, leaving listeners waiting

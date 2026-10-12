"""
API tests for iVIT-SDK.
"""

import re

import pytest
import numpy as np


class TestImport:
    """Test module imports."""

    def test_import_ivit(self):
        """Test main module import."""
        import ivit
        assert hasattr(ivit, '__version__')
        assert re.fullmatch(r"\d+\.\d+\.\d+", ivit.__version__)

    def test_import_core(self):
        """Test core imports."""
        from ivit.core import (
            LoadConfig,
            InferConfig,
            DeviceInfo,
            BBox,
            Detection,
            ClassificationResult,
        )

    def test_import_vision(self):
        """Test vision imports."""
        from ivit.vision import Classifier, Detector, Segmentor

    def test_import_utils(self):
        """Test utils imports."""
        from ivit.utils import Profiler, VideoStream


def make_detection(box, class_id, label, confidence):
    import ivit

    detection = ivit.Detection()
    detection.bbox, detection.class_id = box, class_id
    detection.label, detection.confidence = label, confidence
    return detection


class TestNativeTypes:
    """The types the SDK actually returns (native), not look-alike containers."""

    def test_bbox_creation_and_properties(self):
        import ivit

        bbox = ivit.BBox(10, 20, 110, 70)
        assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (10, 20, 110, 70)
        assert (bbox.width, bbox.height, bbox.area) == (100, 50, 5000)
        assert (bbox.center_x, bbox.center_y) == (60, 45)

    def test_bbox_iou(self):
        import ivit

        box = ivit.BBox(0, 0, 100, 100)
        assert box.iou(ivit.BBox(0, 0, 100, 100)) == 1.0
        assert box.iou(ivit.BBox(200, 200, 300, 300)) == 0.0
        # Intersection 50x50 = 2500; union 100x100 + 100x100 - 2500 = 17500
        assert box.iou(ivit.BBox(50, 50, 150, 150)) == pytest.approx(2500 / 17500, abs=1e-4)

    def test_bbox_conversion(self):
        import ivit

        bbox = ivit.BBox(10, 20, 110, 120)
        assert tuple(bbox.to_xywh()) == (10, 20, 100, 100)
        assert tuple(bbox.to_cxcywh()) == (60, 70, 100, 100)
        for rebuilt in (ivit.BBox.from_xywh(10, 20, 100, 100), ivit.BBox.from_cxcywh(60, 70, 100, 100)):
            assert (rebuilt.x1, rebuilt.y1, rebuilt.x2, rebuilt.y2) == (10, 20, 110, 120)

    def test_load_config_defaults(self):
        import ivit

        config = ivit.LoadConfig()
        assert config.device == "auto"
        assert config.backend == "auto"
        assert config.precision == "fp32"
        assert config.batch_size == 1
        assert config.use_cache is True
        assert config.num_instances == 1

    def test_infer_config_defaults(self):
        import ivit

        config = ivit.InferConfig()
        assert config.conf_threshold == -1.0  # "task default": 0.5 detection, 0 classification
        assert config.iou_threshold == pytest.approx(0.45)
        assert config.max_detections == 100
        assert list(config.classes) == []
        assert config.agnostic_nms is False


class TestNativeResults:
    """Results as produced by inference: filtering and serialisation."""

    def detections(self):
        import ivit

        results = ivit.Results()
        results.detections = [
            make_detection(ivit.BBox(0, 0, 100, 100), 0, "person", 0.9),
            make_detection(ivit.BBox(50, 50, 150, 150), 1, "car", 0.8),
            make_detection(ivit.BBox(100, 100, 200, 200), 0, "person", 0.6),
        ]
        return results

    def test_empty_results(self):
        import ivit

        results = ivit.Results()
        assert results.num_detections() == 0
        assert list(results.detections) == [] and list(results.classifications) == []

    def test_results_hold_detections(self):
        assert self.detections().num_detections() == 3

    def test_results_filter(self):
        results = self.detections()
        assert len(results.filter_by_class([0])) == 2
        high = results.filter_by_confidence(0.85)
        assert len(high) == 1 and high[0].label == "person"
        assert results.num_detections() == 3  # filters return copies, they do not modify the result

    def test_results_to_json(self):
        import json

        data = json.loads(self.detections().to_json())
        assert len(data["detections"]) == 3
        assert {"inference_time_ms", "device_used", "image_size", "classifications"} <= set(data)
        assert data["detections"][0]["label"] == "person"


class TestDevice:
    """Test device management."""

    def test_list_devices(self):
        """Test device listing."""
        from ivit.core.device import list_devices

        devices = list_devices()
        assert isinstance(devices, list)
        # Should at least have CPU fallback
        assert len(devices) >= 1

    @pytest.mark.parametrize("device, backend", [
        ("gpu:0", "openvino"),   # an Intel GPU, not NVIDIA
        ("gpu", "openvino"),
        ("npu", "openvino"),
        ("cpu", "openvino"),
        ("cuda:1", "tensorrt"),
        ("CUDA:0", "tensorrt"),
        ("auto", "auto"),
    ])
    def test_backend_for_unlisted_devices_follows_the_native_rules(self, device, backend, monkeypatch):
        from ivit.core import device as device_module

        monkeypatch.setattr(device_module, "list_devices", lambda refresh=False: [])  # nothing listed
        assert device_module.get_backend_for_device(device) == backend

    def test_get_best_device(self):
        """Test best device selection."""
        from ivit.core.device import get_best_device

        device = get_best_device()
        assert device.id is not None
        assert device.backend is not None
        # The first positional argument is the priority (the ignored task argument was removed).
        assert get_best_device("efficiency").id
        import ivit

        assert ivit.get_best_device(priority="performance").id


class TestProfiler:
    """Test profiler."""

    def test_profiler_timing(self):
        """Test profiler timing."""
        from ivit.utils.profiler import Profiler
        import time

        profiler = Profiler()

        profiler.start()
        time.sleep(0.01)  # 10ms
        profiler.stop()

        elapsed = profiler.elapsed_ms()
        assert elapsed >= 10
        assert elapsed < 50  # Allow some margin

    def test_profiler_stats(self):
        """Test profiler statistics."""
        from ivit.utils.profiler import Profiler
        import time

        profiler = Profiler()

        for _ in range(10):
            profiler.start()
            time.sleep(0.001)  # 1ms
            profiler.stop()

        mean, median, std, min_t, max_t, p95, p99 = profiler.calculate_stats()

        assert mean > 0
        assert median > 0
        assert min_t <= mean <= max_t


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_detector_defaults_match_across_python_and_native():
    import inspect

    import ivit
    from ivit.vision import Detector

    if not ivit.is_cpp_available():
        pytest.skip("C++ bindings are not built")
    wrapper = inspect.signature(Detector.predict).parameters
    assert wrapper["conf_threshold"].default == 0.5
    assert wrapper["iou_threshold"].default == 0.45
    assert wrapper["max_detections"].default == ivit.InferConfig().max_detections == 100
    import re

    native = ivit.Detector.predict.__doc__  # pybind11 signature text carries the defaults
    conf = [float(v) for v in re.findall(r"conf_threshold: [^=]*= ([0-9.]+)", native)]
    iou = [float(v) for v in re.findall(r"iou_threshold: [^=]*= ([0-9.]+)", native)]
    assert conf and all(v == 0.5 for v in conf)
    assert iou and all(abs(v - 0.45) < 1e-6 for v in iou)

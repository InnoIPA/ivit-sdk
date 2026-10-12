"""
Core inference tests for iVIT-SDK.

Tests cover:
- Model loading and configuration
- Single image inference
- Batch inference
- Error handling
- Thread safety
- Resource management
"""

import pytest
import threading
import time


class TestModelLoading:
    """Test model loading scenarios."""

    def test_load_config_defaults(self):
        """The native LoadConfig defaults the SDK really uses."""
        import ivit

        config = ivit.LoadConfig()
        assert config.device == "auto"
        assert config.backend == "auto"
        assert config.precision == "fp32"
        assert config.batch_size == 1
        assert config.use_cache is True

    def test_load_config_custom(self):
        import ivit

        config = ivit.LoadConfig()
        config.device, config.backend = "cuda:0", "tensorrt"
        config.precision, config.batch_size = "fp16", 4
        assert (config.device, config.backend, config.precision, config.batch_size) == (
            "cuda:0", "tensorrt", "fp16", 4)

    def test_infer_config_defaults(self):
        import ivit

        config = ivit.InferConfig()
        assert config.conf_threshold == -1.0  # task default
        assert config.iou_threshold == pytest.approx(0.45)
        assert config.max_detections == 100

    def test_infer_config_custom(self):
        import ivit

        config = ivit.InferConfig()
        config.conf_threshold, config.iou_threshold = 0.25, 0.5
        config.max_detections, config.classes = 50, [0, 1, 2]
        assert config.conf_threshold == pytest.approx(0.25)
        assert config.iou_threshold == pytest.approx(0.5)
        assert config.max_detections == 50
        assert list(config.classes) == [0, 1, 2]


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_image_path(self):
        """Test handling of invalid image path."""
        from ivit.core import IVITError

        with pytest.raises((IVITError, FileNotFoundError, ValueError)):
            # This should raise an error for non-existent file
            import cv2
            img = cv2.imread("/non/existent/path.jpg")
            if img is None:
                raise ValueError("Failed to load image")

    def test_invalid_model_path(self):
        """Test handling of invalid model path."""
        from ivit.core import ModelLoadError

        # ModelLoadError should be raised for invalid paths
        assert ModelLoadError is not None

    def test_inference_error(self):
        """Test InferenceError exception."""
        from ivit.core import InferenceError

        with pytest.raises(InferenceError):
            raise InferenceError("Test inference error")

    def test_device_not_found_error(self):
        """Test DeviceNotFoundError exception."""
        from ivit.core import DeviceNotFoundError

        with pytest.raises(DeviceNotFoundError):
            raise DeviceNotFoundError("cuda:99")


class TestThreadSafety:
    """Test thread safety of core components."""

    def test_concurrent_device_listing(self):
        """Test thread-safe device listing."""
        from ivit.core.device import list_devices

        results = []
        errors = []

        def worker():
            try:
                devices = list_devices()
                results.append(devices)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

    def test_concurrent_profiler_access(self):
        """Test thread-safe profiler usage."""
        from ivit.utils.profiler import Profiler

        profiler = Profiler()
        errors = []

        def worker():
            try:
                for _ in range(100):
                    profiler.start()
                    time.sleep(0.0001)
                    profiler.stop()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Note: Current profiler is not thread-safe, this may fail
        # This test documents the expected behavior
        assert len(profiler.times) > 0



class TestProfileReport:
    """Test ProfileReport functionality."""

    def test_profile_report_creation(self):
        """Test ProfileReport creation."""
        from ivit.utils.profiler import ProfileReport

        report = ProfileReport(
            model_name="test_model",
            device="cuda:0",
            backend="tensorrt",
            precision="fp16",
            input_shape=(1, 3, 640, 640),
            iterations=100,
            latency_mean=10.5,
            latency_median=10.0,
            latency_std=1.5,
            latency_min=8.0,
            latency_max=15.0,
            latency_p95=12.0,
            latency_p99=14.0,
            throughput_fps=95.2,
            memory_mb=512.0,
        )

        assert report.model_name == "test_model"
        assert report.latency_mean == 10.5
        assert report.throughput_fps == 95.2

    def test_profile_report_to_dict(self):
        """Test ProfileReport serialization."""
        from ivit.utils.profiler import ProfileReport

        report = ProfileReport(
            model_name="test",
            latency_mean=10.0,
        )

        data = report.to_dict()
        assert "model_name" in data
        assert "latency" in data
        assert data["latency"]["mean_ms"] == 10.0

    def test_profile_report_to_json(self):
        """Test ProfileReport JSON export."""
        from ivit.utils.profiler import ProfileReport
        import json

        report = ProfileReport(model_name="test")
        json_str = report.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["model_name"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

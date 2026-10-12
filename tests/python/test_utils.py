"""ivit.utils: profiler, video and metrics helpers."""

import json
import threading

import numpy as np
import pytest

import ivit
from ivit.utils import (
    HealthMonitor,
    MetricsCollector,
    MetricsExporter,
    Profiler,
    ProfileReport,
    VideoStream,
    VideoWriter,
)


needs_bindings = pytest.mark.skipif(
    not ivit.is_cpp_available(), reason="C++ bindings are not built"
)


# ------------------------------------------------------------------ profiler


@needs_bindings
def test_profiler_benchmarks_a_loaded_model_through_predict(tmp_path):
    from test_model_contract import model_file

    path = model_file(tmp_path / "cls.onnx", [1, 3, 8, 8], [[0.1, 0.7, 0.2]])
    model = ivit.load(str(path), device="cpu", task="classify", use_cache=False)

    report = Profiler().benchmark(model, iterations=3, warmup=1)

    assert report.iterations == 3
    assert report.input_shape == (8, 8, 3)  # the image fed to predict()
    assert report.device == "cpu" and report.backend == "openvino"
    assert report.precision == "fp32"
    assert 0 < report.latency_min <= report.latency_mean <= report.latency_max
    assert report.throughput_fps > 0
    assert json.loads(report.to_json())["iterations"] == 3


@needs_bindings
def test_profiler_sizes_the_image_from_an_explicit_shape(tmp_path):
    from test_model_contract import model_file

    path = model_file(tmp_path / "cls.onnx", [1, 3, 8, 8], [[0.1, 0.7, 0.2]])
    model = ivit.load(str(path), device="cpu", task="classify", use_cache=False)
    report = Profiler().benchmark(model, input_shape=(1, 3, 16, 24), iterations=1, warmup=0)
    assert report.input_shape == (16, 24, 3)


def test_profiler_timing_and_report_round_trip(tmp_path):
    profiler = Profiler()
    assert profiler.calculate_stats() == (0, 0, 0, 0, 0, 0, 0)
    for _ in range(3):
        profiler.start()
        profiler.stop()
    assert len(profiler.times) == 3 and profiler.elapsed_ms() >= 0
    profiler.stop()  # stop without start records nothing
    assert len(profiler.times) == 3

    out = tmp_path / "report.json"
    ProfileReport(model_name="m", iterations=3, latency_mean=2.0).save(str(out))
    data = json.loads(out.read_text())
    assert data["model_name"] == "m" and data["latency"]["mean_ms"] == 2.0


# ------------------------------------------------------------------ video


def test_video_writer_and_stream_round_trip(tmp_path):
    pytest.importorskip("cv2")
    path = str(tmp_path / "clip.avi")
    frame = np.full((48, 64, 3), 100, np.uint8)
    with VideoWriter(path, fps=10, resolution=(64, 48), codec="MJPG") as writer:
        for _ in range(4):
            writer.write(frame)
        writer.write(np.zeros((10, 10, 3), np.uint8))  # resized to the writer's resolution
        assert writer.frame_count == 5

    with VideoStream(path) as stream:
        assert (stream.width, stream.height) == (64, 48)
        frames = list(stream)
        assert len(frames) == 5 and frames[0].shape == (48, 64, 3)
        assert stream.current_frame == 5
    assert not stream.is_opened


def test_video_stream_rejects_a_missing_source(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(ValueError, match="Failed to open"):
        VideoStream(str(tmp_path / "missing.mp4"))


# ------------------------------------------------------------------ metrics


def test_metrics_collector_counts_and_percentiles():
    collector = MetricsCollector()
    for latency in (10.0, 20.0, 30.0):
        collector.record_inference(latency)
    collector.record_inference(999.0, success=False)

    metrics = collector.get_metrics()
    assert metrics.total_inferences == 4 and metrics.failed_inferences == 1
    assert metrics.avg_latency_ms == 20.0
    assert (metrics.min_latency_ms, metrics.max_latency_ms) == (10.0, 30.0)
    assert metrics.success_rate == 75.0
    assert collector.get_percentile(50) == 20.0

    exported = json.loads(MetricsExporter(collector).to_json())
    assert exported["p50_latency_ms"] == 20.0
    prometheus = MetricsExporter(collector).to_prometheus()
    assert 'ivit_inferences_total{status="failure"} 1' in prometheus

    collector.reset()
    assert collector.get_metrics().to_dict()["min_latency_ms"] == 0.0


def test_metrics_listeners_may_unsubscribe_while_being_notified():
    collector = MetricsCollector()
    seen = []

    def once(event):
        seen.append(event["latency_ms"])
        collector.remove_listener(once)

    collector.add_listener(once)
    collector.record_inference(1.0)
    collector.record_inference(2.0)
    assert seen == [1.0]


def test_metrics_collector_is_thread_safe():
    collector = MetricsCollector()
    threads = [
        threading.Thread(target=lambda: [collector.record_inference(1.0) for _ in range(500)])
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert collector.get_metrics().total_inferences == 2000


def test_health_monitor_reports_status_from_error_rate_and_latency():
    monitor = HealthMonitor(error_threshold=0.5, latency_threshold_ms=100.0)
    monitor.record_inference(10.0)
    assert monitor.is_healthy()
    monitor.record_inference(500.0)
    assert monitor.check_health()["status"] == "degraded"
    monitor.record_inference(0.0, success=False)
    monitor.record_inference(0.0, success=False)
    assert monitor.check_health()["status"] == "unhealthy"

"""
Metrics and monitoring utilities for iVIT-SDK.

Provides inference statistics, performance tracking, and observability.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import threading
import time
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Container for inference metrics."""

    # Counters
    total_inferences: int = 0
    successful_inferences: int = 0
    failed_inferences: int = 0

    # Timing (milliseconds)
    total_time_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0

    # Throughput
    start_time: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.successful_inferences == 0:
            return 0.0
        return self.total_time_ms / self.successful_inferences

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_inferences == 0:
            return 100.0
        return (self.successful_inferences / self.total_inferences) * 100

    @property
    def throughput_fps(self) -> float:
        """Throughput in inferences per second."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.successful_inferences / elapsed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_inferences": self.total_inferences,
            "successful_inferences": self.successful_inferences,
            "failed_inferences": self.failed_inferences,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0,
            "max_latency_ms": self.max_latency_ms,
            "success_rate": self.success_rate,
            "throughput_fps": self.throughput_fps,
        }


class MetricsCollector:
    """
    Collects and aggregates inference metrics.

    Thread-safe metrics collection for monitoring inference performance.

    Examples:
        >>> collector = MetricsCollector()
        >>>
        >>> # Record inference
        >>> collector.record_inference(latency_ms=15.5, success=True)
        >>>
        >>> # Get metrics
        >>> metrics = collector.get_metrics()
        >>> print(f"Avg latency: {metrics.avg_latency_ms:.2f} ms")
        >>> print(f"Throughput: {metrics.throughput_fps:.1f} FPS")
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector.

        Args:
            window_size: Size of sliding window for percentile calculations
        """
        self._lock = threading.Lock()
        self._metrics = InferenceMetrics()
        self._window_size = window_size
        self._latency_window: deque = deque(maxlen=window_size)
        self._listeners: List[Callable[[Dict], None]] = []

    def record_inference(
        self,
        latency_ms: float,
        success: bool = True,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Record an inference event.

        Args:
            latency_ms: Inference latency in milliseconds
            success: Whether inference succeeded
            metadata: Optional metadata
        """
        with self._lock:
            self._metrics.total_inferences += 1

            if success:
                self._metrics.successful_inferences += 1
                self._metrics.total_time_ms += latency_ms
                self._metrics.min_latency_ms = min(
                    self._metrics.min_latency_ms, latency_ms
                )
                self._metrics.max_latency_ms = max(
                    self._metrics.max_latency_ms, latency_ms
                )
                self._latency_window.append(latency_ms)
            else:
                self._metrics.failed_inferences += 1

        # Notify listeners
        event = {
            "type": "inference",
            "latency_ms": latency_ms,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._notify_listeners(event)

    def get_metrics(self) -> InferenceMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            return InferenceMetrics(
                total_inferences=self._metrics.total_inferences,
                successful_inferences=self._metrics.successful_inferences,
                failed_inferences=self._metrics.failed_inferences,
                total_time_ms=self._metrics.total_time_ms,
                min_latency_ms=self._metrics.min_latency_ms,
                max_latency_ms=self._metrics.max_latency_ms,
                start_time=self._metrics.start_time,
            )

    def get_percentile(self, percentile: float) -> float:
        """
        Get latency percentile.

        Args:
            percentile: Percentile value (0-100)

        Returns:
            Latency at given percentile
        """
        with self._lock:
            if not self._latency_window:
                return 0.0

            import numpy as np
            return float(np.percentile(list(self._latency_window), percentile))

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics = InferenceMetrics()
            self._latency_window.clear()

    def add_listener(self, callback: Callable[[Dict], None]) -> None:
        """
        Add event listener for real-time monitoring.

        Args:
            callback: Function called for each inference event
        """
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[Dict], None]) -> None:
        """Remove event listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _notify_listeners(self, event: Dict) -> None:
        """Notify all listeners of an event."""
        with self._lock:
            listeners = list(self._listeners)  # a listener may add or remove listeners
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning(f"Listener error: {e}")


class HealthMonitor:
    """
    Monitor model and system health.

    Tracks error rates, memory usage, and provides health status.

    Examples:
        >>> monitor = HealthMonitor()
        >>> monitor.check_health()
        {'status': 'healthy', 'error_rate': 0.0, ...}
    """

    def __init__(
        self,
        error_threshold: float = 0.1,
        latency_threshold_ms: float = 1000.0
    ):
        """
        Initialize health monitor.

        Args:
            error_threshold: Error rate threshold for unhealthy status (0-1)
            latency_threshold_ms: Latency threshold for degraded status
        """
        self._error_threshold = error_threshold
        self._latency_threshold = latency_threshold_ms
        self._metrics_collector = MetricsCollector()
        self._last_check = time.time()

    @property
    def metrics(self) -> MetricsCollector:
        """Get metrics collector."""
        return self._metrics_collector

    def record_inference(
        self,
        latency_ms: float,
        success: bool = True,
        metadata: Dict = None
    ) -> None:
        """Record inference for health tracking."""
        self._metrics_collector.record_inference(latency_ms, success, metadata)

    def check_health(self) -> Dict[str, Any]:
        """
        Check current health status.

        Returns:
            Health status dictionary with:
            - status: "healthy", "degraded", or "unhealthy"
            - error_rate: Current error rate
            - avg_latency_ms: Average latency
            - memory_usage_mb: Memory usage (if available)
        """
        metrics = self._metrics_collector.get_metrics()

        # Calculate error rate
        error_rate = 1 - (metrics.success_rate / 100) if metrics.total_inferences > 0 else 0

        # Determine status
        if error_rate >= self._error_threshold:
            status = "unhealthy"
        elif metrics.avg_latency_ms > self._latency_threshold:
            status = "degraded"
        else:
            status = "healthy"

        result = {
            "status": status,
            "error_rate": error_rate,
            "avg_latency_ms": metrics.avg_latency_ms,
            "total_inferences": metrics.total_inferences,
            "throughput_fps": metrics.throughput_fps,
            "timestamp": datetime.now().isoformat(),
        }

        # Try to get memory usage
        try:
            import psutil
            process = psutil.Process()
            result["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pass

        return result

    def is_healthy(self) -> bool:
        """Quick health check."""
        return self.check_health()["status"] == "healthy"


class MetricsExporter:
    """
    Export metrics in various formats.

    Supports JSON, Prometheus, and custom formats.

    Examples:
        >>> exporter = MetricsExporter(collector)
        >>> print(exporter.to_prometheus())
    """

    def __init__(self, collector: MetricsCollector):
        """
        Initialize exporter.

        Args:
            collector: Metrics collector to export from
        """
        self._collector = collector

    def to_json(self) -> str:
        """Export metrics as JSON."""
        metrics = self._collector.get_metrics()
        data = metrics.to_dict()
        data["p50_latency_ms"] = self._collector.get_percentile(50)
        data["p95_latency_ms"] = self._collector.get_percentile(95)
        data["p99_latency_ms"] = self._collector.get_percentile(99)
        data["timestamp"] = datetime.now().isoformat()
        return json.dumps(data, indent=2)

    def to_prometheus(self, prefix: str = "ivit") -> str:
        """
        Export metrics in Prometheus format.

        Args:
            prefix: Metric name prefix

        Returns:
            Prometheus-formatted metrics string
        """
        metrics = self._collector.get_metrics()

        lines = [
            f"# HELP {prefix}_inferences_total Total number of inferences",
            f"# TYPE {prefix}_inferences_total counter",
            f'{prefix}_inferences_total{{status="success"}} {metrics.successful_inferences}',
            f'{prefix}_inferences_total{{status="failure"}} {metrics.failed_inferences}',
            "",
            f"# HELP {prefix}_latency_ms Inference latency in milliseconds",
            f"# TYPE {prefix}_latency_ms gauge",
            f"{prefix}_latency_ms{{quantile=\"avg\"}} {metrics.avg_latency_ms:.3f}",
            f"{prefix}_latency_ms{{quantile=\"min\"}} {metrics.min_latency_ms if metrics.min_latency_ms != float('inf') else 0:.3f}",
            f"{prefix}_latency_ms{{quantile=\"max\"}} {metrics.max_latency_ms:.3f}",
            f'{prefix}_latency_ms{{quantile="0.5"}} {self._collector.get_percentile(50):.3f}',
            f'{prefix}_latency_ms{{quantile="0.95"}} {self._collector.get_percentile(95):.3f}',
            f'{prefix}_latency_ms{{quantile="0.99"}} {self._collector.get_percentile(99):.3f}',
            "",
            f"# HELP {prefix}_throughput_fps Inferences per second",
            f"# TYPE {prefix}_throughput_fps gauge",
            f"{prefix}_throughput_fps {metrics.throughput_fps:.3f}",
        ]

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        metrics = self._collector.get_metrics()
        return {
            **metrics.to_dict(),
            "p50_latency_ms": self._collector.get_percentile(50),
            "p95_latency_ms": self._collector.get_percentile(95),
            "p99_latency_ms": self._collector.get_percentile(99),
        }


# Global default metrics collector
_default_collector: Optional[MetricsCollector] = None


def get_default_collector() -> MetricsCollector:
    """Get the default global metrics collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


def record_inference(latency_ms: float, success: bool = True, **kwargs) -> None:
    """Record inference to default collector."""
    get_default_collector().record_inference(latency_ms, success, kwargs)


def get_metrics() -> InferenceMetrics:
    """Get metrics from default collector."""
    return get_default_collector().get_metrics()

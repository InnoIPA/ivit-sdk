"""
Performance profiling utilities.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import time
import json
import numpy as np


def _dummy_image(model, input_shape: Optional[Tuple[int, ...]]) -> np.ndarray:
    """A random BGR uint8 image sized for the model (dynamic dimensions fall back to 640)."""
    shape = tuple(input_shape) if input_shape else tuple(model.input_info[0].shape)
    h, w = 640, 640
    if len(shape) == 4:
        hw = shape[1:3] if shape[3] in (1, 3) else shape[2:4]  # NHWC or NCHW
    elif len(shape) == 3:
        hw = shape[0:2] if shape[2] in (1, 3) else shape[1:3]  # HWC or CHW
    else:
        hw = shape[-2:]
    if len(hw) == 2 and all(int(d) > 0 for d in hw):
        h, w = (int(d) for d in hw)
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


@dataclass
class ProfileReport:
    """Performance profile report."""

    model_name: str = ""
    device: str = ""
    backend: str = ""
    precision: str = ""
    input_shape: Tuple[int, ...] = ()
    iterations: int = 0

    # Latency (milliseconds)
    latency_mean: float = 0.0
    latency_median: float = 0.0
    latency_std: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Throughput
    throughput_fps: float = 0.0

    # Memory (not measured by benchmark(); fill it in yourself if you need it)
    memory_mb: float = 0.0

    def __str__(self) -> str:
        """Format report as string."""
        lines = [
            "",
            "=" * 60,
            "                   Benchmark Report",
            "=" * 60,
            f"  Model:        {self.model_name}",
            f"  Device:       {self.device}",
            f"  Backend:      {self.backend}",
            f"  Precision:    {self.precision}",
            f"  Input Shape:  {self.input_shape}",
            "-" * 60,
            "  Latency:",
            f"    Mean:       {self.latency_mean:.2f} ms",
            f"    Median:     {self.latency_median:.2f} ms",
            f"    Std:        {self.latency_std:.2f} ms",
            f"    Min:        {self.latency_min:.2f} ms",
            f"    Max:        {self.latency_max:.2f} ms",
            f"    P95:        {self.latency_p95:.2f} ms",
            f"    P99:        {self.latency_p99:.2f} ms",
            "-" * 60,
            f"  Throughput:   {self.throughput_fps:.1f} FPS",
            f"  Memory:       {f'{self.memory_mb:.1f} MB' if self.memory_mb else 'not measured'}",
            "=" * 60,
            "",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "backend": self.backend,
            "precision": self.precision,
            "input_shape": list(self.input_shape),
            "iterations": self.iterations,
            "latency": {
                "mean_ms": self.latency_mean,
                "median_ms": self.latency_median,
                "std_ms": self.latency_std,
                "min_ms": self.latency_min,
                "max_ms": self.latency_max,
                "p95_ms": self.latency_p95,
                "p99_ms": self.latency_p99,
            },
            "throughput_fps": self.throughput_fps,
            "memory_mb": self.memory_mb,
        }

    def save(self, path: str) -> None:
        """Save report to file."""
        with open(path, "w") as f:
            f.write(self.to_json())


class Profiler:
    """
    Performance profiler.

    Examples:
        >>> profiler = Profiler()
        >>> report = profiler.benchmark(model, input_shape=(1, 3, 640, 640))
        >>> print(report)
    """

    def __init__(self):
        self._times: List[float] = []
        self._start_time: Optional[float] = None

    def benchmark(
        self,
        model,
        input_shape: Optional[Tuple[int, ...]] = None,
        iterations: int = 100,
        warmup: int = 10,
        image: Optional[np.ndarray] = None,
    ) -> ProfileReport:
        """
        Run benchmark on model.

        Each iteration is a full ``model.predict()`` call (preprocessing, inference and
        decoding), the same path an application uses.

        Args:
            model: Loaded model (``ivit.load`` / ``ivit.zoo.load``)
            input_shape: Model input shape used to size a random image, e.g. (1, 3, 640, 640);
                defaults to the model's own input shape
            iterations: Number of iterations
            warmup: Warmup iterations
            image: BGR uint8 image to use instead of a random one

        Returns:
            Profile report
        """
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if image is None:
            image = _dummy_image(model, input_shape)

        for _ in range(warmup):
            model.predict(image)

        self.reset()
        for _ in range(iterations):
            self.start()
            model.predict(image)
            self.stop()

        mean, median, std, min_t, max_t, p95, p99 = self.calculate_stats()
        backend = model.backend
        return ProfileReport(
            model_name=model.name,
            device=model.device,
            backend=getattr(backend, "name", str(backend)).lower(),
            precision=model.load_config.precision,
            input_shape=tuple(image.shape),
            iterations=iterations,
            latency_mean=mean,
            latency_median=median,
            latency_std=std,
            latency_min=min_t,
            latency_max=max_t,
            latency_p95=p95,
            latency_p99=p99,
            throughput_fps=1000.0 / mean if mean > 0 else 0,
        )

    def start(self) -> None:
        """Start timing."""
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop timing and record."""
        if self._start_time is not None:
            elapsed = (time.perf_counter() - self._start_time) * 1000
            self._times.append(elapsed)
            self._start_time = None

    def elapsed_ms(self) -> float:
        """Get last elapsed time in milliseconds."""
        return self._times[-1] if self._times else 0.0

    def reset(self) -> None:
        """Reset profiler."""
        self._times = []
        self._start_time = None

    @property
    def times(self) -> List[float]:
        """Get all recorded times."""
        return self._times.copy()

    def calculate_stats(self) -> Tuple[float, float, float, float, float, float, float]:
        """
        Calculate statistics.

        Returns:
            (mean, median, std, min, max, p95, p99)
        """
        if not self._times:
            return (0, 0, 0, 0, 0, 0, 0)

        times = np.array(self._times)
        return (
            float(np.mean(times)),
            float(np.median(times)),
            float(np.std(times)),
            float(np.min(times)),
            float(np.max(times)),
            float(np.percentile(times, 95)),
            float(np.percentile(times, 99)),
        )

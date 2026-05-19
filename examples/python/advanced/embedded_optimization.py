#!/usr/bin/env python3
"""
iVIT-SDK Embedded Engineer Optimization Example

Target: Embedded engineers who need to optimize inference performance
        on edge devices.

Features demonstrated:
- Runtime configuration (OpenVINO, TensorRT)
- Custom pre/post processors
- Performance benchmarking
- Model warmup

Usage:
    python embedded_optimization.py
    python embedded_optimization.py --model path/to/model.onnx --device cuda:0
    python embedded_optimization.py --benchmark --iterations 100
"""

import argparse
import time

import numpy as np

import ivit
from ivit.core.processors import (
    BasePreProcessor,
    BasePostProcessor,
    get_preprocessor,
    get_postprocessor,
    register_preprocessor,
    register_postprocessor,
)
from ivit.core.result import Results
from ivit.core.types import Detection, BBox


def configure_runtime(model, backend: str):
    """Step 1: Configure runtime for specific hardware."""
    print("=" * 60)
    print("Step 1: Runtime Configuration")
    print("=" * 60)

    if backend == "openvino":
        # Intel hardware optimization
        print("Configuring OpenVINO (Intel hardware)...")
        model.configure_openvino(
            performance_mode="LATENCY",      # LATENCY or THROUGHPUT
            num_streams=1,                   # Number of inference streams
            inference_precision="FP16",      # Precision
            enable_cpu_pinning=True,         # CPU core pinning
        )
        print("  performance_mode: LATENCY")
        print("  num_streams: 1")
        print("  inference_precision: FP16")
        print("  enable_cpu_pinning: True")

    elif backend == "tensorrt":
        # NVIDIA hardware optimization
        print("Configuring TensorRT (NVIDIA hardware)...")
        model.configure_tensorrt(
            workspace_size=1 << 30,          # 1GB workspace
            enable_fp16=True,                # Enable FP16
            enable_int8=False,               # INT8 requires calibration data
            dla_core=-1,                     # DLA core (for Jetson)
            builder_optimization_level=3,   # Optimization level (0-5)
            enable_sparsity=True,            # Sparse acceleration
        )
        print("  workspace_size: 1GB")
        print("  enable_fp16: True")
        print("  builder_optimization_level: 3")
        print("  enable_sparsity: True")

    # [Future] Add new backend configurations here:
    # elif backend == "new_backend":
    #     model.configure_new_backend(...)

    return model


def warmup_model(model, iterations: int = 10):
    """Step 2: Model warmup (critical for accurate benchmarking)."""
    print("\n" + "=" * 60)
    print("Step 2: Model Warmup")
    print("=" * 60)

    print(f"Running {iterations} warmup iterations...")
    model.warmup(iterations=iterations)
    print("Warmup completed!")
    print("\nNote: First few inferences are typically slower due to:")
    print("  - JIT compilation")
    print("  - Memory allocation")
    print("  - CUDA kernel loading")


def benchmark_preprocessors():
    """Step 3: Benchmark preprocessor performance."""
    print("\n" + "=" * 60)
    print("Step 3: Preprocessor Benchmarking")
    print("=" * 60)

    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Get built-in preprocessors
    letterbox = get_preprocessor("letterbox")
    center_crop = get_preprocessor("center_crop")

    def benchmark(preprocessor, image, iterations=100):
        """Benchmark a preprocessor."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = preprocessor(image)
            times.append((time.perf_counter() - start) * 1000)

        return {
            "mean": np.mean(times),
            "min": np.min(times),
            "max": np.max(times),
            "std": np.std(times),
        }

    # Benchmark Letterbox
    print("Benchmarking Letterbox preprocessor...")
    letterbox_stats = benchmark(letterbox, test_image)
    print(f"  Mean: {letterbox_stats['mean']:.3f} ms")
    print(f"  Min: {letterbox_stats['min']:.3f} ms")
    print(f"  Max: {letterbox_stats['max']:.3f} ms")
    print(f"  Std: {letterbox_stats['std']:.3f} ms")

    # Benchmark CenterCrop
    print("\nBenchmarking CenterCrop preprocessor...")
    center_crop_stats = benchmark(center_crop, test_image)
    print(f"  Mean: {center_crop_stats['mean']:.3f} ms")
    print(f"  Min: {center_crop_stats['min']:.3f} ms")
    print(f"  Max: {center_crop_stats['max']:.3f} ms")
    print(f"  Std: {center_crop_stats['std']:.3f} ms")

    return letterbox_stats, center_crop_stats


class CustomPreProcessor(BasePreProcessor):
    """Step 4: Custom preprocessor example."""

    def __init__(self, target_size=(640, 640), normalize=True):
        self.target_size = target_size
        self.normalize = normalize

    def process(
        self,
        image: np.ndarray,
        target_size: tuple = None,
        **kwargs
    ) -> tuple:
        """
        Pre-process image for inference.

        Args:
            image: Input image (BGR, HWC format)
            target_size: Target size (height, width), uses self.target_size if None
            **kwargs: Additional options

        Returns:
            Tuple of (processed_tensor, preprocess_info)
        """
        import cv2

        if target_size is None:
            target_size = self.target_size

        orig_h, orig_w = image.shape[:2]

        # 1. Resize
        resized = cv2.resize(image, target_size)

        # 2. BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. Normalize
        if self.normalize:
            rgb = rgb.astype(np.float32) / 255.0

        # 4. HWC to NCHW
        transposed = np.transpose(rgb, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        # Build preprocess info for postprocessor
        preprocess_info = {
            "orig_size": (orig_h, orig_w),
            "target_size": target_size,
            "scale": (target_size[0] / orig_w, target_size[1] / orig_h),
        }

        return batched, preprocess_info


class CustomPostProcessor(BasePostProcessor):
    """Step 4: Custom postprocessor example (YOLO output parsing)."""

    def __init__(self, conf_threshold=0.5, iou_threshold=0.45, class_names=None):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []

    def process(
        self,
        outputs: dict,
        orig_size: tuple,
        preprocess_info: dict = None,
        config=None,
        labels: list = None,
    ) -> Results:
        """
        Post-process model outputs.

        Args:
            outputs: Raw model outputs
            orig_size: Original image size (height, width)
            preprocess_info: Info from preprocessing
            config: Inference configuration (uses self.conf_threshold if None)
            labels: Class labels (uses self.class_names if None)

        Returns:
            Results object
        """
        results = Results()
        results.image_size = orig_size

        if preprocess_info is None:
            preprocess_info = {}
        if labels is None:
            labels = self.class_names

        # Get confidence threshold from config or use default
        conf_threshold = self.conf_threshold
        if config is not None and hasattr(config, 'conf_threshold'):
            conf_threshold = config.conf_threshold

        # Parse model output (example)
        predictions = outputs.get("output", outputs[list(outputs.keys())[0]])

        # Filter low confidence predictions
        for pred in predictions:
            confidence = float(pred[4])
            if confidence < conf_threshold:
                continue

            class_id = int(pred[5])
            label = (
                labels[class_id]
                if class_id < len(labels)
                else f"class_{class_id}"
            )

            det = Detection(
                bbox=BBox(pred[0], pred[1], pred[2], pred[3]),
                class_id=class_id,
                label=label,
                confidence=confidence
            )
            results.detections.append(det)

        # NMS
        results.detections = self._nms(results.detections, self.iou_threshold)

        return results

    def _nms(self, detections, iou_threshold):
        """Simple NMS implementation."""
        if not detections:
            return detections

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if d.class_id != best.class_id or best.bbox.iou(d.bbox) < iou_threshold
            ]

        return keep


def demonstrate_custom_processors():
    """Step 4: Demonstrate custom pre/post processors."""
    print("\n" + "=" * 60)
    print("Step 4: Custom Processors")
    print("=" * 60)

    # Register custom preprocessor
    register_preprocessor("custom", CustomPreProcessor)
    print("Registered custom preprocessor: 'custom'")

    # Register custom postprocessor
    register_postprocessor("custom_yolo", CustomPostProcessor)
    print("Registered custom postprocessor: 'custom_yolo'")

    print("\nUsage:")
    print("  model.set_preprocessor(CustomPreProcessor(target_size=(416, 416)))")
    print("  model.set_postprocessor(CustomPostProcessor(conf_threshold=0.6))")

    # Create and test custom preprocessor
    custom_pre = CustomPreProcessor(target_size=(640, 640))
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Call preprocessor (returns tuple of tensor and preprocess_info)
    tensor, preprocess_info = custom_pre(test_image)
    print(f"\nCustom preprocessor output:")
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Expected: (1, 3, 640, 640) for NCHW format")
    print(f"  Preprocess info: {preprocess_info}")


def benchmark_inference(model, iterations: int = 100):
    """Step 5: Benchmark inference performance."""
    print("\n" + "=" * 60)
    print("Step 5: Inference Benchmarking")
    print("=" * 60)

    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    print(f"Warmup: 10 iterations")
    for _ in range(10):
        _ = model(test_image)

    # Benchmark
    print(f"Benchmarking: {iterations} iterations")
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        _ = model(test_image)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{iterations}")

    # Calculate statistics
    mean_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)
    fps = 1000 / mean_time

    print("\nBenchmark Results:")
    print(f"  Mean latency: {mean_time:.2f} ms")
    print(f"  Min latency: {min_time:.2f} ms")
    print(f"  Max latency: {max_time:.2f} ms")
    print(f"  Std deviation: {std_time:.2f} ms")
    print(f"  P95 latency: {p95_time:.2f} ms")
    print(f"  P99 latency: {p99_time:.2f} ms")
    print(f"  Throughput: {fps:.1f} FPS")

    return {
        "mean_ms": mean_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "std_ms": std_time,
        "p95_ms": p95_time,
        "p99_ms": p99_time,
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="iVIT-SDK Embedded Engineer Optimization Example"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file. If not provided, uses Model Zoo."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (default: auto)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openvino", "tensorrt"],
        default="tensorrt",
        help="Backend to configure (default: tensorrt)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("iVIT-SDK Embedded Engineer Optimization Example")
    print("=" * 60)
    print(f"SDK Version: {ivit.__version__}")

    # Select device
    if args.device == "auto":
        device = ivit.devices.best()
        print(f"Auto-selected device: {device.id} ({device.name})")
    else:
        device = args.device
        print(f"Using device: {device}")

    # Load model
    print("\nLoading model...")
    if args.model:
        model = ivit.load(args.model, device=device)
    else:
        model = ivit.zoo.load("yolox-s", device=device)
    print("Model loaded successfully!")

    # Step 1: Configure runtime
    configure_runtime(model, args.backend)

    # Step 2: Warmup
    warmup_model(model, iterations=10)

    # Step 3: Benchmark preprocessors
    benchmark_preprocessors()

    # Step 4: Custom processors
    demonstrate_custom_processors()

    # Step 5: Benchmark inference
    if args.benchmark:
        benchmark_inference(model, iterations=args.iterations)

    print("\n" + "=" * 60)
    print("Embedded Best Practices:")
    print("=" * 60)
    print("1. Always run model.warmup() before benchmarking")
    print("2. Use FP16 quantization for most cases (minimal accuracy loss)")
    print("3. Configure backend based on hardware:")
    print("   - Intel: OpenVINO with LATENCY mode")
    print("   - NVIDIA: TensorRT with FP16 and CUDA Graph")
    print("4. Monitor preprocessing time (can be 30%+ of total)")
    print("5. Use custom processors for non-standard models")


if __name__ == "__main__":
    main()

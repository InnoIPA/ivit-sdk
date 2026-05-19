#!/usr/bin/env python3
"""
iVIT-SDK Backend Engineer Service Example

Target: Backend engineers who need to build stable AI inference services
        with monitoring and high concurrency support.

Features demonstrated:
- Callback system for monitoring
- FPS and latency tracking
- REST API service (FastAPI)
- Health checks and metrics

Usage:
    # Run standalone demo
    python backend_service.py

    # Run as REST API service
    python backend_service.py --serve --port 8080

    # With custom model
    python backend_service.py --serve --model path/to/model.onnx
"""

import argparse
import sys
import time
from typing import Optional

import ivit
from ivit.core.callbacks import (
    CallbackManager,
    CallbackContext,
    FPSCounter,
    LatencyLogger,
)


class PrometheusMetricsCallback:
    """Custom callback for Prometheus metrics integration."""

    def __init__(self):
        self.inference_count = 0
        self.total_latency = 0
        self.error_count = 0

    def __call__(self, ctx: CallbackContext):
        self.inference_count += 1
        self.total_latency += ctx.latency_ms

        # In production, send to Prometheus:
        # prometheus_client.Counter('inference_total').inc()
        # prometheus_client.Histogram('inference_latency').observe(ctx.latency_ms)

    @property
    def average_latency(self):
        if self.inference_count == 0:
            return 0
        return self.total_latency / self.inference_count


class AlertCallback:
    """Custom callback for alerting on high latency."""

    def __init__(self, threshold_ms: float = 100):
        self.threshold_ms = threshold_ms
        self.alert_count = 0

    def __call__(self, ctx: CallbackContext):
        if ctx.latency_ms > self.threshold_ms:
            self.alert_count += 1
            print(f"[ALERT] High latency warning: {ctx.latency_ms:.1f}ms "
                  f"(threshold: {self.threshold_ms}ms)")
            # In production: send Slack/Email notification


def setup_callbacks():
    """Step 1: Setup monitoring callbacks."""
    print("=" * 60)
    print("Step 1: Callback System Setup")
    print("=" * 60)

    callback_manager = CallbackManager()

    # Built-in: FPS Counter
    fps_counter = FPSCounter(window_size=30)
    callback_manager.register("infer_end", fps_counter)
    print("Registered: FPSCounter (window_size=30)")

    # Built-in: Latency Logger
    latency_logger = LatencyLogger()
    callback_manager.register("infer_end", latency_logger)
    print("Registered: LatencyLogger")

    # Custom: Prometheus Metrics
    prometheus_callback = PrometheusMetricsCallback()
    callback_manager.register("infer_end", prometheus_callback)
    print("Registered: PrometheusMetricsCallback")

    # Custom: Alert System
    alert_callback = AlertCallback(threshold_ms=100)
    callback_manager.register("infer_end", alert_callback)
    print("Registered: AlertCallback (threshold=100ms)")

    print("\nAvailable callback events:")
    print("  pre_process    - Before preprocessing")
    print("  post_process   - After postprocessing")
    print("  infer_start    - Inference starts")
    print("  infer_end      - Inference ends (with latency)")
    print("  batch_start    - Batch processing starts")
    print("  batch_end      - Batch processing ends")
    print("  stream_start   - Stream processing starts")
    print("  stream_frame   - Each frame in stream")
    print("  stream_end     - Stream processing ends")
    print("  model_loaded   - Model loaded")
    print("  model_unloaded - Model unloaded")

    return callback_manager, fps_counter, latency_logger, prometheus_callback, alert_callback


def inference_with_monitoring(
    model,
    image,
    callback_manager,
    fps_counter,
    prometheus_callback,
):
    """Step 2: Run inference with monitoring."""
    # Trigger infer_start event
    ctx = CallbackContext(event="infer_start", model_name="yolox-s")
    callback_manager.trigger("infer_start", ctx)

    # Execute inference
    start = time.perf_counter()
    results = model(image)
    latency = (time.perf_counter() - start) * 1000

    # Trigger infer_end event
    ctx = CallbackContext(
        event="infer_end",
        model_name="yolox-s",
        latency_ms=latency,
        detections=len(results)
    )
    callback_manager.trigger("infer_end", ctx)

    return results


def demonstrate_cli_tools():
    """Step 3: Demonstrate CLI tools for backend operations."""
    print("\n" + "=" * 60)
    print("Step 3: CLI Tools")
    print("=" * 60)

    print("Available CLI commands for backend engineers:")
    print()
    print("# System information")
    print("  ivit info")
    print()
    print("# List available devices")
    print("  ivit devices")
    print()
    print("# Performance benchmark")
    print("  ivit benchmark model.onnx --device cuda:0 --iterations 100")
    print()
    print("# Run inference")
    print("  ivit predict model.onnx image.jpg --output result.jpg")
    print()
    print("# Model conversion")
    print("  ivit convert model.onnx model.engine --format tensorrt --fp16")
    print()
    print("# Start REST API service")
    print("  ivit serve model.onnx --port 8080 --device cuda:0")
    print()
    print("# Model Zoo operations")
    print("  ivit zoo list")
    print("  ivit zoo search yolo")
    print("  ivit zoo download yolox-s")


def run_demo():
    """Run standalone demo without REST API."""
    print("=" * 60)
    print("iVIT-SDK Backend Engineer Service Demo")
    print("=" * 60)
    print(f"SDK Version: {ivit.__version__}")

    # Setup callbacks
    (callback_manager, fps_counter, latency_logger,
     prometheus_callback, alert_callback) = setup_callbacks()

    # Load model
    print("\n" + "=" * 60)
    print("Step 2: Model Loading")
    print("=" * 60)

    device = ivit.devices.best()
    print(f"Selected device: {device.id} ({device.name})")

    model = ivit.zoo.load("yolox-s", device=device)
    model.warmup(10)
    print("Model loaded and warmed up!")

    # Run demo inferences
    print("\n" + "=" * 60)
    print("Step 3: Demo Inferences")
    print("=" * 60)

    import numpy as np
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("Running 20 demo inferences...")
    for i in range(20):
        results = inference_with_monitoring(
            model, test_image,
            callback_manager, fps_counter, prometheus_callback
        )
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/20, Current FPS: {fps_counter.fps:.1f}")

    # Show statistics
    print("\n" + "=" * 60)
    print("Step 4: Statistics")
    print("=" * 60)
    print(f"Total inferences: {prometheus_callback.inference_count}")
    print(f"Current FPS: {fps_counter.fps:.1f}")
    print(f"Average latency: {prometheus_callback.average_latency:.1f} ms")
    print(f"High latency alerts: {alert_callback.alert_count}")

    # Show CLI tools
    demonstrate_cli_tools()

    print("\n" + "=" * 60)
    print("Backend Best Practices:")
    print("=" * 60)
    print("1. Use callbacks for monitoring without modifying core logic")
    print("2. Implement health checks for load balancers")
    print("3. Track FPS and latency for SLA compliance")
    print("4. Use warmup before serving production traffic")
    print("5. Consider connection pooling for high concurrency")


def create_fastapi_app(model_path: Optional[str] = None):
    """Create FastAPI application for REST API service."""
    try:
        from fastapi import FastAPI, File, UploadFile, HTTPException
        from fastapi.responses import JSONResponse
        import numpy as np
        import cv2
    except ImportError:
        print("Error: FastAPI not installed.")
        print("Install with: pip install fastapi uvicorn python-multipart")
        sys.exit(1)

    app = FastAPI(
        title="iVIT Inference Service",
        description="REST API for AI inference using iVIT-SDK",
        version=ivit.__version__,
    )

    # Global state
    state = {
        "model": None,
        "fps_counter": None,
        "latency_logger": None,
        "inference_count": 0,
    }

    @app.on_event("startup")
    async def startup():
        """Initialize model on startup."""
        print("Starting iVIT Inference Service...")

        # Select best device
        device = ivit.devices.best()
        print(f"Selected device: {device.id} ({device.name})")

        # Load model
        if model_path:
            state["model"] = ivit.load(model_path, device=device)
        else:
            state["model"] = ivit.zoo.load("yolox-s", device=device)

        # Warmup
        state["model"].warmup(10)

        # Setup callbacks
        state["fps_counter"] = FPSCounter(window_size=100)
        state["latency_logger"] = LatencyLogger()
        state["model"].on("infer_end", state["fps_counter"])
        state["model"].on("infer_end", state["latency_logger"])

        print(f"Model loaded to {state['model'].device}")

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        """Execute object detection on uploaded image."""
        try:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            # Inference
            results = state["model"](image)
            state["inference_count"] += 1

            return JSONResponse({
                "success": True,
                "detections": results.to_dict().get("detections", []),
                "inference_time_ms": results.inference_time_ms,
                "current_fps": state["fps_counter"].fps,
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": state["model"] is not None,
            "device": str(state["model"].device) if state["model"] else None,
        }

    @app.get("/stats")
    async def stats():
        """Performance statistics endpoint."""
        return {
            "inference_count": state["inference_count"],
            "current_fps": state["fps_counter"].fps if state["fps_counter"] else 0,
            "average_latency_ms": (
                state["latency_logger"].average_latency
                if state["latency_logger"] else 0
            ),
            "device": str(state["model"].device) if state["model"] else None,
        }

    @app.get("/devices")
    async def devices():
        """List available devices."""
        device_list = []
        for device in ivit.devices():
            device_list.append({
                "id": device.id,
                "name": device.name,
                "backend": device.backend,
                "type": device.type,
            })
        return {"devices": device_list}

    return app


def run_server(port: int, model_path: Optional[str] = None):
    """Run FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed.")
        print("Install with: pip install uvicorn")
        sys.exit(1)

    app = create_fastapi_app(model_path)

    print("=" * 60)
    print("iVIT Inference Service")
    print("=" * 60)
    print(f"SDK Version: {ivit.__version__}")
    print(f"Starting server on port {port}...")
    print()
    print("API Endpoints:")
    print(f"  POST /predict - Run inference on uploaded image")
    print(f"  GET  /health  - Health check")
    print(f"  GET  /stats   - Performance statistics")
    print(f"  GET  /devices - List available devices")
    print()
    print("Example usage:")
    print(f"  curl -X POST http://localhost:{port}/predict -F 'file=@image.jpg'")
    print(f"  curl http://localhost:{port}/health")
    print(f"  curl http://localhost:{port}/stats")
    print()

    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    parser = argparse.ArgumentParser(
        description="iVIT-SDK Backend Engineer Service Example"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as REST API service"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file. If not provided, uses Model Zoo."
    )
    args = parser.parse_args()

    if args.serve:
        run_server(args.port, args.model)
    else:
        run_demo()


if __name__ == "__main__":
    main()

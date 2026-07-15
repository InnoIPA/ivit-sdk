"""
iVIT-SDK Command Line Interface.

Usage:
    ivit info              Show system and SDK information
    ivit devices           List available inference devices
    ivit benchmark         Run performance benchmark
    ivit predict           Run inference on image/video
    ivit convert           Convert model format
    ivit export            Export model with optimizations
    ivit serve             Start inference server

Examples:
    $ ivit info
    $ ivit devices
    $ ivit benchmark models/yolox-s.onnx --device cuda:0
    $ ivit predict models/yolox-s.onnx image.jpg --conf 0.5
    $ ivit convert models/yolox-s.onnx -f openvino -o models/
    $ ivit serve models/yolox-s.onnx --port 8080
"""

import argparse
import sys
import time
from pathlib import Path


def cmd_info(args):
    """Show system and SDK information."""
    import ivit
    import platform

    print("=" * 60)
    print("iVIT-SDK System Information")
    print("=" * 60)
    print()

    # SDK Info
    print("SDK:")
    print(f"  Version:        {ivit.__version__}")
    print(f"  C++ Bindings:   {'Available' if ivit.is_cpp_available() else 'Not available'}")
    print()

    # System Info
    print("System:")
    print(f"  Platform:       {platform.system()} {platform.release()}")
    print(f"  Architecture:   {platform.machine()}")
    print(f"  Python:         {platform.python_version()}")
    print()

    # Backend availability
    print("Backends:")
    # OpenVINO inference is provided by the C++ backend (_ivit_core), not the
    # Python openvino package, which the project no longer depends on.
    try:
        import ivit
        if ivit.is_cpp_available():
            print("  OpenVINO:       via C++ backend (_ivit_core)")
        else:
            print("  OpenVINO:       C++ backend not built")
    except Exception:
        print("  OpenVINO:       Unknown")

    try:
        import onnxruntime
        print(f"  ONNX Runtime:   {onnxruntime.__version__}")
    except ImportError:
        print("  ONNX Runtime:   Not installed")

    try:
        import tensorrt
        print(f"  TensorRT:       {tensorrt.__version__}")
    except ImportError:
        print("  TensorRT:       Not installed")

    print()

    # Devices summary
    print("Devices:")
    device_list = ivit.list_devices()
    for d in device_list:
        print(f"  [{d.id}] {d.name} ({d.backend})")

    print()
    print("=" * 60)


def cmd_devices(args):
    """List available inference devices."""
    import ivit

    device_list = ivit.list_devices()

    if args.json:
        import json
        data = [
            {
                "id": d.id,
                "name": d.name,
                "type": d.type,
                "backend": d.backend,
                "available": d.is_available,
            }
            for d in device_list
        ]
        print(json.dumps(data, indent=2))
        return

    print()
    print("Available Devices:")
    print("-" * 60)
    print(f"{'ID':<12} {'Name':<30} {'Backend':<12} {'Type':<8}")
    print("-" * 60)

    for d in device_list:
        status = "OK" if d.is_available else "N/A"
        print(f"{d.id:<12} {d.name[:28]:<30} {d.backend:<12} {d.type:<8}")

    print("-" * 60)
    print(f"Total: {len(device_list)} device(s)")
    print()

    # Show best device
    best = ivit.devices.best()
    print(f"Best device (performance): {best.id}")
    best_eff = ivit.devices.best("efficiency")
    print(f"Best device (efficiency):  {best_eff.id}")
    print()


def cmd_benchmark(args):
    """Run performance benchmark."""
    import ivit
    import numpy as np

    model_path = args.model
    device = args.device
    iterations = args.iterations
    warmup = args.warmup

    print()
    print("=" * 60)
    print("iVIT-SDK Benchmark")
    print("=" * 60)
    print()
    print(f"Model:      {model_path}")
    print(f"Device:     {device}")
    print(f"Iterations: {iterations}")
    print(f"Warmup:     {warmup}")
    print()

    # Load model
    print("Loading model...", end=" ", flush=True)
    start = time.time()
    model = ivit.load(model_path, device=device)
    load_time = time.time() - start
    print(f"Done ({load_time:.2f}s)")
    print()

    # Get input shape
    input_shape = model.input_info[0].shape
    print(f"Input shape: {input_shape}")
    print()

    # Create dummy input
    if len(input_shape) == 4:
        _, c, h, w = input_shape
        dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    print(f"Warming up ({warmup} iterations)...", end=" ", flush=True)
    for _ in range(warmup):
        model.predict(dummy_image)
    print("Done")
    print()

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    latencies = []

    for i in range(iterations):
        start = time.perf_counter()
        results = model.predict(dummy_image)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{iterations} completed")

    print()

    # Statistics
    latencies = np.array(latencies)
    print("-" * 40)
    print("Results:")
    print("-" * 40)
    print(f"  Mean:     {np.mean(latencies):.2f} ms")
    print(f"  Std:      {np.std(latencies):.2f} ms")
    print(f"  Min:      {np.min(latencies):.2f} ms")
    print(f"  Max:      {np.max(latencies):.2f} ms")
    print(f"  P50:      {np.percentile(latencies, 50):.2f} ms")
    print(f"  P90:      {np.percentile(latencies, 90):.2f} ms")
    print(f"  P99:      {np.percentile(latencies, 99):.2f} ms")
    print(f"  FPS:      {1000 / np.mean(latencies):.1f}")
    print("-" * 40)
    print()


def cmd_predict(args):
    """Run inference on image/video."""
    import ivit

    model_path = args.model
    source = args.source
    device = args.device
    conf = args.conf
    iou = args.iou
    save = args.save
    show = args.show

    print()
    print(f"Model:  {model_path}")
    print(f"Source: {source}")
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading model...", end=" ", flush=True)
    model = ivit.load(model_path, device=device)
    print("Done")
    print()

    # Check if source is video
    source_path = Path(source)
    is_video = source_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')
    is_camera = source.isdigit()

    if is_video or is_camera:
        # Stream inference
        from ivit.core import InferConfig
        cfg = InferConfig()
        cfg.conf_threshold = conf
        cfg.iou_threshold = iou
        print(f"Streaming inference...")
        print("-" * 40)

        frame_count = 0
        for stream_result in model.stream(str(source), cfg):
            frame_count += 1
            res = stream_result.results
            print(f"Frame {frame_count}: {len(res.detections)} detections, {res.inference_time_ms:.1f}ms")

        print("-" * 40)
        print(f"Total frames: {frame_count}")
    else:
        # Image inference
        print("Running inference...")
        from ivit.core import InferConfig
        cfg = InferConfig()
        cfg.conf_threshold = conf
        cfg.iou_threshold = iou
        results = model.predict(source, cfg)

        print()
        print(f"Inference time: {results.inference_time_ms:.2f} ms")
        print(f"Detections: {len(results)}")
        print()

        for i, det in enumerate(results.detections):
            print(f"  [{i}] {det.label}: {det.confidence:.2%} @ [{det.bbox.x1:.0f}, {det.bbox.y1:.0f}, {det.bbox.x2:.0f}, {det.bbox.y2:.0f}]")

        if save or show:
            import cv2
            image = cv2.imread(str(source))
            vis = results.visualize(image) if image is not None else None

            if save and vis is not None:
                output_path = source_path.stem + "_result" + source_path.suffix
                cv2.imwrite(output_path, vis)
                print()
                print(f"Saved to: {output_path}")

            if show and vis is not None:
                cv2.imshow("iVIT-SDK", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    print()


def cmd_convert(args):
    """Convert model to different format."""
    import shutil

    model_path = args.model
    output_format = args.format.lower()
    output_dir = Path(args.output) if args.output else Path(".")
    precision = args.precision
    calibration_data = args.calibration_data

    print()
    print("=" * 60)
    print("iVIT-SDK Model Converter")
    print("=" * 60)
    print()
    print(f"Input:    {model_path}")
    print(f"Format:   {output_format}")
    print(f"Output:   {output_dir}")
    print(f"Precision: {precision}")
    print()

    # INT8 validation
    if precision == "int8" and output_format == "openvino":
        if calibration_data is None:
            print("Error: INT8 quantization requires calibration data.")
            print("Usage: ivit convert model.onnx -f openvino -p int8 -c ./calibration_images/")
            sys.exit(1)
        cal_path = Path(calibration_data)
        if not cal_path.exists() or not cal_path.is_dir():
            print(f"Error: Calibration data path is not a valid directory: {calibration_data}")
            sys.exit(1)

    if calibration_data is not None and precision != "int8":
        print(f"Warning: --calibration-data is only used with INT8 precision, ignoring.")
        print()

    # Verify input exists
    input_path = Path(model_path)
    if not input_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert based on format
    if output_format == "openvino":
        _convert_to_openvino(input_path, output_dir, precision,
                             calibration_data_path=calibration_data)
    elif output_format == "tensorrt":
        _convert_to_tensorrt(input_path, output_dir, precision)
    elif output_format == "onnx":
        _convert_to_onnx(input_path, output_dir)
    else:
        print(f"Error: Unsupported format: {output_format}")
        print("Supported formats: openvino, tensorrt, onnx")
        sys.exit(1)

    print()
    print("=" * 60)


def _get_onnx_input_shape(input_path: Path):
    """Get input shape (H, W) from an ONNX model.

    Returns (H, W) tuple or None if shape cannot be determined.
    """
    # Strategy 1: Try onnx package
    try:
        import onnx
        model = onnx.load(str(input_path))
        inp = model.graph.input[0]
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 4:
            h = dims[2].dim_value
            w = dims[3].dim_value
            if h > 0 and w > 0:
                return (h, w)
    except Exception:
        pass

    return None


def _load_calibration_images(cal_dir: Path, input_hw):
    """Load and preprocess calibration images from a directory.

    Args:
        cal_dir: Directory containing calibration images.
        input_hw: Tuple of (H, W) for model input size.

    Returns:
        List of np.ndarray in CHW float32 [0,1] format (no batch dim).
    """
    import numpy as np
    import cv2
    import random

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    MAX_IMAGES = 300

    h, w = input_hw

    # Collect image paths (top-level only, no recursion)
    image_paths = sorted([
        p for p in cal_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        print(f"Error: No images found in {cal_dir}")
        print(f"  Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        sys.exit(1)

    # Sample if too many
    if len(image_paths) > MAX_IMAGES:
        total = len(image_paths)
        rng = random.Random(42)
        image_paths = rng.sample(image_paths, MAX_IMAGES)
        print(f"  Sampled {MAX_IMAGES} images from {total} (seed=42)")

    print(f"  Loading {len(image_paths)} calibration images...")

    images = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        images.append(img)

    if not images:
        print(f"Error: Could not read any valid images from {cal_dir}")
        sys.exit(1)

    print(f"  Loaded {len(images)} calibration images ({h}x{w})")
    return images


def _convert_to_openvino_int8(input_path: Path, output_dir: Path,
                               calibration_data_path: str):
    """Quantize an ONNX model to INT8 using NNCF's ONNX backend.

    Produces a quantized ONNX model (with QuantizeLinear/DequantizeLinear
    nodes). The C++ OpenVINO backend loads this ONNX directly for INT8
    inference, so no Python ``openvino`` package is required.
    """
    import numpy as np

    # Check dependencies (nncf ONNX backend needs onnx + nncf, not openvino)
    try:
        import onnx
    except ImportError:
        print("Error: onnx is required for INT8 quantization.")
        print("Install with: pip install onnx>=1.14")
        sys.exit(1)

    try:
        import nncf
    except ImportError:
        print("Error: NNCF is required for INT8 quantization.")
        print("Install with: pip install nncf>=2.7")
        print("Or install all quantization dependencies: pip install -e '.[quantize]'")
        sys.exit(1)

    model_name = input_path.stem
    output_onnx = output_dir / f"{model_name}_int8.onnx"

    print("Quantizing to INT8 (NNCF ONNX backend)...")
    print()

    # Load ONNX model and determine its input tensor name
    onnx_model = onnx.load(str(input_path))
    input_name = onnx_model.graph.input[0].name

    # Get input shape for calibration preprocessing
    input_hw = _get_onnx_input_shape(input_path)
    if input_hw is None:
        input_hw = (640, 640)
        print(f"  Warning: Could not determine input shape, using default {input_hw}")

    # Load calibration images (each returned as (C, H, W) float32, no batch dim)
    cal_dir = Path(calibration_data_path)
    cal_images = _load_calibration_images(cal_dir, input_hw)

    # Build NNCF dataset; transform each sample into a {input_name: NCHW} feed dict
    def transform_fn(image):
        return {input_name: np.expand_dims(image, axis=0)}

    calibration_dataset = nncf.Dataset(cal_images, transform_fn)

    # Quantize
    print("  Running INT8 quantization (this may take a while)...")
    quantized_model = nncf.quantize(
        onnx_model,
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,
    )

    # Save quantized ONNX
    onnx.save(quantized_model, str(output_onnx))

    print()
    print(f"  Created: {output_onnx}")
    print()
    print("INT8 quantization complete!")


def _convert_to_openvino(input_path: Path, output_dir: Path, precision: str,
                          calibration_data_path=None):
    """Convert to OpenVINO IR format."""
    # Route INT8 to dedicated quantization function
    if precision.lower() == "int8" and calibration_data_path is not None:
        _convert_to_openvino_int8(input_path, output_dir, calibration_data_path)
        return

    import subprocess
    import shutil

    model_name = input_path.stem
    output_xml = output_dir / f"{model_name}.xml"
    output_bin = output_dir / f"{model_name}.bin"

    # Strategy 1: Try C++ binding (works with APT-installed OpenVINO)
    try:
        from ivit._ivit_core import convert_model

        print("Converting to OpenVINO IR format...")
        if precision.lower() == "fp16":
            print("  Compressing to FP16...")

        convert_model(str(input_path), str(output_xml), "cpu", precision.lower())

        print(f"  Created: {output_xml}")
        print(f"  Created: {output_bin}")
        print()
        print("Conversion complete!")
        return
    except ImportError:
        pass

    # Strategy 2: Try ovc command-line tool (from APT or pip openvino-dev)
    ovc_path = shutil.which("ovc")
    if ovc_path:
        print("Converting to OpenVINO IR format (using ovc)...")

        cmd = [ovc_path, str(input_path), "--output_model", str(output_xml)]
        if precision.lower() == "fp16":
            print("  Compressing to FP16...")
            cmd.append("--compress_to_fp16")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: ovc conversion failed:\n{result.stderr}")
            sys.exit(1)

        print(f"  Created: {output_xml}")
        print(f"  Created: {output_bin}")
        print()
        print("Conversion complete!")
        return

    # No conversion tool available (Python openvino is intentionally not used)
    print("Error: OpenVINO conversion tools not found.")
    print("Install one of the following:")
    print("  pip install -e .    # Build the C++ binding (provides convert_model)")
    print("  # or install the OpenVINO 'ovc' CLI tool (e.g. via APT openvino package)")
    sys.exit(1)


def _convert_to_tensorrt(input_path: Path, output_dir: Path, precision: str):
    """Convert to TensorRT engine."""
    model_name = input_path.stem
    output_engine = output_dir / f"{model_name}.engine"

    # Strategy 1: Try C++ binding
    try:
        from ivit._ivit_core import convert_model

        print("Converting to TensorRT engine...")
        print(f"  Precision: {precision}")

        convert_model(str(input_path), str(output_engine), "cuda:0", precision.lower())

        print(f"  Created: {output_engine}")
        print()
        print("Conversion complete!")
        return
    except ImportError:
        pass

    # Strategy 2: Fall back to Python tensorrt
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: TensorRT not installed.")
        print("TensorRT requires NVIDIA GPU and proper installation.")
        sys.exit(1)

    print("Converting to TensorRT engine...")
    print(f"  Precision: {precision}")

    # TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Build engine
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print("  Parsing ONNX model...")
    with open(input_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"  Error: {parser.get_error(error)}")
            sys.exit(1)

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if precision.lower() == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled")
        else:
            print("  Warning: FP16 not supported on this platform")

    if precision.lower() == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  INT8 enabled (requires calibration)")
        else:
            print("  Warning: INT8 not supported on this platform")

    # Build engine
    print("  Building engine (this may take a while)...")
    engine_bytes = builder.build_serialized_network(network, config)

    if engine_bytes is None:
        print("Error: Failed to build engine")
        sys.exit(1)

    with open(output_engine, 'wb') as f:
        f.write(engine_bytes)

    print(f"  Created: {output_engine}")
    print()
    print("Conversion complete!")


def _convert_to_onnx(input_path: Path, output_dir: Path):
    """Convert PyTorch model to ONNX."""
    if input_path.suffix.lower() == '.onnx':
        # Already ONNX, just copy
        import shutil
        output_path = output_dir / input_path.name
        shutil.copy(input_path, output_path)
        print(f"  Copied to: {output_path}")
        return

    if input_path.suffix.lower() == '.pt' or input_path.suffix.lower() == '.pth':
        # Generic PyTorch model conversion
        try:
            import torch

            print("Converting PyTorch model to ONNX...")
            print("  Loading model...")

            # Try to load as a complete model first
            try:
                model = torch.load(str(input_path), map_location="cpu")
                if isinstance(model, dict):
                    # State dict - need model architecture
                    print("Error: Cannot convert state dict without model architecture")
                    print("Hint: For YOLOX models, use: ivit zoo download yolox-s")
                    print("Hint: For torchvision models, use: ivit zoo download resnet50")
                    sys.exit(1)
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)

            model.eval()

            # Determine input size (default to 640x640 for detection, 224x224 for classification)
            h, w = 640, 640
            dummy_input = torch.randn(1, 3, h, w)

            output_path = output_dir / input_path.with_suffix('.onnx').name

            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                opset_version=11,
                input_names=["input"],
                output_names=["output"],
            )
            print(f"  Created: {output_path}")

        except ImportError:
            print("Error: PyTorch required for .pt/.pth conversion")
            print("Install with: pip install torch")
            sys.exit(1)
    else:
        print(f"Error: Unsupported input format: {input_path.suffix}")
        print("Supported formats: .onnx, .pt, .pth")
        sys.exit(1)

    print()
    print("Conversion complete!")


def cmd_export(args):
    """Export model with optimizations for target platform."""
    import ivit

    model_path = args.model
    output_path = args.output
    target = args.target
    precision = args.precision

    print()
    print("=" * 60)
    print("iVIT-SDK Model Export")
    print("=" * 60)
    print()
    print(f"Input:    {model_path}")
    print(f"Output:   {output_path}")
    print(f"Target:   {target}")
    print(f"Precision: {precision}")
    print()

    # Load model first to validate
    print("Loading model...", end=" ", flush=True)
    model = ivit.load(model_path)
    print("Done")
    print()

    # Determine output format based on target
    target_formats = {
        "intel_cpu": "openvino",
        "intel_gpu": "openvino",
        "intel_npu": "openvino",
        "nvidia_gpu": "tensorrt",
        "cpu": "onnx",
        "auto": "onnx",
    }

    output_format = target_formats.get(target, "onnx")
    print(f"Export format: {output_format}")

    # Convert
    input_path = Path(model_path)
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "openvino":
        _convert_to_openvino(input_path, output_dir, precision)
    elif output_format == "tensorrt":
        _convert_to_tensorrt(input_path, output_dir, precision)
    else:
        # Just copy ONNX
        import shutil
        shutil.copy(model_path, output_path)
        print(f"Exported to: {output_path}")

    print()
    print("=" * 60)


def cmd_serve(args):
    """Start inference server."""
    import ivit
    import json
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import io
    import base64

    model_path = args.model
    device = args.device
    port = args.port
    host = args.host

    print()
    print("=" * 60)
    print("iVIT-SDK Inference Server")
    print("=" * 60)
    print()
    print(f"Model:  {model_path}")
    print(f"Device: {device}")
    print(f"Host:   {host}")
    print(f"Port:   {port}")
    print()

    # Load model
    print("Loading model...", end=" ", flush=True)
    model = ivit.load(model_path, device=device)
    print("Done")
    print()

    # Create request handler
    class InferenceHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            elif self.path == "/info":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                info = {
                    "model": model.name,
                    "device": model.device,
                    "task": str(model.task),
                    "input_shape": model.input_info[0].shape if model.input_info else [],
                }
                self.wfile.write(json.dumps(info).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/predict":
                try:
                    import numpy as np
                    import cv2

                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)

                    # Try to parse as JSON with base64 image
                    try:
                        data = json.loads(post_data)
                        if "image" in data:
                            # Base64 encoded image
                            img_data = base64.b64decode(data["image"])
                            nparr = np.frombuffer(img_data, np.uint8)
                            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        else:
                            raise ValueError("No image in request")
                    except json.JSONDecodeError:
                        # Raw image bytes
                        nparr = np.frombuffer(post_data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if image is None:
                        raise ValueError("Failed to decode image")

                    # Run inference
                    conf = float(self.headers.get('X-Conf-Threshold', 0.5))
                    from ivit.core import InferConfig
                    _cfg = InferConfig()
                    _cfg.conf_threshold = conf
                    results = model.predict(image, _cfg)

                    # Format response
                    response = {
                        "inference_time_ms": results.inference_time_ms,
                        "detections": [
                            {
                                "label": d.label,
                                "confidence": float(d.confidence),
                                "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                            }
                            for d in results.detections
                        ]
                    }

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())

                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            print(f"[{self.log_date_time_string()}] {format % args}")

    # Start server
    server = HTTPServer((host, port), InferenceHandler)
    print(f"Server started at http://{host}:{port}")
    print()
    print("Endpoints:")
    print(f"  GET  /health  - Health check")
    print(f"  GET  /info    - Model information")
    print(f"  POST /predict - Run inference (send image as raw bytes or JSON with base64)")
    print()
    print("Press Ctrl+C to stop")
    print("-" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        print("Server stopped")
        server.shutdown()


def cmd_eval(args):
    """Evaluate detection model accuracy on a COCO-format dataset."""
    import json
    from pathlib import Path
    from ivit.vision.yolox_predictor import YoloxPredictor

    print(f"[ivit eval] model: {args.model}")
    print(f"[ivit eval] annotation: {args.ann}")
    print(f"[ivit eval] images: {args.images}")
    print(f"[ivit eval] device: {args.device}  conf: {args.conf}  iou: {args.iou}")
    print()

    predictor = YoloxPredictor(args.model, device=args.device)
    report = predictor.evaluate(
        annotation_file=args.ann,
        images_dir=args.images,
        conf=args.conf,
        iou=args.iou,
        max_detections=args.max_det,
        verbose=True,
    )

    print()
    print("=" * 50)
    print(f"  mAP @ IoU=0.50:0.95  : {report.mAP:.4f}")
    print(f"  mAP @ IoU=0.50       : {report.mAP_50:.4f}")
    print(f"  mAP @ IoU=0.75       : {report.mAP_75:.4f}")
    print(f"  AR  @ maxDets=100    : {report.AR_100:.4f}")
    print(f"  images: {report.num_images}  detections: {report.num_detections}")
    print("=" * 50)
    if report.per_class_AP:
        print("Per-class AP (IoU=0.50:0.95):")
        for name, ap in sorted(report.per_class_AP.items(), key=lambda x: -x[1])[:10]:
            print(f"  {name:20s} {ap:.4f}")
        if len(report.per_class_AP) > 10:
            print(f"  ... ({len(report.per_class_AP) - 10} more classes)")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n[ivit eval] saved metrics to {out}")


def cmd_train(args):
    """Train a model (classify / detect / segment)."""
    from pathlib import Path

    task = args.task
    model_name = args.model
    img_size = args.img_size or {"classify": 224, "detect": 640, "segment": 520}[task]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ivit train] task={task} model={model_name} data={args.data}")
    print(f"             epochs={args.epochs} batch={args.batch_size} img={img_size} device={args.device}")

    if task == "classify":
        from ivit.train import ClassificationTrainer, ImageFolderDataset

        train_ds = ImageFolderDataset(args.data, train_split=0.8, split="train")
        val_ds = ImageFolderDataset(args.data, train_split=0.8, split="val")
        lr = args.learning_rate if args.learning_rate is not None else 1e-3
        trainer = ClassificationTrainer(
            model=model_name,
            dataset=train_ds,
            val_dataset=val_ds,
            epochs=args.epochs,
            learning_rate=lr,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            image_size=img_size,
            freeze_backbone=not args.no_pretrained,
        )

    elif task == "segment":
        from ivit.train import SegmentationTrainer, SegmentationDataset

        if args.masks is None:
            raise SystemExit("--masks is required for segmentation")
        train_ds = SegmentationDataset(args.data, args.masks, train_split=0.8, split="train")
        val_ds = SegmentationDataset(args.data, args.masks, train_split=0.8, split="val")
        lr = args.learning_rate if args.learning_rate is not None else 1e-4
        trainer = SegmentationTrainer(
            model=model_name,
            dataset=train_ds,
            val_dataset=val_ds,
            epochs=args.epochs,
            learning_rate=lr,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            image_size=img_size,
            freeze_backbone=args.no_pretrained is False,  # 預設凍結 backbone
        )

    elif task == "detect":
        from ivit.train import DetectionTrainer

        train_ann = args.ann or "instances_train2017.json"
        val_ann = args.val_ann or "instances_val2017.json"
        trainer = DetectionTrainer(
            model=model_name,
            data_dir=args.data,
            train_ann=train_ann,
            val_ann=val_ann,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=img_size,
            device=args.device,
            output_dir=str(output_dir),
        )
    else:
        raise SystemExit(f"Unknown task: {task}")

    history = trainer.fit()
    if history:
        last = history[-1] if isinstance(history, list) else history
        print(f"[ivit train] final metrics: {last}")

    if args.export_onnx:
        out_path = output_dir / f"{model_name.replace('/', '_')}.{args.export_format}"
        produced = trainer.export(str(out_path), format=args.export_format)
        print(f"[ivit train] exported model: {produced}")


def cmd_zoo(args):
    """Model Zoo commands."""
    # Import directly from registry to avoid C++ binding requirement
    from ivit.zoo.registry import (
        list_models,
        search,
        get_model_info,
        download,
        print_models,
    )

    if args.zoo_command == "list":
        task = args.task
        print_models(task)
    elif args.zoo_command == "search":
        results = search(args.query)
        if results:
            print(f"\nFound {len(results)} model(s):")
            for name in results:
                info = get_model_info(name)
                print(f"  {name:<20} {info.task:<12} {info.description[:40]}")
        else:
            print(f"No models found for: {args.query}")
    elif args.zoo_command == "info":
        try:
            info = get_model_info(args.name)
            print()
            print(f"Model: {info.name}")
            print(f"Task: {info.task}")
            print(f"Description: {info.description}")
            print(f"Input size: {info.input_size}")
            print(f"Classes: {info.num_classes}")
            print(f"Formats: {', '.join(info.formats)}")
            print(f"Source: {info.source}")
            print(f"License: {info.license}")
            if info.metrics:
                print(f"Metrics: {info.metrics}")
            if info.tags:
                print(f"Tags: {', '.join(info.tags)}")
            if info.onnx_url:
                print(f"ONNX download: Direct (pre-converted)")
            else:
                print(f"ONNX download: Requires conversion")
            print()
        except KeyError as e:
            print(f"Error: {e}")
    elif args.zoo_command == "download":
        from_source = getattr(args, "from_source", False)
        print(f"Downloading {args.name}...")
        path = download(args.name, format=args.format, from_source=from_source)
        print(f"Downloaded to: {path}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="ivit",
        description="iVIT-SDK Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ivit info                                    Show system info
  ivit devices                                 List devices
  ivit devices --json                          List devices as JSON
  ivit benchmark model.onnx -d cuda:0          Benchmark on GPU
  ivit predict model.onnx image.jpg            Run inference
  ivit predict model.onnx video.mp4 --save     Process video
  ivit convert model.onnx -f openvino          Convert to OpenVINO
  ivit convert model.onnx -f tensorrt -p fp16  Convert to TensorRT FP16
  ivit export model.onnx -o out.xml -t intel_gpu  Export for Intel GPU
  ivit serve model.onnx --port 8080            Start inference server
  ivit zoo list                                List Model Zoo models
  ivit zoo search yolox                        Search models
  ivit zoo download yolox-s                    Download model
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.set_defaults(func=cmd_info)

    # devices command
    devices_parser = subparsers.add_parser("devices", help="List available devices")
    devices_parser.add_argument("--json", action="store_true", help="Output as JSON")
    devices_parser.set_defaults(func=cmd_devices)

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    bench_parser.add_argument("model", help="Model path")
    bench_parser.add_argument("-d", "--device", default="auto", help="Device (default: auto)")
    bench_parser.add_argument("-n", "--iterations", type=int, default=100, help="Iterations (default: 100)")
    bench_parser.add_argument("-w", "--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    bench_parser.set_defaults(func=cmd_benchmark)

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("model", help="Model path")
    predict_parser.add_argument("source", help="Image/video path or camera index")
    predict_parser.add_argument("-d", "--device", default="auto", help="Device (default: auto)")
    predict_parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    predict_parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    predict_parser.add_argument("--save", action="store_true", help="Save results")
    predict_parser.add_argument("--show", action="store_true", help="Display results")
    predict_parser.set_defaults(func=cmd_predict)

    # convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model format")
    convert_parser.add_argument("model", help="Input model path")
    convert_parser.add_argument("-f", "--format", required=True, choices=["openvino", "tensorrt", "onnx"],
                                help="Output format")
    convert_parser.add_argument("-o", "--output", default=".", help="Output directory (default: .)")
    convert_parser.add_argument("-p", "--precision", default="fp32", choices=["fp32", "fp16", "int8"],
                                help="Precision (default: fp32)")
    convert_parser.add_argument("-c", "--calibration-data", default=None,
                                help="Calibration image folder for INT8 quantization")
    convert_parser.set_defaults(func=cmd_convert)

    # export command
    export_parser = subparsers.add_parser("export", help="Export model for target platform")
    export_parser.add_argument("model", help="Input model path")
    export_parser.add_argument("-o", "--output", required=True, help="Output model path")
    export_parser.add_argument("-t", "--target", default="auto",
                               choices=["auto", "cpu", "intel_cpu", "intel_gpu", "intel_npu", "nvidia_gpu"],
                               help="Target platform (default: auto)")
    export_parser.add_argument("-p", "--precision", default="fp32", choices=["fp32", "fp16", "int8"],
                               help="Precision (default: fp32)")
    export_parser.set_defaults(func=cmd_export)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start inference server")
    serve_parser.add_argument("model", help="Model path")
    serve_parser.add_argument("-d", "--device", default="auto", help="Device (default: auto)")
    serve_parser.add_argument("--host", default="127.0.0.1",
                               help="Host (default: 127.0.0.1, use 0.0.0.0 for public access)")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    serve_parser.set_defaults(func=cmd_serve)

    # zoo command
    zoo_parser = subparsers.add_parser("zoo", help="Model Zoo operations")
    zoo_subparsers = zoo_parser.add_subparsers(dest="zoo_command", help="Zoo commands")

    zoo_list_parser = zoo_subparsers.add_parser("list", help="List available models")
    zoo_list_parser.add_argument("-t", "--task", choices=["detect", "classify", "segment", "pose"],
                                 help="Filter by task")

    zoo_search_parser = zoo_subparsers.add_parser("search", help="Search models")
    zoo_search_parser.add_argument("query", help="Search query")

    zoo_info_parser = zoo_subparsers.add_parser("info", help="Show model info")
    zoo_info_parser.add_argument("name", help="Model name")

    zoo_download_parser = zoo_subparsers.add_parser("download", help="Download model")
    zoo_download_parser.add_argument("name", help="Model name")
    zoo_download_parser.add_argument("-f", "--format", default="onnx", help="Format (default: onnx)")
    zoo_download_parser.add_argument("--from-source", action="store_true",
                                     help="Download original weights and convert locally instead of using pre-converted ONNX")

    zoo_parser.set_defaults(func=cmd_zoo)

    # train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--task", required=True, choices=["classify", "detect", "segment"],
                              help="Training task type")
    train_parser.add_argument("--model", required=True,
                              help="Model name (e.g. resnet50 / yolox-s / deeplabv3-mobilenetv3)")
    train_parser.add_argument("--data", required=True,
                              help="Dataset path (image folder / COCO root / seg folder)")
    train_parser.add_argument("--data-format",
                              choices=["imagefolder", "coco", "yolo", "segfolder"],
                              help="Data layout (auto-inferred from --task if omitted)")
    train_parser.add_argument("--masks", help="Mask folder (segmentation only)")
    train_parser.add_argument("--ann", help="COCO annotation file (yolox detection)")
    train_parser.add_argument("--val-ann", help="COCO val annotation (yolox detection)")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--img-size", type=int, default=None,
                              help="Training image size (default: 224 classify / 640 detect / 520 segment)")
    train_parser.add_argument("--learning-rate", type=float, default=None)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--output", default="runs/train", help="Output directory")
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--export-onnx", action="store_true",
                              help="Export model to ONNX after training")
    train_parser.add_argument("--export-format", default="onnx",
                              choices=["onnx", "openvino", "tensorrt", "torchscript"])
    train_parser.set_defaults(func=cmd_train)

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate detection model accuracy (COCO mAP)")
    eval_parser.add_argument("model", help="Model path (.onnx)")
    eval_parser.add_argument("--task", default="detect", choices=["detect"],
                             help="Evaluation task (only detect supported)")
    eval_parser.add_argument("--ann", required=True,
                             help="COCO annotation JSON (e.g. instances_val2017.json)")
    eval_parser.add_argument("--images", required=True,
                             help="Images directory matching the annotation file")
    eval_parser.add_argument("-d", "--device", default="cuda:0")
    eval_parser.add_argument("--conf", type=float, default=0.001,
                             help="Detection confidence threshold (default 0.001 for full PR sweep)")
    eval_parser.add_argument("--iou", type=float, default=0.65,
                             help="NMS IoU threshold (default 0.65, YOLOX eval convention)")
    eval_parser.add_argument("--max-det", type=int, default=300,
                             help="Max detections per image (default 300)")
    eval_parser.add_argument("--output", default=None,
                             help="Optional path to save metrics JSON")
    eval_parser.set_defaults(func=cmd_eval)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

"""
iVIT-SDK Command Line Interface.

Usage:
    ivit info              Show system and SDK information
    ivit devices           List available inference devices
    ivit benchmark         Run performance benchmark
    ivit predict           Run inference on image/video
    ivit convert           Convert model format
    ivit export            Export model with optimizations

Examples:
    $ ivit info
    $ ivit devices
    $ ivit benchmark yolox-s --device cuda:0
    $ ivit predict yolox-s image.jpg --conf 0.5
    $ ivit predict my-detector.onnx image.jpg --task detect
    $ ivit convert models/yolox-s.onnx -f openvino -o models/
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
    print(
        f"  C++ Bindings:   {'Available' if ivit.is_cpp_available() else 'Not available'}"
    )
    print()

    # System Info
    print("System:")
    print(f"  Platform:       {platform.system()} {platform.release()}")
    print(f"  Architecture:   {platform.machine()}")
    print(f"  Python:         {platform.python_version()}")
    print()

    # Backend availability
    print("Backends:")
    try:
        import openvino

        print(f"  OpenVINO:       {openvino.__version__}")
    except ImportError:
        print("  OpenVINO:       Not installed")

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
    print(f"{'ID':<12} {'Name':<30} {'Backend':<12} {'Type':<8} {'Status'}")
    print("-" * 60)

    for d in device_list:
        status = "OK" if d.is_available else "N/A"
        print(f"{d.id:<12} {d.name[:28]:<30} {d.backend:<12} {d.type:<8} {status}")

    print("-" * 60)
    print(f"Total: {len(device_list)} device(s)")
    print()

    # Show best device
    best = ivit.devices.best()
    print(f"Best device (performance): {best.id}")
    best_eff = ivit.devices.best("efficiency")
    print(f"Best device (efficiency):  {best_eff.id}")
    print()


def _load_model(args):
    """Load a Model Zoo name (with its verified settings) or a model file."""
    from ivit.zoo.registry import load

    options = {"task": args.task} if args.task else {}
    return load(args.model, device=args.device, **options)


def _warn_if_raw(model):
    if model.load_config.task == "":
        print("Note: no task set, so predict() returns no decoded results. Use a Model Zoo name or --task.")


def _warn_if_generic(args):
    if args.task and not _is_zoo_name(args.model):
        print(
            "Note: --task uses the generic decoder and preprocessing. If results look wrong, "
            "check the model's decoder, color order and normalization (Model Zoo names apply "
            "verified settings)."
        )


def _is_zoo_name(name):
    from ivit.zoo.registry import MODEL_REGISTRY

    return name in MODEL_REGISTRY


def cmd_benchmark(args):
    """Run performance benchmark."""
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
    model = _load_model(args)
    load_time = time.time() - start
    print(f"Done ({load_time:.2f}s)")
    print(f"Device used: {model.device}")
    _warn_if_raw(model)
    print()

    from ivit.utils.profiler import _dummy_image

    # Get input shape (dynamic dimensions are reported as -1)
    input_shape = list(model.input_info[0].shape)
    print(f"Input shape: {input_shape}")
    print()

    dummy_image = _dummy_image(model, None)

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
        model.predict(dummy_image)
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
    model = _load_model(args)
    print("Done")
    print(f"Device used: {model.device}")
    _warn_if_raw(model)
    _warn_if_generic(args)
    print()

    config = ivit.InferConfig()
    if conf is not None:
        config.conf_threshold = conf
    config.iou_threshold = iou

    # Check if source is video
    source_path = Path(source)
    is_video = source_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    is_camera = source.isdigit()

    if is_video or is_camera:
        import cv2

        print("Streaming inference...")
        print("-" * 40)

        writer = None
        output_path = None
        frame_count = 0
        try:
            for item in model.stream(source, config):
                frame_count += 1
                results = item.results
                print(
                    f"Frame {frame_count}: {_summary(results)}, {results.inference_time_ms:.1f}ms"
                )
                if save or show:
                    drawn = results.visualize(item.frame)
                if save:
                    if writer is None:
                        stem = "camera" + source if is_camera else source_path.stem
                        output_path = f"{stem}_result.mp4"
                        height, width = drawn.shape[:2]
                        fps = item.fps if item.fps > 0 else 30.0
                        writer = cv2.VideoWriter(
                            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                        )
                    writer.write(drawn)
                if show:
                    cv2.imshow("ivit", drawn)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        print("-" * 40)
        print(f"Total frames: {frame_count}")
        if output_path:
            print(f"Saved to: {output_path}")
    else:
        # Image inference
        print("Running inference...")
        results = model.predict(source, config)

        print()
        print(f"Inference time: {results.inference_time_ms:.2f} ms")
        print(f"Results: {_summary(results)}")
        print()

        for i, det in enumerate(results.detections):
            print(
                f"  [{i}] {det.label}: {det.confidence:.2%} @ [{det.bbox.x1:.0f}, {det.bbox.y1:.0f}, {det.bbox.x2:.0f}, {det.bbox.y2:.0f}]"
            )
        for i, cls in enumerate(results.topk(5) if results.classifications else []):
            print(f"  [{i}] {cls.label}: {cls.score:.2%}")

        if save or show:
            import cv2

            image = cv2.imread(source)
            drawn = results.visualize(image)
            if save:
                output_path = source_path.stem + "_result" + source_path.suffix
                cv2.imwrite(output_path, drawn)
                print()
                print(f"Saved to: {output_path}")
            if show:
                cv2.imshow("ivit", drawn)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    print()


def _summary(results):
    if results.classifications:
        top = results.top1()
        return f"top-1 {top.label} ({top.score:.2%})"
    if results.segmentation_mask.size:
        return "segmentation mask"
    return f"{len(results.detections)} detection(s)"


def cmd_convert(args):
    """Convert model to different format."""
    model_path = args.model
    output_format = args.format.lower()
    output_dir = Path(args.output) if args.output else Path(".")
    precision = args.precision

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

    # Verify input exists
    input_path = Path(model_path)
    if not input_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch input: export ONNX first (kept in the output directory), then convert that.
    if input_path.suffix.lower() in _TORCH_SUFFIXES:
        input_path = _torch_to_onnx(
            input_path, output_dir, _parse_input_size(args.input_size), args.trust_pickle
        )
        if output_format == "onnx":
            print()
            print("=" * 60)
            return

    # Convert based on format
    if output_format == "openvino":
        _convert_to_openvino(input_path, output_dir, precision)
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


def _convert_native(input_path: Path, output_path: Path, device: str, precision: str):
    """Use the same native converter as the SDK and training exporter."""
    from ivit import _require_cpp, convert_model

    _require_cpp()
    convert_model(str(input_path), str(output_path), device, precision.lower())
    print(f"Created: {output_path}")


def _convert_to_openvino(input_path: Path, output_dir: Path, precision: str):
    """Convert through the native OpenVINO backend."""
    _convert_native(input_path, output_dir / f"{input_path.stem}.xml", "cpu", precision)


def _convert_to_tensorrt(input_path: Path, output_dir: Path, precision: str):
    """Convert through the native TensorRT backend."""
    _convert_native(
        input_path, output_dir / f"{input_path.stem}.engine", "cuda:0", precision
    )


_TORCH_SUFFIXES = (".pt", ".pth")


def _parse_input_size(text: str):
    """'640' -> (640, 640); '480,640' or '480x640' -> (480, 640) as (H, W)."""
    parts = text.lower().replace("x", ",").split(",")
    try:
        size = tuple(int(p) for p in parts)
    except ValueError:
        size = ()
    if len(size) == 1:
        size = size * 2
    if len(size) != 2 or min(size) <= 0:
        raise ValueError(f"--input-size must be N or H,W (got {text!r})")
    return size


def _load_torch_model(input_path: Path, trust_pickle: bool):
    """Load a TorchScript archive, or a pickled nn.Module when trust_pickle is set."""
    import torch

    try:
        return torch.jit.load(str(input_path), map_location="cpu")
    except RuntimeError:
        pass  # not a TorchScript archive

    try:
        obj = torch.load(str(input_path), map_location="cpu", weights_only=True)
    except Exception as e:
        # A pickled module needs its class; weights_only refuses that as an "Unsupported global".
        if "Unsupported global" not in str(e):
            raise ValueError(
                f"{input_path.name} is not a TorchScript archive or a torch.save() file"
            ) from None
        if not trust_pickle:
            raise ValueError(
                f"{input_path.name} is a pickled PyTorch model; loading it runs code from the file. "
                "Pass --trust-pickle if you trust its source, or save it as TorchScript "
                "(torch.jit.script/trace(...).save()) or export ONNX yourself."
            ) from None
        obj = torch.load(str(input_path), map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        raise ValueError(
            f"{input_path.name} holds weights only (a state dict), not a model. Build the model in "
            "Python, load the weights and export ONNX, or save it as TorchScript. Zoo models are "
            "available ready-made: ivit zoo download <name>"
        )
    raise ValueError(f"{input_path.name} does not contain a PyTorch model ({type(obj).__name__})")


def _torch_to_onnx(input_path: Path, output_dir: Path, input_size, trust_pickle: bool) -> Path:
    """Export a .pt/.pth model to ONNX with a fixed 1x3xHxW float input."""
    try:
        import torch
    except ImportError:
        raise ValueError("PyTorch is required for .pt/.pth conversion (pip install torch)") from None

    print("Converting PyTorch model to ONNX...")
    model = _load_torch_model(input_path, trust_pickle)
    model.eval()
    h, w = input_size
    output_path = output_dir / f"{input_path.stem}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randn(1, 3, h, w),
            str(output_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  Created: {output_path} (input 1x3x{h}x{w})")
    return output_path


def _convert_to_onnx(input_path: Path, output_dir: Path):
    """Copy an ONNX model (PyTorch models are exported by _torch_to_onnx)."""
    if input_path.suffix.lower() != ".onnx":
        raise ValueError(
            f"Unsupported input format for -f onnx: {input_path.suffix} (supported: .onnx, .pt, .pth)"
        )
    import shutil

    output_path = output_dir / input_path.name
    if output_path.resolve() != input_path.resolve():
        shutil.copy(input_path, output_path)
    print(f"  Copied to: {output_path}")


def cmd_export(args):
    """Export model with optimizations for target platform."""
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
    if not input_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if output_format == "openvino":
        _convert_native(input_path, Path(output_path), "cpu", precision)
    elif output_format == "tensorrt":
        _convert_native(input_path, Path(output_path), "cuda:0", precision)
    else:
        # Just copy ONNX
        import shutil

        shutil.copy(model_path, output_path)
        print(f"Exported to: {output_path}")

    print()
    print("=" * 60)


def cmd_zoo(args):
    """Model Zoo commands."""
    # Import directly from registry to avoid C++ binding requirement
    from ivit.zoo.registry import (
        search,
        get_model_info,
        download,
        print_models,
    )

    if args.zoo_command is None:
        args.zoo_parser.print_help()
        sys.exit(1)
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
            print(f"Error: {e.args[0]}", file=sys.stderr)
            sys.exit(1)
    elif args.zoo_command == "download":
        from_source = getattr(args, "from_source", False)
        print(f"Downloading {args.name}...")
        path = download(args.name, from_source=from_source)
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
  ivit benchmark yolox-s -d cuda:0             Benchmark on GPU
  ivit predict yolox-s image.jpg               Run inference
  ivit predict yolox-s video.mp4 --save        Process video
  ivit predict model.onnx image.jpg --task detect  Model file with a decoder
  ivit convert model.onnx -f openvino          Convert to OpenVINO
  ivit convert model.onnx -f tensorrt -p fp16  Convert to TensorRT FP16
  ivit export model.onnx -o out.xml -t intel_gpu  Export for Intel GPU
  ivit zoo list                                List Model Zoo models
  ivit zoo search yolox                        Search models
  ivit zoo download yolox-s                    Download model
        """,
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
    bench_parser.add_argument("model", help="Model path or Model Zoo name")
    bench_parser.add_argument(
        "--task",
        choices=["detect", "classify", "segment"],
        help="Decoder task for a model file (Model Zoo names set it automatically)",
    )
    bench_parser.add_argument(
        "-d", "--device", default="auto", help="Device (default: auto)"
    )
    bench_parser.add_argument(
        "-n", "--iterations", type=int, default=100, help="Iterations (default: 100)"
    )
    bench_parser.add_argument(
        "-w", "--warmup", type=int, default=10, help="Warmup iterations (default: 10)"
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("model", help="Model path or Model Zoo name")
    predict_parser.add_argument(
        "--task",
        choices=["detect", "classify", "segment"],
        help="Decoder task for a model file (Model Zoo names set it automatically)",
    )
    predict_parser.add_argument("source", help="Image/video path or camera index")
    predict_parser.add_argument(
        "-d", "--device", default="auto", help="Device (default: auto)"
    )
    predict_parser.add_argument(
        "--conf", type=float, default=None,
        help="Confidence threshold (default: the task's default)"
    )
    predict_parser.add_argument(
        "--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)"
    )
    predict_parser.add_argument(
        "--save", action="store_true", help="Save the annotated image or video"
    )
    predict_parser.add_argument("--show", action="store_true", help="Display results")
    predict_parser.set_defaults(func=cmd_predict)

    # convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model format")
    convert_parser.add_argument("model", help="Input model path")
    convert_parser.add_argument(
        "-f",
        "--format",
        required=True,
        choices=["openvino", "tensorrt", "onnx"],
        help="Output format",
    )
    convert_parser.add_argument(
        "-o", "--output", default=".", help="Output directory (default: .)"
    )
    convert_parser.add_argument(
        "-p",
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="Precision (default: fp32)",
    )
    convert_parser.add_argument(
        "--input-size",
        default="640",
        help="Input size for .pt/.pth export: N or H,W (default: 640)",
    )
    convert_parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="Allow loading a pickled .pt/.pth model (runs code from the file)",
    )
    convert_parser.set_defaults(func=cmd_convert)

    # export command
    export_parser = subparsers.add_parser(
        "export", help="Export model for target platform"
    )
    export_parser.add_argument("model", help="Input model path")
    export_parser.add_argument(
        "-o", "--output", required=True, help="Output model path"
    )
    export_parser.add_argument(
        "-t",
        "--target",
        default="auto",
        choices=["auto", "cpu", "intel_cpu", "intel_gpu", "intel_npu", "nvidia_gpu"],
        help="Target platform (default: auto)",
    )
    export_parser.add_argument(
        "-p",
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="Precision (default: fp32)",
    )
    export_parser.set_defaults(func=cmd_export)

    # zoo command
    zoo_parser = subparsers.add_parser("zoo", help="Model Zoo operations")
    zoo_subparsers = zoo_parser.add_subparsers(dest="zoo_command", help="Zoo commands")

    zoo_list_parser = zoo_subparsers.add_parser("list", help="List available models")
    zoo_list_parser.add_argument(
        "-t",
        "--task",
        choices=["detect", "classify", "segment"],
        help="Filter by task",
    )

    zoo_search_parser = zoo_subparsers.add_parser("search", help="Search models")
    zoo_search_parser.add_argument("query", help="Search query")

    zoo_info_parser = zoo_subparsers.add_parser("info", help="Show model info")
    zoo_info_parser.add_argument("name", help="Model name")

    zoo_download_parser = zoo_subparsers.add_parser("download", help="Download model")
    zoo_download_parser.add_argument("name", help="Model name")
    zoo_download_parser.add_argument(
        "--from-source",
        action="store_true",
        help="Download original weights and convert locally instead of using pre-converted ONNX",
    )

    zoo_parser.set_defaults(func=cmd_zoo, zoo_parser=zoo_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        # IVITError derives from RuntimeError
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

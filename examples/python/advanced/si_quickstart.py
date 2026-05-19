#!/usr/bin/env python3
"""
iVIT-SDK System Integrator (SI) Quick Start Example

Target: System Integrators who need to quickly integrate AI inference
        into existing systems.

Features demonstrated:
- Device discovery and auto-selection
- Model loading (Model Zoo and local files)
- Structured error handling
- Result serialization

Usage:
    python si_quickstart.py
    python si_quickstart.py --image path/to/image.jpg
    python si_quickstart.py --model path/to/model.onnx --image path/to/image.jpg
"""

import argparse
import json
import sys

import numpy as np

import ivit


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from ivit.core.exceptions import (
    IVITError,
    ModelLoadError,
    DeviceNotFoundError,
    InferenceError,
    ConfigurationError,
)


def discover_devices():
    """Step 1: Discover available devices."""
    print("=" * 60)
    print("Step 1: Device Discovery")
    print("=" * 60)

    # List all available devices
    devices = ivit.devices()
    print(f"Found {len(devices)} available device(s)")

    # Show device details
    for device in devices:
        print(f"  - {device.id}: {device.name} ({device.backend})")

    # Auto-select best device
    best = ivit.devices.best()
    print(f"\nAuto-selected best device: {best.id} ({best.name})")

    # Alternative: select by type
    print("\nDevice selection options:")
    print(f"  ivit.devices.cpu()  -> CPU device")
    print(f"  ivit.devices.cuda() -> NVIDIA GPU (if available)")
    print(f"  ivit.devices.npu()  -> Intel NPU (if available)")
    print(f"  ivit.devices.best() -> Auto-select best")
    print(f"  ivit.devices.best('efficiency') -> Best efficiency")

    return best


def load_model(model_source: str, device):
    """Step 2: Load model."""
    print("\n" + "=" * 60)
    print("Step 2: Model Loading")
    print("=" * 60)

    if model_source:
        # Load from local file (customer-provided model)
        print(f"Loading model from file: {model_source}")
        model = ivit.load(model_source, device=device)
    else:
        # Load from Model Zoo (quick POC)
        print("Loading model from Model Zoo: yolox-s")
        model = ivit.zoo.load("yolox-s", device=device)

    print(f"Model loaded successfully!")
    print(f"  Device: {model.device}")
    print(f"  Input info: {model.input_info}")

    return model


def safe_inference(model, image_path: str) -> dict:
    """
    Step 3: Safe inference with comprehensive error handling.

    This function demonstrates the recommended error handling pattern
    for production deployments.
    """
    print("\n" + "=" * 60)
    print("Step 3: Safe Inference")
    print("=" * 60)

    try:
        # Execute inference
        print(f"Running inference on: {image_path}")
        results = model(image_path)

        # Get structured results
        output = results.to_dict()

        return {
            "success": True,
            "image_path": image_path,
            "inference_time_ms": results.inference_time_ms,
            "detections": output.get("detections", []),
            "detection_count": len(results),
        }

    except ModelLoadError as e:
        return {
            "success": False,
            "error_type": "ModelLoadError",
            "message": str(e),
            "suggestion": "Please verify model path and format (.onnx, .xml, .engine)"
        }

    except DeviceNotFoundError as e:
        return {
            "success": False,
            "error_type": "DeviceNotFoundError",
            "message": str(e),
            "suggestion": "Run ivit.devices() to check available devices"
        }

    except InferenceError as e:
        return {
            "success": False,
            "error_type": "InferenceError",
            "message": str(e),
            "suggestion": "Check input image format and dimensions"
        }

    except ConfigurationError as e:
        return {
            "success": False,
            "error_type": "ConfigurationError",
            "message": str(e),
            "suggestion": "Check model and device configuration"
        }

    except IVITError as e:
        return {
            "success": False,
            "error_type": "IVITError",
            "message": str(e),
            "suggestion": "See error message for details"
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "error_type": "FileNotFoundError",
            "message": str(e),
            "suggestion": "Verify the image file path exists"
        }


def process_results(result: dict):
    """Step 4: Process and display results."""
    print("\n" + "=" * 60)
    print("Step 4: Results")
    print("=" * 60)

    if result["success"]:
        print("Inference successful!")
        print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
        print(f"  Detections: {result['detection_count']}")

        for i, det in enumerate(result.get("detections", [])):
            print(f"    [{i}] {det['label']}: {det['confidence']:.2%}")

        # Export as JSON (for system integration)
        print("\nJSON output (for system integration):")
        print(json.dumps(result, indent=2, ensure_ascii=False, cls=NumpyEncoder))

    else:
        print(f"Inference failed!")
        print(f"  Error type: {result['error_type']}")
        print(f"  Message: {result['message']}")
        print(f"  Suggestion: {result['suggestion']}")


def main():
    parser = argparse.ArgumentParser(
        description="iVIT-SDK System Integrator Quick Start"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (.onnx, .xml, .engine). If not provided, uses Model Zoo."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test_images/bus.jpg",
        help="Path to input image (default: test_images/bus.jpg)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("iVIT-SDK System Integrator (SI) Quick Start")
    print("=" * 60)
    print(f"SDK Version: {ivit.__version__}")

    # Step 1: Discover devices
    best_device = discover_devices()

    # Step 2: Load model
    try:
        model = load_model(args.model, best_device)
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        print("Please ensure the model file exists or Model Zoo is accessible.")
        sys.exit(1)

    # Step 3: Run inference with error handling
    result = safe_inference(model, args.image)

    # Step 4: Process results
    process_results(result)

    print("\n" + "=" * 60)
    print("SI Best Practices:")
    print("=" * 60)
    print("1. Use ivit.devices.best() for auto device selection")
    print("2. Always wrap inference in try-except blocks")
    print("3. Use results.to_dict() for structured output")
    print("4. Test on multiple hardware environments")


if __name__ == "__main__":
    main()

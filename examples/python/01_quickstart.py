#!/usr/bin/env python3
"""
iVIT-SDK Quickstart Example

最簡單的使用範例，直接執行即可體驗 iVIT-SDK 的物件偵測功能。
模型會自動從 Model Zoo 下載（預設使用 yolox-s）。

Usage:
    python examples/python/01_quickstart.py

    # 使用自訂模型
    IVIT_MODEL_PATH=./my_model.onnx python examples/python/01_quickstart.py
"""

import sys
import os

# Get project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Append source path as fallback only — installed package takes precedence
sys.path.append(os.path.join(PROJECT_ROOT, 'python'))

import ivit
from ivit.vision import Detector
import cv2

# Default model from zoo
DEFAULT_MODEL = "yolox-s"


def _ensure_model(model_name):
    """Download model from zoo if not already cached, return path."""
    from ivit.zoo.registry import download, _get_cache_dir
    cache_dir = _get_cache_dir()
    model_path = cache_dir / f"{model_name}.onnx"
    if model_path.exists():
        return str(model_path)
    print(f"\nModel not found locally, downloading {model_name} from Model Zoo...")
    path = download(model_name, format="onnx")
    return str(path)


def main():
    # Print version and available devices
    print(f"iVIT-SDK v{ivit.__version__}")
    print("\nDevices:")
    for d in ivit.list_devices():
        print(f"  - {d.id}: {d.name} [{d.backend}]")

    # Model and image paths
    model_path = os.environ.get("IVIT_MODEL_PATH")
    if model_path is None:
        model_path = _ensure_model(DEFAULT_MODEL)

    image_path = os.environ.get(
        "IVIT_IMAGE_PATH",
        os.path.join(PROJECT_ROOT, "examples", "data", "images", "bus.jpg"),
    )

    # Check files exist
    if not os.path.exists(model_path):
        print(f"\nModel not found: {model_path}")
        return 1
    if not os.path.exists(image_path):
        print(f"\nImage not found: {image_path}")
        return 1

    # Create detector (auto select best device)
    print(f"\nLoading model...")
    device_order = ["cuda:0", "gpu:0", "cpu"]
    detector = None
    used_device = None

    for device in device_order:
        try:
            detector = Detector(model_path, device=device)
            used_device = device
            break
        except Exception as e:
            print(f"  {device}: not available ({type(e).__name__})")

    if detector is None:
        print("Error: No available device for inference")
        return 1

    print(f"  Using: {used_device}")
    print(f"  Input: {detector.input_size}")

    # Load and process image
    image = cv2.imread(image_path)
    print(f"\nProcessing: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # Run detection
    results = detector.predict(image, conf_threshold=0.5)

    # Print results
    print(f"\nFound {len(results.detections)} objects in {results.inference_time_ms:.1f}ms:")
    for det in results.detections:
        print(f"  - {det.label}: {det.confidence:.0%}")

    # Save visualization
    output_path = os.path.join(os.path.dirname(image_path), "quickstart_output.jpg")
    vis = results.visualize(image)
    cv2.imwrite(output_path, vis)
    print(f"\nSaved: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

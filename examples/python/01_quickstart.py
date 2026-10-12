#!/usr/bin/env python3
"""
iVIT-SDK Quickstart Example

最簡單的使用範例，直接執行即可體驗 iVIT-SDK 的物件偵測功能。

Usage:
    python examples/python/01_quickstart.py              # YOLOX-S from the Model Zoo
    IVIT_MODEL=yolox-nano python examples/python/01_quickstart.py
"""

import sys
import os

# Get project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

import ivit
from ivit.vision import Detector
import cv2


def resolve_model(model):
    """A Model Zoo name -> (local file, its verified load settings); a file path -> (path, {}).

    Zoo models need the settings their export was made for (task, decoder, color order,
    normalization, labels); with your own model file, pass the settings that match its export.
    """
    if model in ivit.zoo.list_models():
        return str(ivit.zoo.download(model)), dict(ivit.zoo.get_model_info(model).load_options)
    return model, {}



def main():
    # Print version and available devices
    print(f"iVIT-SDK v{ivit.__version__}")
    print("\nDevices:")
    for d in ivit.list_devices():
        print(f"  - {d.id}: {d.name} [{d.backend}]")

    # A Model Zoo name (downloaded and verified on first use) or your own model file.
    # Behind a firewall, set IVIT_ZOO_MIRROR to a directory or URL holding the zoo files.
    model_path, options = resolve_model(os.environ.get("IVIT_MODEL", "yolox-s"))
    image_path = os.environ.get(
        "IVIT_IMAGE_PATH",
        os.path.join(PROJECT_ROOT, "examples", "data", "images", "bus.jpg"),
    )

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
            detector = Detector(model_path, device=device, **options)
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
    output_path = "quickstart_output.jpg"
    vis = results.visualize(image)
    cv2.imwrite(output_path, vis)
    print(f"\nSaved: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

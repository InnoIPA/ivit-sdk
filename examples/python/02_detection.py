#!/usr/bin/env python3
"""
iVIT-SDK Detection Demo

Demonstrates object detection using the iVIT-SDK Python bindings.
Supports multiple backends (TensorRT, OpenVINO).

Usage:
    # Run from project root
    python examples/python/02_detection.py \
        -m yolox-s \
        -i examples/data/images/bus.jpg

    # With specific device
    python examples/python/02_detection.py \
        -m yolox-s \
        -i examples/data/images/bus.jpg \
        -d cuda:0

    # Benchmark mode
    python examples/python/02_detection.py \
        -m yolox-s \
        -i examples/data/images/bus.jpg \
        -d cuda:0 -b -n 100
"""

import argparse
import time
import sys
import os

# Get project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

import ivit
from ivit.vision import Detector


def resolve_model(model):
    """A Model Zoo name -> (local file, its verified load settings); a file path -> (path, {}).

    Zoo models need the settings their export was made for (task, decoder, color order,
    normalization, labels); with your own model file, pass the settings that match its export.
    """
    if model in ivit.zoo.list_models():
        return str(ivit.zoo.download(model)), dict(ivit.zoo.get_model_info(model).load_options)
    return model, {}
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='iVIT-SDK Detection Demo')
    parser.add_argument('--model', '-m', type=str, default='yolox-s',
                        help='Model Zoo name or model file (default: yolox-s)')
    parser.add_argument('--device', '-d', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda:0, cuda:1)')
    parser.add_argument('--image', '-i', type=str,
                        default=os.path.join(PROJECT_ROOT, 'examples', 'data', 'images', 'bus.jpg'),
                        help='Path to the input image')
    parser.add_argument('--output', '-o', type=str, default='output.jpg',
                        help='Path to the output image')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--benchmark', '-b', action='store_true',
                        help='Run benchmark mode')
    parser.add_argument('--iterations', '-n', type=int, default=100,
                        help='Number of benchmark iterations')
    args = parser.parse_args()

    # Print SDK info
    print(f"iVIT-SDK Version: {ivit.__version__}")
    print(f"C++ bindings: {'Yes' if ivit.is_cpp_available() else 'No'}")
    print()

    # List available devices
    print("Available Devices:")
    for dev in ivit.list_devices():
        mem_gb = dev.memory_total / (1024**3) if dev.memory_total > 0 else 0
        print(f"  - {dev.id}: {dev.name} ({dev.backend}, {mem_gb:.1f} GB)")
    print()

    # Determine device
    device = args.device
    if device == 'auto':
        device = ivit.get_best_device().id
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {args.model}")
    t0 = time.perf_counter()
    model_path, options = resolve_model(args.model)
    for key, value in options.items():
        if key != "labels":  # 80+ names: print the count instead
            print(f"  {key} = {value}")
    detector = Detector(model_path, device=device, **options)
    load_time = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_time:.1f} ms")
    print(f"  Input size: {detector.input_size}")
    print(f"  Classes: {detector.num_classes}")
    print()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Failed to load image: {args.image}")
        return 1
    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")

    # Run inference
    if args.benchmark:
        print(f"\nRunning benchmark ({args.iterations} iterations)...")
        times = []

        # Warmup
        for _ in range(10):
            detector.predict(image, conf_threshold=args.conf, iou_threshold=args.iou)

        # Benchmark
        for i in range(args.iterations):
            results = detector.predict(image, conf_threshold=args.conf, iou_threshold=args.iou)
            times.append(results.inference_time_ms)

        times = np.array(times)
        print(f"\nBenchmark Results:")
        print(f"  Mean:   {times.mean():.2f} ms")
        print(f"  Std:    {times.std():.2f} ms")
        print(f"  Min:    {times.min():.2f} ms")
        print(f"  Max:    {times.max():.2f} ms")
        print(f"  FPS:    {1000 / times.mean():.1f}")
    else:
        # Single inference
        print("\nRunning inference...")
        results = detector.predict(image, conf_threshold=args.conf, iou_threshold=args.iou)

        print(f"\nResults:")
        print(f"  Inference time: {results.inference_time_ms:.2f} ms ({1000/results.inference_time_ms:.1f} FPS)")
        print(f"  Device: {results.device_used}")
        print(f"  Detections: {len(results.detections)}")
        print()

        for i, det in enumerate(results.detections):
            print(f"  [{i}] {det.label}: {det.confidence:.1%}")
            print(f"      BBox: ({det.bbox.x1:.0f}, {det.bbox.y1:.0f}) - "
                  f"({det.bbox.x2:.0f}, {det.bbox.y2:.0f})")
            print(f"      Size: {det.bbox.width:.0f} x {det.bbox.height:.0f}")

        # Save visualization (same directory as input image if relative path)
        if os.path.isabs(args.output):
            output_path = args.output
        else:
            input_dir = os.path.dirname(os.path.abspath(args.image))
            output_path = os.path.join(input_dir, args.output)
        print(f"\nSaving result to: {output_path}")
        vis = results.visualize(image)
        cv2.imwrite(output_path, vis)

        # Also save JSON results
        json_path = output_path.rsplit('.', 1)[0] + '.json'
        results.save(json_path)
        print(f"Saved JSON to: {json_path}")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
iVIT-SDK Classification Demo

Demonstrates image classification using the iVIT-SDK Python bindings.
Supports multiple backends (TensorRT, OpenVINO).

Usage:
    # Run from project root
    python examples/python/02_classification.py \
        -m resnet18 \
        -i examples/data/images/bus.jpg

    # With specific device
    python examples/python/02_classification.py \
        -m resnet18 \
        -i examples/data/images/bus.jpg \
        -d npu

    # Benchmark mode
    python examples/python/02_classification.py \
        -m resnet18 \
        -i examples/data/images/bus.jpg \
        -d npu -b -n 100
"""

import argparse
import time
import sys
import os

# Get project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

import ivit
from ivit.vision import Classifier


def resolve_model(model):
    """A Model Zoo name -> (local file, its verified load settings); a file path -> (path, {}).

    Zoo models need the settings their export was made for (task, decoder, color order,
    normalization, labels); with your own model file, pass the settings that match its export.
    """
    if model in ivit.zoo.list_models():
        try:
            return str(ivit.zoo.download(model)), dict(ivit.zoo.get_model_info(model).load_options)
        except ImportError as error:
            # torchvision models are exported on first use and need PyTorch; say how, no traceback
            sys.exit(f"Error: {error}")
    return model, {}
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='iVIT-SDK Classification Demo')
    parser.add_argument('--model', '-m', type=str, default='resnet18',
                        help='Model Zoo name or model file (default: resnet18)')
    parser.add_argument('--device', '-d', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda:0, npu)')
    parser.add_argument('--image', '-i', type=str,
                        default=os.path.join(PROJECT_ROOT, 'examples', 'data', 'images', 'bus.jpg'),
                        help='Path to the input image')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='Number of top predictions to show')
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
    classifier = Classifier(model_path, device=device, **options)
    load_time = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_time:.1f} ms")
    print(f"  Input size: {classifier.input_size}")
    print(f"  Classes: {classifier.num_classes}")
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
            classifier.predict(image, top_k=args.top_k)

        # Benchmark
        for i in range(args.iterations):
            results = classifier.predict(image, top_k=args.top_k)
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
        results = classifier.predict(image, top_k=args.top_k)

        print(f"\nResults:")
        print(f"  Inference time: {results.inference_time_ms:.2f} ms ({1000/results.inference_time_ms:.1f} FPS)")
        print(f"  Device: {results.device_used}")
        print(f"\nTop-{args.top_k} Predictions:")

        for i, cls_result in enumerate(results.classifications[:args.top_k]):
            bar = "█" * int(cls_result.score * 20)
            print(f"  {i+1}. {cls_result.label:20s} {cls_result.score:6.2%} {bar}")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

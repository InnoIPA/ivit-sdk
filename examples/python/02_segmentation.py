#!/usr/bin/env python3
"""
iVIT-SDK Segmentation Demo

Demonstrates semantic segmentation using the iVIT-SDK Python bindings.
Supports multiple backends (TensorRT, OpenVINO).

Usage:
    # Run from project root
    python examples/python/02_segmentation.py \
        -m deeplabv3-mobilenetv3 \
        -i examples/data/images/bus.jpg

    # With specific device
    python examples/python/02_segmentation.py \
        -m deeplabv3-mobilenetv3 \
        -i examples/data/images/bus.jpg \
        -d npu

    # Benchmark mode
    python examples/python/02_segmentation.py \
        -m deeplabv3-mobilenetv3 \
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
from ivit.vision import Segmentor


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
    parser = argparse.ArgumentParser(description='iVIT-SDK Segmentation Demo')
    parser.add_argument('--model', '-m', type=str, default='deeplabv3-mobilenetv3',
                        help='Model Zoo name or model file (default: deeplabv3-mobilenetv3)')
    parser.add_argument('--device', '-d', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda:0, npu)')
    parser.add_argument('--image', '-i', type=str,
                        default=os.path.join(PROJECT_ROOT, 'examples', 'data', 'images', 'bus.jpg'),
                        help='Path to the input image')
    parser.add_argument('--output', '-o', type=str, default='output_seg.jpg',
                        help='Path to the output image')
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
    segmentor = Segmentor(model_path, device=device, **options)
    load_time = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_time:.1f} ms")
    print(f"  Input size: {segmentor.input_size}")
    print(f"  Classes: {segmentor.num_classes}")
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
            segmentor.predict(image)

        # Benchmark
        for i in range(args.iterations):
            results = segmentor.predict(image)
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
        results = segmentor.predict(image)

        print(f"\nResults:")
        print(f"  Inference time: {results.inference_time_ms:.2f} ms ({1000/results.inference_time_ms:.1f} FPS)")
        print(f"  Device: {results.device_used}")
        print(f"  Mask size: {results.segmentation_mask.shape[1]}x{results.segmentation_mask.shape[0]}")

        # Count classes found
        unique_classes = np.unique(results.segmentation_mask)
        class_labels = segmentor.classes
        print(f"  Classes found: {len(unique_classes)}")
        for cls_id in unique_classes:
            label = class_labels[cls_id] if cls_id < len(class_labels) else f"class_{cls_id}"
            pixel_count = np.sum(results.segmentation_mask == cls_id)
            total_pixels = results.segmentation_mask.size
            print(f"    - {label}: {pixel_count / total_pixels:.1%}")

        # Save visualization
        if os.path.isabs(args.output):
            output_path = args.output
        else:
            input_dir = os.path.dirname(os.path.abspath(args.image))
            output_path = os.path.join(input_dir, args.output)

        print(f"\nSaving result to: {output_path}")
        overlay = results.overlay_mask(image, 0.5)
        cv2.imwrite(output_path, overlay)

        # Save mask
        mask_path = output_path.rsplit('.', 1)[0] + '_mask.png'
        colored_mask = results.colorize_mask()
        cv2.imwrite(mask_path, colored_mask)
        print(f"Saved mask to: {mask_path}")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

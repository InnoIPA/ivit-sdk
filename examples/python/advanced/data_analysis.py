#!/usr/bin/env python3
"""
iVIT-SDK Data Scientist Analysis Example

Target: Data scientists who need to quickly validate models,
        analyze inference results, and conduct experiments.

Features demonstrated:
- Quick model loading and testing
- Results API (filtering, serialization, visualization)
- Classification, Detection, and Segmentation results
- Batch processing and analysis

Usage:
    python data_analysis.py
    python data_analysis.py --image path/to/image.jpg
    python data_analysis.py --task classify --model resnet50
    python data_analysis.py --task segment --model deeplabv3-mobilenetv3
"""

import argparse
import json
from typing import List, Dict, Any

import numpy as np

import ivit


def explore_model_zoo():
    """Step 1: Explore available models in Model Zoo."""
    print("=" * 60)
    print("Step 1: Model Zoo Exploration")
    print("=" * 60)

    # List all models
    all_models = ivit.zoo.list_models()
    print(f"Total available models: {len(all_models)}")
    print(f"Models: {all_models}")

    # List by task
    print("\nModels by task:")
    for task in ["detect", "classify", "segment", "pose"]:
        models = ivit.zoo.list_models(task=task)
        print(f"  {task}: {models}")

    # Search models
    print("\nSearch examples:")
    yolo_models = ivit.zoo.search("yolo")
    print(f"  Search 'yolo': {yolo_models}")

    edge_models = ivit.zoo.search("edge")
    print(f"  Search 'edge': {edge_models}")

    # Get model info
    print("\nModel info example (yolox-s):")
    info = ivit.zoo.get_model_info("yolox-s")
    print(f"  Task: {info.task}")
    print(f"  Input size: {info.input_size}")
    print(f"  Num classes: {info.num_classes}")
    print(f"  Metrics: {info.metrics}")
    print(f"  Tags: {info.tags}")


def analyze_detection_results(model, image):
    """Step 2: Analyze detection results."""
    print("\n" + "=" * 60)
    print("Step 2: Detection Results Analysis")
    print("=" * 60)

    results = model(image)

    # Basic info
    print(f"Detection count: {len(results)}")
    print(f"Inference time: {results.inference_time_ms:.1f} ms")
    print(f"Device used: {results.device_used}")
    print(f"Image size: {results.image_size}")

    # Iterate detections
    print("\nAll detections:")
    for i, det in enumerate(results):
        print(f"  [{i}] {det.label}: {det.confidence:.2%}")
        print(f"       BBox: ({det.bbox.x1:.0f}, {det.bbox.y1:.0f}) - "
              f"({det.bbox.x2:.0f}, {det.bbox.y2:.0f})")
        print(f"       Area: {det.bbox.area:.0f} px^2")

    # Filtering
    print("\nFiltering examples:")

    # Filter by confidence
    high_conf = results.filter(confidence=0.9)
    print(f"  High confidence (>90%): {len(high_conf)} detections")

    # Filter by class
    if len(results) > 0:
        first_class = results.detections[0].label
        class_filtered = results.filter(classes=[first_class])
        print(f"  Class '{first_class}': {len(class_filtered)} detections")

    # Filter by area
    large_objects = results.filter_by_area(min_area=5000)
    print(f"  Large objects (>5000 px^2): {len(large_objects)} detections")

    # Combined filter
    combined = results.filter(confidence=0.5, min_area=1000)
    print(f"  Combined (conf>50%, area>1000): {len(combined)} detections")

    # Serialization
    print("\nSerialization:")
    data = results.to_dict()
    print(f"  to_dict() keys: {list(data.keys())}")

    json_str = results.to_json()
    print(f"  to_json() length: {len(json_str)} chars")

    return results


def analyze_classification_results(image):
    """Step 3: Analyze classification results."""
    print("\n" + "=" * 60)
    print("Step 3: Classification Results Analysis")
    print("=" * 60)

    model = ivit.zoo.load("resnet50")
    results = model(image)

    # Top-1 result
    top1 = results.top1
    print(f"Top-1 prediction: {top1.label} ({top1.score:.2%})")

    # Top-5 results
    print("\nTop-5 predictions:")
    for i, cls in enumerate(results.top5):
        print(f"  {i+1}. {cls.label}: {cls.score:.2%}")

    # Top-K results
    print("\nTop-10 predictions:")
    for i, cls in enumerate(results.topk(10)):
        print(f"  {i+1}. {cls.label}: {cls.score:.2%}")

    return results


def analyze_segmentation_results(image):
    """Step 4: Analyze segmentation results."""
    print("\n" + "=" * 60)
    print("Step 4: Segmentation Results Analysis")
    print("=" * 60)

    model = ivit.zoo.load("deeplabv3-mobilenetv3")
    results = model(image)

    print(f"Detection count: {len(results)}")
    print(f"Inference time: {results.inference_time_ms:.1f} ms")

    # Segmentation mask
    mask = results.segmentation_mask
    print(f"\nSegmentation mask shape: {mask.shape}")
    print(f"Unique classes in mask: {np.unique(mask)}")

    # Colorized mask
    colored_mask = results.colorize_mask()
    print(f"Colorized mask shape: {colored_mask.shape}")

    # Contours
    contours = results.get_contours()
    print(f"Total contours: {len(contours)}")

    # Per-class analysis
    for det in results:
        class_contours = results.get_contours(class_id=det.class_id)
        print(f"  {det.label}: {len(class_contours)} contours")

    return results


def batch_analysis(model, images: List[np.ndarray]) -> Dict[str, Any]:
    """Step 5: Batch processing and analysis."""
    print("\n" + "=" * 60)
    print("Step 5: Batch Processing Analysis")
    print("=" * 60)

    print(f"Processing {len(images)} images...")

    # Batch inference
    all_results = model.predict_batch(images)

    # Aggregate statistics
    total_detections = 0
    class_counts = {}
    latencies = []
    confidences = []

    for i, results in enumerate(all_results):
        total_detections += len(results)
        latencies.append(results.inference_time_ms)

        for det in results:
            class_counts[det.label] = class_counts.get(det.label, 0) + 1
            confidences.append(det.confidence)

    # Statistics
    print(f"\nBatch Statistics:")
    print(f"  Total images: {len(images)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections per image: {total_detections / len(images):.1f}")
    print(f"  Avg latency: {np.mean(latencies):.1f} ms")
    print(f"  Total latency: {np.sum(latencies):.1f} ms")

    if confidences:
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {np.mean(confidences):.2%}")
        print(f"  Std: {np.std(confidences):.2%}")
        print(f"  Min: {np.min(confidences):.2%}")
        print(f"  Max: {np.max(confidences):.2%}")

    print(f"\nClass Distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    return {
        "total_images": len(images),
        "total_detections": total_detections,
        "class_counts": class_counts,
        "avg_latency_ms": np.mean(latencies),
        "confidence_stats": {
            "mean": np.mean(confidences) if confidences else 0,
            "std": np.std(confidences) if confidences else 0,
        }
    }


def export_comparison():
    """Step 6: Model export format comparison."""
    print("\n" + "=" * 60)
    print("Step 6: Export Format Comparison")
    print("=" * 60)

    formats = {
        "onnx": {
            "usage": "Cross-platform deployment",
            "pros": "Best compatibility",
            "quantization": ["fp32", "fp16"],
        },
        "torchscript": {
            "usage": "PyTorch ecosystem",
            "pros": "No ONNX conversion needed",
            "quantization": ["fp32"],
        },
        "openvino": {
            "usage": "Intel hardware optimization",
            "pros": "Best Intel performance",
            "quantization": ["fp32", "fp16", "int8"],
        },
        "tensorrt": {
            "usage": "NVIDIA hardware optimization",
            "pros": "Best NVIDIA performance",
            "quantization": ["fp32", "fp16", "int8"],
        },
    }

    print("Export Format Comparison:")
    print("-" * 70)
    print(f"{'Format':<12} {'Usage':<30} {'Quantization'}")
    print("-" * 70)
    for fmt, info in formats.items():
        quant = ", ".join(info["quantization"])
        print(f"{fmt:<12} {info['usage']:<30} {quant}")
    print("-" * 70)

    print("\nExport examples:")
    print("  trainer.export('model.onnx', format='onnx', quantize='fp16')")
    print("  trainer.export('model.pt', format='torchscript')")
    print("  trainer.export('model.xml', format='openvino', quantize='int8')")
    print("  trainer.export('model.engine', format='tensorrt', quantize='fp16')")


def main():
    parser = argparse.ArgumentParser(
        description="iVIT-SDK Data Scientist Analysis Example"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["detect", "classify", "segment"],
        default="detect",
        help="Task type (default: detect)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name from Model Zoo"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("iVIT-SDK Data Scientist Analysis Example")
    print("=" * 60)
    print(f"SDK Version: {ivit.__version__}")

    # Step 1: Explore Model Zoo
    explore_model_zoo()

    # Prepare test image
    if args.image:
        import cv2
        image = cv2.imread(args.image)
        if image is None:
            print(f"\nError: Cannot read image from {args.image}")
            return
    else:
        # Create synthetic test image
        print("\nNote: Using synthetic test image. Use --image to specify real image.")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Load model based on task
    if args.task == "detect":
        model_name = args.model or "yolox-s"
        print(f"\nLoading detection model: {model_name}")
        model = ivit.zoo.load(model_name)

        # Step 2: Detection analysis
        results = analyze_detection_results(model, image)

        # Step 5: Batch analysis
        batch_images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        batch_analysis(model, batch_images)

    elif args.task == "classify":
        # Step 3: Classification analysis
        analyze_classification_results(image)

    elif args.task == "segment":
        # Step 4: Segmentation analysis
        analyze_segmentation_results(image)

    # Step 6: Export comparison
    export_comparison()

    print("\n" + "=" * 60)
    print("Data Scientist Best Practices:")
    print("=" * 60)
    print("1. Use ivit.zoo for quick model exploration")
    print("2. Use results.filter() for targeted analysis")
    print("3. Use results.to_dict()/to_json() for data export")
    print("4. Use predict_batch() for efficient batch processing")
    print("5. Compare export formats based on deployment target")


if __name__ == "__main__":
    main()

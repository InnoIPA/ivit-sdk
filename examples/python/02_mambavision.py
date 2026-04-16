#!/usr/bin/env python3
"""
MambaVision Classification Demo

Runs MambaVision (Hybrid Mamba+Transformer) image classification
using onnxruntime, following the same style as 02_classification.py.

Note: Does not require iVIT C++ bindings — uses onnxruntime directly.

Usage:
    # Export model first (one-time)
    python -c "
    import sys; sys.path.insert(0,'python')
    from ivit.train.exporter import load_mambavision_for_export
    import torch, os
    model = load_mambavision_for_export('t')
    class W(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)['logits']
    os.makedirs(os.path.expanduser('~/.cache/ivit/models'), exist_ok=True)
    path = os.path.expanduser('~/.cache/ivit/models/mambavision-t.onnx')
    torch.onnx.export(W(model.cpu().eval()), torch.randn(1,3,224,224), path,
        input_names=['input'], output_names=['logits'], opset_version=17)
    print('Exported:', path)
    "

    # Run classification
    python examples/python/02_mambavision.py \\
        -m ~/.cache/ivit/models/mambavision-t.onnx \\
        -i examples/data/images/bus.jpg

    # With GPU (TensorRT provider)
    python examples/python/02_mambavision.py \\
        -m ~/.cache/ivit/models/mambavision-t.onnx \\
        -i examples/data/images/bus.jpg \\
        -d cuda:0

    # Benchmark
    python examples/python/02_mambavision.py \\
        -m ~/.cache/ivit/models/mambavision-t.onnx \\
        -i examples/data/images/bus.jpg \\
        -b -n 100
"""

import argparse
import time
import sys
import os

import numpy as np
import cv2

# ImageNet 1K labels (abbreviated — first 10 and last 10 shown, full list loaded from file)
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'python'))


def load_imagenet_labels():
    """Load ImageNet 1K class labels."""
    labels_path = os.path.join(PROJECT_ROOT, "models", "imagenet_labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            return [l.strip() for l in f if l.strip()]

    # Try HuggingFace cache (MambaVision model ships id2label)
    hf_cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--MambaVision-T-1K"
    )
    for root, _, files in os.walk(hf_cache):
        if "config.json" in files:
            import json
            cfg = json.load(open(os.path.join(root, "config.json")))
            if "id2label" in cfg:
                return [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]

    # Fallback: unnamed labels
    return [f"class_{i}" for i in range(1000)]


def preprocess(image: np.ndarray, input_size=(224, 224)) -> np.ndarray:
    """Standard ImageNet preprocessing: resize → RGB → normalize → NCHW."""
    img = cv2.resize(image, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def get_providers(device: str):
    """Map iVIT-style device string to onnxruntime ExecutionProviders."""
    import onnxruntime as ort
    available = ort.get_available_providers()

    if device.startswith("cuda") or device == "gpu":
        dev_id = int(device.split(":")[-1]) if ":" in device else 0
        if "CUDAExecutionProvider" in available:
            return [("CUDAExecutionProvider", {"device_id": dev_id}),
                    "CPUExecutionProvider"]
        print(f"  [warn] CUDAExecutionProvider not available, falling back to CPU")

    if device == "npu":
        if "OpenVINOExecutionProvider" in available:
            return [("OpenVINOExecutionProvider", {"device_type": "NPU"}),
                    "CPUExecutionProvider"]
        print(f"  [warn] OpenVINO NPU not available, falling back to CPU")

    return ["CPUExecutionProvider"]


def main():
    parser = argparse.ArgumentParser(description="MambaVision Classification Demo")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to MambaVision ONNX file")
    parser.add_argument("--image", "-i", required=True,
                        help="Path to input image")
    parser.add_argument("--device", "-d", default="cpu",
                        help="Device: cpu | cuda:0 | npu")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Number of top predictions to show")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Run benchmark mode")
    parser.add_argument("--iterations", "-n", type=int, default=100,
                        help="Benchmark iterations")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed. Run: pip install onnxruntime")
        return 1

    # ── Model info ──────────────────────────────────────────────────────────
    print("=" * 50)
    print("  MambaVision Classification Demo")
    print("=" * 50)
    print(f"  Model  : {args.model}")
    print(f"  Device : {args.device}")
    print(f"  onnxruntime: {ort.__version__}")
    print()

    # ── Load labels ──────────────────────────────────────────────────────────
    labels = load_imagenet_labels()
    print(f"  Labels : {len(labels)} classes")
    print(f"  Sample : {labels[:3]} ...")
    print()

    # ── Load model ───────────────────────────────────────────────────────────
    providers = get_providers(args.device)
    print(f"Loading model... (providers: {[p if isinstance(p,str) else p[0] for p in providers]})")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(args.model, providers=providers)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Loaded in {load_ms:.0f} ms")

    inp_name  = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    out_name  = sess.get_outputs()[0].name
    print(f"  Input  : {inp_name} {inp_shape}")
    print(f"  Output : {out_name}")
    print()

    # ── Load image ───────────────────────────────────────────────────────────
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Failed to load image: {args.image}")
        return 1
    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")

    inp_tensor = preprocess(image)

    # ── Inference ────────────────────────────────────────────────────────────
    if args.benchmark:
        print(f"\nRunning benchmark ({args.iterations} iterations)...")
        times = []

        # Warmup
        for _ in range(10):
            sess.run([out_name], {inp_name: inp_tensor})

        for _ in range(args.iterations):
            t0 = time.perf_counter()
            sess.run([out_name], {inp_name: inp_tensor})
            times.append((time.perf_counter() - t0) * 1000)

        times = np.array(times)
        print(f"\nBenchmark Results:")
        print(f"  Mean   : {times.mean():.2f} ms")
        print(f"  Std    : {times.std():.2f} ms")
        print(f"  Min    : {times.min():.2f} ms")
        print(f"  Max    : {times.max():.2f} ms")
        print(f"  FPS    : {1000 / times.mean():.1f}")

    else:
        print("\nRunning inference...")
        t0 = time.perf_counter()
        logits = sess.run([out_name], {inp_name: inp_tensor})[0][0]
        infer_ms = (time.perf_counter() - t0) * 1000

        probs = softmax(logits)
        top_indices = probs.argsort()[::-1][:args.top_k]

        print(f"\nResults:")
        print(f"  Inference time : {infer_ms:.2f} ms  ({1000/infer_ms:.1f} FPS)")
        print(f"  Device         : {args.device}")
        print(f"\nTop-{args.top_k} Predictions:")
        for rank, idx in enumerate(top_indices):
            label = labels[idx] if idx < len(labels) else f"class_{idx}"
            bar = "█" * int(probs[idx] * 40)
            print(f"  {rank+1}. {label:30s}  {probs[idx]:6.2%}  {bar}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

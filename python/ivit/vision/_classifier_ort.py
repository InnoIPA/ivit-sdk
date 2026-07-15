"""
Pure-Python Classifier using onnxruntime.

Used as fallback when C++ bindings (_ivit_core) cannot be loaded.
Provides the same interface as the C++ Classifier wrapper.
"""

import os
import time
from typing import Union, List, Tuple

import numpy as np
import cv2


# ── Lightweight result types ──────────────────────────────────────────────────

class _ClassificationResult:
    __slots__ = ("label", "score", "class_id")

    def __init__(self, label: str, score: float, class_id: int):
        self.label = label
        self.score = float(score)
        self.class_id = class_id

    def __repr__(self):
        return f"ClassificationResult(label={self.label!r}, score={self.score:.4f})"


class _Results:
    __slots__ = ("classifications", "inference_time_ms", "device_used")

    def __init__(self, classifications, inference_time_ms: float, device_used: str):
        self.classifications = classifications
        self.inference_time_ms = inference_time_ms
        self.device_used = device_used


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_providers(device: str):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "The pure-Python classifier fallback requires onnxruntime. Install it "
            "with `pip install onnxruntime`, or build the C++ bindings."
        ) from e
    available = ort.get_available_providers()

    if device.startswith("cuda") or device == "gpu":
        dev_id = int(device.split(":")[-1]) if ":" in device else 0
        if "CUDAExecutionProvider" in available:
            return [("CUDAExecutionProvider", {"device_id": dev_id}),
                    "CPUExecutionProvider"]

    if device == "npu":
        if "OpenVINOExecutionProvider" in available:
            return [("OpenVINOExecutionProvider", {"device_type": "NPU"}),
                    "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def _resolve_device_label(providers) -> str:
    first = providers[0]
    name = first if isinstance(first, str) else first[0]
    if "CUDA" in name:
        return "cuda:0"
    if "OpenVINO" in name:
        return "npu"
    return "cpu"


def _load_labels(model_path: str) -> List[str]:
    """Load labels from a .txt file next to the model, or return unnamed fallback."""
    txt = os.path.splitext(model_path)[0] + ".txt"
    if os.path.exists(txt):
        with open(txt) as f:
            labels = [line.strip() for line in f if line.strip()]
        if labels:
            return labels
    return [f"class_{i}" for i in range(1000)]


def _preprocess(image: np.ndarray, input_hw: Tuple[int, int]) -> np.ndarray:
    """Standard ImageNet preprocessing → NCHW float32."""
    h, w = input_hw
    img = cv2.resize(image, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)[np.newaxis]   # (1, C, H, W)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── Classifier ────────────────────────────────────────────────────────────────

class Classifier:
    """
    Pure-Python image classifier backed by onnxruntime.

    Drop-in replacement for the C++ Classifier when bindings are unavailable.
    """

    def __init__(self, model: str, device: str = "auto", **kwargs):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "The pure-Python classifier fallback requires onnxruntime. Install "
                "it with `pip install onnxruntime`, or build the C++ bindings."
            ) from e

        if device == "auto":
            device = "cpu"

        providers = _get_providers(device)
        self._sess = ort.InferenceSession(model, providers=providers)
        self._input_name  = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        inp_shape = self._sess.get_inputs()[0].shape   # [1, C, H, W]
        self._input_hw = (int(inp_shape[2]), int(inp_shape[3]))   # (H, W)
        self._labels = _load_labels(model)
        self._device_label = _resolve_device_label(providers)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def input_size(self) -> Tuple[int, int]:
        """(width, height)"""
        h, w = self._input_hw
        return (w, h)

    @property
    def num_classes(self) -> int:
        return len(self._labels)

    @property
    def classes(self) -> List[str]:
        return list(self._labels)

    @property
    def device(self) -> str:
        return self._device_label

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, image: Union[str, np.ndarray], top_k: int = 5) -> _Results:
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")

        tensor = _preprocess(image, self._input_hw)

        t0 = time.perf_counter()
        logits = self._sess.run([self._output_name], {self._input_name: tensor})[0][0]
        infer_ms = (time.perf_counter() - t0) * 1000

        probs = _softmax(logits)
        top_k = min(top_k, len(probs))
        top_idx = probs.argsort()[::-1][:top_k]

        classifications = [
            _ClassificationResult(
                label=self._labels[i] if i < len(self._labels) else f"class_{i}",
                score=float(probs[i]),
                class_id=int(i),
            )
            for i in top_idx
        ]

        return _Results(
            classifications=classifications,
            inference_time_ms=infer_ms,
            device_used=self._device_label,
        )

    def __call__(self, image, top_k: int = 5) -> _Results:
        return self.predict(image, top_k=top_k)

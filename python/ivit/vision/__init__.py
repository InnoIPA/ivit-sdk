"""
Vision modules for iVIT-SDK.

High-level task APIs, each a thin wrapper over the native C++ implementation:
- Detector: Object detection
- Classifier: Image classification
- Segmentor: Semantic segmentation

These require the compiled bindings (``ivit.is_cpp_available()``); there is no pure-Python
fallback, so importing them without the bindings raises ImportError.
"""

from .detector import Detector
from .classifier import Classifier
from .segmentor import Segmentor

__all__ = [
    "Detector",
    "Classifier",
    "Segmentor",
]

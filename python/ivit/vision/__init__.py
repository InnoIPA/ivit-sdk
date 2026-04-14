"""
Vision modules for iVIT-SDK.

This module provides high-level APIs for computer vision tasks:
- Detector: Object detection
- Classifier: Image classification
- Segmentor: Semantic segmentation

The implementation automatically uses C++ bindings when available
for optimal performance, with pure Python fallback.
"""

from typing import TYPE_CHECKING

# Check if C++ bindings are available
_CPP_ERROR = None
try:
    from .._ivit_core import (
        Detector as _CppDetector,
        Classifier as _CppClassifier,
        Segmentor as _CppSegmentor,
    )
    _HAS_CPP_VISION = True
except ImportError as _e:
    _HAS_CPP_VISION = False
    _CPP_ERROR = str(_e)

if _HAS_CPP_VISION:
    # Use C++ implementations with Python wrappers for better ergonomics
    from .detector import Detector
    from .classifier import Classifier
    from .segmentor import Segmentor
else:
    # Fall back to pure-Python (onnxruntime) implementations
    from ._classifier_ort import Classifier

    # Detector and Segmentor have no pure-Python fallback yet; provide stubs
    # so that importing the module doesn't fail when only Classifier is needed.
    class Detector:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError(
                "Detector requires C++ bindings (not available on this system). "
                "Reason: " + (_CPP_ERROR or "unknown")
            )

    class Segmentor:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError(
                "Segmentor requires C++ bindings (not available on this system). "
                "Reason: " + (_CPP_ERROR or "unknown")
            )


__all__ = [
    "Detector",
    "Classifier",
    "Segmentor",
]

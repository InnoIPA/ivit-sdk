"""
Runtime backends for iVIT-SDK.

.. deprecated:: 1.0.0
    Pure Python runtimes are deprecated. All inference now uses C++ bindings.
    These modules are kept for reference only and will be removed in a future version.
"""

import warnings

warnings.warn(
    "ivit.runtime is deprecated. All inference now uses C++ bindings. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import Optional, List
import logging

from .base import BaseRuntime, BaseModel

logger = logging.getLogger(__name__)

# Registry of available runtimes
_runtimes = {}


def get_runtime(backend: str):
    """
    Get runtime instance for backend.

    Args:
        backend: Backend name (openvino, tensorrt, snpe)

    Returns:
        Runtime instance

    Raises:
        ValueError: If backend not found
    """
    backend = backend.lower()

    if backend in _runtimes:
        return _runtimes[backend]

    # Try to create runtime
    runtime = None

    if backend == "openvino":
        runtime = _create_openvino_runtime()
    elif backend == "tensorrt":
        runtime = _create_tensorrt_runtime()
    elif backend == "qnn":
        runtime = _create_qnn_runtime()
    elif backend == "snpe":
        # Legacy support - redirect to QNN
        runtime = _create_qnn_runtime()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if runtime is None:
        raise ValueError(f"Failed to create runtime for backend: {backend}")

    _runtimes[backend] = runtime
    return runtime


def _create_openvino_runtime():
    """The pure-Python OpenVINO runtime has been removed.

    OpenVINO inference now runs through the C++ backend (``_ivit_core``).
    """
    logger.warning(
        "The pure-Python OpenVINO runtime has been removed. "
        "OpenVINO inference now uses the C++ backend (_ivit_core)."
    )
    return None


def _create_tensorrt_runtime():
    """Create TensorRT runtime."""
    try:
        from .tensorrt_runtime import TensorRTRuntime
        return TensorRTRuntime()
    except ImportError as e:
        logger.warning(f"TensorRT not available: {e}")
        return None


def _create_qnn_runtime():
    """Create QNN (Qualcomm AI Engine Direct) runtime."""
    try:
        from .qnn_runtime import QNNRuntime
        return QNNRuntime()
    except ImportError as e:
        logger.warning(f"QNN not available: {e}")
        return None


# Legacy alias
_create_snpe_runtime = _create_qnn_runtime


def list_available_backends():
    """List available backends."""
    available = []

    try:
        import tensorrt
        available.append("tensorrt")
    except ImportError:
        pass

    return available

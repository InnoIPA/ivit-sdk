"""
iVIT-SDK: Innodisk Vision Intelligence Toolkit

Unified Computer Vision SDK with extensible hardware support.
Currently supports Intel (OpenVINO) and NVIDIA (TensorRT) platforms.

Note: C++ bindings are required. Install with: pip install ivit-sdk

Example:
    >>> import ivit
    >>>
    >>> # One-liner inference (auto device selection)
    >>> model = ivit.load("yolox-s.onnx")
    >>> results = model("image.jpg")
    >>> results.show()
    >>>
    >>> # Task-specific classes
    >>> detector = ivit.Detector("yolox-s.onnx", device="cuda:0")
    >>> results = detector.predict("image.jpg")
"""

__version__ = "1.0.0"
__author__ = "Innodisk AI Team"

# Pre-load native library if it exists in the package directory
import os as _os
import sys as _sys
import ctypes as _ctypes

_package_dir = _os.path.dirname(_os.path.abspath(__file__))

if _sys.platform == 'win32':
    # Windows: use os.add_dll_directory() so Python can find ivit.dll and OpenCV DLLs
    _dll_dirs = [
        _package_dir,                          # ivit.dll installed here (RUNTIME DESTINATION .)
        _os.path.join(_package_dir, "bin"),   # fallback for older wheels
        _os.path.join(_package_dir, "lib"),   # fallback
    ]
    # Also search OpenCV_DIR env if set
    _opencv_dir = _os.environ.get("OpenCV_DIR", "")
    if _opencv_dir:
        for _sub in ["bin\\Release", "bin\\Debug", "bin", "x64\\vc17\\bin", "x64\\vc18\\bin"]:
            _dll_dirs.append(_os.path.join(_opencv_dir, _sub))

    # Conda environments: add base env Library\bin for avif.dll and other conda-packaged libs
    # Active env's Library\bin (sibling of Python.exe's Scripts/)
    _python_prefix = _os.path.dirname(_sys.executable)  # e.g. envs\OpenV\
    _dll_dirs.append(_os.path.join(_python_prefix, "Library", "bin"))
    # Anaconda base Library\bin (parent of envs\)
    _conda_base = _os.environ.get("CONDA_ROOT", "") or _os.environ.get("CONDA_PREFIX_1", "")
    if not _conda_base:
        # Heuristic: base env is 3 levels up from envs/<name>/ → base/Library/bin
        _candidate = _os.path.dirname(_os.path.dirname(_python_prefix))
        if _os.path.isdir(_os.path.join(_candidate, "Library", "bin")):
            _conda_base = _candidate
    if _conda_base:
        _dll_dirs.append(_os.path.join(_conda_base, "Library", "bin"))

    # OpenVINO pip package: openvino.dll + device plugins live in openvino/libs/
    import site as _site
    for _sp in _site.getsitepackages():
        _ov_libs = _os.path.join(_sp, "openvino", "libs")
        if _os.path.isdir(_ov_libs):
            _dll_dirs.append(_ov_libs)
            break

    for _d in _dll_dirs:
        if _os.path.isdir(_d):
            try:
                _os.add_dll_directory(_d)
            except (OSError, AttributeError):
                pass
else:
    # Linux/macOS: pre-load libivit.so with RTLD_LOCAL to avoid symbol conflicts
    _libivit_path = _os.path.join(_package_dir, "libivit.so")
    if _os.path.exists(_libivit_path):
        try:
            _ctypes.CDLL(_libivit_path, mode=_ctypes.RTLD_LOCAL)
        except OSError:
            pass

# C++ bindings - optional for some use cases (e.g., zoo commands)
_CPP_AVAILABLE = False
_CPP_ERROR = None

try:
    from ._ivit_core import (
        # Enums
        DataType,
        Precision,
        Layout,

        # Configuration
        LoadConfig,
        InferConfig,

        # Data structures
        TensorInfo,
        DeviceInfo,
        BBox,
        Detection,
        ClassificationResult,
        Keypoint,
        Pose,

        # Results
        Results,

        # Vision models
        Classifier,
        Detector,
        Segmentor,

        # Model
        Model,
        load_model,

        # Callback system
        CallbackEvent,
        CallbackContext,

        # Runtime config
        OpenVINOConfig,
        TensorRTConfig,

        QNNConfig,
        HailoConfig,

        # Stream
        StreamResult,
        StreamIterator,

        # Functions
        version,
        set_log_level,
        set_cache_dir,

        # Exceptions
        IVITError,
        ModelLoadError,
        InferenceError,
        DeviceNotFoundError,
        UnsupportedFormatError,
    )
    _CPP_AVAILABLE = True
except ImportError as e:
    _CPP_ERROR = str(e)
    # Don't raise - allow package to load for zoo/CLI functionality
    # Functions that require C++ will check _CPP_AVAILABLE


def is_cpp_available():
    """Check if C++ bindings are available."""
    return _CPP_AVAILABLE


# ── Device discovery ──────────────────────────────────────────────────────────
# devices.py has the full Device dataclass and DeviceManager (OpenVINO-based
# discovery that works with or without C++ bindings). Always import it so both
# paths return the same Device type with all attributes (available, type, etc.)
from .devices import DeviceManager as _DeviceManager, Device, devices, D

_device_manager = _DeviceManager()


def _require_cpp():
    """Raise error if C++ bindings are not available."""
    if not _CPP_AVAILABLE:
        raise ImportError(
            "This feature requires C++ bindings. "
            "Please build and install with: pip install -e . "
            f"(Original error: {_CPP_ERROR})"
        )


def get_runtime_info():
    """Get runtime information about iVIT SDK."""
    return {
        "version": __version__,
        "cpp_bindings": _CPP_AVAILABLE,
        "mode": "C++ accelerated" if _CPP_AVAILABLE else "Python only (limited)",
    }


# Import extended Python exceptions (supplement C++ exceptions)
try:
    from .core.exceptions import (
        BackendNotAvailableError,
        InvalidInputError,
        ModelNotLoadedError,
        ConfigurationError,
        ModelConversionError,
        ResourceExhaustedError,
        UnsupportedOperationError,
        wrap_error,
    )
except ImportError:
    pass

# Import devices module for device discovery (requires C++ bindings)
devices = None
# Always use DeviceManager for list_devices / get_best_device.
# DeviceManager wraps C++ DeviceInfo → Python Device (normalising is_available
# → available, adding vendor/type/is_discrete) and falls back to OpenVINO
# Python API when C++ bindings are unavailable. This gives a single, consistent
# Device type regardless of backend availability.
list_devices = _device_manager
get_best_device = _device_manager.best

# Import Model Zoo (works without C++ bindings)
from . import zoo

# Import Training module (lazy load to avoid torch dependency)
try:
    from . import train
except ImportError:
    train = None


def load(
    source: str,
    device: str = "auto",
    task: str = None,
    **kwargs
):
    """
    Load a model for inference.

    Args:
        source: Model path (.onnx, .xml, .engine) or Model Zoo name
        device: Target device ("auto", "cpu", "cuda:0", "npu")
        task: Task type hint ("detect", "classify", "segment", "pose")

    Returns:
        Model: Loaded model ready for inference

    Examples:
        >>> model = ivit.load("yolox-s.onnx")
        >>> results = model.predict("image.jpg")
    """
    _require_cpp()

    if hasattr(device, 'id'):
        device = device.id
    elif device == "auto":
        best = devices.best()
        device = best.id

    config = LoadConfig()
    config.device = device
    if task:
        config.task = task
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return load_model(source, config)


__all__ = [
    # Version
    "__version__",

    # Main API
    "load",
    "load_model",
    "devices",
    "Device",
    "D",

    # Training module
    "train",

    # Enums
    "DataType",
    "Precision",
    "Layout",

    # Configuration
    "LoadConfig",
    "InferConfig",
    "OpenVINOConfig",
    "TensorRTConfig",

    "QNNConfig",
    "HailoConfig",

    # Data structures
    "TensorInfo",
    "DeviceInfo",
    "BBox",
    "Detection",
    "ClassificationResult",
    "Keypoint",
    "Pose",

    # Results
    "Results",

    # Model
    "Model",

    # Vision models
    "Classifier",
    "Detector",
    "Segmentor",

    # Callback
    "CallbackEvent",
    "CallbackContext",

    # Stream
    "StreamResult",
    "StreamIterator",

    # Functions
    "version",
    "list_devices",
    "get_best_device",
    "set_log_level",
    "set_cache_dir",
    "is_cpp_available",
    "get_runtime_info",

    # Exceptions
    "IVITError",
    "ModelLoadError",
    "InferenceError",
    "DeviceNotFoundError",
    "BackendNotAvailableError",
    "InvalidInputError",
    "ModelNotLoadedError",
    "ConfigurationError",
    "ModelConversionError",
    "ResourceExhaustedError",
    "UnsupportedOperationError",
    "UnsupportedFormatError",
]

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

# Pre-load libivit.so if it exists in the package directory
import os as _os
import ctypes as _ctypes

_package_dir = _os.path.dirname(_os.path.abspath(__file__))
_libivit_path = _os.path.join(_package_dir, "libivit.so")

if _os.path.exists(_libivit_path):
    try:
        # Use RTLD_LOCAL (not RTLD_GLOBAL) to avoid polluting the global symbol
        # table with OpenVINO / libcudart symbols, which conflict with PyTorch's
        # own CUDA runtime and cause munmap heap-corruption in torch.jit.load.
        # _ivit_core.so has a RUNPATH pointing to the build lib dir, so it
        # resolves libivit.so independently without RTLD_GLOBAL.
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
        list_devices,
        get_best_device,
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


# ── Python-only device discovery (used when C++ bindings are unavailable) ────

class _DeviceInfo:
    """Minimal DeviceInfo for the Python fallback path."""
    __slots__ = ("id", "name", "backend", "memory_total")

    def __init__(self, id: str, name: str, backend: str, memory_total: int = 0):
        self.id = id
        self.name = name
        self.backend = backend
        self.memory_total = memory_total

    def __repr__(self):
        return f"DeviceInfo(id={self.id!r}, name={self.name!r}, backend={self.backend!r})"


def _py_list_devices():
    """Return available devices using onnxruntime provider discovery."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        available = []

    devs = [_DeviceInfo("cpu", "CPU", "cpu", 0)]

    if "CUDAExecutionProvider" in available:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip().splitlines()
            for i, line in enumerate(out):
                parts = [p.strip() for p in line.split(",")]
                name = parts[0] if parts else "NVIDIA GPU"
                mem  = int(parts[1]) * 1024 * 1024 if len(parts) > 1 else 0
                devs.append(_DeviceInfo(f"cuda:{i}", name, "tensorrt", mem))
        except Exception:
            devs.append(_DeviceInfo("cuda:0", "NVIDIA GPU", "tensorrt", 0))

    if "OpenVINOExecutionProvider" in available:
        devs.append(_DeviceInfo("npu", "Intel NPU/iGPU", "openvino", 0))

    return devs


def _py_get_best_device():
    """Return best available device (GPU > NPU > CPU)."""
    devs = _py_list_devices()
    for dev in devs:
        if dev.id.startswith("cuda"):
            return dev
    for dev in devs:
        if dev.id == "npu":
            return dev
    return devs[0]


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
Device = None
D = None
if _CPP_AVAILABLE:
    try:
        from .devices import devices, Device, D
    except ImportError:
        pass

# When C++ is unavailable, expose Python-only list_devices / get_best_device
if not _CPP_AVAILABLE:
    list_devices = _py_list_devices
    get_best_device = _py_get_best_device

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

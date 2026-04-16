"""
Runtime configuration classes for hardware-specific settings.

Each backend has its own configuration class with tunable parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpenVINOConfig:
    """
    OpenVINO runtime configuration.

    Provides access to Intel-specific optimizations.

    Examples:
        >>> model.configure_openvino(
        ...     performance_mode="LATENCY",
        ...     num_streams=4,
        ...     inference_precision="FP16",
        ... )
    """
    # Performance hint
    performance_mode: str = "LATENCY"
    """Performance hint: "LATENCY" (default), "THROUGHPUT", or "CUMULATIVE_THROUGHPUT" """

    # Number of inference streams
    num_streams: int = 1
    """Number of inference streams for parallel execution"""

    # Inference precision
    inference_precision: str = "FP32"
    """Inference precision: "FP32", "FP16", or "INT8" """

    # CPU-specific settings
    enable_cpu_pinning: bool = False
    """Pin threads to CPU cores for better cache utilization"""

    num_threads: int = 0
    """Number of CPU threads (0 = auto)"""

    # NPU-specific settings
    npu_compilation_mode: str = "DefaultHW"
    """NPU compilation mode: "DefaultHW", "DefaultSW" """

    # Cache settings
    cache_dir: Optional[str] = None
    """Directory for model compilation cache"""

    # Device-specific properties
    device_properties: Dict[str, Any] = field(default_factory=dict)
    """Additional device-specific properties"""

    def to_ov_config(self) -> Dict[str, Any]:
        """Convert to OpenVINO config dict."""
        config = {}

        # Performance hint
        config["PERFORMANCE_HINT"] = self.performance_mode

        # Streams
        if self.num_streams > 0:
            config["NUM_STREAMS"] = str(self.num_streams)

        # Precision
        if self.inference_precision != "FP32":
            config["INFERENCE_PRECISION_HINT"] = self.inference_precision

        # CPU pinning
        if self.enable_cpu_pinning:
            config["ENABLE_CPU_PINNING"] = "YES"

        # Threads
        if self.num_threads > 0:
            config["INFERENCE_NUM_THREADS"] = str(self.num_threads)

        # Cache
        if self.cache_dir:
            config["CACHE_DIR"] = self.cache_dir

        # Additional properties
        config.update(self.device_properties)

        return config


@dataclass
class TensorRTConfig:
    """
    TensorRT runtime configuration.

    Provides access to NVIDIA-specific optimizations.

    Examples:
        >>> model.configure_tensorrt(
        ...     workspace_size=1 << 30,  # 1GB
        ...     dla_core=0,
        ...     enable_fp16=True,
        ... )
    """
    # Workspace size in bytes
    workspace_size: int = 1 << 30  # 1GB default
    """Maximum workspace size for TensorRT (bytes)"""

    # DLA (Deep Learning Accelerator) settings
    dla_core: int = -1
    """DLA core to use (-1 = disabled, 0-1 on Jetson)"""

    allow_gpu_fallback: bool = True
    """Allow GPU fallback for unsupported DLA layers"""

    # Precision settings
    enable_fp16: bool = False
    """Enable FP16 inference"""

    enable_int8: bool = False
    """Enable INT8 inference (requires calibration)"""

    strict_type_constraints: bool = False
    """Enforce strict type constraints"""

    # Optimization settings
    builder_optimization_level: int = 3
    """Builder optimization level (0-5, higher = more optimization)"""

    enable_sparsity: bool = False
    """Enable sparsity optimizations (Ampere+)"""

    # Timing cache
    timing_cache_path: Optional[str] = None
    """Path to timing cache file"""

    # Engine settings
    max_batch_size: int = 1
    """Maximum batch size"""

    # Profiling
    enable_profiling: bool = False
    """Enable layer profiling"""

    def to_trt_config(self) -> Dict[str, Any]:
        """Convert to TensorRT config dict."""
        return {
            "workspace_size": self.workspace_size,
            "dla_core": self.dla_core,
            "allow_gpu_fallback": self.allow_gpu_fallback,
            "enable_fp16": self.enable_fp16,
            "enable_int8": self.enable_int8,
            "strict_type_constraints": self.strict_type_constraints,
            "builder_optimization_level": self.builder_optimization_level,
            "enable_sparsity": self.enable_sparsity,
            "timing_cache_path": self.timing_cache_path,
            "max_batch_size": self.max_batch_size,
            "enable_profiling": self.enable_profiling,
        }


@dataclass
class QNNConfig:
    """
    Qualcomm QNN (AI Engine Direct) runtime configuration.

    Provides access to Hexagon Tensor Processor (HTP) optimizations
    for Qualcomm IQ Series (QCS9075, QCS8550, etc.).

    Examples:
        >>> model.configure_qnn(
        ...     backend="htp",
        ...     performance_profile="HIGH_PERFORMANCE",
        ... )
    """
    # Backend selection
    backend: str = "htp"
    """Backend: "cpu", "gpu", "htp" (Hexagon Tensor Processor) """

    # Performance profile
    performance_profile: str = "DEFAULT"
    """Performance profile: "DEFAULT", "BALANCED", "HIGH_PERFORMANCE", "POWER_SAVER", "SUSTAINED_HIGH_PERFORMANCE", "BURST" """

    # HTP-specific options
    htp_precision: str = "fp16"
    """HTP precision: "fp16", "int8" """

    htp_use_fold_relu: bool = True
    """Fold ReLU activations for better performance"""

    # Buffer optimization
    use_native_memory: bool = True
    """Use native memory for zero-copy between backends"""

    # Profiling
    enable_profiling: bool = False
    """Enable profiling"""

    profiling_level: str = "BASIC"
    """Profiling level: "OFF", "BASIC", "MODERATE", "DETAILED" """

    # Context caching
    cache_context: bool = True
    """Cache compiled context for faster subsequent loads"""

    def to_qnn_config(self) -> Dict[str, Any]:
        """Convert to QNN config dict."""
        return {
            "backend": self.backend,
            "performance_profile": self.performance_profile,
            "htp_precision": self.htp_precision,
            "htp_use_fold_relu": self.htp_use_fold_relu,
            "use_native_memory": self.use_native_memory,
            "enable_profiling": self.enable_profiling,
            "profiling_level": self.profiling_level,
            "cache_context": self.cache_context,
        }


# Backward compatibility alias
SNPEConfig = QNNConfig


@dataclass
class HailoConfig:
    """
    HailoRT runtime configuration for Hailo-8 AI acceleration card.

    Examples:
        >>> config = HailoConfig(batch_size=1, timeout_ms=1000)
    """
    batch_size: int = 1
    """Batch size for inference (default 1)"""

    timeout_ms: int = 1000
    """Inference timeout in milliseconds (default 1000)"""

    def to_hailo_config(self) -> Dict[str, Any]:
        """Convert to HailoRT config dict."""
        return {
            "batch_size": self.batch_size,
            "timeout_ms": self.timeout_ms,
        }

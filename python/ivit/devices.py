"""
iVIT Device Discovery and Selection API

Provides user-friendly device enumeration and selection with
intelligent vendor-aware priority ordering.

Design Principles:
1. Platform-Adaptive: Selects best device on current platform
2. Vendor-Aware: NVIDIA dGPU > Intel dGPU > Intel iGPU > NPU > CPU
3. Strategy-Based: Multiple selection strategies (latency, efficiency, balanced)

Examples:
    # List all devices
    >>> import ivit
    >>> ivit.devices()
    [Device(cpu), Device(cuda:0), Device(npu)]

    # Get specific device types
    >>> ivit.devices.cuda()      # First CUDA GPU
    >>> ivit.devices.cpu()       # CPU device
    >>> ivit.devices.npu()       # NPU device
    >>> ivit.devices.best()      # Auto-select best (latency-optimized)

    # Strategy-based selection
    >>> ivit.devices.best(strategy="latency")     # Default: lowest latency
    >>> ivit.devices.best(strategy="efficiency")  # Power-efficient (NPU preferred)

    # Use in model loading
    >>> model = ivit.load("model.onnx", device=ivit.devices.cuda())
    >>> model = ivit.load("model.onnx", device=ivit.devices.best())
"""

from typing import List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class DeviceType(Enum):
    """Device type enumeration."""
    CPU = "cpu"
    CUDA = "cuda"      # NVIDIA GPU
    GPU = "gpu"        # Intel GPU (OpenVINO)
    NPU = "npu"        # Intel NPU
    AUTO = "auto"


@dataclass
class Device:
    """
    Represents an inference device.

    Attributes:
        id: Device identifier (e.g., "cuda:0", "cpu", "npu")
        name: Human-readable device name
        type: Device type (cpu, cuda, npu, etc.)
        backend: Inference backend (openvino, tensorrt)
        vendor: Hardware vendor (nvidia, intel, amd)
        available: Whether device is currently available
        memory_total: Total memory in bytes (if applicable)
        compute_capability: Compute capability (for CUDA devices)
        is_discrete: True for discrete GPU (dGPU)
    """
    id: str
    name: str
    type: str
    backend: str
    vendor: str = "unknown"
    available: bool = True
    memory_total: int = 0
    compute_capability: str = ""
    is_discrete: bool = False

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        status = "available" if self.available else "unavailable"
        gpu_type = " (dGPU)" if self.is_discrete else ""
        return f"Device({self.id}, {self.vendor}{gpu_type}, {status})"


# The C++ DeviceInfo does not report a vendor or whether a GPU is discrete, but
# best() needs both to rank devices (e.g. NVIDIA dGPU above an Intel iGPU).
_VENDOR_BY_BACKEND = {"tensorrt": "nvidia", "openvino": "intel"}
_INTEL_DISCRETE_HINTS = ("arc", "a770", "a750", "a380")


def _vendor_and_discrete(dev_type: str, backend: str, name: str):
    """Infer (vendor, is_discrete) for a device reported by the C++ bindings."""
    if dev_type == "cpu":
        return "unknown", False  # CPU vendor is not implied by the backend
    vendor = _VENDOR_BY_BACKEND.get(backend, "unknown")
    if vendor == "nvidia":
        return vendor, True  # NVIDIA GPUs are discrete
    is_discrete = (
        vendor == "intel"
        and dev_type == "gpu"
        and any(hint in name.lower() for hint in _INTEL_DISCRETE_HINTS)
    )
    return vendor, is_discrete


class DeviceManager:
    """
    Device discovery and selection manager.

    Provides convenient methods to discover and select inference devices.

    Examples:
        >>> devices = DeviceManager()
        >>> devices()              # List all devices
        >>> devices.cuda()         # Get first CUDA device
        >>> devices.cuda(0)        # Get CUDA device 0
        >>> devices.cuda(1)        # Get CUDA device 1
        >>> devices.cpu()          # Get CPU device
        >>> devices.npu()          # Get NPU device
        >>> devices.best()         # Auto-select best device
        >>> devices.best("efficiency")  # Best for efficiency
    """

    _cache: Optional[List[Device]] = None
    _cache_valid: bool = False

    def __call__(self, refresh: bool = False) -> List[Device]:
        """
        Get list of all available devices.

        Args:
            refresh: Force refresh device list

        Returns:
            List of Device objects
        """
        if not self._cache_valid or refresh:
            self._discover_devices()
        return self._cache or []

    def _discover_devices(self) -> None:
        """Discover all available devices."""
        self._cache = []

        # Try to use C++ bindings if available
        try:
            from ivit import _ivit_core as core
            raw_devices = core.list_devices()

            for dev in raw_devices:
                vendor, is_discrete = _vendor_and_discrete(dev.type, dev.backend, dev.name)
                self._cache.append(Device(
                    id=dev.id,
                    name=dev.name,
                    type=dev.type,
                    backend=dev.backend,
                    vendor=vendor,
                    available=dev.is_available,
                    memory_total=getattr(dev, 'memory_total', 0),
                    is_discrete=is_discrete,
                ))
        except ImportError:
            # Fallback to Python-based discovery
            self._discover_openvino_devices()
            self._discover_cuda_devices()
            self._add_cpu_fallback()

        self._cache_valid = True

    def _discover_openvino_devices(self) -> None:
        """Discover OpenVINO devices (Intel hardware)."""
        try:
            from openvino import Core
            core = Core()

            for device_name in core.available_devices:
                try:
                    full_name = core.get_property(device_name, "FULL_DEVICE_NAME")
                except:
                    full_name = device_name

                # Determine type, ID, and check if discrete
                is_discrete = False
                if "CPU" in device_name:
                    dev_id = "cpu"
                    dev_type = "cpu"
                elif "GPU" in device_name:
                    dev_id = "gpu:0"
                    dev_type = "gpu"
                    # Check for discrete GPU (Arc)
                    name_lower = full_name.lower()
                    is_discrete = any(x in name_lower for x in ["arc", "a770", "a750", "a380"])
                elif "NPU" in device_name:
                    dev_id = "npu"
                    dev_type = "npu"
                else:
                    continue

                self._cache.append(Device(
                    id=dev_id,
                    name=full_name,
                    type=dev_type,
                    backend="openvino",
                    vendor="intel",
                    available=True,
                    is_discrete=is_discrete,
                ))
        except ImportError:
            pass

    def _discover_cuda_devices(self) -> None:
        """Discover CUDA devices (NVIDIA hardware)."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    self._cache.append(Device(
                        id=f"cuda:{i}",
                        name=props.name,
                        type="cuda",
                        backend="tensorrt",
                        vendor="nvidia",
                        available=True,
                        memory_total=props.total_memory,
                        compute_capability=f"{props.major}.{props.minor}",
                        is_discrete=True,  # NVIDIA GPUs are discrete
                    ))
        except ImportError:
            # Try pycuda
            try:
                import pycuda.driver as cuda
                cuda.init()

                for i in range(cuda.Device.count()):
                    dev = cuda.Device(i)
                    self._cache.append(Device(
                        id=f"cuda:{i}",
                        name=dev.name(),
                        type="cuda",
                        backend="tensorrt",
                        vendor="nvidia",
                        available=True,
                        memory_total=dev.total_memory(),
                        is_discrete=True,
                    ))
            except ImportError:
                pass

    def _add_cpu_fallback(self) -> None:
        """Add CPU as fallback if not already present."""
        has_cpu = any(d.type == "cpu" for d in self._cache)
        if not has_cpu:
            import platform
            cpu_info = platform.processor().lower()
            vendor = "intel" if "intel" in cpu_info else \
                     "amd" if "amd" in cpu_info else "unknown"

            self._cache.append(Device(
                id="cpu",
                name=platform.processor() or "CPU",
                type="cpu",
                backend="openvino",
                vendor=vendor,
                available=True,
            ))

    def cpu(self) -> Device:
        """
        Get CPU device.

        Returns:
            CPU Device object

        Example:
            >>> model = ivit.load("model.onnx", device=ivit.devices.cpu())
        """
        for dev in self():
            if dev.type == "cpu":
                return dev
        # Return default CPU device
        return Device(
            id="cpu",
            name="CPU",
            type="cpu",
            backend="openvino",
            available=True,
        )

    def cuda(self, index: int = 0) -> Optional[Device]:
        """
        Get CUDA (NVIDIA GPU) device.

        Args:
            index: GPU index (default: 0)

        Returns:
            CUDA Device object, or None if not available

        Example:
            >>> model = ivit.load("model.onnx", device=ivit.devices.cuda())
            >>> model = ivit.load("model.onnx", device=ivit.devices.cuda(1))
        """
        target_id = f"cuda:{index}"
        for dev in self():
            if dev.id == target_id:
                return dev
        return None

    def gpu(self, index: int = 0) -> Optional[Device]:
        """
        Get Intel GPU device (OpenVINO).

        Args:
            index: GPU index (default: 0)

        Returns:
            Intel GPU Device object, or None if not available

        Example:
            >>> model = ivit.load("model.onnx", device=ivit.devices.gpu())
        """
        target_id = f"gpu:{index}"
        for dev in self():
            if dev.id == target_id or (dev.type == "gpu" and dev.backend == "openvino"):
                return dev
        return None

    def npu(self) -> Optional[Device]:
        """
        Get NPU (Neural Processing Unit) device.

        Returns:
            NPU Device object, or None if not available

        Example:
            >>> model = ivit.load("model.onnx", device=ivit.devices.npu())
        """
        for dev in self():
            if dev.type == "npu":
                return dev
        return None

    def best(self, strategy: str = "latency") -> Device:
        """
        Auto-select the best available device using intelligent scoring.

        The selection considers:
        1. Vendor priority (NVIDIA dGPU > Intel dGPU > Intel iGPU > NPU > CPU)
        2. Device availability and resources
        3. Selection strategy

        Args:
            strategy: Selection strategy
                - "latency": Lowest inference latency (default, recommended)
                - "throughput": Highest batch processing capacity
                - "efficiency": Best performance per watt (NPU preferred)
                - "balanced": Balance between latency and efficiency

        Returns:
            Best Device object based on strategy

        Example:
            >>> model = ivit.load("model.onnx", device=ivit.devices.best())
            >>> model = ivit.load("model.onnx", device=ivit.devices.best("efficiency"))
        """
        # Same ranking as the C++ get_best_device (src/core/device_ranking.hpp) for
        # "latency"/"throughput" (= C++ "performance") and "efficiency"; keep them in
        # step. "balanced" exists only here.
        available = [d for d in self() if d.available]

        if not available:
            return self.cpu()

        if strategy in ("latency", "throughput"):
            # Order: NVIDIA dGPU > Intel dGPU > Intel iGPU > NPU > CPU
            def score(d):
                base = 0
                if d.vendor == "nvidia":
                    base = 100
                elif d.vendor == "intel" and d.is_discrete:
                    base = 80
                elif d.type == "gpu":
                    base = 60
                elif d.type == "npu":
                    base = 40
                else:
                    base = 10
                return base

            available.sort(key=score, reverse=True)

        elif strategy == "efficiency":
            # Order: NPU > iGPU > CPU > dGPU
            def score(d):
                if d.type == "npu":
                    return 100
                elif d.type == "gpu" and not d.is_discrete:
                    return 60
                elif d.type == "cpu":
                    return 40
                else:
                    return 20

            available.sort(key=score, reverse=True)

        else:  # balanced
            # Order: NPU > NVIDIA > Intel GPU > CPU
            def score(d):
                if d.type == "npu":
                    return 90
                elif d.vendor == "nvidia":
                    return 85
                elif d.type == "gpu":
                    return 70
                else:
                    return 30

            available.sort(key=score, reverse=True)

        return available[0]

    def filter(
        self,
        type: Optional[str] = None,
        backend: Optional[str] = None,
        available_only: bool = True,
    ) -> List[Device]:
        """
        Filter devices by criteria.

        Args:
            type: Filter by device type (cuda, cpu, npu, etc.)
            backend: Filter by backend (tensorrt, openvino, snpe)
            available_only: Only return available devices

        Returns:
            List of matching Device objects

        Example:
            >>> gpus = ivit.devices.filter(type="cuda")
            >>> ov_devices = ivit.devices.filter(backend="openvino")
        """
        result = self()

        if type:
            result = [d for d in result if d.type == type]
        if backend:
            result = [d for d in result if d.backend == backend]
        if available_only:
            result = [d for d in result if d.available]

        return result

    def refresh(self) -> List[Device]:
        """
        Force refresh device list.

        Returns:
            Updated list of Device objects
        """
        return self(refresh=True)

    def summary(self) -> str:
        """
        Get formatted summary of available devices.

        Returns:
            Formatted string summary

        Example:
            >>> print(ivit.devices.summary())
        """
        lines = ["Available Devices:", "=" * 50]

        for dev in self():
            status = "✓" if dev.available else "✗"
            mem = f" ({dev.memory_total // 1024 // 1024} MB)" if dev.memory_total else ""
            lines.append(f"  {status} {dev.id:12} {dev.name}{mem}")
            lines.append(f"    Backend: {dev.backend}, Type: {dev.type}")

        if not self._cache:
            lines.append("  No devices found")

        return "\n".join(lines)


# Global device manager instance
devices = DeviceManager()


# Convenience constants for common devices
class D:
    """
    Device constants for quick access.

    Example:
        >>> model = ivit.load("model.onnx", device=ivit.D.AUTO)
        >>> model = ivit.load("model.onnx", device=ivit.D.CPU)
        >>> model = ivit.load("model.onnx", device=ivit.D.CUDA)
    """
    # Auto selection (recommended)
    AUTO = "auto"

    # NVIDIA CUDA devices
    CUDA = "cuda:0"
    CUDA_0 = "cuda:0"
    CUDA_1 = "cuda:1"

    # Intel OpenVINO devices
    CPU = "cpu"
    GPU = "gpu:0"      # Intel iGPU/dGPU
    GPU_0 = "gpu:0"
    NPU = "npu"        # Intel NPU

"""
Device management for iVIT-SDK.

Delegates to C++ bindings for device discovery and selection.
"""

from typing import List, Optional

from .._ivit_core import (
    list_devices as _cpp_list_devices,
    get_best_device as _cpp_get_best_device,
    DeviceInfo,
)


def list_devices(refresh: bool = False) -> List[DeviceInfo]:
    """
    List all available inference devices.

    Args:
        refresh: Force refresh device list (reserved for future use)

    Returns:
        List of available DeviceInfo objects
    """
    return _cpp_list_devices()


def get_best_device(priority: str = "performance") -> DeviceInfo:
    """
    Get the best available device.

    Args:
        priority: Selection priority ("performance" or "efficiency")

    Returns:
        Best available DeviceInfo
    """
    return _cpp_get_best_device(priority)


def get_device(device_id: str) -> Optional[DeviceInfo]:
    """
    Get device info by ID.

    Args:
        device_id: Device identifier (e.g., "cuda:0", "cpu")

    Returns:
        DeviceInfo or None if not found
    """
    for dev in list_devices():
        if dev.id == device_id:
            return dev
    return None


def get_backend_for_device(device_id: str) -> str:
    """
    Get the backend name for a device.

    Args:
        device_id: Device identifier

    Returns:
        Backend name string
    """
    dev = get_device(device_id)
    if dev:
        return dev.backend
    # Same rules as the C++ parse_device_string(): "gpu" is an Intel GPU (OpenVINO); only
    # "cuda" means NVIDIA (TensorRT).
    name = device_id.lower()
    if name == "auto":
        return "auto"
    if name.startswith("cuda"):
        return "tensorrt"
    return "openvino"

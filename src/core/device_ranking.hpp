#pragma once
// Device ranking shared by get_best_device(). Kept in step with the Python ranking in
// python/ivit/devices.py (DeviceManager.best), so both APIs pick the same device.

#include "ivit/core/device.hpp"
#include <algorithm>
#include <cctype>
#include <string>

namespace ivit::detail {

inline bool is_discrete_gpu(const DeviceInfo &device) {
    if (device.type != "gpu")
        return false;
    if (device.backend == "tensorrt")
        return true; // NVIDIA GPUs are discrete
    std::string name = device.name;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const char *hint : {"arc", "a770", "a750", "a380"}) // Intel discrete GPUs
        if (name.find(hint) != std::string::npos)
            return true;
    return false;
}

/// Higher is better. Unknown priorities fall back to "performance".
inline int device_score(const DeviceInfo &device, const std::string &priority) {
    const bool gpu = device.type == "gpu";
    const bool discrete = is_discrete_gpu(device);
    if (priority == "efficiency") { // NPU > iGPU > CPU > discrete GPU
        if (device.type == "npu")
            return 100;
        if (gpu && !discrete)
            return 60;
        if (device.type == "cpu")
            return 40;
        return 20;
    }
    if (priority == "memory") { // the most memory first: discrete GPU > iGPU > CPU > NPU
        if (discrete)
            return 100;
        if (gpu)
            return 60;
        if (device.type == "cpu")
            return 40;
        return 20;
    }
    // performance: NVIDIA GPU > Intel discrete GPU > Intel iGPU > NPU > CPU
    if (gpu && device.backend == "tensorrt")
        return 100;
    if (discrete)
        return 80;
    if (gpu)
        return 60;
    if (device.type == "npu")
        return 40;
    return 10;
}

} // namespace ivit::detail

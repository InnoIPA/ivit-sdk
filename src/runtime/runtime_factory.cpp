/**
 * @file runtime_factory.cpp
 * @brief Runtime factory implementation
 */

#include "ivit/runtime/runtime.hpp"
#include "ivit/core/device.hpp"
#include <algorithm>
#include <filesystem>

namespace ivit {

// ============================================================================
// RuntimeFactory implementation
// ============================================================================

RuntimeFactory& RuntimeFactory::instance() {
    static RuntimeFactory instance;
    return instance;
}

RuntimeFactory::RuntimeFactory() {
    // Runtimes are registered by their respective implementations
}

void RuntimeFactory::register_runtime(std::shared_ptr<IRuntime> runtime) {
    if (runtime && runtime->is_available()) {
        runtimes_[runtime->type()] = runtime;
    }
}

std::shared_ptr<IRuntime> RuntimeFactory::get_runtime(BackendType type) {
    auto it = runtimes_.find(type);
    if (it != runtimes_.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<IRuntime> RuntimeFactory::get_runtime_for_device(const std::string& device) {
    // Parse device string to determine backend
    BackendType backend = get_backend_for_device(device);

    if (backend == BackendType::Auto) {
        // Try to find the best available backend
        // Priority: TensorRT > OpenVINO
        if (runtimes_.count(BackendType::TensorRT)) {
            return runtimes_[BackendType::TensorRT];
        }

        if (runtimes_.count(BackendType::OpenVINO)) {
            return runtimes_[BackendType::OpenVINO];
        }

        return nullptr;
    }

    return get_runtime(backend);
}

std::shared_ptr<IRuntime> RuntimeFactory::get_best_runtime(
    const std::string& model_path,
    const std::string& device
) {
    namespace fs = std::filesystem;

    std::string ext = fs::path(model_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Determine backend based on model format
    if (ext == ".engine" || ext == ".trt") {
        auto rt = get_runtime(BackendType::TensorRT);
        if (rt) return rt;
    } else if (ext == ".xml" || ext == ".bin") {
        auto rt = get_runtime(BackendType::OpenVINO);
        if (rt) return rt;
    } else if (ext == ".hef") {
        auto rt = get_runtime(BackendType::HailoRT);
        if (rt) return rt;
    } else if (ext == ".dlc" || ext == ".serialized") {
        // QNN supports DLC (legacy SNPE) and serialized context formats
        auto rt = get_runtime(BackendType::QNN);
        if (rt) return rt;
    } else if (ext == ".onnx") {
        // ONNX can be used by multiple backends
        // Respect device selection: cpu/npu/gpu -> OpenVINO, cuda -> TensorRT
        if (device == "cpu" || device.find("npu") == 0 || device.find("gpu") == 0) {
            auto rt = get_runtime(BackendType::OpenVINO);
            if (rt) return rt;
        }

        // Priority: TensorRT > OpenVINO
        auto rt = get_runtime(BackendType::TensorRT);
        if (rt) return rt;

        rt = get_runtime(BackendType::OpenVINO);
        if (rt) return rt;
    }

    // Fallback to device-based selection
    return get_runtime_for_device(device);
}

std::vector<BackendType> RuntimeFactory::list_available_runtimes() const {
    std::vector<BackendType> result;
    result.reserve(runtimes_.size());
    for (const auto& [type, runtime] : runtimes_) {
        if (runtime->is_available()) {
            result.push_back(type);
        }
    }
    return result;
}

} // namespace ivit

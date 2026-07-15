/**
 * @file device_manager.cpp
 * @brief Device manager implementation
 */

#include "ivit/ivit.hpp"
#include "ivit/core/device.hpp"
#include "ivit/runtime/runtime.hpp"
#include <algorithm>
#include <mutex>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "ivit/core/platform_compat.hpp"

#ifdef IVIT_HAS_OPENVINO
#include <openvino/openvino.hpp>
#endif

#ifdef IVIT_HAS_TENSORRT
#include <cuda_runtime.h>
#endif

#ifdef IVIT_HAS_QNN
#include <dlfcn.h>
#endif

#ifdef IVIT_HAS_HAILORT
// <filesystem> already included above
#endif

namespace ivit {

// ============================================================================
// DeviceManager implementation
// ============================================================================

DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

DeviceManager::DeviceManager() {
    discover_devices();
}

std::vector<DeviceInfo> DeviceManager::list_devices() {
    if (!initialized_) {
        discover_devices();
    }
    return devices_;
}

DeviceInfo DeviceManager::get_device(const std::string& device_id) {
    auto devices = list_devices();

    for (const auto& dev : devices) {
        if (dev.id == device_id) {
            return dev;
        }
    }

    throw DeviceNotFoundError(device_id);
}

DeviceInfo DeviceManager::get_best_device(
    const std::string& task,
    const std::string& priority
) {
    auto devices = list_devices();

    if (devices.empty()) {
        throw DeviceError("No devices available");
    }

    // Priority order based on selection criteria
    std::vector<std::string> type_order;
    if (priority == "performance") {
        type_order = {"gpu", "npu", "vpu", "cpu"};
    } else if (priority == "efficiency") {
        type_order = {"npu", "vpu", "gpu", "cpu"};
    } else { // memory
        type_order = {"gpu", "cpu", "npu", "vpu"};
    }

    // Sort devices by priority
    std::sort(devices.begin(), devices.end(),
        [&type_order](const DeviceInfo& a, const DeviceInfo& b) {
            auto it_a = std::find(type_order.begin(), type_order.end(), a.type);
            auto it_b = std::find(type_order.begin(), type_order.end(), b.type);

            int idx_a = (it_a != type_order.end()) ?
                        std::distance(type_order.begin(), it_a) :
                        type_order.size();
            int idx_b = (it_b != type_order.end()) ?
                        std::distance(type_order.begin(), it_b) :
                        type_order.size();

            return idx_a < idx_b;
        });

    // Return first available device
    for (const auto& dev : devices) {
        if (dev.is_available) {
            return dev;
        }
    }

    return devices[0];
}

DeviceStatus DeviceManager::get_device_status(const std::string& device_id) {
    DeviceStatus status;
    status.id = device_id;

    // Determine backend from device
    auto [backend, device_index] = parse_device_string(device_id);

    switch (backend) {
        case BackendType::TensorRT:
#ifdef IVIT_HAS_TENSORRT
            {
                // Get CUDA device index
                int dev_idx = device_index.empty() ? 0 : std::stoi(device_index);

                // Set device and query properties
                cudaError_t err = cudaSetDevice(dev_idx);
                if (err != cudaSuccess) {
                    status.is_available = false;
                    break;
                }

                // Query device properties for name
                cudaDeviceProp props;
                if (cudaGetDeviceProperties(&props, dev_idx) == cudaSuccess) {
                    status.name = props.name;
                    status.compute_capability = std::to_string(props.major) + "." +
                                                std::to_string(props.minor);
                }

                // Query memory info
                size_t free_mem = 0, total_mem = 0;
                if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                    status.memory_total = total_mem;
                    status.memory_used = total_mem - free_mem;
                    status.memory_free = free_mem;
                    // Calculate utilization percentage
                    if (total_mem > 0) {
                        status.memory_utilization = static_cast<float>(status.memory_used) /
                                                    static_cast<float>(total_mem) * 100.0f;
                    }
                }

                status.is_available = true;
                status.backend = "tensorrt";

                // Note: GPU utilization, temperature, and power require NVML library
                // which is not included as a dependency to keep the SDK lightweight.
                // Users can query these metrics separately using nvidia-smi or NVML.
            }
#endif
            break;

        case BackendType::OpenVINO:
#ifdef IVIT_HAS_OPENVINO
            try {
                ov::Core core;

                // Map device_id to OpenVINO device name
                std::string ov_device;
                if (device_id == "cpu") {
                    ov_device = "CPU";
                } else if (device_id == "gpu:0" || device_id == "gpu") {
                    ov_device = "GPU";
                } else if (device_id == "npu") {
                    ov_device = "NPU";
                } else {
                    ov_device = "CPU";
                }

                // Query device full name
                try {
                    status.name = core.get_property(ov_device, ov::device::full_name);
                } catch (...) {
                    status.name = ov_device;
                }

                // Query available device capabilities
                try {
                    auto caps = core.get_property(ov_device, ov::device::capabilities);
                    for (const auto& cap : caps) {
                        if (cap == "FP16") status.supports_fp16 = true;
                        if (cap == "INT8") status.supports_int8 = true;
                        if (cap == "FP32") status.supports_fp32 = true;
                    }
                } catch (...) {
                    // Capabilities query not supported
                    status.supports_fp32 = true;  // Default
                }

                status.is_available = true;
                status.backend = "openvino";

            } catch (...) {
                status.is_available = false;
            }
#endif
            break;

        default:
            status.is_available = false;
            break;
    }

    return status;
}

bool DeviceManager::supports_precision(
    const std::string& device_id,
    Precision precision
) {
    auto dev = get_device(device_id);

    for (const auto& p : dev.supported_precisions) {
        if (p == precision) {
            return true;
        }
    }

    return false;
}

bool DeviceManager::supports_format(
    const std::string& device_id,
    const std::string& format
) {
    // Determine backend from device
    auto [backend, _] = parse_device_string(device_id);

    // Normalize format string (ensure leading dot, lowercase)
    std::string fmt = format;
    if (!fmt.empty() && fmt[0] != '.') {
        fmt = "." + fmt;
    }
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);

    // Define supported formats per backend
    // [Future] Add new backend format support here
    switch (backend) {
        case BackendType::OpenVINO:
            // OpenVINO supports IR format (.xml + .bin) and ONNX
            return (fmt == ".xml" || fmt == ".bin" || fmt == ".onnx" || fmt == ".pdmodel");

        case BackendType::TensorRT:
            // TensorRT supports compiled engines and ONNX (for runtime compilation)
            return (fmt == ".engine" || fmt == ".trt" || fmt == ".plan" || fmt == ".onnx");

        case BackendType::QNN:
            // QNN supports DLC (legacy SNPE), serialized context, and ONNX
            return (fmt == ".dlc" || fmt == ".serialized" || fmt == ".bin" || fmt == ".onnx");

        case BackendType::Auto:
            // Auto backend - accept common formats
            return (fmt == ".onnx" || fmt == ".xml" || fmt == ".engine" ||
                    fmt == ".trt" || fmt == ".plan" || fmt == ".dlc");

        default:
            return false;
    }
}

void DeviceManager::refresh() {
    devices_.clear();
    initialized_ = false;
    discover_devices();
}

void DeviceManager::discover_devices() {
    devices_.clear();

    // ========================================================================
    // Device Discovery - Add new hardware platforms here
    // ========================================================================

    // Intel: OpenVINO (CPU, iGPU, NPU, VPU)
    discover_openvino_devices();

    // NVIDIA: TensorRT/CUDA (dGPU, Jetson)
    discover_tensorrt_devices();

    // Qualcomm: QNN / AI Engine Direct (HTP, Adreno GPU)
    discover_qnn_devices();

    // Hailo: HailoRT (Hailo-8 PCIe AI accelerator)
    discover_hailort_devices();

    // [Future] Add new hardware platforms here:
    // discover_xxx_devices();

    // ========================================================================
    // CPU Fallback - Available when OpenVINO is installed
    // ========================================================================
    bool has_cpu = false;
    for (const auto& dev : devices_) {
        if (dev.type == "cpu") {
            has_cpu = true;
            break;
        }
    }

    if (!has_cpu && openvino_is_available()) {
        DeviceInfo cpu;
        cpu.id = "cpu";
        cpu.name = "CPU";
        cpu.backend = "openvino";
        cpu.type = "cpu";
        cpu.is_available = true;
        cpu.supported_precisions = {Precision::FP32};
        devices_.push_back(cpu);
    }

    initialized_ = true;
}

void DeviceManager::discover_openvino_devices() {
#ifdef IVIT_HAS_OPENVINO
    try {
        // Suppress stderr noise from Intel OpenCL compiler (IGC) during device enumeration
        fflush(stderr);
        int saved_stderr = ivit_dup(IVIT_STDERR_FILENO);
        int devnull = ivit_open(IVIT_DEVNULL, O_WRONLY);
        ivit_dup2(devnull, IVIT_STDERR_FILENO);
        ivit_close(devnull);

        ov::Core core;
        auto available = core.get_available_devices();

        // Restore stderr
        fflush(stderr);
        ivit_dup2(saved_stderr, IVIT_STDERR_FILENO);
        ivit_close(saved_stderr);

        for (const auto& device_name : available) {
            DeviceInfo info;

            // Get full device name
            try {
                info.name = core.get_property(device_name, ov::device::full_name);
            } catch (...) {
                info.name = device_name;
            }

            // Determine device type and ID
            if (device_name.find("CPU") != std::string::npos) {
                info.id = "cpu";
                info.type = "cpu";
            } else if (device_name.find("GPU") != std::string::npos) {
                // Filter out NVIDIA GPUs — they should use TensorRT, not OpenVINO
                std::string name_upper = info.name;
                std::transform(name_upper.begin(), name_upper.end(), name_upper.begin(), ::toupper);
                if (name_upper.find("NVIDIA") != std::string::npos ||
                    name_upper.find("GEFORCE") != std::string::npos) {
                    continue;
                }
                info.id = "gpu:0";
                info.type = "gpu";
            } else if (device_name.find("NPU") != std::string::npos) {
                info.id = "npu";
                info.type = "npu";
            } else if (device_name.find("MYRIAD") != std::string::npos ||
                       device_name.find("VPU") != std::string::npos) {
                info.id = "vpu";
                info.type = "vpu";
            } else {
                info.id = device_name;
                info.type = "unknown";
            }

            info.backend = "openvino";
            info.is_available = true;
            info.supported_precisions = {
                Precision::FP32,
                Precision::FP16,
                Precision::INT8
            };

            devices_.push_back(info);
        }
    } catch (...) {
        // OpenVINO not available or error
    }
#endif
}

void DeviceManager::discover_tensorrt_devices() {
#ifdef IVIT_HAS_TENSORRT
    try {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);

        if (err == cudaSuccess && device_count > 0) {
            for (int i = 0; i < device_count; i++) {
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, i);

                DeviceInfo info;
                info.id = "cuda:" + std::to_string(i);
                info.name = props.name;
                info.backend = "tensorrt";
                info.type = "gpu";
                info.memory_total = props.totalGlobalMem;
                info.is_available = true;
                info.supported_precisions = {
                    Precision::FP32,
                    Precision::FP16,
                    Precision::INT8
                };

                devices_.push_back(info);
            }
        }
    } catch (...) {
        // CUDA/TensorRT not available
    }
#endif
}

void DeviceManager::discover_qnn_devices() {
#ifdef IVIT_HAS_QNN
    // Probe backend libraries via dlopen — no QNN init required just for discovery.
    struct {
        const char* lib;
        const char* id;
        const char* display_name;
        const char* type;
    } candidates[] = {
        {"libQnnHtp.so", "htp",     "Qualcomm Hexagon HTP (NPU)", "npu"},
        {"libQnnGpu.so", "qnn:gpu", "Qualcomm Adreno GPU",         "gpu"},
        {"libQnnCpu.so", "qnn:cpu", "Qualcomm QNN CPU",            "cpu"},
    };

    for (const auto& c : candidates) {
        void* h = dlopen(c.lib, RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        dlclose(h);

        DeviceInfo info;
        info.id        = c.id;
        info.name      = c.display_name;
        info.backend   = "qnn";
        info.type      = c.type;
        info.is_available = true;

        if (std::string(c.type) == "npu") {
            info.supported_precisions = {Precision::FP32, Precision::FP16, Precision::INT8};
        } else if (std::string(c.type) == "gpu") {
            info.supported_precisions = {Precision::FP32, Precision::FP16};
        } else {
            info.supported_precisions = {Precision::FP32};
        }

        devices_.push_back(info);
    }
#endif
}

// ============================================================================
// Helper functions
// ============================================================================

std::pair<BackendType, std::string> parse_device_string(const std::string& device) {
    std::string dev_lower = device;
    std::transform(dev_lower.begin(), dev_lower.end(), dev_lower.begin(), ::tolower);

    if (dev_lower == "auto") {
        return {BackendType::Auto, ""};
    }

    if (dev_lower.rfind("cuda", 0) == 0) {
        // Extract device index if present
        if (dev_lower.length() > 5 && dev_lower[4] == ':') {
            return {BackendType::TensorRT, dev_lower.substr(5)};
        }
        return {BackendType::TensorRT, "0"};
    }

    if (dev_lower == "cpu" || dev_lower.rfind("cpu:", 0) == 0) {
        return {BackendType::OpenVINO, "CPU"};
    }

    if (dev_lower == "gpu" || dev_lower.rfind("gpu:", 0) == 0) {
        return {BackendType::OpenVINO, "GPU"};
    }

    if (dev_lower == "npu") {
        return {BackendType::OpenVINO, "NPU"};
    }

    if (dev_lower == "vpu" || dev_lower == "myriad") {
        return {BackendType::OpenVINO, "MYRIAD"};
    }

    // Qualcomm IQ Series (QCS9075, QCS8550, etc.) via QNN
    // Support both iQ series names and internal Qualcomm names
    if (dev_lower == "iq9" || dev_lower == "iq8" || dev_lower == "iq6" ||
        dev_lower.rfind("iq", 0) == 0 ||  // Any "iq*" pattern
        dev_lower == "hexagon" || dev_lower == "htp" || dev_lower == "dsp") {
        return {BackendType::QNN, "HTP"};  // Hexagon Tensor Processor
    }

    // Hailo-8 PCIe AI accelerator via HailoRT
    if (dev_lower == "hailo" || dev_lower.rfind("hailo:", 0) == 0) {
        std::string idx = "0";
        auto pos = dev_lower.find(':');
        if (pos != std::string::npos) idx = dev_lower.substr(pos + 1);
        return {BackendType::HailoRT, idx};
    }

    // Default to CPU (OpenVINO fallback)
    return {BackendType::OpenVINO, "CPU"};
}

BackendType get_backend_for_device(const std::string& device) {
    return parse_device_string(device).first;
}

bool cuda_is_available() {
#ifdef IVIT_HAS_TENSORRT
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

int cuda_device_count() {
#ifdef IVIT_HAS_TENSORRT
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
#else
    return 0;
#endif
}

bool openvino_is_available() {
#ifdef IVIT_HAS_OPENVINO
    try {
        ov::Core core;
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool qnn_is_available() {
#ifdef IVIT_HAS_QNN
    for (const char* lib : {"libQnnHtp.so", "libQnnCpu.so", "libQnnGpu.so"}) {
        void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (h) { dlclose(h); return true; }
    }
#endif
    return false;
}
void DeviceManager::discover_hailort_devices() {
#ifdef IVIT_HAS_HAILORT
    for (int i = 0; i < 8; ++i) {
        std::string dev_node = "/dev/hailo" + std::to_string(i);
        if (!std::filesystem::exists(dev_node)) break;

        DeviceInfo info;
        info.id           = "hailo:" + std::to_string(i);
        info.name         = "Hailo-8 (device " + std::to_string(i) + ")";
        info.backend      = "hailort";
        info.type         = "npu";
        info.is_available = true;
        info.supported_precisions = {Precision::FP32, Precision::FP16, Precision::INT8};

        devices_.push_back(info);
    }
#endif
}

bool hailo_is_available() {
#ifdef IVIT_HAS_HAILORT
    return std::filesystem::exists("/dev/hailo0");
#else
    return false;
#endif
}

// =============================================================================
// Template for new hardware platform availability check
// =============================================================================
//
// bool xxx_is_available() {
// #ifdef IVIT_HAS_XXX
//     try {
//         // Initialize XXX SDK and check availability
//         return xxx_init() == XXX_SUCCESS;
//     } catch (...) {
//         return false;
//     }
// #else
//     return false;
// #endif
// }

// ============================================================================
// Free functions (declared in ivit.hpp)
// ============================================================================

std::vector<DeviceInfo> list_devices() {
    return DeviceManager::instance().list_devices();
}

DeviceInfo get_best_device(
    const std::string& task,
    const std::string& priority
) {
    return DeviceManager::instance().get_best_device(task, priority);
}

// Global log level setting
static std::string g_log_level = "info";

static LogLevel parse_log_level(const std::string& level) {
    std::string l = level;
    std::transform(l.begin(), l.end(), l.begin(), ::tolower);

    if (l == "debug" || l == "verbose") return LogLevel::DEBUG;
    if (l == "info") return LogLevel::INFO;
    if (l == "warning" || l == "warn") return LogLevel::WARNING;
    if (l == "error") return LogLevel::ERROR;
    if (l == "off" || l == "none") return LogLevel::OFF;

    return LogLevel::INFO;  // Default
}

// Global log level for iVIT SDK
static LogLevel g_parsed_log_level = LogLevel::INFO;

LogLevel get_parsed_log_level() {
    return g_parsed_log_level;
}

static void configure_openvino_logging([[maybe_unused]] LogLevel level) {
#ifdef IVIT_HAS_OPENVINO
    // OpenVINO logging can be configured via environment variables
    // Set environment variable to control OpenVINO logging level
    switch (level) {
        case LogLevel::DEBUG:
            ivit_setenv("OPENVINO_LOG_LEVEL", "0", 1);  // TRACE/DEBUG
            break;
        case LogLevel::INFO:
            ivit_setenv("OPENVINO_LOG_LEVEL", "1", 1);  // INFO
            break;
        case LogLevel::WARNING:
            ivit_setenv("OPENVINO_LOG_LEVEL", "2", 1);  // WARNING
            break;
        case LogLevel::ERROR:
            ivit_setenv("OPENVINO_LOG_LEVEL", "3", 1);  // ERROR
            break;
        case LogLevel::OFF:
            ivit_setenv("OPENVINO_LOG_LEVEL", "4", 1);  // NO_LOG
            break;
    }
#endif
}

static void configure_tensorrt_logging([[maybe_unused]] LogLevel level) {
#ifdef IVIT_HAS_TENSORRT
    // TensorRT logging is configured through ILogger interface
    // The TensorRT runtime uses a global logger that respects g_parsed_log_level
    // (see tensorrt_runtime.cpp for TRTLogger implementation)
    //
    // TensorRT severity levels:
    //   kINTERNAL_ERROR = 0  - internal errors
    //   kERROR = 1           - errors
    //   kWARNING = 2         - warnings
    //   kINFO = 3            - informational messages
    //   kVERBOSE = 4         - verbose debug messages
    //
    // The mapping is handled in TRTLogger::log() by checking g_parsed_log_level
#endif
}

void set_log_level(const std::string& level) {
    g_log_level = level;
    LogLevel parsed = parse_log_level(level);
    g_parsed_log_level = parsed;

    // Configure backend-specific logging
    configure_openvino_logging(parsed);
    configure_tensorrt_logging(parsed);

    // Log the level change (if not turning off logging)
    if (parsed != LogLevel::OFF && parsed != LogLevel::ERROR) {
        std::cout << "[iVIT] Log level set to: " << level << std::endl;
    }
}

std::string get_log_level() {
    return g_log_level;
}

// Global cache directory
static std::string g_cache_dir = "";

void set_cache_dir(const std::string& path) {
    g_cache_dir = path;
}

std::string get_cache_dir() {
    if (g_cache_dir.empty()) {
        // Default cache directory
        const char* home = std::getenv("HOME");
        if (home) {
            return std::string(home) + "/.cache/ivit";
        }
        return "/tmp/ivit_cache";
    }
    return g_cache_dir;
}

void convert_model(
    const std::string& src_path,
    const std::string& dst_path,
    const std::string& device,
    const std::string& precision
) {
    namespace fs = std::filesystem;

    // Determine target backend from device
    BackendType backend = get_backend_for_device(device);

    if (backend == BackendType::Auto) {
        // Determine from output extension
        std::string ext = fs::path(dst_path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".engine" || ext == ".trt" || ext == ".plan") {
            backend = BackendType::TensorRT;
        } else if (ext == ".xml") {
            backend = BackendType::OpenVINO;
        } else {
            throw IVITError("Cannot determine target backend from extension: " + ext);
        }
    }

    // Create runtime config
    RuntimeConfig config;
    config.device = device;

    // Parse precision
    if (precision == "fp16" || precision == "FP16") {
        config.precision = Precision::FP16;
    } else if (precision == "int8" || precision == "INT8") {
        config.precision = Precision::INT8;
    } else {
        config.precision = Precision::FP32;
    }

    // Get runtime and convert
    auto& factory = RuntimeFactory::instance();
    auto runtime = factory.get_runtime(backend);

    if (!runtime) {
        throw IVITError("Backend not available: " + to_string(backend));
    }

    runtime->convert_model(src_path, dst_path, config);
}

void clear_cache(const std::string& cache_dir) {
    namespace fs = std::filesystem;

    std::string dir = cache_dir.empty() ? get_cache_dir() : cache_dir;

    if (fs::exists(dir)) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".engine" || ext == ".trt" || ext == ".blob") {
                    fs::remove(entry.path());
                }
            }
        }
    }
}

std::shared_ptr<Model> load_model(
    const std::string& path,
    const LoadConfig& config
) {
    return ModelManager::instance().load(path, config);
}

} // namespace ivit

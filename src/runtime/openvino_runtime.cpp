/**
 * @file openvino_runtime.cpp
 * @brief OpenVINO backend runtime implementation
 */

#include "ivit/runtime/openvino_runtime.hpp"

#ifdef IVIT_HAS_OPENVINO

#include <algorithm>
#include <filesystem>
#include <cstdio>
#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#define STDERR_FILENO 2
#define dup      _dup
#define dup2     _dup2
#define close    _close
static inline int open_devnull() { return _open("NUL", _O_WRONLY); }
#else
#include <unistd.h>
#include <fcntl.h>
static inline int open_devnull() { return open("/dev/null", O_WRONLY); }
#endif

namespace ivit {

// ============================================================================
// OpenVINORuntime implementation
// ============================================================================

OpenVINORuntime::OpenVINORuntime() {
    initialize();
}

OpenVINORuntime::~OpenVINORuntime() = default;

void OpenVINORuntime::initialize() {
    if (initialized_) return;

    // Note: Do NOT set ov::log::level globally — GPU plugin does not support
    // LOG_LEVEL and will fail to load. Stderr noise from OpenCL compiler is
    // suppressed via stderr redirection in device_manager.cpp instead.

    // Enable model caching if cache directory is set
    if (!cache_dir_.empty()) {
        core_.set_property(ov::cache_dir(cache_dir_));
    }

    initialized_ = true;
}

bool OpenVINORuntime::is_available() const {
    try {
        ov::Core core;
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<std::string> OpenVINORuntime::supported_formats() const {
    return {".onnx", ".xml", ".bin", ".pdmodel"};
}

std::vector<DeviceInfo> OpenVINORuntime::get_devices() const {
    std::vector<DeviceInfo> devices;

    try {
        // Suppress stderr noise from Intel OpenCL compiler (IGC) during device enumeration
        fflush(stderr);
        int saved_stderr = dup(STDERR_FILENO);
        int devnull = open_devnull();
        dup2(devnull, STDERR_FILENO);
        close(devnull);

        ov::Core core;
        auto available = core.get_available_devices();

        // Restore stderr
        fflush(stderr);
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);

        for (const auto& device_name : available) {
            DeviceInfo info;

            // Get full device name
            try {
                info.name = core.get_property(device_name, ov::device::full_name);
            } catch (...) {
                info.name = device_name;
            }

            // Map device types
            if (device_name.find("CPU") != std::string::npos) {
                info.id = "cpu";
                info.type = "cpu";
            } else if (device_name.find("GPU") != std::string::npos) {
                // Extract GPU index if present
                size_t dot_pos = device_name.find('.');
                if (dot_pos != std::string::npos) {
                    std::string index = device_name.substr(dot_pos + 1);
                    info.id = "gpu:" + index;
                } else {
                    info.id = "gpu:0";
                }
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

            // Set supported precisions based on device
            info.supported_precisions = {Precision::FP32};

            if (info.type == "gpu" || info.type == "npu") {
                info.supported_precisions.push_back(Precision::FP16);
            }

            if (info.type == "cpu" || info.type == "gpu") {
                info.supported_precisions.push_back(Precision::INT8);
            }

            devices.push_back(info);
        }
    } catch (...) {
        // OpenVINO not available
    }

    return devices;
}

void* OpenVINORuntime::load_model(
    const std::string& path,
    const RuntimeConfig& config
) {
    std::lock_guard<std::mutex> lock(mutex_);
    namespace fs = std::filesystem;

    auto wrapper = new OpenVINOModel();

    // Map device name
    std::string device = map_device_name(config.device);
    wrapper->device = device;

    // Configure properties
    ov::AnyMap properties;

    // Set precision hints
    if (config.precision == Precision::FP16) {
        properties[ov::hint::inference_precision.name()] = ov::element::f16;
    } else if (config.precision == Precision::INT8) {
        properties[ov::hint::inference_precision.name()] = ov::element::i8;
    }

    // Set performance hint
    properties[ov::hint::performance_mode.name()] =
        ov::hint::PerformanceMode::LATENCY;

    // Set number of inference threads for CPU
    if (device == "CPU" && config.num_threads > 0) {
        properties[ov::inference_num_threads.name()] = config.num_threads;
    }

    // Enable profiling if requested
    if (config.enable_profiling) {
        properties[ov::enable_profiling.name()] = true;
    }

    try {
        // Read model
        std::shared_ptr<ov::Model> model;

        std::string ext = fs::path(path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".xml") {
            // OpenVINO IR format
            model = core_.read_model(path);
        } else if (ext == ".onnx") {
            // ONNX format
            model = core_.read_model(path);
        } else if (ext == ".pdmodel") {
            // PaddlePaddle format
            model = core_.read_model(path);
        } else {
            throw UnsupportedFormatError(ext);
        }

        // Compile model
        wrapper->compiled_model = core_.compile_model(model, device, properties);

        // Create inference request
        wrapper->infer_request = wrapper->compiled_model.create_infer_request();

        // Extract input info
        for (const auto& input : model->inputs()) {
            auto info = extract_tensor_info(input);
            wrapper->input_infos.push_back(info);
        }

        // Extract output info
        for (const auto& output : model->outputs()) {
            auto info = extract_tensor_info(output);
            wrapper->output_infos.push_back(info);
        }

    } catch (const ov::Exception& e) {
        delete wrapper;
        throw ModelLoadError(e.what());
    }

    return wrapper;
}

void OpenVINORuntime::unload_model(void* handle) {
    std::lock_guard<std::mutex> lock(mutex_);

    OpenVINOModel* model = static_cast<OpenVINOModel*>(handle);
    delete model;
}

std::vector<TensorInfo> OpenVINORuntime::get_input_info(void* handle) const {
    OpenVINOModel* model = static_cast<OpenVINOModel*>(handle);
    return model->input_infos;
}

std::vector<TensorInfo> OpenVINORuntime::get_output_info(void* handle) const {
    OpenVINOModel* model = static_cast<OpenVINOModel*>(handle);
    return model->output_infos;
}

std::map<std::string, Tensor> OpenVINORuntime::infer(
    void* handle,
    const std::map<std::string, Tensor>& inputs
) {
    OpenVINOModel* model = static_cast<OpenVINOModel*>(handle);

    // Set input tensors
    for (const auto& info : model->input_infos) {
        auto it = inputs.find(info.name);
        if (it == inputs.end()) {
            throw InferenceError("Missing input: " + info.name);
        }

        const Tensor& input = it->second;

        // Create OpenVINO tensor from input data
        ov::Shape ov_shape(input.shape().begin(), input.shape().end());
        ov::element::Type ov_type = dtype_to_ov(input.dtype());

        ov::Tensor ov_tensor(ov_type, ov_shape, const_cast<void*>(input.data()));

        model->infer_request.set_tensor(info.name, ov_tensor);
    }

    // Run inference
    if (use_async_) {
        model->infer_request.start_async();
        model->infer_request.wait();
    } else {
        model->infer_request.infer();
    }

    // Get output tensors
    std::map<std::string, Tensor> outputs;

    for (const auto& info : model->output_infos) {
        ov::Tensor ov_tensor = model->infer_request.get_tensor(info.name);

        // Convert to iVIT tensor
        Shape shape(ov_tensor.get_shape().begin(), ov_tensor.get_shape().end());
        DataType dtype = ov_to_dtype(ov_tensor.get_element_type());
        Tensor output(shape, dtype);
        output.set_name(info.name);

        // Copy data
        std::memcpy(output.data(), ov_tensor.data(), output.byte_size());

        outputs[info.name] = std::move(output);
    }

    return outputs;
}

void OpenVINORuntime::convert_model(
    const std::string& src_path,
    const std::string& dst_path,
    const RuntimeConfig& config
) {
    namespace fs = std::filesystem;

    // Read source model
    auto model = core_.read_model(src_path);

    // Apply precision conversion
    if (config.precision == Precision::INT8) {
        throw IVITError(
            "INT8 quantization requires calibration data and is not supported "
            "by simple model conversion. Please use NNCF toolkit: "
            "https://github.com/openvinotoolkit/nncf");
    }

    // Serialize to OpenVINO IR format
    // ov::save_model handles FP16 compression correctly, including
    // inserting proper Convert nodes at input/output boundaries.
    bool compress_fp16 = (config.precision == Precision::FP16);
    ov::save_model(model, dst_path, compress_fp16);
}

std::string OpenVINORuntime::map_device_name(const std::string& device) const {
    std::string dev_lower = device;
    std::transform(dev_lower.begin(), dev_lower.end(), dev_lower.begin(), ::tolower);

    if (dev_lower == "auto" || dev_lower.empty()) {
        return "AUTO";
    }

    if (dev_lower == "cpu") {
        return "CPU";
    }

    if (dev_lower.rfind("gpu", 0) == 0) {
        if (dev_lower.find(":") != std::string::npos) {
            std::string index = dev_lower.substr(dev_lower.find(":") + 1);
            return "GPU." + index;
        }
        return "GPU";
    }

    if (dev_lower == "npu") {
        return "NPU";
    }

    if (dev_lower == "vpu" || dev_lower == "myriad") {
        return "MYRIAD";
    }

    // Return as-is for unknown devices
    return device;
}

std::string OpenVINORuntime::get_version() const {
    return ov::get_openvino_version().buildNumber;
}

void OpenVINORuntime::set_cache_dir(const std::string& path) {
    cache_dir_ = path;
    if (!path.empty()) {
        core_.set_property(ov::cache_dir(path));
    }
}

TensorInfo OpenVINORuntime::extract_tensor_info(const ov::Output<ov::Node>& port) {
    TensorInfo info;

    // Get name
    info.name = port.get_any_name();

    // Get shape
    auto ov_shape = port.get_partial_shape();
    if (ov_shape.is_static()) {
        auto shape = ov_shape.get_shape();
        for (auto dim : shape) {
            info.shape.push_back(static_cast<int64_t>(dim));
        }
    } else {
        // Handle dynamic dimensions
        for (auto dim : ov_shape) {
            if (dim.is_static()) {
                info.shape.push_back(static_cast<int64_t>(dim.get_length()));
            } else {
                info.shape.push_back(-1);  // Dynamic dimension
            }
        }
    }

    // Get data type
    info.dtype = ov_to_dtype(port.get_element_type());

    // Determine layout
    if (info.shape.size() == 4) {
        info.layout = Layout::NCHW;
    } else if (info.shape.size() == 2) {
        info.layout = Layout::NC;
    } else {
        info.layout = Layout::Unknown;
    }

    return info;
}

DataType OpenVINORuntime::ov_to_dtype(ov::element::Type ov_type) const {
    if (ov_type == ov::element::f32) return DataType::Float32;
    if (ov_type == ov::element::f16) return DataType::Float16;
    if (ov_type == ov::element::i8) return DataType::Int8;
    if (ov_type == ov::element::u8) return DataType::UInt8;
    if (ov_type == ov::element::i32) return DataType::Int32;
    if (ov_type == ov::element::i64) return DataType::Int64;
    if (ov_type == ov::element::boolean) return DataType::Bool;
    return DataType::Unknown;
}

ov::element::Type OpenVINORuntime::dtype_to_ov(DataType dtype) const {
    switch (dtype) {
        case DataType::Float32: return ov::element::f32;
        case DataType::Float16: return ov::element::f16;
        case DataType::Int8: return ov::element::i8;
        case DataType::UInt8: return ov::element::u8;
        case DataType::Int32: return ov::element::i32;
        case DataType::Int64: return ov::element::i64;
        case DataType::Bool: return ov::element::boolean;
        default: return ov::element::f32;
    }
}

OpenVINORuntime& get_openvino_runtime() {
    static OpenVINORuntime instance;
    return instance;
}

} // namespace ivit

#endif // IVIT_HAS_OPENVINO

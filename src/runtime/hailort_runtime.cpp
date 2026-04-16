/**
 * @file hailort_runtime.cpp
 * @brief HailoRT backend runtime implementation
 */

#ifdef IVIT_HAS_HAILORT

#include "ivit/runtime/hailort_runtime.hpp"
#include "ivit/core/runtime_config.hpp"
#include <hailo/hailort.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ivit {

// ============================================================================
// Constructor
// ============================================================================

HailoRTRuntime::HailoRTRuntime() {}

// ============================================================================
// IRuntime: availability & discovery
// ============================================================================

bool HailoRTRuntime::is_available() const {
    return hailo_is_available();
}

std::vector<std::string> HailoRTRuntime::supported_formats() const {
    return {".hef"};
}

std::vector<DeviceInfo> HailoRTRuntime::get_devices() const {
    std::vector<DeviceInfo> result;

    auto scan = hailort::Device::scan();
    if (!scan) {
        return result;
    }

    int idx = 0;
    for (const auto& dev_str : scan.value()) {
        DeviceInfo info;
        info.id           = "hailo:" + std::to_string(idx);
        info.name         = "Hailo-8 [" + dev_str + "]";
        info.backend      = "hailort";
        info.type         = "npu";
        info.is_available = true;
        info.supported_precisions = {Precision::FP32, Precision::FP16, Precision::INT8};
        result.push_back(info);
        ++idx;
    }

    return result;
}

// ============================================================================
// IRuntime: load_model
// ============================================================================

void* HailoRTRuntime::load_model(
    const std::string& path,
    const RuntimeConfig& config
) {
    // Validate format
    std::filesystem::path fpath(path);
    if (fpath.extension() != ".hef") {
        throw UnsupportedFormatError(fpath.extension().string());
    }

    if (!std::filesystem::exists(fpath)) {
        throw ModelLoadError("File not found: " + path);
    }

    // Use defaults (RuntimeConfig has no backend_config variant field)
    int batch_size = 1;
    int timeout_ms = 1000;

    auto model_handle = std::make_unique<HailoRTModel>();
    model_handle->device_id  = config.device;
    model_handle->device_idx = parse_device_index(config.device);

    // Create VDevice
    hailo_vdevice_params_t params;
    auto init_status = hailo_init_vdevice_params(&params);
    if (init_status != HAILO_SUCCESS) {
        throw ModelLoadError("hailo_init_vdevice_params failed: " +
                             std::to_string(static_cast<int>(init_status)));
    }
    params.device_count = 1;

    auto vdevice = hailort::VDevice::create(params);
    if (!vdevice) {
        throw DeviceNotFoundError("hailo:" + std::to_string(model_handle->device_idx));
    }
    model_handle->vdevice = std::move(vdevice.value());

    // Create InferModel from HEF file
    auto infer_model = model_handle->vdevice->create_infer_model(path);
    if (!infer_model) {
        throw ModelLoadError("create_infer_model failed for: " + path);
    }
    model_handle->infer_model = std::move(infer_model.value());

    // Configure batch size
    model_handle->infer_model->set_batch_size(batch_size);

    // Configure and obtain a ConfiguredInferModel
    auto configured = model_handle->infer_model->configure();
    if (!configured) {
        throw ModelLoadError("InferModel::configure() failed: " + path);
    }
    model_handle->configured_model = std::move(configured.value());
    model_handle->timeout_ms = timeout_ms;

    // Extract tensor info
    // Inputs: report as NCHW float32 so the generic Vision layer preprocess works correctly.
    // The actual HailoRT dtype (e.g. UInt8) is stored separately for conversion in infer().
    for (const auto& input : model_handle->infer_model->inputs()) {
        auto hailo_shape = input.shape();
        TensorInfo info;
        info.name   = input.name();
        info.shape  = std::vector<int64_t>{
            1LL,
            static_cast<int64_t>(hailo_shape.features),
            static_cast<int64_t>(hailo_shape.height),
            static_cast<int64_t>(hailo_shape.width)
        };
        info.dtype  = DataType::Float32;
        info.layout = Layout::NCHW;
        model_handle->input_infos.push_back(info);
        model_handle->input_hailo_dtypes.push_back(hailort_to_dtype(input.format().type));
    }
    // Outputs: use real HailoRT format + frame_size for correct buffer allocation.
    for (const auto& output : model_handle->infer_model->outputs()) {
        model_handle->output_infos.push_back(make_tensor_info(output));
        model_handle->output_frame_sizes.push_back(output.get_frame_size());
    }

    return model_handle.release();
}

// ============================================================================
// IRuntime: unload_model
// ============================================================================

void HailoRTRuntime::unload_model(void* handle) {
    if (!handle) return;
    auto* m = static_cast<HailoRTModel*>(handle);
    // Explicit teardown order required by HailoRT:
    //   ConfiguredInferModel → InferModel → VDevice
    // Move configured_model into a local scope so it is destroyed first,
    // before the shared_ptrs below are reset.
    { auto _cfg = std::move(m->configured_model); }
    m->infer_model.reset();
    m->vdevice.reset();
    delete m;
}

// ============================================================================
// IRuntime: tensor info
// ============================================================================

std::vector<TensorInfo> HailoRTRuntime::get_input_info(void* handle) const {
    return static_cast<HailoRTModel*>(handle)->input_infos;
}

std::vector<TensorInfo> HailoRTRuntime::get_output_info(void* handle) const {
    return static_cast<HailoRTModel*>(handle)->output_infos;
}

// ============================================================================
// IRuntime: infer
// ============================================================================

std::map<std::string, Tensor> HailoRTRuntime::infer(
    void*  handle,
    const std::map<std::string, Tensor>& inputs
) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto* m = static_cast<HailoRTModel*>(handle);
    int timeout_ms = m->timeout_ms;

    // Create bindings
    auto bindings_exp = m->configured_model.create_bindings();
    if (!bindings_exp) {
        throw InferenceError("HailoRT create_bindings() failed");
    }
    auto bindings = std::move(bindings_exp.value());

    // Copy input tensors into bindings.
    // When the Vision layer sends NCHW float32 [0,1] but HailoRT expects NHWC uint8 [0,255],
    // convert on-the-fly.
    for (size_t idx = 0; idx < m->input_infos.size(); ++idx) {
        const std::string& name    = m->input_infos[idx].name;
        const Tensor&      tensor  = inputs.at(name);
        DataType           hailo_dtype = m->input_hailo_dtypes[idx];

        auto input_buffer = bindings.input(name);
        if (!input_buffer) {
            throw InferenceError("HailoRT: unknown input name: " + name);
        }

        std::vector<uint8_t> converted;
        const void* data_ptr  = tensor.data();
        size_t      data_size = tensor.byte_size();

        if (hailo_dtype == DataType::UInt8 && tensor.dtype() == DataType::Float32) {
            // NCHW float32 [0,1] → NHWC uint8 [0,255]
            const auto& s = tensor.shape();  // {1, C, H, W}
            if (s.size() == 4) {
                const int64_t C = s[1], H = s[2], W = s[3];
                converted.resize(static_cast<size_t>(H * W * C));
                const float* src = tensor.data_ptr<float>();
                uint8_t*     dst = converted.data();
                for (int64_t h = 0; h < H; ++h)
                    for (int64_t w = 0; w < W; ++w)
                        for (int64_t c = 0; c < C; ++c)
                            dst[h * W * C + w * C + c] = static_cast<uint8_t>(
                                std::clamp(src[c * H * W + h * W + w] * 255.0f,
                                           0.0f, 255.0f));
                data_ptr  = converted.data();
                data_size = converted.size();
            }
        }

        auto set_status = input_buffer->set_buffer(
            hailort::MemoryView(const_cast<void*>(data_ptr), data_size)
        );
        if (set_status != HAILO_SUCCESS) {
            throw InferenceError("HailoRT: set_buffer failed for input: " + name);
        }
    }

    // Allocate output buffers and bind them
    std::map<std::string, std::vector<uint8_t>> out_buffers;
    for (size_t i = 0; i < m->output_infos.size(); ++i) {
        const auto& info = m->output_infos[i];
        size_t nbytes = m->output_frame_sizes[i];

        out_buffers[info.name].resize(nbytes);

        auto output_buffer = bindings.output(info.name);
        if (!output_buffer) {
            throw InferenceError("HailoRT: unknown output name: " + info.name);
        }
        auto set_status = output_buffer->set_buffer(
            hailort::MemoryView(out_buffers[info.name].data(), nbytes)
        );
        if (set_status != HAILO_SUCCESS) {
            throw InferenceError("HailoRT: set_buffer failed for output: " + info.name);
        }
    }

    // Run inference
    auto run_status = m->configured_model.run(
        std::move(bindings),
        std::chrono::milliseconds(timeout_ms)
    );
    if (run_status != HAILO_SUCCESS) {
        if (run_status == HAILO_TIMEOUT) {
            throw InferenceError("HailoRT inference timeout (" +
                                 std::to_string(timeout_ms) + " ms)");
        }
        throw InferenceError("HailoRT run() failed: " +
                             std::to_string(static_cast<int>(run_status)));
    }

    // Collect outputs into Tensor map
    std::map<std::string, Tensor> results;
    for (size_t i = 0; i < m->output_infos.size(); ++i) {
        const auto& info = m->output_infos[i];
        size_t frame_size = m->output_frame_sizes[i];

        Tensor t;
        if (info.layout == Layout::HailoNMS) {
            // NMS output: shape is {num_classes, 1+max_bboxes*5}, dtype Float32.
            // byte_size() == frame_size so the memcpy below is safe.
            t = Tensor(info.shape, DataType::Float32);
            t.set_layout(Layout::HailoNMS);
        } else {
            t = Tensor(Shape{static_cast<int64_t>(frame_size)}, DataType::UInt8);
        }
        t.set_name(info.name);
        auto& buf = out_buffers[info.name];
        std::memcpy(t.data(), buf.data(), buf.size());
        results[info.name] = std::move(t);
    }
    return results;
}

// ============================================================================
// IRuntime: convert_model
// ============================================================================

void HailoRTRuntime::convert_model(
    const std::string& src_path,
    const std::string& dst_path,
    const RuntimeConfig& /*config*/
) {
    // Check for hailo_model_zoo CLI
    if (std::system("hailomz --version > /dev/null 2>&1") != 0 &&
        std::system("python -m hailo_model_zoo --help > /dev/null 2>&1") != 0) {
        throw IVITError(
            "Hailo Dataflow Compiler (hailo_model_zoo) is not installed.\n"
            "Install it from:\n"
            "  https://developer.hailo.ai/developer-zone/software-downloads/\n"
            "Then run:\n"
            "  python -m hailo_model_zoo compile " + src_path +
            " --hw-arch hailo8 -o " + dst_path
        );
    }

    std::string cmd = "python -m hailo_model_zoo compile \"" + src_path +
                      "\" --hw-arch hailo8 -o \"" + dst_path + "\"";

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw IVITError("hailo_model_zoo compile failed (exit code: " +
                        std::to_string(ret) + ")\nCommand: " + cmd);
    }
}

// ============================================================================
// Private helpers
// ============================================================================

int HailoRTRuntime::parse_device_index(const std::string& device_id) const {
    if (device_id == "hailo") return 0;
    auto pos = device_id.find(':');
    if (pos != std::string::npos) {
        try { return std::stoi(device_id.substr(pos + 1)); }
        catch (...) {}
    }
    return 0;
}

DataType HailoRTRuntime::hailort_to_dtype(hailo_format_type_t fmt) const {
    switch (fmt) {
        case HAILO_FORMAT_TYPE_FLOAT32: return DataType::Float32;
        case HAILO_FORMAT_TYPE_UINT8:   return DataType::UInt8;
        case HAILO_FORMAT_TYPE_UINT16:  return DataType::Unknown;
        default:                        return DataType::Unknown;
    }
}

TensorInfo HailoRTRuntime::make_tensor_info(
    const hailort::InferModel::InferStream& stream
) const {
    TensorInfo info;
    info.name = stream.name();

    if (stream.format().order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS) {
        // On-device NMS output: shape encodes {num_classes, 1 + max_bboxes_per_class * 5}
        // so that byte_size() == frame_size and data is directly readable as float32.
        auto nms_exp = stream.get_nms_shape();
        if (nms_exp) {
            const auto& nms = nms_exp.value();
            info.shape = std::vector<int64_t>{
                static_cast<int64_t>(nms.number_of_classes),
                static_cast<int64_t>(1 + nms.max_bboxes_per_class * 5)
            };
        } else {
            // Fallback: 80 classes × (1 + 100×5) = 80 × 501 floats
            info.shape = {80LL, 501LL};
        }
        info.dtype  = DataType::Float32;
        info.layout = Layout::HailoNMS;
    } else {
        auto shape = stream.shape();
        info.shape = std::vector<int64_t>{
            static_cast<int64_t>(shape.height),
            static_cast<int64_t>(shape.width),
            static_cast<int64_t>(shape.features)
        };
        info.dtype  = hailort_to_dtype(stream.format().type);
        info.layout = Layout::NHWC;
    }
    return info;
}

} // namespace ivit

#endif // IVIT_HAS_HAILORT

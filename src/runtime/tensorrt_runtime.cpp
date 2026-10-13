/**
 * @file tensorrt_runtime.cpp
 * @brief TensorRT backend runtime implementation
 */

#include "ivit/runtime/tensorrt_runtime.hpp"
#include "ivit/ivit.hpp"
#include "utils/fingerprint.hpp"

#ifdef IVIT_HAS_TENSORRT

// Suppress deprecation warnings from TensorRT 10.12 headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <NvOnnxParser.h>
#include <opencv2/core.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace ivit {

#if NV_TENSORRT_MAJOR >= 11
namespace {
// TensorRT 11 networks are strongly typed: precision comes from tensor types, and the FP16
// builder flag is gone. precision=fp16 therefore rewrites the parsed FP32 network itself:
// float weights become half, explicit casts to float target half, and casts at the network
// boundary keep the engine's inputs and outputs float32 as the SDK expects.
using HalfStore = std::vector<std::vector<uint16_t>>; // IEEE half bit patterns

nvinfer1::Weights to_half(nvinfer1::Weights weights, HalfStore &store) {
    if (weights.type != nvinfer1::DataType::kFLOAT || weights.count <= 0 || !weights.values)
        return weights;
    const int count = static_cast<int>(weights.count);
    // Clamp first: values outside the half range would otherwise become infinity.
    cv::Mat source(1, count, CV_32F, const_cast<void *>(weights.values));
    cv::Mat clamped;
    cv::min(cv::max(source, -65504.0), 65504.0, clamped);
    store.emplace_back(static_cast<size_t>(count));
    cv::Mat target(1, count, CV_16F, store.back().data());
    clamped.convertTo(target, CV_16F);
    return nvinfer1::Weights{nvinfer1::DataType::kHALF, store.back().data(), weights.count};
}

void convert_network_to_fp16(nvinfer1::INetworkDefinition &network, HalfStore &store) {
    using nvinfer1::DataType;
    using nvinfer1::LayerType;

    // Remember the float outputs (and the output order) before anything changes.
    std::vector<nvinfer1::ITensor *> outputs;
    std::vector<bool> float_output;
    for (int i = 0; i < network.getNbOutputs(); ++i) {
        outputs.push_back(network.getOutput(i));
        float_output.push_back(outputs.back()->getType() == DataType::kFLOAT);
    }

    const int original_layers = network.getNbLayers();
    for (int i = 0; i < original_layers; ++i) {
        nvinfer1::ILayer *layer = network.getLayer(i);
        switch (layer->getType()) {
        case LayerType::kCONSTANT: {
            auto *constant = static_cast<nvinfer1::IConstantLayer *>(layer);
            constant->setWeights(to_half(constant->getWeights(), store));
            break;
        }
        case LayerType::kCONVOLUTION: {
            auto *conv = static_cast<nvinfer1::IConvolutionLayer *>(layer);
            conv->setKernelWeights(to_half(conv->getKernelWeights(), store));
            conv->setBiasWeights(to_half(conv->getBiasWeights(), store));
            break;
        }
        case LayerType::kDECONVOLUTION: {
            auto *deconv = static_cast<nvinfer1::IDeconvolutionLayer *>(layer);
            deconv->setKernelWeights(to_half(deconv->getKernelWeights(), store));
            deconv->setBiasWeights(to_half(deconv->getBiasWeights(), store));
            break;
        }
        case LayerType::kSCALE: {
            auto *scale = static_cast<nvinfer1::IScaleLayer *>(layer);
            scale->setShift(to_half(scale->getShift(), store));
            scale->setScale(to_half(scale->getScale(), store));
            scale->setPower(to_half(scale->getPower(), store));
            break;
        }
        case LayerType::kCAST: {
            auto *cast = static_cast<nvinfer1::ICastLayer *>(layer);
            if (cast->getToType() == DataType::kFLOAT)
                cast->setToType(DataType::kHALF);
            break;
        }
        case LayerType::kFILL: {
            auto *fill = static_cast<nvinfer1::IFillLayer *>(layer);
            if (fill->getToType() == DataType::kFLOAT)
                fill->setToType(DataType::kHALF);
            break;
        }
        default:
            break;
        }
    }

    // Float inputs: cast to half once and feed every consumer from the cast.
    for (int i = 0; i < network.getNbInputs(); ++i) {
        nvinfer1::ITensor *input = network.getInput(i);
        if (input->getType() != DataType::kFLOAT)
            continue;
        nvinfer1::ICastLayer *cast = network.addCast(*input, DataType::kHALF);
        cast->setName((std::string(input->getName()) + "_to_fp16").c_str());
        for (int l = 0; l < original_layers; ++l) {
            nvinfer1::ILayer *layer = network.getLayer(l);
            for (int j = 0; j < layer->getNbInputs(); ++j)
                if (layer->getInput(j) == input)
                    layer->setInput(j, *cast->getOutput(0));
        }
    }

    // Float outputs: cast back to float under the original name, keeping the output order.
    for (nvinfer1::ITensor *output : outputs)
        network.unmarkOutput(*output);
    for (size_t i = 0; i < outputs.size(); ++i) {
        nvinfer1::ITensor *output = outputs[i];
        if (!float_output[i]) {
            network.markOutput(*output);
            continue;
        }
        const std::string name = output->getName();
        output->setName((name + "_fp16").c_str());
        nvinfer1::ICastLayer *cast = network.addCast(*output, DataType::kFLOAT);
        cast->setName((name + "_to_fp32").c_str());
        cast->getOutput(0)->setName(name.c_str());
        network.markOutput(*cast->getOutput(0));
    }
}
} // namespace
#endif

// ============================================================================
// TensorRTLogger implementation
// ============================================================================

void TensorRTLogger::log(Severity severity, const char *msg) noexcept {
    // Get global log level
    LogLevel log_level = get_parsed_log_level();

    // Map TensorRT severity to our log level filtering
    // TensorRT: kINTERNAL_ERROR=0, kERROR=1, kWARNING=2, kINFO=3, kVERBOSE=4
    bool should_log = false;
    switch (log_level) {
    case LogLevel::DEBUG:
        should_log = true; // Log everything
        break;
    case LogLevel::INFO:
        should_log = (severity <= Severity::kINFO);
        break;
    case LogLevel::WARNING:
        should_log = (severity <= Severity::kWARNING);
        break;
    case LogLevel::ERROR:
        should_log = (severity <= Severity::kERROR);
        break;
    case LogLevel::OFF:
        should_log = false;
        break;
    }

    // TensorRT's kVERBOSE output (per-tactic builder traces) is never forwarded, even at DEBUG.
    if (severity == Severity::kVERBOSE) {
        should_log = false;
    }

    if (!should_log) {
        return;
    }

    LogLevel level = LogLevel::DEBUG;
    if (severity <= Severity::kERROR)
        level = LogLevel::ERROR;
    else if (severity == Severity::kWARNING)
        level = LogLevel::WARNING;
    else if (severity == Severity::kINFO)
        level = LogLevel::INFO;
    detail::log(level, std::string("[TRT] ") + msg); // level already carries the severity
}

// ============================================================================
// TensorRTEngine implementation
// ============================================================================

TensorRTEngine::~TensorRTEngine() {
    int previous = 0;
    const bool have_device = cudaGetDevice(&previous) == cudaSuccess;
    const bool selected = cudaSetDevice(device_id) == cudaSuccess;
    if (selected)
        for (auto &slot : slots)
            if (slot->stream)
                cudaStreamSynchronize(slot->stream);
    // TRT requires every context to die before its engine and parent runtime.
    for (auto &slot : slots)
        slot->context.reset();
    engine.reset();
    runtime.reset();
    if (selected) {
        for (auto &slot : slots) {
            for (void *buffer : slot->device_buffers)
                if (buffer)
                    cudaFree(buffer);
            if (slot->stream)
                cudaStreamDestroy(slot->stream);
        }
    }
    if (have_device && previous != device_id)
        cudaSetDevice(previous);
}

TensorRTSlot *TensorRTEngine::acquire_slot() {
    std::unique_lock<std::mutex> lock(slot_mutex);
    slot_available.wait(lock, [this] { return !free_slots.empty(); });
    TensorRTSlot *slot = free_slots.back(); // most recently used: warmest caches
    free_slots.pop_back();
    return slot;
}

void TensorRTEngine::release_slot(TensorRTSlot *slot) {
    {
        std::lock_guard<std::mutex> lock(slot_mutex);
        free_slots.push_back(slot);
    }
    slot_available.notify_one();
}

namespace {
// Returns the slot on every exit path, including exceptions.
class SlotLease {
  public:
    explicit SlotLease(TensorRTEngine &engine) : engine_(engine), slot_(engine.acquire_slot()) {}
    ~SlotLease() { engine_.release_slot(slot_); }
    SlotLease(const SlotLease &) = delete;
    SlotLease &operator=(const SlotLease &) = delete;
    TensorRTSlot &operator*() const { return *slot_; }

  private:
    TensorRTEngine &engine_;
    TensorRTSlot *slot_;
};
} // namespace

// ============================================================================
// TensorRTRuntime implementation
// ============================================================================

TensorRTRuntime::TensorRTRuntime() { initialize(); }

TensorRTRuntime::~TensorRTRuntime() = default;

void TensorRTRuntime::initialize() {
    if (initialized_)
        return;

    runtime_.reset(nvinfer1::createInferRuntime(logger_), [](nvinfer1::IRuntime *r) { delete r; });

    if (!runtime_) {
        throw IVITError("Failed to create TensorRT runtime");
    }

    initialized_ = true;
}

bool TensorRTRuntime::is_available() const {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

std::vector<std::string> TensorRTRuntime::supported_formats() const {
    return {".onnx", ".engine", ".trt", ".plan"};
}

std::vector<DeviceInfo> TensorRTRuntime::get_devices() const {
    std::vector<DeviceInfo> devices;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        return devices;
    }

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

        // Check supported precisions based on compute capability
        info.supported_precisions = {Precision::FP32};

        // FP16 support (Compute Capability >= 5.3 for full support)
        if (props.major > 5 || (props.major == 5 && props.minor >= 3)) {
            info.supported_precisions.push_back(Precision::FP16);
        }

        // INT8 support (Compute Capability >= 6.1)
        if (props.major > 6 || (props.major == 6 && props.minor >= 1)) {
            info.supported_precisions.push_back(Precision::INT8);
        }

        devices.push_back(info);
    }

    return devices;
}

std::string TensorRTRuntime::get_cache_path(const std::string &model_path,
                                            const RuntimeConfig &config) const {
    namespace fs = std::filesystem;

    // Generate cache filename based on model path, precision, and GPU
    std::string model_name = fs::path(model_path).stem().string();
    std::string precision_str = "fp32";
    if (config.precision == Precision::FP16)
        precision_str = "fp16";
    else if (config.precision == Precision::INT8)
        precision_str = "int8";

    // Get GPU info for cache key
    cudaDeviceProp props;
    int device_id = 0;
    if (config.device.find("cuda:") == 0) {
        device_id = std::stoi(config.device.substr(5));
    }
    if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess)
        throw ModelLoadError("Cannot inspect CUDA device " + std::to_string(device_id));

    // Use compute capability as part of cache key
    std::string gpu_key = std::to_string(props.major) + std::to_string(props.minor);

    // Determine cache directory: LoadConfig.cache_dir, else set_cache_dir(), else
    // ~/.cache/ivit/engines. Engines are built on the target GPU, so they belong in a per-user
    // writable cache rather than next to the model, which is often read-only (/opt, images).
    std::string cache_dir = config.cache_dir;
    if (cache_dir.empty())
        cache_dir = (fs::path(ivit::get_cache_dir()) / "engines").string();

    int driver = 0, cuda_version = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&cuda_version);
    std::string compatibility =
        detail::model_fingerprint(model_path) + precision_str + get_version() + ":" +
        std::to_string(driver) + ":" + std::to_string(cuda_version) + ":" + props.name + ":" +
        gpu_key + ":profile-v2-1-" + std::to_string(config.batch_size) + "-" +
        std::to_string(config.max_batch_size) + ":" + std::to_string(config.workspace_size);
    compatibility.append(reinterpret_cast<const char *>(&props.uuid), sizeof(props.uuid));
    std::string cache_name = model_name + "_trt_" + detail::fingerprint(compatibility) + ".engine";
    return (fs::path(cache_dir) / cache_name).string();
}

void *TensorRTRuntime::load_model(const std::string &path, const RuntimeConfig &config) {
    std::lock_guard<std::mutex> lock(mutex_);
    namespace fs = std::filesystem;

    // Parse device ID
    int device_id = 0;
    if (config.device.find("cuda:") == 0) {
        device_id = std::stoi(config.device.substr(5));
    }

    // Set CUDA device
    if (cudaSetDevice(device_id) != cudaSuccess)
        throw ModelLoadError("Cannot select CUDA device " + std::to_string(device_id));

    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    std::shared_ptr<nvinfer1::ICudaEngine> engine;

    if (ext == ".onnx") {
        // Check for cached engine
        if (config.use_cache) {
            std::string cache_path = get_cache_path(path, config);

            if (fs::exists(cache_path)) {
                try {
                    detail::log(LogLevel::INFO, "[TensorRT] Loading cached engine: " + cache_path);
                    engine = load_engine(cache_path);
                } catch (const std::exception &error) {
                    detail::log(LogLevel::WARNING, std::string("[TensorRT] Rebuilding invalid cache: ") +
                                                       error.what());
                }
            }
        }

        // Build engine if no cache
        if (!engine) {
            detail::log(LogLevel::INFO,
                        "[TensorRT] Building engine from ONNX (this may take a while)...");
            engine = build_engine_from_onnx(path, config);

            // Save to cache. A cache that cannot be written must not fail the load: the engine
            // is already built and usable; it is only rebuilt again next time.
            if (config.use_cache && engine) {
                std::string cache_path = get_cache_path(path, config);
                try {
                    save_engine(engine.get(), cache_path);
                    detail::log(LogLevel::INFO, "[TensorRT] Saved engine to cache: " + cache_path);
                } catch (const std::exception &error) {
                    detail::log(LogLevel::WARNING,
                                std::string("[TensorRT] Engine cache not written (the engine will be "
                                            "rebuilt on the next load; set cache_dir to a writable "
                                            "directory): ") +
                                    error.what());
                }
            }
        }
    } else if (ext == ".engine" || ext == ".trt" || ext == ".plan") {
        // Load serialized engine
        engine = load_engine(path);
    } else {
        throw UnsupportedFormatError(ext);
    }

    if (!engine) {
        throw ModelLoadError("Failed to create TensorRT engine");
    }

    return create_engine_wrapper(engine, device_id, std::max(1, config.num_instances));
}

void TensorRTRuntime::unload_model(void *handle) {
    std::lock_guard<std::mutex> lock(mutex_);

    TensorRTEngine *engine = static_cast<TensorRTEngine *>(handle);
    delete engine;
}

std::vector<TensorInfo> TensorRTRuntime::get_input_info(void *handle) const {
    TensorRTEngine *engine = static_cast<TensorRTEngine *>(handle);
    return engine->input_infos;
}

std::vector<TensorInfo> TensorRTRuntime::get_output_info(void *handle) const {
    TensorRTEngine *engine = static_cast<TensorRTEngine *>(handle);
    return engine->output_infos;
}

namespace {
std::string describe(const Shape &shape) {
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i)
        text += (i ? "," : "") + std::to_string(shape[i]);
    return text + "]";
}

// Fixed dimensions must match exactly; dynamic ones (negative in the engine) are validated by
// setInputShape against the optimization profile.
void validate_input(const TensorInfo &info, const Tensor &input) {
    if (input.dtype() != info.dtype) {
        throw InferenceError("Input '" + info.name + "' has data type " + to_string(input.dtype()) +
                             " but the engine expects " + to_string(info.dtype));
    }
    if (input.shape().size() != info.shape.size()) {
        throw InferenceError("Input '" + info.name + "' has shape " + describe(input.shape()) +
                             " but the engine expects " + describe(info.shape));
    }
    for (size_t d = 0; d < info.shape.size(); ++d) {
        if (info.shape[d] >= 0 && input.shape()[d] != info.shape[d]) {
            throw InferenceError("Input '" + info.name + "' has shape " + describe(input.shape()) +
                                 " but the engine expects " + describe(info.shape));
        }
    }
}
} // namespace

std::map<std::string, Tensor> TensorRTRuntime::infer(void *handle,
                                                     const std::map<std::string, Tensor> &inputs) {
    TensorRTEngine *engine = static_cast<TensorRTEngine *>(handle);

    // Each call owns one slot (context + stream + buffers); with num_instances > 1 that many
    // calls run on the GPU at the same time, otherwise they take turns.
    SlotLease lease(*engine);
    TensorRTSlot &slot = *lease;

    // Set CUDA device
    cudaSetDevice(engine->device_id);

    // Copy input data to device
    for (size_t i = 0; i < engine->input_infos.size(); i++) {
        const auto &info = engine->input_infos[i];
        auto it = inputs.find(info.name);

        if (it == inputs.end()) {
            throw InferenceError("Missing input: " + info.name);
        }

        const Tensor &input = it->second;

        // The copy below only checks that the bytes fit. A wrong dtype or shape that happens to
        // fit would run on stale or reinterpreted device memory and return wrong numbers without
        // any error, so check them against the engine (OpenVINO already rejects these).
        validate_input(info, input);

        // Dynamic inputs: tell the context the actual shape for this call
        if (info.is_dynamic()) {
            nvinfer1::Dims dims;
            dims.nbDims = static_cast<int32_t>(input.shape().size());
            if (dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
                throw InferenceError("Too many dimensions for input: " + info.name);
            }
            for (int d = 0; d < dims.nbDims; d++) {
                dims.d[d] = input.shape()[d];
            }
            if (!slot.context->setInputShape(info.name.c_str(), dims)) {
                throw InferenceError("Input shape for '" + info.name +
                                     "' is outside the engine's optimization profile");
            }
        }

        // Never copy more than the device buffer can hold
        if (input.byte_size() > slot.buffer_bytes[i]) {
            throw InferenceError("Input '" + info.name + "' is " +
                                 std::to_string(input.byte_size()) +
                                 " bytes, but the device buffer holds only " +
                                 std::to_string(slot.buffer_bytes[i]) + " bytes");
        }

        cudaError_t copy_err =
            cudaMemcpyAsync(slot.device_buffers[i], input.data(), input.byte_size(),
                            cudaMemcpyHostToDevice, slot.stream);
        if (copy_err != cudaSuccess) {
            throw InferenceError(std::string("Failed to copy input to device: ") +
                                 cudaGetErrorString(copy_err));
        }
    }

    // Run inference (TensorRT 10 API uses enqueueV3)
    bool success = slot.context->enqueueV3(slot.stream);

    if (!success) {
        throw InferenceError("TensorRT inference failed");
    }

    // Copy output data to host
    std::map<std::string, Tensor> outputs;
    size_t input_count = engine->input_infos.size();

    for (size_t i = 0; i < engine->output_infos.size(); i++) {
        const auto &info = engine->output_infos[i];

        // Output shape may depend on the input shapes set above
        Shape out_shape = info.shape;
        if (info.is_dynamic()) {
            nvinfer1::Dims dims = slot.context->getTensorShape(info.name.c_str());
            out_shape.assign(dims.d, dims.d + dims.nbDims);
        }

        Tensor output(out_shape, info.dtype); // throws if still dynamic
        output.set_name(info.name);

        if (output.byte_size() > slot.buffer_bytes[input_count + i]) {
            throw InferenceError("Output '" + info.name + "' needs " +
                                 std::to_string(output.byte_size()) +
                                 " bytes, but the device buffer holds only " +
                                 std::to_string(slot.buffer_bytes[input_count + i]) + " bytes");
        }

        cudaError_t copy_err =
            cudaMemcpyAsync(output.data(), slot.device_buffers[input_count + i],
                            output.byte_size(), cudaMemcpyDeviceToHost, slot.stream);
        if (copy_err != cudaSuccess) {
            throw InferenceError(std::string("Failed to copy output to host: ") +
                                 cudaGetErrorString(copy_err));
        }

        outputs[info.name] = std::move(output);
    }

    // Synchronize with error checking
    cudaError_t sync_err = cudaStreamSynchronize(slot.stream);
    if (sync_err != cudaSuccess) {
        throw InferenceError(std::string("CUDA stream synchronization failed: ") +
                             cudaGetErrorString(sync_err));
    }

    return outputs;
}

void TensorRTRuntime::convert_model(const std::string &src_path, const std::string &dst_path,
                                    const RuntimeConfig &config) {
    namespace fs = std::filesystem;

    std::string src_ext = fs::path(src_path).extension().string();
    std::transform(src_ext.begin(), src_ext.end(), src_ext.begin(), ::tolower);

    if (src_ext != ".onnx") {
        throw IVITError("TensorRT conversion only supports ONNX input");
    }

    auto engine = build_engine_from_onnx(src_path, config);
    save_engine(engine.get(), dst_path);
}

std::shared_ptr<nvinfer1::ICudaEngine>
TensorRTRuntime::build_engine_from_onnx(const std::string &onnx_path, const RuntimeConfig &config) {
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        throw IVITError("Failed to create TensorRT builder");
    }

    // Create network
#if NV_TENSORRT_MAJOR >= 11
    const uint32_t network_flags = 0; // strongly typed is the only mode
#else
    const auto network_flags =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    auto network =
        std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(network_flags));
    if (!network) {
        throw IVITError("Failed to create TensorRT network");
    }

    // Create ONNX parser
    auto parser =
        std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        throw IVITError("Failed to create ONNX parser");
    }

    // Parse ONNX model
    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::string errors;
        for (int i = 0; i < parser->getNbErrors(); i++) {
            errors += parser->getError(i)->desc();
            errors += "\n";
        }
        throw ModelLoadError("ONNX parsing failed: " + errors);
    }

    // Create builder config
    auto builder_config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builder_config) {
        throw IVITError("Failed to create builder config");
    }

    // Dynamic inputs need an optimization profile. Only a dynamic batch
    // dimension can be given a sensible default; any other dynamic dimension
    // must be fixed when the model is exported.
    nvinfer1::IOptimizationProfile *profile = nullptr;
    for (int i = 0; i < network->getNbInputs(); i++) {
        nvinfer1::ITensor *input = network->getInput(i);
        nvinfer1::Dims dims = input->getDimensions();

        nvinfer1::Dims min_dims = dims, opt_dims = dims, max_dims = dims;
        bool dynamic = false;
        for (int d = 0; d < dims.nbDims; d++) {
            if (dims.d[d] >= 0)
                continue;
            dynamic = true;
            if (d != 0) {
                throw ModelLoadError(std::string("Input '") + input->getName() +
                                     "' has a dynamic "
                                     "non-batch dimension; export the ONNX model with fixed "
                                     "height/width for TensorRT");
            }
            min_dims.d[d] = 1;
            opt_dims.d[d] = config.batch_size;
            max_dims.d[d] = config.max_batch_size;
        }
        if (!dynamic)
            continue;

        if (!profile)
            profile = builder->createOptimizationProfile();
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims);
    }
    if (profile) {
        builder_config->addOptimizationProfile(profile);
    }

    // Set precision
#if NV_TENSORRT_MAJOR >= 11
    HalfStore half_weights; // must outlive buildSerializedNetwork()
#endif
    if (config.precision == Precision::FP16) {
#if NV_TENSORRT_MAJOR >= 11
        convert_network_to_fp16(*network, half_weights);
#else
        if (builder->platformHasFastFp16()) {
            builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
#endif
    } else if (config.precision == Precision::INT8) {
        bool quantized = false;
        for (int i = 0; i < network->getNbLayers(); ++i)
            quantized |= network->getLayer(i)->getType() == nvinfer1::LayerType::kQUANTIZE;
        if (!quantized)
            throw ModelLoadError("INT8 requires a quantized ONNX model with explicit Q/DQ; "
                                 "calibration is not implemented");
    }

    // Set memory pool limit (use 1GB as default)
    builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config.workspace_size);

    // Build engine
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *builder_config));
    if (!serialized_engine) {
        throw IVITError("Failed to build TensorRT engine");
    }

    // Deserialize engine
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()),
        [](nvinfer1::ICudaEngine *e) { delete e; });

    return engine;
}

std::shared_ptr<nvinfer1::ICudaEngine>
TensorRTRuntime::load_engine(const std::string &engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw ModelLoadError("Cannot open engine file: " + engine_path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw ModelLoadError("Cannot read engine file: " + engine_path);
    }

    auto engine =
        std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer.data(), size),
                                               [](nvinfer1::ICudaEngine *e) { delete e; });

    if (!engine) {
        throw ModelLoadError("Failed to deserialize TensorRT engine");
    }

    return engine;
}

void TensorRTRuntime::save_engine(nvinfer1::ICudaEngine *engine, const std::string &path) {
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(engine->serialize());

    if (!serialized)
        throw IVITError("Cannot serialize TensorRT engine");
    namespace fs = std::filesystem;
    const fs::path destination(path);
    if (!destination.parent_path().empty())
        fs::create_directories(destination.parent_path());
    static std::atomic<uint64_t> sequence{0};
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::string temporary =
        path + ".tmp-" + std::to_string(stamp) + "-" + std::to_string(sequence++);
    std::ofstream file(temporary, std::ios::binary);
    if (!file) {
        throw IVITError("Cannot open file for writing: " + path);
    }

    file.write(static_cast<const char *>(serialized->data()), serialized->size());
    file.close();
    if (!file) {
        fs::remove(temporary);
        throw IVITError("Cannot write TensorRT engine: " + path);
    }
    try {
        fs::rename(temporary, destination);
    } catch (...) {
        fs::remove(temporary);
        throw;
    }
}

std::string TensorRTRuntime::get_version() const {
    return std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." +
           std::to_string(NV_TENSORRT_PATCH);
}

TensorRTEngine *
TensorRTRuntime::create_engine_wrapper(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                                       int device_id, int num_instances) {
    // Owned until fully set up. ~TensorRTEngine frees whatever device buffers
    // and streams were created, so error paths must not free them by hand.
    auto wrapper = std::make_unique<TensorRTEngine>();
    wrapper->runtime = runtime_; // prevent static destruction order issue
    wrapper->engine = engine;
    wrapper->device_id = device_id;

    int num_io_tensors = engine->getNbIOTensors();
    auto to_shape = [](const nvinfer1::Dims &dims) { return Shape(dims.d, dims.d + dims.nbDims); };

    for (int instance = 0; instance < num_instances; ++instance) {
        // Registered immediately so a failure part-way still releases what exists.
        wrapper->slots.push_back(std::make_unique<TensorRTSlot>());
        TensorRTSlot *slot = wrapper->slots.back().get();
        const bool first = instance == 0; // tensor metadata is identical for every slot

        // Create execution context
        slot->context.reset(engine->createExecutionContext(),
                            [](nvinfer1::IExecutionContext *c) { delete c; });

        if (!slot->context) {
            throw IVITError("Failed to create execution context");
        }

        // Create CUDA stream with error checking
        cudaError_t stream_err = cudaStreamCreate(&slot->stream);
        if (stream_err != cudaSuccess) {
            slot->stream = nullptr;
            throw IVITError(std::string("Failed to create CUDA stream: ") +
                            cudaGetErrorString(stream_err));
        }

        slot->device_buffers.resize(num_io_tensors, nullptr); // Initialize to nullptr
        slot->buffer_bytes.resize(num_io_tensors, 0);

        // Pass 1: give every dynamic input its largest profile shape, so that
        // output shapes (and therefore buffer sizes) can be resolved in pass 2.
        for (int i = 0; i < num_io_tensors; i++) {
            const char *name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
                continue;

            TensorInfo probe;
            probe.shape = to_shape(engine->getTensorShape(name));
            if (!probe.is_dynamic())
                continue;

            if (engine->getNbOptimizationProfiles() < 1) {
                throw ModelLoadError(std::string("Input '") + name +
                                     "' has a dynamic shape but the "
                                     "engine has no optimization profile");
            }
            nvinfer1::Dims max_dims =
                engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            if (!slot->context->setInputShape(name, max_dims)) {
                throw ModelLoadError(std::string("Cannot apply the maximum profile shape to input '") +
                                     name + "'");
            }
        }

        for (int i = 0; i < num_io_tensors; i++) {
            const char *name = engine->getIOTensorName(i);
            nvinfer1::Dims dims = engine->getTensorShape(name);
            nvinfer1::DataType nv_dtype = engine->getTensorDataType(name);
            nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(name);

            TensorInfo info;
            info.name = name;
            info.dtype = nv_to_dtype(nv_dtype);
            if (info.dtype == DataType::Unknown) {
                // Buffers and host tensors are sized from the dtype; Unknown would be wrong.
                throw ModelLoadError(std::string("Tensor '") + name + "' has a data type iVIT "
                                     "cannot represent (e.g. bfloat16 or FP8); keep model "
                                     "inputs and outputs in float32, float16, int8, int32 or int64");
            }

            // Keep the declared shape (negative = dynamic) so callers can tell
            info.shape = to_shape(dims);

            // The device buffer must hold the largest shape the engine can see
            TensorInfo alloc_info = info;
            if (info.is_dynamic()) {
                if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
                    alloc_info.shape = to_shape(
                        engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX));
                } else {
                    alloc_info.shape = to_shape(slot->context->getTensorShape(name));
                }
                if (alloc_info.is_dynamic()) {
                    throw ModelLoadError(std::string("Cannot determine the size of tensor '") +
                                         name + "' (data-dependent output shapes are not supported)");
                }
            }

            // Determine layout
            if (dims.nbDims == 4) {
                info.layout = Layout::NCHW;
            } else if (dims.nbDims == 2) {
                info.layout = Layout::NC;
            } else {
                info.layout = Layout::Unknown;
            }

            // Allocate device buffer with error checking
            size_t buffer_size = alloc_info.byte_size();
            cudaError_t malloc_err = cudaMalloc(&slot->device_buffers[i], buffer_size);
            if (malloc_err != cudaSuccess) {
                // A failed cudaMalloc may leave garbage in the pointer; make sure
                // ~TensorRTEngine does not try to free it.
                slot->device_buffers[i] = nullptr;
                throw IVITError(std::string("Failed to allocate CUDA memory (") +
                                std::to_string(buffer_size) +
                                " bytes): " + cudaGetErrorString(malloc_err));
            }
            slot->buffer_bytes[i] = buffer_size;

            // Set tensor address in context
            slot->context->setTensorAddress(name, slot->device_buffers[i]);

            // Store info (once)
            if (first) {
                if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
                    wrapper->input_infos.push_back(info);
                } else {
                    wrapper->output_infos.push_back(info);
                }
            }
        }
    }

    for (auto &slot : wrapper->slots)
        wrapper->free_slots.push_back(slot.get());
    return wrapper.release();
}

DataType TensorRTRuntime::nv_to_dtype(nvinfer1::DataType nv_type) const {
    switch (nv_type) {
    case nvinfer1::DataType::kFLOAT:
        return DataType::Float32;
    case nvinfer1::DataType::kHALF:
        return DataType::Float16;
    case nvinfer1::DataType::kINT8:
        return DataType::Int8;
    case nvinfer1::DataType::kINT32:
        return DataType::Int32;
#if NV_TENSORRT_MAJOR >=                                                                           \
    9 // kINT64 (9.0) and kUINT8 (8.6) are missing on older TensorRT (e.g. JetPack 5)
    case nvinfer1::DataType::kINT64:
        return DataType::Int64;
    case nvinfer1::DataType::kUINT8:
        return DataType::UInt8;
#endif
    case nvinfer1::DataType::kBOOL:
        return DataType::Bool;
    default:
        return DataType::Unknown;
    }
}

TensorRTRuntime &get_tensorrt_runtime() {
    static TensorRTRuntime instance;
    return instance;
}

} // namespace ivit

#pragma GCC diagnostic pop

#endif // IVIT_HAS_TENSORRT

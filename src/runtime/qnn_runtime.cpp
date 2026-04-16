/**
 * @file qnn_runtime.cpp
 * @brief Qualcomm QNN backend runtime implementation
 *
 * Implements IRuntime for Qualcomm AI Engine Direct (QNN SDK 2.x).
 * Loads serialized context binaries (.bin) and runs inference on
 * HTP (Hexagon), Adreno GPU, or CPU backends.
 */

#include "ivit/runtime/qnn_runtime.hpp"

#ifdef IVIT_HAS_QNN

#include "ivit/core/tensor.hpp"
#include "ivit/core/common.hpp"

#include <dlfcn.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace ivit {

// ============================================================================
// Internal helpers
// ============================================================================

static std::string backend_to_lib(const std::string& b) {
    std::string bl = b;
    std::transform(bl.begin(), bl.end(), bl.begin(), ::tolower);
    if (bl == "gpu")  return "libQnnGpu.so";
    if (bl == "cpu")  return "libQnnCpu.so";
    return "libQnnHtp.so";   // default: HTP
}

static std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        throw IVITError("QNNRuntime: cannot open: " + path);
    }
    auto sz = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    if (!ifs.read(reinterpret_cast<char*>(buf.data()), sz)) {
        throw IVITError("QNNRuntime: read failed: " + path);
    }
    return buf;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

QNNRuntime::QNNRuntime() = default;

QNNRuntime::~QNNRuntime() {
    if (shared_backend_ && qnn_interface_) {
        qnn_interface_->backendFree(shared_backend_);
    }
    if (backend_lib_handle_) dlclose(backend_lib_handle_);
    if (sys_lib_handle_)     dlclose(sys_lib_handle_);
}

// ============================================================================
// is_available
// ============================================================================

bool QNNRuntime::is_available() const {
    // Probe for at least one QNN backend library
    for (const char* lib : {"libQnnHtp.so", "libQnnCpu.so", "libQnnGpu.so"}) {
        void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (h) { dlclose(h); return true; }
    }
    return false;
}

// ============================================================================
// supported_formats / get_devices
// ============================================================================

std::vector<std::string> QNNRuntime::supported_formats() const {
    return {".bin", ".serialized"};
}

std::vector<DeviceInfo> QNNRuntime::get_devices() const {
    std::vector<DeviceInfo> devices;

    struct {
        const char* lib;
        const char* id;
        const char* display_name;
        const char* type;
        std::vector<Precision> precisions;
    } candidates[] = {
        {"libQnnHtp.so", "htp",     "Qualcomm Hexagon HTP (NPU)", "npu",
         {Precision::FP32, Precision::FP16, Precision::INT8}},
        {"libQnnGpu.so", "qnn:gpu", "Qualcomm Adreno GPU",         "gpu",
         {Precision::FP32, Precision::FP16}},
        {"libQnnCpu.so", "qnn:cpu", "Qualcomm QNN CPU",            "cpu",
         {Precision::FP32}},
    };

    for (const auto& c : candidates) {
        void* h = dlopen(c.lib, RTLD_NOW | RTLD_LOCAL);
        if (h) {
            DeviceInfo d;
            d.id   = c.id;
            d.name = c.display_name;
            d.backend = "qnn";
            d.type    = c.type;
            d.is_available        = true;
            d.supported_precisions = c.precisions;
            devices.push_back(d);
            dlclose(h);
        }
    }
    return devices;
}

// ============================================================================
// Backend library loading
// ============================================================================

bool QNNRuntime::load_backend_library(const std::string& backend_name,
                                       const std::string& skel_dir) {
    // For HTP: set skel library search path so the DSP driver can find it
    if (!skel_dir.empty()) {
        setenv("ADSP_LIBRARY_PATH", skel_dir.c_str(), 1);
    }

    std::string lib = backend_to_lib(backend_name);
    backend_lib_handle_ = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!backend_lib_handle_) {
        return false;
    }

    // Resolve QNN backend provider list
    using GetProvidersFn = Qnn_ErrorHandle_t (*)(
        const QnnInterface_t***, uint32_t*);
    auto get_providers = reinterpret_cast<GetProvidersFn>(
        dlsym(backend_lib_handle_, "QnnInterface_getProviders"));
    if (!get_providers) {
        dlclose(backend_lib_handle_);
        backend_lib_handle_ = nullptr;
        return false;
    }

    const QnnInterface_t** providers = nullptr;
    uint32_t num_providers = 0;
    if (get_providers(&providers, &num_providers) != QNN_SUCCESS ||
        num_providers == 0 || !providers[0]) {
        dlclose(backend_lib_handle_);
        backend_lib_handle_ = nullptr;
        return false;
    }
    qnn_interface_ = providers[0];

    // Optionally load QnnSystem for binary metadata extraction
    sys_lib_handle_ = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (sys_lib_handle_) {
        using SysGetProvidersFn = Qnn_ErrorHandle_t (*)(
            const QnnSystemInterface_t***, uint32_t*);
        auto sys_get = reinterpret_cast<SysGetProvidersFn>(
            dlsym(sys_lib_handle_, "QnnSystemInterface_getProviders"));
        if (sys_get) {
            const QnnSystemInterface_t** sp = nullptr;
            uint32_t ns = 0;
            if (sys_get(&sp, &ns) == QNN_SUCCESS && ns > 0 && sp[0]) {
                sys_interface_ = sp[0];
            }
        }
    }

    return true;
}

// ============================================================================
// initialize — create backend handle
// ============================================================================

void QNNRuntime::initialize(const std::string& backend_name,
                             const std::string& skel_dir) {
    if (initialized_ && backend_name_ == backend_name) return;

    // Tear down previous backend if switching
    if (shared_backend_ && qnn_interface_) {
        qnn_interface_->backendFree(shared_backend_);
        shared_backend_ = nullptr;
    }
    if (backend_lib_handle_) {
        dlclose(backend_lib_handle_);
        backend_lib_handle_ = nullptr;
        qnn_interface_      = nullptr;
    }
    if (sys_lib_handle_) {
        dlclose(sys_lib_handle_);
        sys_lib_handle_ = nullptr;
        sys_interface_  = nullptr;
    }

    if (!load_backend_library(backend_name, skel_dir)) {
        std::string err = dlerror() ? dlerror() : "unknown";
        throw IVITError("QNNRuntime: failed to load " +
                        backend_to_lib(backend_name) + ": " + err);
    }

    Qnn_ErrorHandle_t rc = qnn_interface_->backendCreate(
        nullptr,   // log handle
        nullptr,   // config (null → defaults)
        &shared_backend_
    );
    if (rc != QNN_SUCCESS) {
        throw IVITError("QNNRuntime: backendCreate failed (" +
                        std::to_string(rc) + ")");
    }

    backend_name_ = backend_name;
    initialized_  = true;
}

// ============================================================================
// Backend selection from RuntimeConfig
// ============================================================================

std::string QNNRuntime::select_backend(const RuntimeConfig& config) const {
    std::string dev = config.device;
    std::transform(dev.begin(), dev.end(), dev.begin(), ::tolower);
    if (dev.find("gpu") != std::string::npos) return "GPU";
    if (dev.find("cpu") != std::string::npos) return "CPU";
    return "HTP";   // default for Qualcomm NPU devices
}

// ============================================================================
// extract_tensor_infos — uses QnnSystem to read binary metadata
// ============================================================================

void QNNRuntime::extract_tensor_infos(QNNModel* model,
                                       const std::vector<uint8_t>& binary) {
    if (!sys_interface_) return;

    QnnSystemContext_Handle_t sys_ctx = nullptr;
    if (sys_interface_->systemContextCreate(&sys_ctx) != QNN_SUCCESS) return;

    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t info_size = 0;

    Qnn_ErrorHandle_t rc = sys_interface_->systemContextGetBinaryInfo(
        sys_ctx,
        const_cast<void*>(reinterpret_cast<const void*>(binary.data())),
        static_cast<Qnn_ContextBinarySize_t>(binary.size()),
        &info,
        &info_size
    );
    if (rc != QNN_SUCCESS || !info) {
        sys_interface_->systemContextFree(sys_ctx);
        return;
    }

    if (info->version != QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1 ||
        info->contextBlobInfo.numGraphs == 0) {
        sys_interface_->systemContextFree(sys_ctx);
        return;
    }

    // Use the first graph in the context binary
    const auto& g = info->contextBlobInfo.graphs[0];
    if (g.version != QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        sys_interface_->systemContextFree(sys_ctx);
        return;
    }
    const auto& gv1 = g.graphInfoV1;

    if (gv1.graphName) {
        model->graph_name = gv1.graphName;
    }

    for (uint32_t i = 0; i < gv1.numGraphInputs; ++i) {
        model->input_infos.push_back(make_tensor_info(gv1.graphInputs[i]));
        model->input_tensors.push_back(gv1.graphInputs[i]);
    }
    for (uint32_t i = 0; i < gv1.numGraphOutputs; ++i) {
        model->output_infos.push_back(make_tensor_info(gv1.graphOutputs[i]));
        model->output_tensors.push_back(gv1.graphOutputs[i]);
    }

    sys_interface_->systemContextFree(sys_ctx);
}

// ============================================================================
// load_model
// ============================================================================

void* QNNRuntime::load_model(const std::string& path,
                              const RuntimeConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string backend = select_backend(config);
    initialize(backend);

    auto* model = new QNNModel();
    model->binary_data  = read_binary_file(path);
    model->backend_name = backend;

    // Build context from serialized binary
    Qnn_ErrorHandle_t rc = qnn_interface_->contextCreateFromBinary(
        shared_backend_,
        nullptr,    // device handle (optional)
        nullptr,    // config (null → defaults)
        model->binary_data.data(),
        static_cast<Qnn_ContextBinarySize_t>(model->binary_data.size()),
        &model->context,
        nullptr     // profile handle
    );
    if (rc != QNN_SUCCESS) {
        delete model;
        throw ModelLoadError("QNN contextCreateFromBinary failed (" +
                             std::to_string(rc) + "): " + path);
    }

    // Extract tensor shapes and graph name via QnnSystem metadata
    extract_tensor_infos(model, model->binary_data);

    // Retrieve graph by name
    rc = qnn_interface_->graphRetrieve(
        model->context,
        model->graph_name.empty() ? nullptr : model->graph_name.c_str(),
        &model->graph
    );
    if (rc != QNN_SUCCESS) {
        qnn_interface_->contextFree(model->context, nullptr);
        delete model;
        throw ModelLoadError("QNN graphRetrieve failed (" +
                             std::to_string(rc) + "): " + path);
    }

    // Pre-allocate output buffers and bind client-buffer pointers on output tensors
    model->output_buffers.resize(model->output_infos.size());
    for (size_t i = 0; i < model->output_infos.size(); ++i) {
        size_t bytes = model->output_infos[i].byte_size();
        model->output_buffers[i].resize(bytes, 0);

        QNN_TENSOR_SET_MEM_TYPE(model->output_tensors[i], QNN_TENSORMEMTYPE_RAW);
        Qnn_ClientBuffer_t buf;
        buf.data     = model->output_buffers[i].data();
        buf.dataSize = static_cast<uint32_t>(bytes);
        QNN_TENSOR_SET_CLIENT_BUF(model->output_tensors[i], buf);
    }

    return model;
}

// ============================================================================
// unload_model
// ============================================================================

void QNNRuntime::unload_model(void* handle) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto* model = static_cast<QNNModel*>(handle);
    if (model->context && qnn_interface_) {
        qnn_interface_->contextFree(model->context, nullptr);
    }
    delete model;
}

// ============================================================================
// get_input_info / get_output_info
// ============================================================================

std::vector<TensorInfo> QNNRuntime::get_input_info(void* handle) const {
    return static_cast<const QNNModel*>(handle)->input_infos;
}

std::vector<TensorInfo> QNNRuntime::get_output_info(void* handle) const {
    return static_cast<const QNNModel*>(handle)->output_infos;
}

// ============================================================================
// infer
// ============================================================================

std::map<std::string, Tensor> QNNRuntime::infer(
    void* handle,
    const std::map<std::string, Tensor>& inputs
) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* model = static_cast<QNNModel*>(handle);

    // Bind input client-buffer pointers (zero-copy: point into caller Tensor)
    for (size_t i = 0; i < model->input_tensors.size(); ++i) {
        const std::string& tname = model->input_infos[i].name;

        // Prefer name lookup, fall back to positional
        const Tensor* src = nullptr;
        auto it = inputs.find(tname);
        if (it != inputs.end()) {
            src = &it->second;
        } else if (i < inputs.size()) {
            src = &std::next(inputs.begin(), i)->second;
        } else {
            throw InferenceError("QNNRuntime: missing input: " + tname);
        }

        QNN_TENSOR_SET_MEM_TYPE(model->input_tensors[i], QNN_TENSORMEMTYPE_RAW);
        Qnn_ClientBuffer_t buf;
        buf.data     = const_cast<void*>(src->data());
        buf.dataSize = static_cast<uint32_t>(src->byte_size());
        QNN_TENSOR_SET_CLIENT_BUF(model->input_tensors[i], buf);
    }

    // Execute graph
    Qnn_ErrorHandle_t rc = qnn_interface_->graphExecute(
        model->graph,
        model->input_tensors.data(),
        static_cast<uint32_t>(model->input_tensors.size()),
        model->output_tensors.data(),
        static_cast<uint32_t>(model->output_tensors.size()),
        nullptr,   // profile handle
        nullptr    // signal handle
    );
    if (rc != QNN_SUCCESS) {
        throw InferenceError("QNNRuntime: graphExecute failed (" +
                             std::to_string(rc) + ")");
    }

    // Copy output buffers into iVIT Tensors
    std::map<std::string, Tensor> outputs;
    for (size_t i = 0; i < model->output_infos.size(); ++i) {
        const TensorInfo& info = model->output_infos[i];
        Tensor out(info.shape, info.dtype);
        out.set_name(info.name);
        out.set_layout(info.layout);
        std::memcpy(out.data(), model->output_buffers[i].data(),
                    model->output_buffers[i].size());
        outputs[info.name] = std::move(out);
    }

    return outputs;
}

// ============================================================================
// convert_model
// ============================================================================

void QNNRuntime::convert_model(
    const std::string& src_path,
    const std::string& dst_path,
    const RuntimeConfig& /*config*/
) {
    throw IVITError(
        "QNNRuntime: runtime model conversion is not supported.\n"
        "Convert offline with:\n"
        "  qnn-onnx-converter --input_network " + src_path +
        " --output_path " + dst_path + "\n"
        "  qnn-context-binary-generator --model <lib.so> --backend libQnnHtp.so "
        "--output_dir <dir>"
    );
}

// ============================================================================
// get_version
// ============================================================================

std::string QNNRuntime::get_version() const {
    if (!qnn_interface_ || !shared_backend_) return "unknown";
    Qnn_ApiVersion_t ver{};
    if (qnn_interface_->backendGetApiVersion(shared_backend_, &ver) == QNN_SUCCESS) {
        return std::to_string(ver.coreApiVersion.major) + "." +
               std::to_string(ver.coreApiVersion.minor) + "." +
               std::to_string(ver.coreApiVersion.patch);
    }
    return "unknown";
}

// ============================================================================
// Type mapping
// ============================================================================

DataType QNNRuntime::qnn_to_dtype(Qnn_DataType_t t) const {
    switch (t) {
        case QNN_DATATYPE_FLOAT_32: return DataType::Float32;
        case QNN_DATATYPE_FLOAT_16: return DataType::Float16;
        case QNN_DATATYPE_INT_8:    return DataType::Int8;
        case QNN_DATATYPE_UINT_8:   return DataType::UInt8;
        case QNN_DATATYPE_INT_32:   return DataType::Int32;
        case QNN_DATATYPE_INT_64:   return DataType::Int64;
        case QNN_DATATYPE_BOOL_8:   return DataType::Bool;
        default:                    return DataType::Unknown;
    }
}

size_t QNNRuntime::element_size(Qnn_DataType_t t) const {
    switch (t) {
        case QNN_DATATYPE_FLOAT_32: return 4;
        case QNN_DATATYPE_FLOAT_16: return 2;
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_BOOL_8:   return 1;
        case QNN_DATATYPE_INT_32:   return 4;
        case QNN_DATATYPE_INT_64:   return 8;
        default:                    return 4;
    }
}

TensorInfo QNNRuntime::make_tensor_info(const Qnn_Tensor_t& t) const {
    TensorInfo info;
    const char* n = QNN_TENSOR_GET_NAME(t);
    info.name  = n ? n : "";
    info.dtype = qnn_to_dtype(QNN_TENSOR_GET_DATA_TYPE(t));

    uint32_t  rank = QNN_TENSOR_GET_RANK(t);
    uint32_t* dims = QNN_TENSOR_GET_DIMENSIONS(t);
    for (uint32_t i = 0; i < rank; ++i) {
        info.shape.push_back(static_cast<int64_t>(dims[i]));
    }

    // QNN defaults to NHWC layout for 4-D tensors
    info.layout = (rank == 4) ? Layout::NHWC : Layout::Unknown;
    return info;
}

// ============================================================================
// Global instance
// ============================================================================

QNNRuntime& get_qnn_runtime() {
    static QNNRuntime instance;
    return instance;
}

} // namespace ivit

#endif // IVIT_HAS_QNN

/**
 * @file qnn_runtime.hpp
 * @brief Qualcomm QNN backend runtime
 *
 * Supports Qualcomm AI Engine Direct (QNN) for inference on:
 *   - HTP (Hexagon Tensor Processor) — NPU acceleration
 *   - GPU (Adreno GPU)               — GPU acceleration
 *   - CPU                            — CPU fallback
 *
 * Target hardware: Qualcomm IQ6/IQ8/IQ9 series, QCS9075, QCS8550, etc.
 * Requires: Qualcomm AI Engine Direct SDK (QNN SDK 2.x)
 */

#ifndef IVIT_RUNTIME_QNN_RUNTIME_HPP
#define IVIT_RUNTIME_QNN_RUNTIME_HPP

#include "ivit/runtime/runtime.hpp"

#ifdef IVIT_HAS_QNN

#include <QnnInterface.h>
#include <QnnBackend.h>
#include <QnnContext.h>
#include <QnnGraph.h>
#include <QnnTensor.h>
#include <System/QnnSystemInterface.h>
#include <System/QnnSystemContext.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <map>

namespace ivit {

/**
 * @brief QNN loaded model wrapper
 *
 * Holds QNN context/graph handles and I/O buffers for one loaded
 * serialized context binary (.bin / .serialized).
 */
struct QNNModel {
    Qnn_ContextHandle_t context  = nullptr;
    Qnn_GraphHandle_t   graph    = nullptr;

    std::string graph_name;          ///< Graph name retrieved from binary metadata
    std::string backend_name;        ///< "HTP", "GPU", or "CPU"

    std::vector<TensorInfo>    input_infos;
    std::vector<TensorInfo>    output_infos;

    // Tensor descriptors (mutable — client-buffer pointers updated per inference)
    std::vector<Qnn_Tensor_t>  input_tensors;
    std::vector<Qnn_Tensor_t>  output_tensors;

    // Host-side output buffers (allocated at load time)
    std::vector<std::vector<uint8_t>> output_buffers;

    // Raw binary kept alive for the context lifetime
    std::vector<uint8_t> binary_data;
};

/**
 * @brief Qualcomm QNN runtime implementation
 *
 * Loads serialized QNN context binaries and executes inference via
 * Qualcomm AI Engine Direct.
 *
 * Device strings recognised:
 *   "htp" | "iq6" | "iq8" | "iq9" | "hexagon" | "dsp" → HTP backend
 *   "qnn:gpu"                                           → Adreno GPU backend
 *   "qnn:cpu"                                           → QNN CPU backend
 */
class QNNRuntime : public IRuntime {
public:
    QNNRuntime();
    ~QNNRuntime() override;

    // ----- IRuntime interface -----
    BackendType              type()              const override { return BackendType::QNN; }
    std::string              name()              const override { return "QNN"; }
    bool                     is_available()      const override;
    std::vector<std::string> supported_formats() const override;
    std::vector<DeviceInfo>  get_devices()       const override;

    void* load_model(
        const std::string& path,
        const RuntimeConfig& config
    ) override;

    void unload_model(void* handle) override;

    std::vector<TensorInfo> get_input_info(void* handle)  const override;
    std::vector<TensorInfo> get_output_info(void* handle) const override;

    std::map<std::string, Tensor> infer(
        void* handle,
        const std::map<std::string, Tensor>& inputs
    ) override;

    void convert_model(
        const std::string& src_path,
        const std::string& dst_path,
        const RuntimeConfig& config
    ) override;

    // ----- QNN-specific -----
    std::string get_version()      const;
    std::string get_backend_name() const { return backend_name_; }

private:
    void initialize(const std::string& backend_name,
                    const std::string& skel_dir = "");

    bool load_backend_library(const std::string& backend_name,
                              const std::string& skel_dir);

    void extract_tensor_infos(QNNModel* model,
                              const std::vector<uint8_t>& binary);

    std::string select_backend(const RuntimeConfig& config) const;

    DataType       qnn_to_dtype(Qnn_DataType_t qnn_type) const;
    TensorInfo     make_tensor_info(const Qnn_Tensor_t& t) const;
    size_t         element_size(Qnn_DataType_t dtype) const;

    // QNN backend library (dynamically loaded)
    void*                 backend_lib_handle_ = nullptr;
    const QnnInterface_t* qnn_interface_      = nullptr;
    Qnn_BackendHandle_t   shared_backend_     = nullptr;

    // QNN System library (for binary metadata extraction)
    void*                        sys_lib_handle_ = nullptr;
    const QnnSystemInterface_t*  sys_interface_  = nullptr;

    std::string backend_name_ = "HTP";
    std::mutex  mutex_;
    bool        initialized_  = false;
};

/**
 * @brief Get global QNN runtime instance
 */
QNNRuntime& get_qnn_runtime();

} // namespace ivit

#endif // IVIT_HAS_QNN
#endif // IVIT_RUNTIME_QNN_RUNTIME_HPP

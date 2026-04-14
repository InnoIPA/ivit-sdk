/**
 * @file hailort_runtime.hpp
 * @brief HailoRT backend runtime for Hailo-8 AI accelerator
 *
 * Supports Hailo-8 PCIe M.2 AI accelerator via HailoRT 4.18+
 * InferModel high-level C++ API.
 *
 * Device strings recognised:
 *   "hailo"    → first Hailo-8 card (device_id = 0)
 *   "hailo:0"  → explicit first card
 *   "hailo:1"  → second card (multi-card environment)
 *
 * Model format: .hef (Hailo Executable Format)
 * Conversion: hailo_model_zoo compile <onnx> --hw-arch hailo8 -o <hef>
 */

#ifndef IVIT_RUNTIME_HAILORT_RUNTIME_HPP
#define IVIT_RUNTIME_HAILORT_RUNTIME_HPP

#include "ivit/runtime/runtime.hpp"

#ifdef IVIT_HAS_HAILORT

#include <hailo/hailort.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <map>

namespace ivit {

/**
 * @brief HailoRT loaded model wrapper
 *
 * Holds all HailoRT InferModel handles and tensor metadata
 * for one loaded .hef model.
 */
struct HailoRTModel {
    std::shared_ptr<hailort::VDevice>    vdevice;
    std::shared_ptr<hailort::InferModel> infer_model;
    hailort::ConfiguredInferModel        configured_model;

    std::vector<TensorInfo> input_infos;
    std::vector<TensorInfo> output_infos;

    std::string device_id;  ///< e.g. "hailo:0"
    int         device_idx; ///< Physical device index (0, 1, …)
};

/**
 * @brief Hailo-8 runtime implementation via HailoRT InferModel API
 */
class HailoRTRuntime : public IRuntime {
public:
    HailoRTRuntime();
    ~HailoRTRuntime() override = default;

    // ----- IRuntime interface -----
    BackendType              type()              const override { return BackendType::HailoRT; }
    std::string              name()              const override { return "HailoRT"; }
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

private:
    int  parse_device_index(const std::string& device_id) const;

    DataType hailort_to_dtype(hailo_format_type_t fmt) const;
    TensorInfo make_tensor_info(const hailort::InferModel::InferStream& stream) const;

    mutable std::mutex mutex_;
};

/**
 * @brief Check if any Hailo device is accessible on this machine
 */
bool hailo_is_available();

} // namespace ivit

#endif // IVIT_HAS_HAILORT
#endif // IVIT_RUNTIME_HAILORT_RUNTIME_HPP

/**
 * @file openvino_runtime.hpp
 * @brief OpenVINO backend runtime
 */

#ifndef IVIT_RUNTIME_OPENVINO_RUNTIME_HPP
#define IVIT_RUNTIME_OPENVINO_RUNTIME_HPP

#include "ivit/runtime/runtime.hpp"

#ifdef IVIT_HAS_OPENVINO

#include <openvino/openvino.hpp>
#include <memory>
#include <vector>
#include <string>
#include <mutex>

namespace ivit {

/**
 * @brief OpenVINO compiled model wrapper
 */
struct OpenVINOModel {
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::mutex infer_mutex;  ///< Serializes use of the single infer_request
    std::vector<TensorInfo> input_infos;
    std::vector<TensorInfo> output_infos;
    std::string device;
};

/**
 * @brief OpenVINO runtime implementation
 */
class OpenVINORuntime : public IRuntime {
public:
    OpenVINORuntime();
    ~OpenVINORuntime() override;

    // IRuntime interface
    BackendType type() const override { return BackendType::OpenVINO; }
    std::string name() const override { return "OpenVINO"; }
    bool is_available() const override;
    std::vector<std::string> supported_formats() const override;
    std::vector<DeviceInfo> get_devices() const override;

    void* load_model(
        const std::string& path,
        const RuntimeConfig& config
    ) override;

    void unload_model(void* handle) override;

    std::vector<TensorInfo> get_input_info(void* handle) const override;
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

    // OpenVINO-specific methods

    /**
     * @brief Get OpenVINO Core instance
     */
    ov::Core& get_core() { return core_; }

    /**
     * @brief Map device string to OpenVINO device name
     */
    std::string map_device_name(const std::string& device) const;

    /**
     * @brief Get OpenVINO version
     */
    std::string get_version() const;

    /**
     * @brief Set cache directory for compiled models
     */
    void set_cache_dir(const std::string& path);

    /**
     * @brief Enable/disable async inference
     */

private:
    void initialize();

    DataType ov_to_dtype(ov::element::Type ov_type) const;
    ov::element::Type dtype_to_ov(DataType dtype) const;

    TensorInfo extract_tensor_info(const ov::Output<ov::Node>& port);

    ov::Core core_;
    std::mutex mutex_;
    bool initialized_ = false;
    std::string cache_dir_;
};

/**
 * @brief Get global OpenVINO runtime instance
 */
OpenVINORuntime& get_openvino_runtime();

} // namespace ivit

#endif // IVIT_HAS_OPENVINO

#endif // IVIT_RUNTIME_OPENVINO_RUNTIME_HPP

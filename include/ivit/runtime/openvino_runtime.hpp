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
using namespace std;
namespace ivit {

/**
 * @brief OpenVINO compiled model wrapper
 */
struct OpenVINOModel {
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    vector<TensorInfo> input_infos;
    vector<TensorInfo> output_infos;
    string device;
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
    string name() const override { return "OpenVINO"; }
    bool is_available() const override;
    vector<string> supported_formats() const override;
    vector<DeviceInfo> get_devices() const override;

    void* load_model(
        const string& path,
        const RuntimeConfig& config
    ) override;

    void unload_model(void* handle) override;

    vector<TensorInfo> get_input_info(void* handle) const override;
    vector<TensorInfo> get_output_info(void* handle) const override;

    map<string, Tensor> infer(
        void* handle,
        const map<string, Tensor>& inputs
    ) override;

    void convert_model(
        const string& src_path,
        const string& dst_path,
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
    string map_device_name(const string& device) const;

    /**
     * @brief Get OpenVINO version
     */
    string get_version() const;

    /**
     * @brief Set cache directory for compiled models
     */
    void set_cache_dir(const string& path);

    /**
     * @brief Enable/disable async inference
     */
    void set_async(bool enable) { use_async_ = enable; }

private:
    void initialize();

    DataType ov_to_dtype(ov::element::Type ov_type) const;
    ov::element::Type dtype_to_ov(DataType dtype) const;

    TensorInfo extract_tensor_info(const ov::Output<ov::Node>& port);

    ov::Core core_;
    mutex mutex_;
    bool initialized_ = false;
    bool use_async_ = false;
    string cache_dir_;
};

/**
 * @brief Get global OpenVINO runtime instance
 */
OpenVINORuntime& get_openvino_runtime();

} // namespace ivit

#endif // IVIT_HAS_OPENVINO

#endif // IVIT_RUNTIME_OPENVINO_RUNTIME_HPP

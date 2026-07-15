/**
 * @file runtime.hpp
 * @brief Runtime abstraction interface
 */

#ifndef IVIT_RUNTIME_RUNTIME_HPP
#define IVIT_RUNTIME_RUNTIME_HPP

#include "ivit/core/common.hpp"
#include "ivit/core/tensor.hpp"
#include "ivit/core/device.hpp"
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace ivit {

/**
 * @brief Runtime configuration
 */
struct RuntimeConfig {
    std::string device = "auto";
    int num_threads = 0;               ///< 0 = auto
    bool enable_profiling = false;
    std::string cache_dir = "";
    Precision precision = Precision::FP32;
    bool use_cache = true;             ///< Enable automatic caching for compiled engines
};

/**
 * @brief Abstract runtime interface
 *
 * All backend runtimes (OpenVINO, TensorRT, QNN) implement this interface.
 */
class IRuntime {
public:
    virtual ~IRuntime() = default;

    /**
     * @brief Get backend type
     */
    virtual BackendType type() const = 0;

    /**
     * @brief Get backend name
     */
    virtual std::string name() const = 0;

    /**
     * @brief Check if runtime is available
     */
    virtual bool is_available() const = 0;

    /**
     * @brief Get supported model formats
     */
    virtual std::vector<std::string> supported_formats() const = 0;

    /**
     * @brief Get available devices
     */
    virtual std::vector<DeviceInfo> get_devices() const = 0;

    /**
     * @brief Load model
     *
     * @param path Model file path
     * @param config Runtime configuration
     * @return Internal model handle
     */
    virtual void* load_model(
        const std::string& path,
        const RuntimeConfig& config
    ) = 0;

    /**
     * @brief Unload model
     */
    virtual void unload_model(void* handle) = 0;

    /**
     * @brief Get model input info
     */
    virtual std::vector<TensorInfo> get_input_info(void* handle) const = 0;

    /**
     * @brief Get model output info
     */
    virtual std::vector<TensorInfo> get_output_info(void* handle) const = 0;

    /**
     * @brief Run inference
     *
     * @param handle Model handle
     * @param inputs Input tensors
     * @return Output tensors
     */
    virtual std::map<std::string, Tensor> infer(
        void* handle,
        const std::map<std::string, Tensor>& inputs
    ) = 0;

    /**
     * @brief Convert model to optimized format
     *
     * @param src_path Source model path
     * @param dst_path Destination path
     * @param config Configuration
     */
    virtual void convert_model(
        const std::string& src_path,
        const std::string& dst_path,
        const RuntimeConfig& config
    ) = 0;
};

/**
 * @brief Runtime factory
 */
class RuntimeFactory {
public:
    /**
     * @brief Get singleton instance
     */
    static RuntimeFactory& instance();

    /**
     * @brief Get runtime by type
     */
    std::shared_ptr<IRuntime> get_runtime(BackendType type);

    /**
     * @brief Get runtime for device
     */
    std::shared_ptr<IRuntime> get_runtime_for_device(const std::string& device);

    /**
     * @brief Get best runtime for model
     */
    std::shared_ptr<IRuntime> get_best_runtime(
        const std::string& model_path,
        const std::string& device = "auto"
    );

    /**
     * @brief Register runtime
     */
    void register_runtime(std::shared_ptr<IRuntime> runtime);

    /**
     * @brief List available runtimes
     */
    std::vector<BackendType> list_available_runtimes() const;

private:
    RuntimeFactory();
    ~RuntimeFactory() = default;

    std::map<BackendType, std::shared_ptr<IRuntime>> runtimes_;
};

/**
 * @brief Lazily register all compiled-in runtimes (thread-safe, once).
 *
 * Registration probes CUDA/TensorRT/OpenVINO, which must NOT run during DLL
 * static initialization on Windows (doing so under the loader lock deadlocks).
 * This is invoked on first runtime lookup instead of at library load.
 */
void ensure_runtimes_registered();

} // namespace ivit

#endif // IVIT_RUNTIME_RUNTIME_HPP

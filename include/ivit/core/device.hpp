/**
 * @file device.hpp
 * @brief Device management
 */

#ifndef IVIT_CORE_DEVICE_HPP
#define IVIT_CORE_DEVICE_HPP

#include "ivit/core/common.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ivit {

/**
 * @brief Device information structure
 */
struct DeviceInfo {
    std::string id;                              ///< Device ID (e.g., "cpu", "cuda:0", "npu")
    std::string name;                            ///< Device name (e.g., "NVIDIA GeForce RTX 4090")
    std::string backend;                         ///< Backend name (e.g., "tensorrt")
    std::string type;                            ///< Device type (e.g., "gpu", "cpu", "npu")
    size_t memory_total = 0;                     ///< Total memory in bytes
    size_t memory_available = 0;                 ///< Available memory in bytes
    std::vector<Precision> supported_precisions; ///< Supported precisions
    bool is_available = true;                    ///< Whether device is available
};

/**
 * @brief Device status information
 */
struct DeviceStatus {
    std::string id;                 ///< Device ID
    std::string name;               ///< Device name
    std::string backend;            ///< Backend name
    std::string compute_capability; ///< Compute capability (CUDA only)

    // Memory status
    size_t memory_total = 0;      ///< Total memory in bytes
    size_t memory_used = 0;       ///< Memory used in bytes
    size_t memory_free = 0;       ///< Free memory in bytes
    float memory_utilization = 0; ///< Memory utilization percentage (0-100)

    // Utilization (requires NVML for CUDA)
    float utilization = 0; ///< GPU utilization (0.0 - 100.0)
    float temperature = 0; ///< Temperature in Celsius
    float power_usage = 0; ///< Power usage in Watts

    // Precision support
    bool supports_fp32 = false; ///< FP32 precision support
    bool supports_fp16 = false; ///< FP16 precision support
    bool supports_int8 = false; ///< INT8 precision support

    bool is_available = false; ///< Whether device is available
};

/**
 * @brief Device manager singleton
 */
class DeviceManager {
  public:
    /**
     * @brief Get singleton instance
     */
    static DeviceManager &instance();

    /**
     * @brief List all available devices
     */
    std::vector<DeviceInfo> list_devices();

    /**
     * @brief Get device by ID
     */
    DeviceInfo get_device(const std::string &device_id);

    /**
     * @brief Get the best available device
     *
     * @param priority Selection priority: "performance", "efficiency", "memory"
     * @return Best device information
     */
    DeviceInfo get_best_device(const std::string &priority = "performance");

    /**
     * @brief Get device status
     */
    DeviceStatus get_device_status(const std::string &device_id);

    /**
     * @brief Check if device supports precision
     */
    bool supports_precision(const std::string &device_id, Precision precision);

    /**
     * @brief Check if device supports format
     */
    bool supports_format(const std::string &device_id, const std::string &format);

    /**
     * @brief Refresh device list
     */
    void refresh();

  private:
    DeviceManager();
    ~DeviceManager() = default;

    void discover_devices();
    void discover_openvino_devices();
    void discover_tensorrt_devices();

    std::vector<DeviceInfo> devices_;
    mutable std::mutex mutex_;
    bool initialized_ = false;
};

// ============================================================================
// Helper functions
// ============================================================================

/**
 * @brief Parse device string to backend and device index
 *
 * Examples:
 *   "cpu" -> (OpenVINO, CPU)
 *   "gpu:0" -> (OpenVINO, GPU.0)
 *   "cuda:0" -> (TensorRT, 0)
 *   "npu" -> (OpenVINO, NPU)
 */
std::pair<BackendType, std::string> parse_device_string(const std::string &device);

/**
 * @brief Get backend for device
 */
BackendType get_backend_for_device(const std::string &device);

/**
 * @brief Check if CUDA is available
 */
bool cuda_is_available();

/**
 * @brief Get CUDA device count
 */
int cuda_device_count();

/**
 * @brief Check if OpenVINO is available
 */
bool openvino_is_available();

} // namespace ivit

#endif // IVIT_CORE_DEVICE_HPP

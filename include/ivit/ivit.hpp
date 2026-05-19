/**
 * @file ivit.hpp
 * @brief iVIT-SDK Main Header
 *
 * iVIT-SDK: Innodisk Vision Intelligence Toolkit
 * Unified Computer Vision SDK for Intel/NVIDIA/Qualcomm platforms
 *
 * @copyright Copyright (c) 2024 Innodisk Corporation
 * @license Apache-2.0
 */

#ifndef IVIT_IVIT_HPP
#define IVIT_IVIT_HPP

// Version
#define IVIT_VERSION_MAJOR 1
#define IVIT_VERSION_MINOR 0
#define IVIT_VERSION_PATCH 0
#define IVIT_VERSION_STRING "1.0.0"

// Core headers
#include "ivit/core/common.hpp"
#include "ivit/core/tensor.hpp"
#include "ivit/core/model.hpp"
#include "ivit/core/result.hpp"
#include "ivit/core/device.hpp"

// Vision headers
#include "ivit/vision/classifier.hpp"
#include "ivit/vision/detector.hpp"
#include "ivit/vision/segmentor.hpp"

// Utils headers
#include "ivit/utils/visualizer.hpp"
#include "ivit/utils/profiler.hpp"

// Runtime headers
#include "ivit/runtime/runtime.hpp"

namespace ivit {

/**
 * @brief Get iVIT-SDK version string
 * @return Version string (e.g., "1.0.0")
 */
inline const char* version() {
    return IVIT_VERSION_STRING;
}

/**
 * @brief Load a model from file or Model Zoo
 *
 * @param path Model file path or Model Zoo name
 * @param config Load configuration
 * @return Loaded model
 *
 * @example
 * @code
 * // Load from file
 * auto model = ivit::load_model("yolox-s.onnx");
 *
 * // Load with specific device
 * auto model = ivit::load_model("yolox-s.onnx", {
 *     .device = "cuda:0",
 *     .backend = "tensorrt"
 * });
 *
 * // Load from Model Zoo
 * auto model = ivit::load_model("yolox-s", {.task = "detection"});
 * @endcode
 */
std::shared_ptr<Model> load_model(
    const std::string& path,
    const LoadConfig& config = LoadConfig{}
);

/**
 * @brief List all available inference devices
 * @return Vector of device information
 */
std::vector<DeviceInfo> list_devices();

/**
 * @brief Get the best device for a specific task
 * @param task Task type (optional)
 * @param priority Selection priority
 * @return Recommended device information
 */
DeviceInfo get_best_device(
    const std::string& task = "",
    const std::string& priority = "performance"
);

/**
 * @brief Log level enumeration
 */
enum class LogLevel {
    DEBUG,      ///< Verbose debug messages
    INFO,       ///< Informational messages
    WARNING,    ///< Warning messages
    ERROR,      ///< Error messages
    OFF         ///< No logging
};

/**
 * @brief Set global log level
 * @param level Log level: "debug", "info", "warning", "error", "off"
 */
void set_log_level(const std::string& level);

/**
 * @brief Get current log level string
 * @return Current log level
 */
std::string get_log_level();

/**
 * @brief Get current parsed log level
 * @return Current LogLevel enum value
 */
LogLevel get_parsed_log_level();

/**
 * @brief Set global cache directory
 * @param path Cache directory path
 */
void set_cache_dir(const std::string& path);

/**
 * @brief Get global cache directory
 * @return Current cache directory path
 */
std::string get_cache_dir();

/**
 * @brief Convert model to optimized format
 *
 * @param src_path Source model path (ONNX)
 * @param dst_path Destination path (.engine for TensorRT, .xml for OpenVINO)
 * @param device Target device (determines output format)
 * @param precision Target precision ("fp32", "fp16", "int8")
 *
 * @example
 * @code
 * // Convert to TensorRT engine
 * ivit::convert_model("yolox-s.onnx", "yolox-s.engine", "cuda:0", "fp16");
 *
 * // Convert to OpenVINO IR
 * ivit::convert_model("yolox-s.onnx", "yolox-s.xml", "cpu", "fp16");
 * @endcode
 */
void convert_model(
    const std::string& src_path,
    const std::string& dst_path,
    const std::string& device = "auto",
    const std::string& precision = "fp16"
);

/**
 * @brief Clear model cache
 * @param cache_dir Cache directory (empty for default)
 */
void clear_cache(const std::string& cache_dir = "");

} // namespace ivit

#endif // IVIT_IVIT_HPP

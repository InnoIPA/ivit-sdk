/**
 * @file common.hpp
 * @brief Common definitions and types
 */

#ifndef IVIT_CORE_COMMON_HPP
#define IVIT_CORE_COMMON_HPP

#include <algorithm>
#include <cctype>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace ivit {

// ============================================================================
// Forward declarations
// ============================================================================
class Model;
class Results;
class Tensor;
struct DeviceInfo;
struct LoadConfig;
struct InferConfig;

// ============================================================================
// Enums
// ============================================================================

/**
 * @brief Data type enumeration
 */
enum class DataType { Float32, Float16, Int8, UInt8, Int32, Int64, Bool, Unknown };

/**
 * @brief Tensor layout enumeration
 */
enum class Layout {
    NCHW, ///< Batch, Channel, Height, Width
    NHWC, ///< Batch, Height, Width, Channel
    NC,   ///< Batch, Channel
    CHW,  ///< Channel, Height, Width
    HWC,  ///< Height, Width, Channel
    Unknown
};

/**
 * @brief Precision enumeration
 */
enum class Precision { FP32, FP16, INT8, INT4, Unknown };

/**
 * @brief Task type enumeration
 */
enum class TaskType {
    Classification,
    Detection,
    Segmentation,
    Unknown
};

/**
 * @brief Backend type enumeration
 *
 * This enum is extensible - new backends can be added as the SDK
 * supports additional hardware platforms.
 */
enum class BackendType {
    OpenVINO, ///< Intel: CPU, iGPU, NPU
    TensorRT, ///< NVIDIA: CUDA GPUs, Jetson
    Auto,     ///< Automatic selection
    Unknown
};

// ============================================================================
// Configurations
// ============================================================================

/**
 * @brief Model loading configuration
 */
struct LoadConfig {
    std::string device = "auto";   ///< Target device: "auto", "cpu", "gpu:0", "cuda:0", "npu"
    std::string backend = "auto";  ///< Backend: "auto", "openvino", "tensorrt"
    std::string task = "";         ///< Task hint: "classification", "detection", etc.
    int batch_size = 1;            ///< Preferred dynamic TensorRT batch (profile OPT)
    int max_batch_size = 8;        ///< Dynamic TensorRT batch profile MAX; MIN is 1
    int num_threads = 0;           ///< OpenVINO CPU threads; 0 = automatic
    int num_instances = 1;         ///< TensorRT: execution contexts for concurrent callers (1-16)
    bool enable_profiling = false; ///< OpenVINO per-layer profiling
    std::string performance_mode = "LATENCY"; ///< OpenVINO LATENCY or THROUGHPUT
    size_t workspace_size = 1ULL << 30;       ///< TensorRT workspace limit, bytes
    std::string precision = "fp32";           ///< Precision: "fp32", "fp16", "int8"
    std::string cache_dir = "";               ///< Cache directory
    bool use_cache = true;                    ///< Enable model caching
    std::string decoder = "auto";      ///< auto, logits, probabilities, yolov5, yolov8, legacy, raw
    std::string input_color = "rgb";   ///< Color order expected by the model: rgb or bgr
    std::string input_layout = "nchw"; ///< Image tensor layout: nchw or nhwc
    std::string resize_mode = "auto";  ///< auto: letterbox for detection, stretch otherwise
    float input_scale = 1.0f / 255.0f;
    std::vector<float> mean = {0.0f, 0.0f, 0.0f};
    std::vector<float> std = {1.0f, 1.0f, 1.0f}; ///< (pixel * input_scale - mean) / std
    std::vector<std::string> labels;
    std::string labels_path = ""; ///< Explicit UTF-8 text file, one label per line
    std::string output_name = ""; ///< Required when decoding ambiguous multiple outputs
};

/**
 * @brief Inference configuration
 */
struct InferConfig {
    /// Score threshold in [0,1], or -1 (the default) for the task default: 0.5 for detection,
    /// 0 for classification so that every class is returned (filter or take top-k afterwards).
    /// Other negative values are rejected.
    float conf_threshold = -1.0f;
    float iou_threshold = 0.45f;   ///< NMS IoU threshold
    int max_detections = 100;      ///< Maximum detections
    std::vector<int> classes;      ///< Filter by class IDs (empty = all)
    bool agnostic_nms = false;     ///< Suppress overlapping boxes across classes (default: per class)
};

// ============================================================================
// Exceptions
// ============================================================================

/**
 * @brief Base exception for iVIT-SDK
 */
class IVITError : public std::runtime_error {
  public:
    explicit IVITError(const std::string &message) : std::runtime_error(message) {}
};

/**
 * @brief Model-related errors
 */
class ModelError : public IVITError {
  public:
    explicit ModelError(const std::string &message) : IVITError(message) {}
};

/**
 * @brief Model load error
 */
class ModelLoadError : public ModelError {
  public:
    explicit ModelLoadError(const std::string &message)
        : ModelError("Failed to load model: " + message) {}
};

/**
 * @brief Unsupported format error
 */
class UnsupportedFormatError : public ModelError {
  public:
    explicit UnsupportedFormatError(const std::string &format)
        : ModelError("Unsupported model format: " + format) {}
};

/**
 * @brief Device-related errors
 */
class DeviceError : public IVITError {
  public:
    explicit DeviceError(const std::string &message) : IVITError(message) {}
};

/**
 * @brief Device not found error
 */
class DeviceNotFoundError : public DeviceError {
  public:
    explicit DeviceNotFoundError(const std::string &device)
        : DeviceError("Device not found: " + device) {}
};

/**
 * @brief Inference-related errors
 */
class InferenceError : public IVITError {
  public:
    explicit InferenceError(const std::string &message) : IVITError(message) {}
};

// ============================================================================
// Type aliases
// ============================================================================

using Shape = std::vector<int64_t>;
using Callback = std::function<void(const Results &)>;

// ============================================================================
// Utility functions
// ============================================================================

/**
 * @brief Convert DataType to string
 */
inline std::string to_string(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return "float32";
    case DataType::Float16:
        return "float16";
    case DataType::Int8:
        return "int8";
    case DataType::UInt8:
        return "uint8";
    case DataType::Int32:
        return "int32";
    case DataType::Int64:
        return "int64";
    case DataType::Bool:
        return "bool";
    default:
        return "unknown";
    }
}

/**
 * @brief Convert TaskType to string
 */
inline std::string to_string(TaskType task) {
    switch (task) {
    case TaskType::Classification:
        return "classification";
    case TaskType::Detection:
        return "detection";
    case TaskType::Segmentation:
        return "segmentation";
    default:
        return "unknown";
    }
}

/**
 * @brief Convert BackendType to string
 */
inline std::string to_string(BackendType backend) {
    switch (backend) {
    case BackendType::OpenVINO:
        return "openvino";
    case BackendType::TensorRT:
        return "tensorrt";
    case BackendType::Auto:
        return "auto";
    default:
        return "unknown";
    }
}

/**
 * @brief Convert a backend name to BackendType (case-insensitive)
 *
 * An empty string means "auto". Throws IVITError for unknown names.
 */
inline BackendType backend_from_string(const std::string &name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (lower.empty() || lower == "auto")
        return BackendType::Auto;
    if (lower == "openvino")
        return BackendType::OpenVINO;
    if (lower == "tensorrt")
        return BackendType::TensorRT;
    throw IVITError("Unknown backend '" + name + "' (expected auto, openvino or tensorrt)");
}

} // namespace ivit

#endif // IVIT_CORE_COMMON_HPP

/**
 * @file common.hpp
 * @brief Common definitions and types
 */

#ifndef IVIT_CORE_COMMON_HPP
#define IVIT_CORE_COMMON_HPP

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <functional>
#include <stdexcept>

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
enum class DataType {
    Float32,
    Float16,
    Int8,
    UInt8,
    Int32,
    Int64,
    Bool,
    Unknown
};

/**
 * @brief Tensor layout enumeration
 */
enum class Layout {
    NCHW,   ///< Batch, Channel, Height, Width
    NHWC,   ///< Batch, Height, Width, Channel
    NC,     ///< Batch, Channel
    CHW,    ///< Channel, Height, Width
    HWC,    ///< Height, Width, Channel
    Unknown
};

/**
 * @brief Precision enumeration
 */
enum class Precision {
    FP32,
    FP16,
    INT8,
    INT4,
    Unknown
};

/**
 * @brief Task type enumeration
 */
enum class TaskType {
    Classification,
    Detection,
    Segmentation,
    InstanceSegmentation,
    PoseEstimation,
    FaceDetection,
    FaceRecognition,
    OCR,
    AnomalyDetection,
    Unknown
};

/**
 * @brief Backend type enumeration
 *
 * This enum is extensible - new backends can be added as the SDK
 * supports additional hardware platforms.
 */
enum class BackendType {
    OpenVINO,    ///< Intel: CPU, iGPU, NPU, VPU
    TensorRT,    ///< NVIDIA: CUDA GPUs, Jetson
    QNN,         ///< Qualcomm: IQ Series NPU (Hexagon) via AI Engine Direct
    HailoRT,     ///< Hailo: Hailo-8 PCIe AI accelerator
    Auto,        ///< Automatic selection
    Unknown
};

// ============================================================================
// Configurations
// ============================================================================

/**
 * @brief Model loading configuration
 */
struct LoadConfig {
    std::string device = "auto";       ///< Target device: "auto", "cpu", "gpu:0", "cuda:0", "npu"
    std::string backend = "auto";      ///< Backend: "auto", "openvino", "tensorrt"
    std::string task = "";             ///< Task hint: "classification", "detection", etc.
    int batch_size = 1;                ///< Batch size
    std::string precision = "fp32";    ///< Precision: "fp32", "fp16", "int8"
    std::string cache_dir = "";        ///< Cache directory
    bool use_cache = true;             ///< Enable model caching
};

/**
 * @brief Inference configuration
 */
struct InferConfig {
    float conf_threshold = 0.5f;       ///< Confidence threshold
    float iou_threshold = 0.45f;       ///< NMS IoU threshold
    int max_detections = 100;          ///< Maximum detections
    std::vector<int> classes;          ///< Filter by class IDs (empty = all)
    bool enable_profiling = false;     ///< Enable profiling
};

// ============================================================================
// Exceptions
// ============================================================================

/**
 * @brief Base exception for iVIT-SDK
 */
class IVITError : public std::runtime_error {
public:
    explicit IVITError(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief Model-related errors
 */
class ModelError : public IVITError {
public:
    explicit ModelError(const std::string& message)
        : IVITError(message) {}
};

/**
 * @brief Model load error
 */
class ModelLoadError : public ModelError {
public:
    explicit ModelLoadError(const std::string& message)
        : ModelError("Failed to load model: " + message) {}
};

/**
 * @brief Unsupported format error
 */
class UnsupportedFormatError : public ModelError {
public:
    explicit UnsupportedFormatError(const std::string& format)
        : ModelError("Unsupported model format: " + format) {}
};

/**
 * @brief Device-related errors
 */
class DeviceError : public IVITError {
public:
    explicit DeviceError(const std::string& message)
        : IVITError(message) {}
};

/**
 * @brief Device not found error
 */
class DeviceNotFoundError : public DeviceError {
public:
    explicit DeviceNotFoundError(const std::string& device)
        : DeviceError("Device not found: " + device) {}
};

/**
 * @brief Inference-related errors
 */
class InferenceError : public IVITError {
public:
    explicit InferenceError(const std::string& message)
        : IVITError(message) {}
};

// ============================================================================
// Type aliases
// ============================================================================

using Shape = std::vector<int64_t>;
using Callback = std::function<void(const Results&)>;

// ============================================================================
// Utility functions
// ============================================================================

/**
 * @brief Convert DataType to string
 */
inline std::string to_string(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float16: return "float16";
        case DataType::Int8: return "int8";
        case DataType::UInt8: return "uint8";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::Bool: return "bool";
        default: return "unknown";
    }
}

/**
 * @brief Convert TaskType to string
 */
inline std::string to_string(TaskType task) {
    switch (task) {
        case TaskType::Classification: return "classification";
        case TaskType::Detection: return "detection";
        case TaskType::Segmentation: return "segmentation";
        case TaskType::InstanceSegmentation: return "instance_segmentation";
        case TaskType::PoseEstimation: return "pose_estimation";
        case TaskType::FaceDetection: return "face_detection";
        case TaskType::FaceRecognition: return "face_recognition";
        case TaskType::OCR: return "ocr";
        case TaskType::AnomalyDetection: return "anomaly_detection";
        default: return "unknown";
    }
}

/**
 * @brief Convert BackendType to string
 */
inline std::string to_string(BackendType backend) {
    switch (backend) {
        case BackendType::OpenVINO: return "openvino";
        case BackendType::TensorRT: return "tensorrt";
        case BackendType::QNN: return "qnn";
        case BackendType::HailoRT: return "hailort";
        case BackendType::Auto: return "auto";
        default: return "unknown";
    }
}

} // namespace ivit

#endif // IVIT_CORE_COMMON_HPP

/**
 * @file runtime_config.hpp
 * @brief Hardware-specific runtime configurations
 */

#ifndef IVIT_CORE_RUNTIME_CONFIG_HPP
#define IVIT_CORE_RUNTIME_CONFIG_HPP

#include <string>
#include <map>

namespace ivit {

/**
 * @brief OpenVINO runtime configuration
 */
struct OpenVINOConfig {
    std::string performance_mode = "LATENCY";  ///< "LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"
    int num_streams = 1;                        ///< Number of inference streams
    std::string inference_precision = "FP32";   ///< "FP32", "FP16", "INT8"
    bool enable_cpu_pinning = false;            ///< Pin threads to CPU cores
    int num_threads = 0;                        ///< CPU threads (0 = auto)
    std::string npu_compilation_mode = "DefaultHW"; ///< NPU: "DefaultHW", "DefaultSW"
    std::string cache_dir;                      ///< Model compilation cache directory
    std::map<std::string, std::string> device_properties; ///< Additional properties

    std::map<std::string, std::string> to_ov_config() const;
};

/**
 * @brief TensorRT runtime configuration
 */
struct TensorRTConfig {
    bool enable_fp16 = false;        ///< Enable FP16 mode
    bool enable_int8 = false;        ///< Enable INT8 mode
    size_t workspace_size = 1ULL << 30; ///< Max workspace size in bytes (default 1GB)
    bool enable_dla = false;         ///< Enable DLA (Deep Learning Accelerator)
    int dla_core = 0;                ///< DLA core index
    int max_batch_size = 1;          ///< Maximum batch size
    bool enable_sparsity = false;    ///< Enable structured sparsity
    bool strict_types = false;       ///< Strict type constraints
    std::string calibration_cache;   ///< INT8 calibration cache file path
    std::string timing_cache;        ///< Timing cache file path
    std::map<std::string, std::string> extra_options; ///< Additional options
};

/**
 * @brief Qualcomm QNN runtime configuration
 */
struct QNNConfig {
    std::string backend = "HTP";     ///< "HTP" (Hexagon), "GPU", "CPU"
    std::string performance_mode = "high_performance"; ///< Performance profile
    std::string precision = "fp16";  ///< "fp16", "int8"
    bool enable_htp_fp16 = true;     ///< Enable FP16 on HTP
    int htp_socs = 0;               ///< Target SoC ID (0 = auto)
    std::string skel_library_dir;    ///< Skeleton library directory
    std::map<std::string, std::string> extra_options; ///< Additional options
};

/**
 * @brief HailoRT runtime configuration for Hailo-8 AI accelerator
 */
struct HailoConfig {
    int batch_size = 1;      ///< Batch size (default 1)
    int timeout_ms = 1000;   ///< Inference timeout in milliseconds (default 1000)
};

} // namespace ivit

#endif // IVIT_CORE_RUNTIME_CONFIG_HPP

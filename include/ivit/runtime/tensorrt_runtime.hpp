/**
 * @file tensorrt_runtime.hpp
 * @brief TensorRT backend runtime
 */

#ifndef IVIT_RUNTIME_TENSORRT_RUNTIME_HPP
#define IVIT_RUNTIME_TENSORRT_RUNTIME_HPP

#include "ivit/runtime/runtime.hpp"

#ifdef IVIT_HAS_TENSORRT

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>
#include <condition_variable>
#include <mutex>

namespace ivit {

/**
 * @brief TensorRT logger implementation
 */
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    bool verbose_ = false;
};

/**
 * @brief TensorRT engine wrapper
 */
/**
 * @brief One independent way to run the engine: its own execution context, CUDA stream and
 * device buffers. A call owns a slot for its whole duration, so slots never share state.
 */
struct TensorRTSlot {
    std::shared_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    std::vector<void*> device_buffers;
    std::vector<size_t> buffer_bytes;  ///< Allocated size of each device buffer
};

struct TensorRTEngine {
    // Keep a reference to IRuntime so it outlives engine destruction
    // (prevents static destruction order crash)
    std::shared_ptr<nvinfer1::IRuntime> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<TensorInfo> input_infos;
    std::vector<TensorInfo> output_infos;
    int device_id = 0;

    /// num_instances slots; callers block (FIFO is not guaranteed) while all are in use.
    std::vector<std::unique_ptr<TensorRTSlot>> slots;
    std::mutex slot_mutex;
    std::condition_variable slot_available;
    std::vector<TensorRTSlot*> free_slots;

    TensorRTSlot* acquire_slot();
    void release_slot(TensorRTSlot* slot);

    ~TensorRTEngine();
};

/**
 * @brief TensorRT runtime implementation
 */
class TensorRTRuntime : public IRuntime {
public:
    TensorRTRuntime();
    ~TensorRTRuntime() override;

    // IRuntime interface
    BackendType type() const override { return BackendType::TensorRT; }
    std::string name() const override { return "TensorRT"; }
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

    // TensorRT-specific methods

    /**
     * @brief Build TensorRT engine from ONNX model
     */
    std::shared_ptr<nvinfer1::ICudaEngine> build_engine_from_onnx(
        const std::string& onnx_path,
        const RuntimeConfig& config
    );

    /**
     * @brief Load serialized TensorRT engine
     */
    std::shared_ptr<nvinfer1::ICudaEngine> load_engine(
        const std::string& engine_path
    );

    /**
     * @brief Save engine to file
     */
    void save_engine(
        nvinfer1::ICudaEngine* engine,
        const std::string& path
    );

    /**
     * @brief Set device ID
     */
    void set_device(int device_id);

    /**
     * @brief Get TensorRT version
     */
    std::string get_version() const;

private:
    void initialize();
    TensorRTEngine* create_engine_wrapper(
        std::shared_ptr<nvinfer1::ICudaEngine> engine,
        int device_id,
        int num_instances
    );

    /**
     * @brief Get cache path for a model
     */
    std::string get_cache_path(
        const std::string& model_path,
        const RuntimeConfig& config
    ) const;

    DataType nv_to_dtype(nvinfer1::DataType nv_type) const;
    nvinfer1::DataType dtype_to_nv(DataType dtype) const;

    TensorRTLogger logger_;
    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::mutex mutex_;
    bool initialized_ = false;
    int default_device_ = 0;
};

/**
 * @brief Get global TensorRT runtime instance
 */
TensorRTRuntime& get_tensorrt_runtime();

} // namespace ivit

#endif // IVIT_HAS_TENSORRT

#endif // IVIT_RUNTIME_TENSORRT_RUNTIME_HPP

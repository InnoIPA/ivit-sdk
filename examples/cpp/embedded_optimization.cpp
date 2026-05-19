/**
 * @file embedded_optimization.cpp
 * @brief iVIT-SDK Embedded Engineer Optimization Example
 *
 * Target: Embedded engineers who need to optimize inference performance
 *         on edge devices.
 *
 * Features demonstrated:
 * - Model loading with configuration
 * - Performance benchmarking
 * - Model warmup
 * - Timing analysis
 *
 * Note: Runtime-specific configuration (OpenVINO, TensorRT settings)
 *       is available through the Python API. C++ focuses on
 *       inference execution and benchmarking.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make embedded_optimization
 *
 * Usage:
 *   ./embedded_optimization <model_path>
 *   ./embedded_optimization <model_path> --benchmark
 *   ./embedded_optimization <model_path> --device cuda:0 --iterations 100
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "ivit/ivit.hpp"

using namespace ivit;

/**
 * Step 1: Model warmup (critical for accurate benchmarking)
 */
void warmup_model(Model& model, int iterations = 10) {
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Step 1: Model Warmup" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "Running " << iterations << " warmup iterations..." << std::endl;

    // Create dummy image for warmup
    cv::Mat dummy(480, 640, CV_8UC3);
    cv::randu(dummy, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    for (int i = 0; i < iterations; ++i) {
        model.predict(dummy);
    }

    std::cout << "Warmup completed!" << std::endl;
    std::cout << "\nNote: First few inferences are typically slower due to:" << std::endl;
    std::cout << "  - JIT compilation" << std::endl;
    std::cout << "  - Memory allocation" << std::endl;
    std::cout << "  - CUDA kernel loading" << std::endl;
}

/**
 * Step 2: Benchmark preprocessor performance
 */
void benchmark_preprocessor() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Step 2: Preprocessor Benchmarking" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Create test image
    cv::Mat test_image(480, 640, CV_8UC3);
    cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    const int iterations = 100;

    // Benchmark resize + normalize
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // Typical preprocessing steps
        cv::Mat resized;
        cv::resize(test_image, resized, cv::Size(640, 640));

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat normalized;
        rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(elapsed);
    }

    // Calculate statistics
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());

    double sq_sum = 0;
    for (double t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_sum / times.size());

    std::cout << "Preprocessing benchmark (resize + RGB + normalize):" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Mean: " << mean << " ms" << std::endl;
    std::cout << "  Min: " << min_time << " ms" << std::endl;
    std::cout << "  Max: " << max_time << " ms" << std::endl;
    std::cout << "  Std: " << std_dev << " ms" << std::endl;
}

/**
 * Step 3: Benchmark inference performance
 */
struct BenchmarkStats {
    double mean_ms;
    double min_ms;
    double max_ms;
    double std_ms;
    double p95_ms;
    double p99_ms;
    double fps;
};

BenchmarkStats benchmark_inference(Model& model, int iterations = 100) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Step 3: Inference Benchmarking" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Create test image
    cv::Mat test_image(480, 640, CV_8UC3);
    cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    // Warmup
    std::cout << "Warmup: 10 iterations" << std::endl;
    for (int i = 0; i < 10; ++i) {
        model.predict(test_image);
    }

    // Benchmark
    std::cout << "Benchmarking: " << iterations << " iterations" << std::endl;
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        model.predict(test_image);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(elapsed);

        if ((i + 1) % 20 == 0) {
            std::cout << "  Progress: " << (i + 1) << "/" << iterations << std::endl;
        }
    }

    // Calculate statistics
    std::sort(times.begin(), times.end());

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_time = times.front();
    double max_time = times.back();

    double sq_sum = 0;
    for (double t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_sum / times.size());

    double p95 = times[static_cast<size_t>(iterations * 0.95)];
    double p99 = times[static_cast<size_t>(iterations * 0.99)];
    double fps = 1000.0 / mean;

    BenchmarkStats stats = {mean, min_time, max_time, std_dev, p95, p99, fps};

    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Mean latency: " << stats.mean_ms << " ms" << std::endl;
    std::cout << "  Min latency: " << stats.min_ms << " ms" << std::endl;
    std::cout << "  Max latency: " << stats.max_ms << " ms" << std::endl;
    std::cout << "  Std deviation: " << stats.std_ms << " ms" << std::endl;
    std::cout << "  P95 latency: " << stats.p95_ms << " ms" << std::endl;
    std::cout << "  P99 latency: " << stats.p99_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << stats.fps << " FPS" << std::endl;

    return stats;
}

void print_runtime_configuration_info() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Runtime Configuration (Python API)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "Advanced runtime configuration is available via Python:" << std::endl;
    std::cout << std::endl;
    std::cout << "# OpenVINO Configuration" << std::endl;
    std::cout << "model.configure_openvino(" << std::endl;
    std::cout << "    performance_mode='LATENCY'," << std::endl;
    std::cout << "    num_streams=1," << std::endl;
    std::cout << "    inference_precision='FP16'" << std::endl;
    std::cout << ")" << std::endl;
    std::cout << std::endl;
    std::cout << "# TensorRT Configuration" << std::endl;
    std::cout << "model.configure_tensorrt(" << std::endl;
    std::cout << "    workspace_size=1<<30," << std::endl;
    std::cout << "    fp16=True," << std::endl;
    std::cout << "    builder_optimization_level=3" << std::endl;
    std::cout << ")" << std::endl;
}

void print_best_practices() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Embedded Best Practices:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "1. Always run warmup before benchmarking" << std::endl;
    std::cout << "2. Use FP16 quantization for most cases (minimal accuracy loss)" << std::endl;
    std::cout << "3. Configure backend based on hardware:" << std::endl;
    std::cout << "   - Intel: OpenVINO with LATENCY mode" << std::endl;
    std::cout << "   - NVIDIA: TensorRT with FP16 and CUDA Graph" << std::endl;
    std::cout << "   - Qualcomm: SNPE with DSP runtime" << std::endl;
    std::cout << "4. Monitor preprocessing time (can be 30%+ of total)" << std::endl;
    std::cout << "5. Use Python API for advanced runtime configuration" << std::endl;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model_path> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --device <device>     Target device (auto, cpu, cuda:0, npu)" << std::endl;
    std::cout << "  --benchmark           Run inference benchmark" << std::endl;
    std::cout << "  --iterations <n>      Benchmark iterations (default: 100)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program << " yolox-s.onnx --device cuda:0 --benchmark" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string device = "auto";
    bool run_benchmark = false;
    int iterations = 100;

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        }
    }

    std::cout << std::string(60, '=') << std::endl;
    std::cout << "iVIT-SDK Embedded Engineer Optimization Example" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "SDK Version: " << ivit::version() << std::endl;

    // Select device
    DeviceInfo target_device;
    if (device == "auto") {
        target_device = get_best_device();
        std::cout << "Auto-selected device: " << target_device.id
                  << " (" << target_device.name << ")" << std::endl;
    } else {
        target_device = DeviceManager::instance().get_device(device);
        std::cout << "Using device: " << target_device.id << std::endl;
    }

    // Load model
    std::cout << "\nLoading model: " << model_path << std::endl;

    LoadConfig config;
    config.device = target_device.id;

    auto model = load_model(model_path, config);
    std::cout << "Model loaded successfully!" << std::endl;

    // Step 1: Warmup
    warmup_model(*model, 10);

    // Step 2: Benchmark preprocessor
    benchmark_preprocessor();

    // Step 3: Benchmark inference
    if (run_benchmark) {
        benchmark_inference(*model, iterations);
    }

    // Runtime configuration info
    print_runtime_configuration_info();

    // Best practices
    print_best_practices();

    return 0;
}

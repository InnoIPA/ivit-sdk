/**
 * @file backend_service.cpp
 * @brief iVIT-SDK Backend Engineer Service Example
 *
 * Target: Backend engineers who need to build stable AI inference services
 *         with monitoring and performance tracking.
 *
 * Features demonstrated:
 * - Service wrapper with monitoring
 * - FPS and latency tracking
 * - Thread-safe inference service pattern
 * - Statistics collection
 *
 * Note: Advanced callback system is available via Python API.
 *       C++ demonstrates manual monitoring patterns.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make backend_service
 *
 * Usage:
 *   ./backend_service <model_path>
 *   ./backend_service <model_path> --demo
 *   ./backend_service <model_path> --device cuda:0
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <numeric>
#include <deque>

#include <opencv2/opencv.hpp>

#include "ivit/ivit.hpp"

using namespace ivit;

/**
 * FPS Counter - Sliding window FPS calculation
 */
class FPSCounter {
public:
    explicit FPSCounter(size_t window_size = 30)
        : window_size_(window_size) {}

    void record(double latency_ms) {
        std::lock_guard<std::mutex> lock(mutex_);

        latencies_.push_back(latency_ms);

        while (latencies_.size() > window_size_) {
            latencies_.pop_front();
        }
    }

    double fps() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latencies_.empty()) return 0.0;

        double total_ms = std::accumulate(latencies_.begin(), latencies_.end(), 0.0);
        double avg_ms = total_ms / latencies_.size();
        return avg_ms > 0 ? 1000.0 / avg_ms : 0.0;
    }

    double avg_latency_ms() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latencies_.empty()) return 0.0;
        return std::accumulate(latencies_.begin(), latencies_.end(), 0.0) / latencies_.size();
    }

private:
    size_t window_size_;
    std::deque<double> latencies_;
    mutable std::mutex mutex_;
};

/**
 * Latency Logger - Tracks inference latency statistics
 */
class LatencyLogger {
public:
    void record(double latency_ms) {
        std::lock_guard<std::mutex> lock(mutex_);

        latencies_.push_back(latency_ms);
        total_latency_ += latency_ms;
        inference_count_++;

        // Keep last 1000 samples for rolling statistics
        if (latencies_.size() > 1000) {
            total_latency_ -= latencies_.front();
            latencies_.pop_front();
        }
    }

    double average_latency() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latencies_.empty()) return 0.0;
        return total_latency_ / latencies_.size();
    }

    size_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return inference_count_;
    }

    double min_latency() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latencies_.empty()) return 0.0;
        return *std::min_element(latencies_.begin(), latencies_.end());
    }

    double max_latency() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latencies_.empty()) return 0.0;
        return *std::max_element(latencies_.begin(), latencies_.end());
    }

private:
    std::deque<double> latencies_;
    double total_latency_ = 0.0;
    size_t inference_count_ = 0;
    mutable std::mutex mutex_;
};

/**
 * Alert Monitor - Monitors for high latency events
 */
class AlertMonitor {
public:
    explicit AlertMonitor(double threshold_ms = 100.0)
        : threshold_ms_(threshold_ms), alert_count_(0) {}

    void check(double latency_ms) {
        if (latency_ms > threshold_ms_) {
            alert_count_++;
            std::cout << "[ALERT] High latency warning: " << std::fixed
                      << std::setprecision(1) << latency_ms << "ms "
                      << "(threshold: " << threshold_ms_ << "ms)" << std::endl;
        }
    }

    size_t alert_count() const { return alert_count_.load(); }
    double threshold() const { return threshold_ms_; }

private:
    double threshold_ms_;
    std::atomic<size_t> alert_count_;
};

/**
 * Inference Service - Wraps model with monitoring
 */
class InferenceService {
public:
    InferenceService(const std::string& model_path, const DeviceInfo& device)
        : fps_counter_(30),
          alert_monitor_(100.0) {

        // Load model using load_model API
        LoadConfig config;
        config.device = device.id;
        model_ = load_model(model_path, config);

        device_id_ = device.id;
        device_name_ = device.name;

        // Warmup
        warmup(10);
    }

    Results infer(const cv::Mat& image) {
        auto start = std::chrono::high_resolution_clock::now();

        auto results = model_->predict(image);

        auto end = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end - start).count();

        // Record metrics
        fps_counter_.record(latency);
        latency_logger_.record(latency);
        alert_monitor_.check(latency);

        return results;
    }

    void warmup(int iterations = 10) {
        std::cout << "Running " << iterations << " warmup iterations..." << std::endl;

        cv::Mat dummy(480, 640, CV_8UC3);
        cv::randu(dummy, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        for (int i = 0; i < iterations; ++i) {
            model_->predict(dummy);
        }

        std::cout << "Warmup completed!" << std::endl;
    }

    // Statistics getters
    double fps() const { return fps_counter_.fps(); }
    double average_latency() const { return latency_logger_.average_latency(); }
    double min_latency() const { return latency_logger_.min_latency(); }
    double max_latency() const { return latency_logger_.max_latency(); }
    size_t inference_count() const { return latency_logger_.count(); }
    size_t alert_count() const { return alert_monitor_.alert_count(); }

    const std::string& device_id() const { return device_id_; }
    const std::string& device_name() const { return device_name_; }

private:
    std::shared_ptr<Model> model_;
    std::string device_id_;
    std::string device_name_;

    FPSCounter fps_counter_;
    LatencyLogger latency_logger_;
    AlertMonitor alert_monitor_;
};

void demonstrate_monitoring_patterns() {
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Step 1: Monitoring Patterns" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "Monitoring components demonstrated:" << std::endl;
    std::cout << "  - FPSCounter: Sliding window FPS calculation" << std::endl;
    std::cout << "  - LatencyLogger: Latency statistics (min/max/avg)" << std::endl;
    std::cout << "  - AlertMonitor: High latency alerting" << std::endl;

    std::cout << "\nFor advanced callback system, use Python API:" << std::endl;
    std::cout << "  from ivit.core.callbacks import CallbackManager, FPSCounter" << std::endl;
    std::cout << "  manager = CallbackManager()" << std::endl;
    std::cout << "  manager.register('infer_end', FPSCounter())" << std::endl;
}

void demonstrate_cli_tools() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Step 3: CLI Tools" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "Available CLI commands for backend engineers:" << std::endl;
    std::cout << std::endl;
    std::cout << "# System information" << std::endl;
    std::cout << "  ivit info" << std::endl;
    std::cout << std::endl;
    std::cout << "# List available devices" << std::endl;
    std::cout << "  ivit devices" << std::endl;
    std::cout << std::endl;
    std::cout << "# Performance benchmark" << std::endl;
    std::cout << "  ivit benchmark model.onnx --device cuda:0 --iterations 100" << std::endl;
    std::cout << std::endl;
    std::cout << "# Run inference" << std::endl;
    std::cout << "  ivit predict model.onnx image.jpg --output result.jpg" << std::endl;
    std::cout << std::endl;
    std::cout << "# Model conversion" << std::endl;
    std::cout << "  ivit convert model.onnx model.engine --format tensorrt --fp16" << std::endl;
    std::cout << std::endl;
    std::cout << "# Start REST API service" << std::endl;
    std::cout << "  ivit serve model.onnx --port 8080 --device cuda:0" << std::endl;
}

void run_demo(InferenceService& service) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Step 2: Demo Inferences" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Create test image
    cv::Mat test_image(480, 640, CV_8UC3);
    cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    std::cout << "Running 20 demo inferences..." << std::endl;

    for (int i = 0; i < 20; ++i) {
        auto results = service.infer(test_image);

        if ((i + 1) % 5 == 0) {
            std::cout << "  Completed " << (i + 1) << "/20, "
                      << "Current FPS: " << std::fixed << std::setprecision(1)
                      << service.fps() << std::endl;
        }
    }

    // Show statistics
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Statistics:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total inferences: " << service.inference_count() << std::endl;
    std::cout << "Current FPS: " << service.fps() << std::endl;
    std::cout << "Average latency: " << service.average_latency() << " ms" << std::endl;
    std::cout << "Min latency: " << service.min_latency() << " ms" << std::endl;
    std::cout << "Max latency: " << service.max_latency() << " ms" << std::endl;
    std::cout << "High latency alerts: " << service.alert_count() << std::endl;
}

void print_best_practices() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Backend Best Practices:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "1. Wrap model in service class for monitoring" << std::endl;
    std::cout << "2. Use sliding window for stable FPS calculation" << std::endl;
    std::cout << "3. Track latency statistics (min/max/avg)" << std::endl;
    std::cout << "4. Implement alerting for SLA compliance" << std::endl;
    std::cout << "5. Use warmup before serving production traffic" << std::endl;
    std::cout << "6. For advanced callbacks, use Python API" << std::endl;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model_path> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --device <device>  Target device (auto, cpu, cuda:0)" << std::endl;
    std::cout << "  --demo             Run demo inferences" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program << " yolox-s.onnx --device cuda:0 --demo" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string device = "auto";
    bool run_demo_mode = false;

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg == "--demo") {
            run_demo_mode = true;
        }
    }

    std::cout << std::string(60, '=') << std::endl;
    std::cout << "iVIT-SDK Backend Engineer Service Example" << std::endl;
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

    // Step 1: Demonstrate monitoring patterns
    demonstrate_monitoring_patterns();

    // Create service
    std::cout << "\nCreating inference service..." << std::endl;
    InferenceService service(model_path, target_device);
    std::cout << "Service created on device: " << service.device_id()
              << " (" << service.device_name() << ")" << std::endl;

    // Step 2: Run demo
    if (run_demo_mode) {
        run_demo(service);
    }

    // Step 3: CLI tools
    demonstrate_cli_tools();

    // Best practices
    print_best_practices();

    return 0;
}

/**
 * @file simple_inference.cpp
 * @brief Simple inference example demonstrating iVIT-SDK usage
 *
 * This example shows how to use iVIT-SDK for:
 * - Image classification
 * - Object detection
 * - Semantic segmentation
 */

#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "ivit/ivit.hpp"
#include "ivit/vision/classifier.hpp"
#include "ivit/vision/detector.hpp"
#include "ivit/vision/segmentor.hpp"
#include "ivit/core/device.hpp"

using namespace ivit;
using namespace ivit::vision;

// ============================================================================
// Helper functions
// ============================================================================

void print_separator(const std::string& title) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " " << title << "\n";
    std::cout << "========================================\n";
}

void print_device_info() {
    print_separator("Available Devices");

    auto& manager = DeviceManager::instance();
    auto devices = manager.list_devices();

    std::cout << "Found " << devices.size() << " device(s):\n\n";

    for (const auto& dev : devices) {
        std::cout << "  ID:      " << dev.id << "\n";
        std::cout << "  Name:    " << dev.name << "\n";
        std::cout << "  Backend: " << dev.backend << "\n";
        std::cout << "  Type:    " << dev.type << "\n";

        if (dev.memory_total > 0) {
            std::cout << "  Memory:  " << (dev.memory_total / 1024 / 1024) << " MB\n";
        }

        std::cout << "  Status:  " << (dev.is_available ? "Available" : "Unavailable") << "\n";
        std::cout << "\n";
    }
}

// ============================================================================
// Classification Example
// ============================================================================

void run_classification(const std::string& model_path,
                        const std::string& image_path,
                        const std::string& device) {
    print_separator("Image Classification");

    try {
        std::cout << "Model:  " << model_path << "\n";
        std::cout << "Image:  " << image_path << "\n";
        std::cout << "Device: " << device << "\n\n";

        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Cannot load image: " << image_path << "\n";
            return;
        }

        // Create classifier
        std::cout << "Loading model...\n";
        Classifier classifier(model_path, device);

        std::cout << "Input size: " << classifier.input_size().width
                  << "x" << classifier.input_size().height << "\n";
        std::cout << "Classes: " << classifier.num_classes() << "\n\n";

        // Run inference
        std::cout << "Running inference...\n";
        auto results = classifier.predict(image, 5);  // Top-5

        // Print results
        std::cout << "\nTop-5 Predictions:\n";
        std::cout << "-------------------\n";

        for (size_t i = 0; i < results.classifications.size(); i++) {
            const auto& cls = results.classifications[i];
            std::cout << "  " << (i + 1) << ". " << cls.label
                      << " (class " << cls.class_id << ")"
                      << " - " << std::fixed << std::setprecision(2)
                      << (cls.score * 100) << "%\n";
        }

        std::cout << "\nInference time: " << results.inference_time_ms << " ms\n";
        std::cout << "Device used: " << results.device_used << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Classification error: " << e.what() << "\n";
    }
}

// ============================================================================
// Detection Example
// ============================================================================

void run_detection(const std::string& model_path,
                   const std::string& image_path,
                   const std::string& device,
                   const std::string& output_path = "") {
    print_separator("Object Detection");

    try {
        std::cout << "Model:  " << model_path << "\n";
        std::cout << "Image:  " << image_path << "\n";
        std::cout << "Device: " << device << "\n\n";

        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Cannot load image: " << image_path << "\n";
            return;
        }

        // Create detector
        std::cout << "Loading model...\n";
        Detector detector(model_path, device);

        std::cout << "Input size: " << detector.input_size().width
                  << "x" << detector.input_size().height << "\n";
        std::cout << "Classes: " << detector.num_classes() << "\n\n";

        // Run inference
        std::cout << "Running inference...\n";
        auto results = detector.predict(image, 0.5f, 0.45f);

        // Print results
        std::cout << "\nDetected " << results.detections.size() << " object(s):\n";
        std::cout << "--------------------------------\n";

        for (size_t i = 0; i < results.detections.size(); i++) {
            const auto& det = results.detections[i];
            std::cout << "  " << (i + 1) << ". " << det.label
                      << " - " << std::fixed << std::setprecision(1)
                      << (det.confidence * 100) << "%"
                      << " @ [" << (int)det.bbox.x1 << ", " << (int)det.bbox.y1
                      << ", " << (int)det.bbox.x2 << ", " << (int)det.bbox.y2 << "]\n";
        }

        std::cout << "\nInference time: " << results.inference_time_ms << " ms\n";
        std::cout << "Device used: " << results.device_used << "\n";

        // Save visualization
        if (!output_path.empty()) {
            cv::Mat vis = results.visualize(image);
            cv::imwrite(output_path, vis);
            std::cout << "\nVisualization saved to: "
                      << std::filesystem::absolute(output_path).string() << "\n";
        }

        // Save JSON results
        std::string json_path = output_path.empty() ? "detections.json" : output_path + ".json";
        results.save(json_path, "json");
        std::cout << "Results saved to: "
                  << std::filesystem::absolute(json_path).string() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Detection error: " << e.what() << "\n";
    }
}

// ============================================================================
// Segmentation Example
// ============================================================================

void run_segmentation(const std::string& model_path,
                      const std::string& image_path,
                      const std::string& device,
                      const std::string& output_path = "") {
    print_separator("Semantic Segmentation");

    try {
        std::cout << "Model:  " << model_path << "\n";
        std::cout << "Image:  " << image_path << "\n";
        std::cout << "Device: " << device << "\n\n";

        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Cannot load image: " << image_path << "\n";
            return;
        }

        // Create segmentor
        std::cout << "Loading model...\n";
        Segmentor segmentor(model_path, device);

        std::cout << "Input size: " << segmentor.input_size().width
                  << "x" << segmentor.input_size().height << "\n";
        std::cout << "Classes: " << segmentor.num_classes() << "\n\n";

        // Run inference
        std::cout << "Running inference...\n";
        auto results = segmentor.predict(image);

        // Print results
        std::cout << "\nSegmentation complete.\n";
        std::cout << "Mask size: " << results.segmentation_mask.cols
                  << "x" << results.segmentation_mask.rows << "\n";
        std::cout << "\nInference time: " << results.inference_time_ms << " ms\n";
        std::cout << "Device used: " << results.device_used << "\n";

        // Save visualization
        if (!output_path.empty()) {
            cv::Mat overlay = results.overlay_mask(image, 0.5);
            cv::imwrite(output_path, overlay);
            std::cout << "\nOverlay saved to: " << output_path << "\n";

            // Also save colorized mask
            std::string mask_path = output_path.substr(0, output_path.rfind('.')) + "_mask.png";
            cv::Mat colored = results.colorize_mask();
            cv::imwrite(mask_path, colored);
            std::cout << "Mask saved to: " << mask_path << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Segmentation error: " << e.what() << "\n";
    }
}

// ============================================================================
// Benchmark Example
// ============================================================================

void run_benchmark(const std::string& model_path,
                   const std::string& device,
                   int iterations = 100) {
    print_separator("Benchmark");

    try {
        std::cout << "Model:      " << model_path << "\n";
        std::cout << "Device:     " << device << "\n";
        std::cout << "Iterations: " << iterations << "\n\n";

        // Create detector for benchmark
        Detector detector(model_path, device);

        // Create dummy image
        cv::Mat dummy(640, 640, CV_8UC3, cv::Scalar(128, 128, 128));

        // Warmup
        std::cout << "Warming up...\n";
        for (int i = 0; i < 10; i++) {
            detector.predict(dummy);
        }

        // Benchmark
        std::cout << "Running benchmark...\n";
        std::vector<double> times;
        times.reserve(iterations);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            auto results = detector.predict(dummy);
            times.push_back(results.inference_time_ms);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Calculate statistics
        std::sort(times.begin(), times.end());
        double sum = 0;
        for (double t : times) sum += t;

        double avg = sum / times.size();
        double min = times.front();
        double max = times.back();
        double p50 = times[times.size() / 2];
        double p95 = times[static_cast<size_t>(times.size() * 0.95)];
        double p99 = times[static_cast<size_t>(times.size() * 0.99)];
        double fps = 1000.0 / avg;

        // Print results
        std::cout << "\nResults:\n";
        std::cout << "--------\n";
        std::cout << "  Total time:  " << total << " ms\n";
        std::cout << "  Average:     " << std::fixed << std::setprecision(2) << avg << " ms\n";
        std::cout << "  Min:         " << min << " ms\n";
        std::cout << "  Max:         " << max << " ms\n";
        std::cout << "  P50:         " << p50 << " ms\n";
        std::cout << "  P95:         " << p95 << " ms\n";
        std::cout << "  P99:         " << p99 << " ms\n";
        std::cout << "  Throughput:  " << fps << " FPS\n";

    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << "\n";
    }
}

// ============================================================================
// Model Conversion
// ============================================================================

void run_convert(const std::string& src_path,
                 const std::string& dst_path,
                 const std::string& device,
                 const std::string& precision) {
    print_separator("Model Conversion");

    try {
        std::cout << "Source:    " << src_path << "\n";
        std::cout << "Output:    " << dst_path << "\n";
        std::cout << "Device:    " << device << "\n";
        std::cout << "Precision: " << precision << "\n\n";

        std::cout << "Converting model...\n";
        auto start = std::chrono::high_resolution_clock::now();

        ivit::convert_model(src_path, dst_path, device, precision);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "\nConversion completed successfully!\n";
        std::cout << "Time taken: " << duration << " ms\n";
        std::cout << "Output saved to: " << dst_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Conversion error: " << e.what() << "\n";
    }
}

// ============================================================================
// Cache Management
// ============================================================================

void run_clear_cache(const std::string& cache_dir) {
    print_separator("Clear Cache");

    try {
        std::string dir = cache_dir.empty() ? ivit::get_cache_dir() : cache_dir;
        std::cout << "Clearing cache directory: " << dir << "\n";

        ivit::clear_cache(cache_dir);

        std::cout << "Cache cleared successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Clear cache error: " << e.what() << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* program) {
    std::cout << "iVIT-SDK Simple Inference Example\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program << " <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  devices                           List available devices\n";
    std::cout << "  classify <model> <image> [device] Run image classification\n";
    std::cout << "  detect <model> <image> [device] [output]  Run object detection\n";
    std::cout << "  segment <model> <image> [device] [output] Run segmentation\n";
    std::cout << "  benchmark <model> [device] [iterations]   Run benchmark\n";
    std::cout << "  convert <src> <dst> [device] [precision]  Convert model format\n";
    std::cout << "  clear-cache [cache_dir]           Clear cached engine files\n";
    std::cout << "\n";
    std::cout << "Device options:\n";
    std::cout << "  auto    - Automatically select best device (default)\n";
    std::cout << "  cpu     - Use CPU (OpenVINO)\n";
    std::cout << "  gpu:0   - Use Intel GPU (OpenVINO)\n";
    std::cout << "  cuda:0  - Use NVIDIA GPU (TensorRT)\n";
    std::cout << "  npu     - Use Intel NPU (OpenVINO)\n";
    std::cout << "\n";
    std::cout << "Precision options:\n";
    std::cout << "  fp32    - Full precision (default)\n";
    std::cout << "  fp16    - Half precision (recommended for GPU)\n";
    std::cout << "  int8    - Integer 8-bit (requires calibration)\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program << " devices\n";
    std::cout << "  " << program << " classify resnet50.onnx cat.jpg cpu\n";
    std::cout << "  " << program << " detect yolox-s.onnx street.jpg cuda:0 output.jpg\n";
    std::cout << "  " << program << " segment deeplabv3.onnx scene.jpg gpu:0\n";
    std::cout << "  " << program << " benchmark yolox-s.onnx cuda:0 100\n";
    std::cout << "  " << program << " convert model.onnx model.engine cuda:0 fp16\n";
    std::cout << "  " << program << " convert model.onnx model.xml cpu fp32\n";
    std::cout << "  " << program << " clear-cache\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "devices") {
        print_device_info();
    }
    else if (command == "classify") {
        if (argc < 4) {
            std::cerr << "Error: classify requires <model> and <image>\n";
            return 1;
        }
        std::string model = argv[2];
        std::string image = argv[3];
        std::string device = (argc > 4) ? argv[4] : "auto";
        run_classification(model, image, device);
    }
    else if (command == "detect") {
        if (argc < 4) {
            std::cerr << "Error: detect requires <model> and <image>\n";
            return 1;
        }
        std::string model = argv[2];
        std::string image = argv[3];
        std::string device = (argc > 4) ? argv[4] : "auto";
        std::string output = (argc > 5) ? argv[5] : "";
        run_detection(model, image, device, output);
    }
    else if (command == "segment") {
        if (argc < 4) {
            std::cerr << "Error: segment requires <model> and <image>\n";
            return 1;
        }
        std::string model = argv[2];
        std::string image = argv[3];
        std::string device = (argc > 4) ? argv[4] : "auto";
        std::string output = (argc > 5) ? argv[5] : "";
        run_segmentation(model, image, device, output);
    }
    else if (command == "benchmark") {
        if (argc < 3) {
            std::cerr << "Error: benchmark requires <model>\n";
            return 1;
        }
        std::string model = argv[2];
        std::string device = (argc > 3) ? argv[3] : "auto";
        int iterations = (argc > 4) ? std::stoi(argv[4]) : 100;
        run_benchmark(model, device, iterations);
    }
    else if (command == "convert") {
        if (argc < 4) {
            std::cerr << "Error: convert requires <src_path> and <dst_path>\n";
            std::cerr << "Usage: " << argv[0] << " convert <src> <dst> [device] [precision]\n";
            return 1;
        }
        std::string src = argv[2];
        std::string dst = argv[3];
        std::string device = (argc > 4) ? argv[4] : "auto";
        std::string precision = (argc > 5) ? argv[5] : "fp16";
        run_convert(src, dst, device, precision);
    }
    else if (command == "clear-cache") {
        std::string cache_dir = (argc > 2) ? argv[2] : "";
        run_clear_cache(cache_dir);
    }
    else if (command == "-h" || command == "--help" || command == "help") {
        print_usage(argv[0]);
    }
    else {
        std::cerr << "Unknown command: " << command << "\n";
        print_usage(argv[0]);
        return 1;
    }

    // Use _exit() to skip static destruction, which can cause segfault
    // with TensorRT on newer GPUs during cleanup.
    std::cout.flush();
    std::cerr.flush();
    _exit(0);
}

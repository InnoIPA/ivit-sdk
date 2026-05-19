/**
 * @file data_analysis.cpp
 * @brief iVIT-SDK Data Scientist Analysis Example
 *
 * Target: Data scientists who need to quickly validate models,
 *         analyze inference results, and conduct experiments.
 *
 * Features demonstrated:
 * - Quick model loading and testing
 * - Results analysis (filtering, statistics)
 * - Batch processing
 * - JSON serialization
 *
 * Note: Model Zoo is available via Python API (ivit.zoo).
 *       C++ focuses on inference and analysis.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make data_analysis
 *
 * Usage:
 *   ./data_analysis <model_path>
 *   ./data_analysis <model_path> <image_path>
 *   ./data_analysis <model_path> --batch
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "ivit/ivit.hpp"

using namespace ivit;

/**
 * Step 1: Explore available devices and model info
 */
void explore_system(Model& model) {
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Step 1: System & Model Exploration" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // List all devices
    auto devices = list_devices();
    std::cout << "Available devices: " << devices.size() << std::endl;
    for (const auto& dev : devices) {
        std::cout << "  - " << dev.id << ": " << dev.name
                  << " (" << dev.backend << ")" << std::endl;
    }

    // Model info
    std::cout << "\nModel Information:" << std::endl;
    auto inputs = model.input_info();
    if (!inputs.empty()) {
        std::cout << "  Input: " << inputs[0].name << " [";
        for (size_t i = 0; i < inputs[0].shape.size(); ++i) {
            std::cout << inputs[0].shape[i];
            if (i < inputs[0].shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    auto outputs = model.output_info();
    if (!outputs.empty()) {
        std::cout << "  Output: " << outputs[0].name << " [";
        for (size_t i = 0; i < outputs[0].shape.size(); ++i) {
            std::cout << outputs[0].shape[i];
            if (i < outputs[0].shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Python Model Zoo reference
    std::cout << "\nFor Model Zoo, use Python API:" << std::endl;
    std::cout << "  import ivit" << std::endl;
    std::cout << "  ivit.zoo.list()           # List all models" << std::endl;
    std::cout << "  ivit.zoo.load('yolox-s')  # Load pre-trained model" << std::endl;
}

/**
 * Convert detection results to JSON string
 */
std::string results_to_json(const Results& results, int image_width, int image_height) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"inference_time_ms\": " << std::fixed << std::setprecision(2)
        << results.inference_time_ms << ",\n";
    oss << "  \"image_size\": [" << image_width << ", " << image_height << "],\n";
    oss << "  \"detection_count\": " << results.detections.size() << ",\n";
    oss << "  \"detections\": [\n";

    for (size_t i = 0; i < results.detections.size(); ++i) {
        const auto& det = results.detections[i];
        oss << "    {\n";
        oss << "      \"label\": \"" << det.label << "\",\n";
        oss << "      \"class_id\": " << det.class_id << ",\n";
        oss << "      \"confidence\": " << std::fixed << std::setprecision(4)
            << det.confidence << ",\n";
        oss << "      \"bbox\": [" << det.bbox.x1 << ", " << det.bbox.y1
            << ", " << det.bbox.x2 << ", " << det.bbox.y2 << "]\n";
        oss << "    }" << (i < results.detections.size() - 1 ? "," : "") << "\n";
    }

    oss << "  ]\n";
    oss << "}";
    return oss.str();
}

/**
 * Step 2: Analyze detection results
 */
void analyze_detection_results(Model& model, const cv::Mat& image) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Step 2: Detection Results Analysis" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    auto results = model.predict(image);

    // Basic info
    std::cout << "Detection count: " << results.detections.size() << std::endl;
    std::cout << "Inference time: " << std::fixed << std::setprecision(1)
              << results.inference_time_ms << " ms" << std::endl;
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;

    // Iterate detections
    std::cout << "\nAll detections:" << std::endl;
    for (size_t i = 0; i < results.detections.size(); ++i) {
        const auto& det = results.detections[i];
        std::cout << "  [" << i << "] " << det.label << ": "
                  << std::fixed << std::setprecision(1)
                  << (det.confidence * 100) << "%" << std::endl;
        std::cout << "       BBox: (" << static_cast<int>(det.bbox.x1) << ", "
                  << static_cast<int>(det.bbox.y1) << ") - ("
                  << static_cast<int>(det.bbox.x2) << ", "
                  << static_cast<int>(det.bbox.y2) << ")" << std::endl;

        double area = (det.bbox.x2 - det.bbox.x1) * (det.bbox.y2 - det.bbox.y1);
        std::cout << "       Area: " << static_cast<int>(area) << " px^2" << std::endl;
    }

    // Filtering examples
    std::cout << "\nFiltering examples:" << std::endl;

    // Filter by confidence
    int high_conf_count = 0;
    for (const auto& det : results.detections) {
        if (det.confidence > 0.9) high_conf_count++;
    }
    std::cout << "  High confidence (>90%): " << high_conf_count << " detections" << std::endl;

    // Filter by area
    int large_obj_count = 0;
    for (const auto& det : results.detections) {
        double area = (det.bbox.x2 - det.bbox.x1) * (det.bbox.y2 - det.bbox.y1);
        if (area > 5000) large_obj_count++;
    }
    std::cout << "  Large objects (>5000 px^2): " << large_obj_count << " detections" << std::endl;

    // JSON serialization
    std::cout << "\nSerialization:" << std::endl;
    std::string json = results_to_json(results, image.cols, image.rows);
    std::cout << "  to_json() length: " << json.length() << " chars" << std::endl;
    std::cout << "\nJSON Output:" << std::endl;
    std::cout << json << std::endl;
}

/**
 * Step 3: Batch processing and analysis
 */
struct BatchStats {
    int total_images;
    int total_detections;
    std::map<std::string, int> class_counts;
    double avg_latency_ms;
    double confidence_mean;
    double confidence_std;
};

BatchStats batch_analysis(Model& model, const std::vector<cv::Mat>& images) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Step 3: Batch Processing Analysis" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "Processing " << images.size() << " images..." << std::endl;

    BatchStats stats;
    stats.total_images = static_cast<int>(images.size());
    stats.total_detections = 0;

    std::vector<double> latencies;
    std::vector<double> confidences;

    for (size_t i = 0; i < images.size(); ++i) {
        auto results = model.predict(images[i]);

        stats.total_detections += static_cast<int>(results.detections.size());
        latencies.push_back(results.inference_time_ms);

        for (const auto& det : results.detections) {
            stats.class_counts[det.label]++;
            confidences.push_back(det.confidence);
        }

        std::cout << "  Image " << (i + 1) << "/" << images.size()
                  << ": " << results.detections.size() << " detections, "
                  << std::fixed << std::setprecision(1) << results.inference_time_ms
                  << " ms" << std::endl;
    }

    // Calculate statistics
    stats.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                           / latencies.size();

    if (!confidences.empty()) {
        stats.confidence_mean = std::accumulate(confidences.begin(), confidences.end(), 0.0)
                                / confidences.size();

        double sq_sum = 0;
        for (double c : confidences) {
            sq_sum += (c - stats.confidence_mean) * (c - stats.confidence_mean);
        }
        stats.confidence_std = std::sqrt(sq_sum / confidences.size());
    } else {
        stats.confidence_mean = 0;
        stats.confidence_std = 0;
    }

    // Print statistics
    std::cout << "\nBatch Statistics:" << std::endl;
    std::cout << "  Total images: " << stats.total_images << std::endl;
    std::cout << "  Total detections: " << stats.total_detections << std::endl;
    std::cout << "  Avg detections per image: " << std::fixed << std::setprecision(1)
              << static_cast<double>(stats.total_detections) / stats.total_images << std::endl;
    std::cout << "  Avg latency: " << std::fixed << std::setprecision(1)
              << stats.avg_latency_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
              << (1000.0 / stats.avg_latency_ms) << " FPS" << std::endl;

    if (!confidences.empty()) {
        std::cout << "\nConfidence Statistics:" << std::endl;
        std::cout << "  Mean: " << std::fixed << std::setprecision(1)
                  << (stats.confidence_mean * 100) << "%" << std::endl;
        std::cout << "  Std: " << std::fixed << std::setprecision(1)
                  << (stats.confidence_std * 100) << "%" << std::endl;
    }

    std::cout << "\nClass Distribution:" << std::endl;
    // Sort by count
    std::vector<std::pair<std::string, int>> sorted_classes(
        stats.class_counts.begin(), stats.class_counts.end());
    std::sort(sorted_classes.begin(), sorted_classes.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& [cls, count] : sorted_classes) {
        std::cout << "  " << cls << ": " << count << std::endl;
    }

    return stats;
}

/**
 * Step 4: Export format comparison
 */
void export_comparison() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Step 4: Export Format Comparison" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "Export Format Comparison:" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left << std::setw(12) << "Format"
              << std::setw(30) << "Usage"
              << "Quantization" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::setw(12) << "onnx"
              << std::setw(30) << "Cross-platform deployment"
              << "fp32, fp16" << std::endl;
    std::cout << std::setw(12) << "torchscript"
              << std::setw(30) << "PyTorch ecosystem"
              << "fp32" << std::endl;
    std::cout << std::setw(12) << "openvino"
              << std::setw(30) << "Intel hardware optimization"
              << "fp32, fp16, int8" << std::endl;
    std::cout << std::setw(12) << "tensorrt"
              << std::setw(30) << "NVIDIA hardware optimization"
              << "fp32, fp16, int8" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::cout << "\nPython export examples:" << std::endl;
    std::cout << "  model.export('model.onnx', format='onnx', quantization='fp16')" << std::endl;
    std::cout << "  model.export('model.xml', format='openvino', quantization='int8')" << std::endl;
    std::cout << "  model.export('model.engine', format='tensorrt', quantization='fp16')" << std::endl;

    std::cout << "\nCLI export examples:" << std::endl;
    std::cout << "  ivit convert model.onnx model.engine --format tensorrt --fp16" << std::endl;
    std::cout << "  ivit convert model.onnx model.xml --format openvino --int8" << std::endl;
}

void print_best_practices() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Data Scientist Best Practices:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "1. Use Python ivit.zoo for quick model exploration" << std::endl;
    std::cout << "2. Use filtering for targeted analysis" << std::endl;
    std::cout << "3. Use JSON serialization for data export" << std::endl;
    std::cout << "4. Use batch processing for efficient analysis" << std::endl;
    std::cout << "5. Compare export formats based on deployment target" << std::endl;
    std::cout << "6. Track confidence statistics for model quality assessment" << std::endl;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model_path> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  <image_path>   Path to input image" << std::endl;
    std::cout << "  --batch        Run batch processing demo" << std::endl;
    std::cout << "  --device       Target device (auto, cpu, cuda:0)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program << " yolox-s.onnx image.jpg" << std::endl;
    std::cout << "  " << program << " yolox-s.onnx --batch" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path;
    std::string device = "auto";
    bool batch_mode = false;

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--batch") {
            batch_mode = true;
        } else if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg[0] != '-') {
            image_path = arg;
        }
    }

    std::cout << std::string(60, '=') << std::endl;
    std::cout << "iVIT-SDK Data Scientist Analysis Example" << std::endl;
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

    // Step 1: Explore system
    explore_system(*model);

    // Prepare image
    cv::Mat image;
    if (!image_path.empty()) {
        image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Cannot read image from " << image_path << std::endl;
            return 1;
        }
        std::cout << "\nLoaded image: " << image_path
                  << " (" << image.cols << "x" << image.rows << ")" << std::endl;
    } else {
        // Create synthetic test image
        std::cout << "\nNote: Using synthetic test image. Provide image path for real data." << std::endl;
        image = cv::Mat(480, 640, CV_8UC3);
        cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    }

    // Step 2: Analyze detection results
    analyze_detection_results(*model, image);

    // Step 3: Batch analysis
    if (batch_mode) {
        std::vector<cv::Mat> batch_images;
        for (int i = 0; i < 5; ++i) {
            cv::Mat img(480, 640, CV_8UC3);
            cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            batch_images.push_back(img);
        }
        batch_analysis(*model, batch_images);
    }

    // Step 4: Export comparison
    export_comparison();

    // Best practices
    print_best_practices();

    return 0;
}

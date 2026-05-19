/**
 * @file si_quickstart.cpp
 * @brief iVIT-SDK System Integrator (SI) Quick Start Example
 *
 * Target: System Integrators who need to quickly integrate AI inference
 *         into existing systems.
 *
 * Features demonstrated:
 * - Device discovery and auto-selection
 * - Model loading with load_model()
 * - Structured error handling
 * - Result serialization (JSON)
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make si_quickstart
 *
 * Usage:
 *   ./si_quickstart
 *   ./si_quickstart <image_path>
 *   ./si_quickstart <image_path> <model_path>
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "ivit/ivit.hpp"

using namespace ivit;

/**
 * Step 1: Discover available devices
 */
void discover_devices() {
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Step 1: Device Discovery" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // List all available devices
    auto devices = list_devices();
    std::cout << "Found " << devices.size() << " available device(s)" << std::endl;

    for (const auto& device : devices) {
        std::cout << "  - " << device.id << ": " << device.name
                  << " (" << device.backend << ")" << std::endl;
    }

    // Auto-select best device
    auto best = get_best_device();
    std::cout << "\nAuto-selected best device: " << best.id
              << " (" << best.name << ")" << std::endl;

    std::cout << "\nDevice selection options:" << std::endl;
    std::cout << "  get_best_device()                    -> Auto-select best" << std::endl;
    std::cout << "  get_best_device(\"\", \"efficiency\")    -> Best efficiency" << std::endl;
    std::cout << "  DeviceManager::instance().get_device(\"cpu\") -> Specific device" << std::endl;
}

/**
 * Step 2: Load model with error handling
 */
std::shared_ptr<Model> load_model_safe(const std::string& model_path,
                                        const DeviceInfo& device) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Step 2: Model Loading" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "Loading model: " << model_path << std::endl;
    std::cout << "Target device: " << device.id << std::endl;

    LoadConfig config;
    config.device = device.id;

    auto model = load_model(model_path, config);

    std::cout << "Model loaded successfully!" << std::endl;

    // Get model info
    auto input_info = model->input_info();
    if (!input_info.empty()) {
        std::cout << "  Input: " << input_info[0].name
                  << " [" << input_info[0].shape[0];
        for (size_t i = 1; i < input_info[0].shape.size(); ++i) {
            std::cout << ", " << input_info[0].shape[i];
        }
        std::cout << "]" << std::endl;
    }

    return model;
}

/**
 * Step 3: Safe inference result structure
 */
struct InferenceResult {
    bool success;
    std::string error_type;
    std::string message;
    std::string suggestion;
    double inference_time_ms;
    int detection_count;
    std::vector<Detection> detections;

    // Convert to JSON string
    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"success\": " << (success ? "true" : "false") << ",\n";

        if (success) {
            oss << "  \"inference_time_ms\": " << std::fixed
                << std::setprecision(2) << inference_time_ms << ",\n";
            oss << "  \"detection_count\": " << detection_count << ",\n";
            oss << "  \"detections\": [\n";

            for (size_t i = 0; i < detections.size(); ++i) {
                const auto& det = detections[i];
                oss << "    {\n";
                oss << "      \"label\": \"" << det.label << "\",\n";
                oss << "      \"confidence\": " << std::fixed
                    << std::setprecision(4) << det.confidence << ",\n";
                oss << "      \"bbox\": [" << det.bbox.x1 << ", " << det.bbox.y1
                    << ", " << det.bbox.x2 << ", " << det.bbox.y2 << "]\n";
                oss << "    }" << (i < detections.size() - 1 ? "," : "") << "\n";
            }

            oss << "  ]\n";
        } else {
            oss << "  \"error_type\": \"" << error_type << "\",\n";
            oss << "  \"message\": \"" << message << "\",\n";
            oss << "  \"suggestion\": \"" << suggestion << "\"\n";
        }

        oss << "}";
        return oss.str();
    }
};

/**
 * Step 3: Safe inference with comprehensive error handling
 */
InferenceResult safe_inference(Model& model, const std::string& image_path) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Step 3: Safe Inference" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    InferenceResult result;

    try {
        std::cout << "Running inference on: " << image_path << std::endl;

        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }

        // Run inference
        auto results = model.predict(image);

        result.success = true;
        result.inference_time_ms = results.inference_time_ms;
        result.detection_count = static_cast<int>(results.detections.size());
        result.detections = results.detections;

    } catch (const ModelLoadError& e) {
        result.success = false;
        result.error_type = "ModelLoadError";
        result.message = e.what();
        result.suggestion = "Verify model path and format (.onnx, .xml, .engine)";

    } catch (const DeviceNotFoundError& e) {
        result.success = false;
        result.error_type = "DeviceNotFoundError";
        result.message = e.what();
        result.suggestion = "Run list_devices() to check available devices";

    } catch (const InferenceError& e) {
        result.success = false;
        result.error_type = "InferenceError";
        result.message = e.what();
        result.suggestion = "Check input image format and dimensions";

    } catch (const IVITError& e) {
        result.success = false;
        result.error_type = "IVITError";
        result.message = e.what();
        result.suggestion = "See error message for details";

    } catch (const std::exception& e) {
        result.success = false;
        result.error_type = "StdException";
        result.message = e.what();
        result.suggestion = "Unexpected error occurred";
    }

    return result;
}

/**
 * Step 4: Process and display results
 */
void process_results(const InferenceResult& result) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Step 4: Results" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    if (result.success) {
        std::cout << "Inference successful!" << std::endl;
        std::cout << "  Inference time: " << std::fixed << std::setprecision(2)
                  << result.inference_time_ms << " ms" << std::endl;
        std::cout << "  Detections: " << result.detection_count << std::endl;

        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& det = result.detections[i];
            std::cout << "    [" << i << "] " << det.label << ": "
                      << std::fixed << std::setprecision(1)
                      << (det.confidence * 100) << "%" << std::endl;
        }

        std::cout << "\nJSON output (for system integration):" << std::endl;
        std::cout << result.to_json() << std::endl;

    } else {
        std::cout << "Inference failed!" << std::endl;
        std::cout << "  Error type: " << result.error_type << std::endl;
        std::cout << "  Message: " << result.message << std::endl;
        std::cout << "  Suggestion: " << result.suggestion << std::endl;
    }
}

void print_best_practices() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "SI Best Practices:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "1. Use get_best_device() for auto device selection" << std::endl;
    std::cout << "2. Always wrap inference in try-catch blocks" << std::endl;
    std::cout << "3. Use to_json() for structured output" << std::endl;
    std::cout << "4. Test on multiple hardware environments" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "iVIT-SDK System Integrator (SI) Quick Start" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "SDK Version: " << ivit::version() << std::endl;

    std::string image_path = argc > 1 ? argv[1] : "test_images/bus.jpg";
    std::string model_path = argc > 2 ? argv[2] : "models/yolox-s.onnx";

    // Step 1: Discover devices
    discover_devices();

    // Get best device
    auto best_device = get_best_device();

    // Step 2: Load model
    std::shared_ptr<Model> model;
    try {
        model = load_model_safe(model_path, best_device);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        std::cerr << "Please ensure the model file exists." << std::endl;
        return 1;
    }

    // Step 3: Run inference with error handling
    auto result = safe_inference(*model, image_path);

    // Step 4: Process results
    process_results(result);

    // Best practices
    print_best_practices();

    return result.success ? 0 : 1;
}

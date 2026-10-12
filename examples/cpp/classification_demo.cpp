/**
 * @file classification_demo.cpp
 * @brief Image classification demo
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "ivit/vision/classifier.hpp"

// A window needs a display; without one (SSH, CI, containers) OpenCV's GUI aborts the process.
static bool has_display() {
#ifdef _WIN32
    return true;
#else
    return std::getenv("DISPLAY") || std::getenv("WAYLAND_DISPLAY");
#endif
}

using namespace ivit;
using namespace ivit::vision;

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model_path> <image_path> [device] [imagenet]\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  model_path  Path to classification model (ONNX, OpenVINO XML, etc.)\n";
    std::cout << "  image_path  Path to input image\n";
    std::cout << "  device      Target device (auto, cpu, gpu:0, npu) [default: auto]\n";
    std::cout << "  imagenet    Apply ImageNet mean/std (torchvision models, e.g. the Model Zoo's)\n";
    std::cout << "\n";
    std::cout << "Example:\n";
    std::cout << "  " << program << " resnet18.onnx bus.jpg cpu imagenet\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string device = argc > 3 ? argv[3] : "auto";
    const bool imagenet = argc > 4 && std::string(argv[4]) == "imagenet";

    try {
        // Load classifier
        std::cout << "Loading model: " << model_path << std::endl;
        std::cout << "Device: " << device << std::endl;

        // The input contract must match how the model was exported (see the SDK manual, 4.2).
        LoadConfig config;
        if (imagenet) {
            config.mean = {0.485f, 0.456f, 0.406f};
            config.std = {0.229f, 0.224f, 0.225f};
        }
        Classifier classifier(model_path, device, config);

        // Load and classify image
        std::cout << "Classifying: " << image_path << std::endl;

        auto results = classifier.predict(image_path, 5);

        // Print results
        std::cout << "\n=== Classification Results ===" << std::endl;
        std::cout << "Inference time: " << results.inference_time_ms << " ms" << std::endl;
        std::cout << "Device used: " << results.device_used << std::endl;
        std::cout << "\nTop-5 predictions:" << std::endl;

        for (size_t i = 0; i < results.classifications.size(); i++) {
            const auto& cls = results.classifications[i];
            std::cout << "  " << (i + 1) << ". " << cls.label
                      << " (class " << cls.class_id << "): "
                      << std::fixed << std::setprecision(2)
                      << (cls.score * 100) << "%" << std::endl;
        }

        // Display image with results
        cv::Mat image = cv::imread(image_path);
        if (!image.empty()) {
            // Draw top prediction on image
            if (!results.classifications.empty()) {
                const auto& top = results.classifications[0];
                std::string text = top.label + ": " +
                    std::to_string(static_cast<int>(top.score * 100)) + "%";

                cv::putText(image, text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            }

            cv::imwrite("classification_result.jpg", image);
            std::cout << "\nSaved: classification_result.jpg" << std::endl;
            if (has_display()) {
                cv::imshow("Classification Result", image);
                std::cout << "Press any key to exit..." << std::endl;
                cv::waitKey(0);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

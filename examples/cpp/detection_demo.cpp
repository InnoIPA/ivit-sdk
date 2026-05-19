/**
 * @file detection_demo.cpp
 * @brief Object detection demo
 */

#include <iostream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "ivit/vision/detector.hpp"

using namespace ivit;
using namespace ivit::vision;

// Color palette for visualization
static const std::vector<cv::Scalar> COLORS = {
    {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0},
    {255, 0, 255}, {0, 255, 255}, {128, 0, 0}, {0, 128, 0},
    {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128}
};

void draw_detections(cv::Mat& image, const Results& results) {
    for (const auto& det : results.detections) {
        // Get color for this class
        cv::Scalar color = COLORS[det.class_id % COLORS.size()];

        // Draw bounding box
        cv::rectangle(image,
            cv::Point(static_cast<int>(det.bbox.x1), static_cast<int>(det.bbox.y1)),
            cv::Point(static_cast<int>(det.bbox.x2), static_cast<int>(det.bbox.y2)),
            color, 2);

        // Draw label background
        std::string label = det.label + ": " +
            std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
            0.5, 1, &baseline);

        cv::rectangle(image,
            cv::Point(static_cast<int>(det.bbox.x1),
                      static_cast<int>(det.bbox.y1) - text_size.height - 5),
            cv::Point(static_cast<int>(det.bbox.x1) + text_size.width,
                      static_cast<int>(det.bbox.y1)),
            color, -1);

        // Draw label text
        cv::putText(image, label,
            cv::Point(static_cast<int>(det.bbox.x1),
                      static_cast<int>(det.bbox.y1) - 3),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model_path> <image_path> [device] [conf_threshold]\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  model_path      Path to detection model (YOLO, SSD, etc.)\n";
    std::cout << "  image_path      Path to input image\n";
    std::cout << "  device          Target device (auto, cpu, gpu:0, npu) [default: auto]\n";
    std::cout << "  conf_threshold  Confidence threshold [default: 0.5]\n";
    std::cout << "\n";
    std::cout << "Example:\n";
    std::cout << "  " << program << " yolox-s.onnx street.jpg gpu:0 0.5\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string device = argc > 3 ? argv[3] : "auto";
    float conf_threshold = argc > 4 ? std::stof(argv[4]) : 0.5f;

    try {
        // Load detector
        std::cout << "Loading model: " << model_path << std::endl;
        std::cout << "Device: " << device << std::endl;

        Detector detector(model_path, device);

        // Load and detect
        std::cout << "Detecting objects in: " << image_path << std::endl;

        auto results = detector.predict(image_path, conf_threshold);

        // Print results
        std::cout << "\n=== Detection Results ===" << std::endl;
        std::cout << "Inference time: " << results.inference_time_ms << " ms" << std::endl;
        std::cout << "Device used: " << results.device_used << std::endl;
        std::cout << "Objects detected: " << results.detections.size() << std::endl;
        std::cout << std::endl;

        for (size_t i = 0; i < results.detections.size(); i++) {
            const auto& det = results.detections[i];
            std::cout << "  " << (i + 1) << ". " << det.label
                      << " (class " << det.class_id << "): "
                      << std::fixed << std::setprecision(1)
                      << (det.confidence * 100) << "% at ["
                      << static_cast<int>(det.bbox.x1) << ", "
                      << static_cast<int>(det.bbox.y1) << ", "
                      << static_cast<int>(det.bbox.x2) << ", "
                      << static_cast<int>(det.bbox.y2) << "]" << std::endl;
        }

        // Display image with detections
        cv::Mat image = cv::imread(image_path);
        if (!image.empty()) {
            draw_detections(image, results);

            cv::imshow("Detection Result", image);
            std::cout << "\nPress any key to exit..." << std::endl;
            cv::waitKey(0);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

/**
 * @file video_demo.cpp
 * @brief Video object detection demo
 */

#include <iostream>
#include <string>
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
        cv::Scalar color = COLORS[det.class_id % COLORS.size()];

        cv::rectangle(image,
            cv::Point(static_cast<int>(det.bbox.x1), static_cast<int>(det.bbox.y1)),
            cv::Point(static_cast<int>(det.bbox.x2), static_cast<int>(det.bbox.y2)),
            color, 2);

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

        cv::putText(image, label,
            cv::Point(static_cast<int>(det.bbox.x1),
                      static_cast<int>(det.bbox.y1) - 3),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model_path> <video_source> [device] [conf_threshold]\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  model_path      Path to detection model\n";
    std::cout << "  video_source    Video file path or camera ID (0, 1, etc.)\n";
    std::cout << "  device          Target device (auto, cpu, gpu:0, npu) [default: auto]\n";
    std::cout << "  conf_threshold  Confidence threshold [default: 0.5]\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program << " yolox-s.onnx video.mp4 gpu:0 0.5\n";
    std::cout << "  " << program << " yolox-s.onnx 0 gpu:0 0.5  # Use camera 0\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string video_source = argv[2];
    std::string device = argc > 3 ? argv[3] : "auto";
    float conf_threshold = argc > 4 ? std::stof(argv[4]) : 0.5f;

    try {
        // Load detector
        std::cout << "Loading model: " << model_path << std::endl;
        std::cout << "Device: " << device << std::endl;

        Detector detector(model_path, device);

        // Open video source
        cv::VideoCapture cap;
        bool is_camera = false;

        try {
            int cam_id = std::stoi(video_source);
            cap.open(cam_id);
            is_camera = true;
            std::cout << "Opening camera: " << cam_id << std::endl;
        } catch (...) {
            cap.open(video_source);
            std::cout << "Opening video: " << video_source << std::endl;
        }

        if (!cap.isOpened()) {
            std::cerr << "Failed to open video source: " << video_source << std::endl;
            return 1;
        }

        // Get video properties
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;

        std::cout << "Video: " << frame_width << "x" << frame_height
                  << " @ " << fps << " FPS" << std::endl;
        std::cout << "\nPress 'q' or ESC to quit..." << std::endl;

        // Process frames
        cv::Mat frame;
        int frame_count = 0;
        double total_inference_time = 0;

        while (cap.read(frame)) {
            if (frame.empty()) break;

            // Run detection
            auto results = detector.predict(frame, conf_threshold);

            // Update statistics
            frame_count++;
            total_inference_time += results.inference_time_ms;

            // Draw results
            draw_detections(frame, results);

            // Draw FPS
            double avg_inference = total_inference_time / frame_count;
            double inference_fps = 1000.0 / avg_inference;

            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(inference_fps)) +
                " | Latency: " + std::to_string(static_cast<int>(results.inference_time_ms)) + "ms";

            cv::putText(frame, fps_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            // Display
            cv::imshow("Video Detection", frame);

            // Check for quit
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {  // 'q' or ESC
                break;
            }
        }

        // Print statistics
        std::cout << "\n=== Statistics ===" << std::endl;
        std::cout << "Total frames: " << frame_count << std::endl;
        std::cout << "Average inference time: "
                  << (total_inference_time / frame_count) << " ms" << std::endl;
        std::cout << "Average FPS: "
                  << (1000.0 * frame_count / total_inference_time) << std::endl;

        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

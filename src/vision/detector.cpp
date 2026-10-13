/**
 * @file detector.cpp
 * @brief Object detector implementation
 */

#include "ivit/vision/detector.hpp"
#include "ivit/runtime/runtime.hpp"
#include "vision/processing.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace ivit {
namespace vision {

// ============================================================================
// COCO class labels (80 classes)
// ============================================================================

// ============================================================================
// Detector implementation
// ============================================================================

Detector::Detector(const std::string &model_path, const std::string &device,
                   const LoadConfig &config) {
    // Create load config with detection task hint
    LoadConfig load_config = config;
    load_config.device = device;
    load_config.task = "detection";

    // Load model
    model_ = ModelManager::instance().load(model_path, load_config);

    // Get input size from model
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto &shape = input_info[0].shape;
        const bool nhwc = model_->load_config().input_layout == "nhwc";
        const int h = nhwc ? 1 : 2, w = nhwc ? 2 : 3;
        if (shape.size() == 4 && shape[h] > 0 && shape[w] > 0) {
            input_size_ = cv::Size(static_cast<int>(shape[w]), static_cast<int>(shape[h]));
        } else {
            input_size_ = cv::Size(640, 640);
        }
    } else {
        input_size_ = cv::Size(640, 640);
    }

    labels_ = model_->labels();
}

Detector::Detector(std::shared_ptr<Model> model) : model_(model) {
    if (!model_ || model_->task() != TaskType::Detection)
        throw IVITError("Detector requires a non-null detector model");

    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto &shape = input_info[0].shape;
        const bool nhwc = model_->load_config().input_layout == "nhwc";
        const int h = nhwc ? 1 : 2, w = nhwc ? 2 : 3;
        if (shape.size() == 4 && shape[h] > 0 && shape[w] > 0) {
            input_size_ = cv::Size(static_cast<int>(shape[w]), static_cast<int>(shape[h]));
        } else {
            input_size_ = cv::Size(640, 640);
        }
    } else {
        input_size_ = cv::Size(640, 640);
    }

    labels_ = model_->labels();
}

Results Detector::predict(const cv::Mat &image, float conf_threshold, float iou_threshold) {
    InferConfig config;
    config.conf_threshold = conf_threshold;
    config.iou_threshold = iou_threshold;
    return predict(image, config);
}

Results Detector::predict(const std::string &image_path, float conf_threshold,
                          float iou_threshold) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw IVITError("Failed to load image: " + image_path);
    }
    return predict(image, conf_threshold, iou_threshold);
}

Results Detector::predict(const cv::Mat &image, const InferConfig &config) {
    return model_->predict(image, config);
}

std::vector<Results> Detector::predict_batch(const std::vector<cv::Mat> &images,
                                             const InferConfig &config) {
    std::vector<Results> results_list;
    results_list.reserve(images.size());

    for (const auto &image : images) {
        results_list.push_back(predict(image, config));
    }

    return results_list;
}

void Detector::predict_video(const std::string &source,
                             std::function<void(const Results &, const cv::Mat &)> callback,
                             const InferConfig &config) {
    cv::VideoCapture cap;

    // Only an all-digit string is a camera index: std::stoi("720p.mp4") parses "720" and used to
    // open camera 720 instead of the file.
    const bool is_camera =
        !source.empty() && std::all_of(source.begin(), source.end(),
                                       [](unsigned char c) { return std::isdigit(c) != 0; });
    if (is_camera)
        cap.open(std::stoi(source));
    else
        cap.open(source);

    if (!cap.isOpened()) {
        throw IVITError("Failed to open video source: " + source);
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty())
            break;

        auto results = predict(frame, config);
        callback(results, frame);
    }

    cap.release();
}

} // namespace vision
} // namespace ivit

/**
 * @file classifier.cpp
 * @brief Image classifier implementation
 */

#include "ivit/vision/classifier.hpp"
#include "ivit/runtime/runtime.hpp"
#include "vision/processing.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace ivit {
namespace vision {

// ============================================================================
// Default ImageNet labels (first 10 for demo, full list would be loaded)
// ============================================================================

// ============================================================================
// Classifier implementation
// ============================================================================

Classifier::Classifier(const std::string &model_path, const std::string &device,
                       const LoadConfig &config) {
    // Create load config with classification task hint
    LoadConfig load_config = config;
    load_config.device = device;
    load_config.task = "classification";

    // Load model
    model_ = ModelManager::instance().load(model_path, load_config);

    // Get input size from model
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto &shape = input_info[0].shape;
        const bool nhwc = model_->load_config().input_layout == "nhwc";
        const int h = nhwc ? 1 : 2, w = nhwc ? 2 : 3;
        if (shape.size() == 4 && shape[h] > 0 && shape[w] > 0) {
            // NCHW format
            input_size_ = cv::Size(static_cast<int>(shape[w]), // width
                                   static_cast<int>(shape[h])  // height
            );
        } else {
            input_size_ = cv::Size(224, 224); // Default
        }
    } else {
        input_size_ = cv::Size(224, 224);
    }

    labels_ = model_->labels();
}

Classifier::Classifier(std::shared_ptr<Model> model) : model_(model) {
    if (!model_ || model_->task() != TaskType::Classification)
        throw IVITError("Classifier requires a non-null classifier model");

    // Get input size from model
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto &shape = input_info[0].shape;
        const bool nhwc = model_->load_config().input_layout == "nhwc";
        const int h = nhwc ? 1 : 2, w = nhwc ? 2 : 3;
        if (shape.size() == 4 && shape[h] > 0 && shape[w] > 0) {
            input_size_ = cv::Size(static_cast<int>(shape[w]), static_cast<int>(shape[h]));
        } else {
            input_size_ = cv::Size(224, 224);
        }
    } else {
        input_size_ = cv::Size(224, 224);
    }

    labels_ = model_->labels();
}

Results Classifier::predict(const cv::Mat &image, int top_k) {
    if (top_k <= 0)
        throw IVITError("top_k must be positive");
    InferConfig config;
    config.conf_threshold = 0;
    auto result = model_->predict(image, config);
    if (result.classifications.size() > static_cast<size_t>(top_k))
        result.classifications.resize(top_k);
    return result;
}

Results Classifier::predict(const std::string &image_path, int top_k) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw IVITError("Failed to load image: " + image_path);
    }
    return predict(image, top_k);
}

std::vector<Results> Classifier::predict_batch(const std::vector<cv::Mat> &images, int top_k) {
    if (top_k <= 0)
        throw IVITError("top_k must be positive");
    InferConfig config;
    config.conf_threshold = 0;
    auto results = model_->predict_batch(images, config);
    for (auto &result : results)
        if (result.classifications.size() > static_cast<size_t>(top_k))
            result.classifications.resize(top_k);
    return results;
}

} // namespace vision
} // namespace ivit

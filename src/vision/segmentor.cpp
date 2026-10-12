/**
 * @file segmentor.cpp
 * @brief Semantic segmentor implementation
 */

#include "ivit/vision/segmentor.hpp"
#include "ivit/runtime/runtime.hpp"
#include "vision/processing.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace ivit {
namespace vision {

// ============================================================================
// Common segmentation class labels
// ============================================================================

// PASCAL VOC labels (21 classes)

// Cityscapes labels (19 classes)

// ADE20K labels (150 classes) - abbreviated

// ============================================================================
// Segmentor implementation
// ============================================================================

Segmentor::Segmentor(const std::string &model_path, const std::string &device,
                     const LoadConfig &config) {
    // Create load config with segmentation task hint
    LoadConfig load_config = config;
    load_config.device = device;
    load_config.task = "segmentation";

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
            input_size_ = cv::Size(512, 512);
        }
    } else {
        input_size_ = cv::Size(512, 512);
    }

    labels_ = model_->labels();

    // Initialize colormap
    init_default_colormap();
}

Segmentor::Segmentor(std::shared_ptr<Model> model) : model_(model) {
    if (!model_ || model_->task() != TaskType::Segmentation)
        throw IVITError("Segmentor requires a non-null segmentation model");

    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto &shape = input_info[0].shape;
        const bool nhwc = model_->load_config().input_layout == "nhwc";
        const int h = nhwc ? 1 : 2, w = nhwc ? 2 : 3;
        if (shape.size() == 4 && shape[h] > 0 && shape[w] > 0) {
            input_size_ = cv::Size(static_cast<int>(shape[w]), static_cast<int>(shape[h]));
        } else {
            input_size_ = cv::Size(512, 512);
        }
    } else {
        input_size_ = cv::Size(512, 512);
    }

    labels_ = model_->labels();

    init_default_colormap();
}

Results Segmentor::predict(const cv::Mat &image) { return model_->predict(image); }

Results Segmentor::predict(const std::string &image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw IVITError("Failed to load image: " + image_path);
    }
    return predict(image);
}

std::vector<Results> Segmentor::predict_batch(const std::vector<cv::Mat> &images) {
    std::vector<Results> results_list;
    results_list.reserve(images.size());

    for (const auto &image : images) {
        results_list.push_back(predict(image));
    }

    return results_list;
}

void Segmentor::init_default_colormap() {
    // Generate distinct colors for each class
    int num_classes = static_cast<int>(labels_.size());

    for (int i = 0; i < std::max(num_classes, 256); i++) {
        // Use HSV color space for distinct colors
        float hue = static_cast<float>(i) / std::max(num_classes, 256) * 180.0f;

        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 200));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);

        colormap_[i] = rgb.at<cv::Vec3b>(0, 0);
    }

    // Special color for background (class 0)
    colormap_[0] = cv::Vec3b(0, 0, 0);

    // Use standard colors for common classes if using VOC
    if (labels_.size() == 21) {
        // VOC colormap
        std::vector<cv::Vec3b> voc_colors = {
            {0, 0, 0},       // background
            {128, 0, 0},     // aeroplane
            {0, 128, 0},     // bicycle
            {128, 128, 0},   // bird
            {0, 0, 128},     // boat
            {128, 0, 128},   // bottle
            {0, 128, 128},   // bus
            {128, 128, 128}, // car
            {64, 0, 0},      // cat
            {192, 0, 0},     // chair
            {64, 128, 0},    // cow
            {192, 128, 0},   // diningtable
            {64, 0, 128},    // dog
            {192, 0, 128},   // horse
            {64, 128, 128},  // motorbike
            {192, 128, 128}, // person
            {0, 64, 0},      // pottedplant
            {128, 64, 0},    // sheep
            {0, 192, 0},     // sofa
            {128, 192, 0},   // train
            {0, 64, 128}     // tvmonitor
        };

        for (size_t i = 0; i < voc_colors.size(); i++) {
            colormap_[static_cast<int>(i)] = voc_colors[i];
        }
    }
}

} // namespace vision
} // namespace ivit

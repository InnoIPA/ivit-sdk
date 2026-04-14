/**
 * @file segmentor.cpp
 * @brief Semantic segmentor implementation
 */

#include "ivit/vision/segmentor.hpp"
#include "ivit/runtime/runtime.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>

namespace ivit {
namespace vision {

// ============================================================================
// Common segmentation class labels
// ============================================================================

// PASCAL VOC labels (21 classes)
static const std::vector<std::string> VOC_LABELS = {
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

// Cityscapes labels (19 classes)
static const std::vector<std::string> CITYSCAPES_LABELS = {
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
    "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
};

// ADE20K labels (150 classes) - abbreviated
static const std::vector<std::string> ADE20K_LABELS = {
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair"
    // ... 150 classes total
};

// ============================================================================
// Segmentor implementation
// ============================================================================

Segmentor::Segmentor(
    const std::string& model_path,
    const std::string& device,
    const LoadConfig& config
) {
    // Create load config with segmentation task hint
    LoadConfig load_config = config;
    load_config.device = device;
    load_config.task = "segmentation";

    // Load model
    model_ = ModelManager::instance().load(model_path, load_config);

    // Get input size from model (guard against dynamic dims: -1)
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto& shape = input_info[0].shape;
        if (shape.size() >= 4) {
            int w = static_cast<int>(shape[3]);
            int h = static_cast<int>(shape[2]);
            input_size_ = cv::Size(
                w > 0 ? w : 512,
                h > 0 ? h : 512
            );
        } else {
            input_size_ = cv::Size(512, 512);
        }
    } else {
        input_size_ = cv::Size(512, 512);
    }

    // Load labels
    labels_ = model_->labels();
    if (labels_.empty()) {
        namespace fs = std::filesystem;
        fs::path model_dir = fs::path(model_path).parent_path();
        fs::path labels_path = model_dir / "labels.txt";

        if (fs::exists(labels_path)) {
            std::ifstream file(labels_path);
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) {
                    labels_.push_back(line);
                }
            }
        }

        // Determine default labels based on model name
        std::string name_lower = model_path;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

        if (labels_.empty()) {
            if (name_lower.find("cityscapes") != std::string::npos) {
                labels_ = CITYSCAPES_LABELS;
            } else if (name_lower.find("ade") != std::string::npos) {
                labels_ = ADE20K_LABELS;
            } else {
                labels_ = VOC_LABELS;
            }
        }
    }

    // Initialize colormap
    init_default_colormap();
}

Segmentor::Segmentor(std::shared_ptr<Model> model)
    : model_(model)
{
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto& shape = input_info[0].shape;
        if (shape.size() >= 4) {
            int w = static_cast<int>(shape[3]);
            int h = static_cast<int>(shape[2]);
            input_size_ = cv::Size(
                w > 0 ? w : 512,
                h > 0 ? h : 512
            );
        } else {
            input_size_ = cv::Size(512, 512);
        }
    } else {
        input_size_ = cv::Size(512, 512);
    }

    labels_ = model_->labels();
    if (labels_.empty()) {
        labels_ = VOC_LABELS;
    }

    init_default_colormap();
}

Results Segmentor::predict(const cv::Mat& image) {
    Results results;
    results.image_size = image.size();

    auto start = std::chrono::high_resolution_clock::now();

    // Preprocess
    cv::Mat processed = preprocess(image);

    // Create input tensor
    auto input_info = model_->input_info();
    if (input_info.empty()) {
        throw InferenceError("Model has no inputs");
    }

    const auto& info = input_info[0];
    Shape input_shape = info.shape;
    for (auto& d : input_shape) { if (d < 0) d = 1; }
    Tensor input(input_shape, DataType::Float32);
    input.set_name(info.name);

    std::memcpy(input.data(), processed.data, input.byte_size());

    // Run inference
    std::map<std::string, Tensor> inputs;
    inputs[info.name] = std::move(input);

    auto outputs = model_->infer_raw(inputs);

    // Postprocess
    results = postprocess(outputs, image.size());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    results.inference_time_ms = duration.count() / 1000.0f;
    results.device_used = model_->device();
    results.image_size = image.size();

    return results;
}

Results Segmentor::predict(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw IVITError("Failed to load image: " + image_path);
    }
    return predict(image);
}

std::vector<Results> Segmentor::predict_batch(const std::vector<cv::Mat>& images) {
    std::vector<Results> results_list;
    results_list.reserve(images.size());

    for (const auto& image : images) {
        results_list.push_back(predict(image));
    }

    return results_list;
}

cv::Mat Segmentor::preprocess(const cv::Mat& image) {
    cv::Mat processed;

    // Resize to input size
    cv::resize(image, processed, input_size_);

    // Convert BGR to RGB
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    // Convert to float and normalize
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);

    // Apply ImageNet normalization
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);

    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }

    cv::Mat normalized;
    cv::merge(channels, normalized);

    // Convert to NCHW format
    int h = normalized.rows;
    int w = normalized.cols;
    int c = normalized.channels();

    cv::Mat nchw(1, c * h * w, CV_32F);
    float* ptr = nchw.ptr<float>();

    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                ptr[ch * h * w + y * w + x] = normalized.at<cv::Vec3f>(y, x)[ch];
            }
        }
    }

    return nchw;
}

Results Segmentor::postprocess(
    const std::map<std::string, Tensor>& outputs,
    const cv::Size& orig_size
) {
    Results results;

    if (outputs.empty()) {
        return results;
    }

    const Tensor& output = outputs.begin()->second;
    const float* data = output.data_ptr<float>();
    const auto& shape = output.shape();

    // Output shape: [1, num_classes, height, width] or [1, height, width]
    int num_classes, height, width;

    if (shape.size() == 4) {
        num_classes = static_cast<int>(shape[1]);
        height = static_cast<int>(shape[2]);
        width = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        // Already argmax'd
        num_classes = 1;
        height = static_cast<int>(shape[1]);
        width = static_cast<int>(shape[2]);
    } else {
        throw InferenceError("Unexpected output shape for segmentation");
    }

    // Create segmentation mask
    cv::Mat mask(height, width, CV_32S);

    if (num_classes > 1) {
        // Argmax over classes
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int best_class = 0;
                float best_score = -std::numeric_limits<float>::max();

                for (int c = 0; c < num_classes; c++) {
                    float score = data[c * height * width + y * width + x];
                    if (score > best_score) {
                        best_score = score;
                        best_class = c;
                    }
                }

                mask.at<int>(y, x) = best_class;
            }
        }
    } else {
        // Already class indices
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mask.at<int>(y, x) = static_cast<int>(data[y * width + x]);
            }
        }
    }

    // Resize mask to original size
    if (mask.size() != orig_size) {
        cv::resize(mask, mask, orig_size, 0, 0, cv::INTER_NEAREST);
    }

    results.segmentation_mask = mask;

    return results;
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
    if (labels_.size() == VOC_LABELS.size()) {
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

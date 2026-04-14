/**
 * @file classifier.cpp
 * @brief Image classifier implementation
 */

#include "ivit/vision/classifier.hpp"
#include "ivit/runtime/runtime.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>

namespace ivit {
namespace vision {

// ============================================================================
// Default ImageNet labels (first 10 for demo, full list would be loaded)
// ============================================================================
static const std::vector<std::string> IMAGENET_LABELS = {
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
    "electric_ray", "stingray", "cock", "hen", "ostrich"
    // ... 1000 classes total
};

// ============================================================================
// Classifier implementation
// ============================================================================

Classifier::Classifier(
    const std::string& model_path,
    const std::string& device,
    const LoadConfig& config
) {
    // Create load config with classification task hint
    LoadConfig load_config = config;
    load_config.device = device;
    load_config.task = "classification";

    // Load model
    model_ = ModelManager::instance().load(model_path, load_config);

    // Get input size from model (guard against dynamic dims: -1)
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto& shape = input_info[0].shape;
        if (shape.size() >= 4) {
            // NCHW format
            int w = static_cast<int>(shape[3]);
            int h = static_cast<int>(shape[2]);
            input_size_ = cv::Size(
                w > 0 ? w : 224,
                h > 0 ? h : 224
            );
        } else {
            input_size_ = cv::Size(224, 224);  // Default
        }
    } else {
        input_size_ = cv::Size(224, 224);
    }

    // Set default normalization (ImageNet)
    mean_ = {0.485f, 0.456f, 0.406f};
    std_ = {0.229f, 0.224f, 0.225f};

    // Try to load labels
    // Priority: 1) model built-in  2) {model_stem}.txt  3) labels.txt  4) IMAGENET_LABELS
    labels_ = model_->labels();
    if (labels_.empty()) {
        namespace fs = std::filesystem;
        fs::path model_file = fs::path(model_path);
        fs::path model_dir = model_file.parent_path();

        // Try model-specific label file (e.g. resnet18.onnx -> resnet18.txt)
        fs::path stem_labels_path = model_file;
        stem_labels_path.replace_extension(".txt");

        // Try generic labels.txt in same directory
        fs::path generic_labels_path = model_dir / "labels.txt";

        for (const auto& lpath : {stem_labels_path, generic_labels_path}) {
            if (fs::exists(lpath)) {
                std::ifstream file(lpath);
                std::string line;
                while (std::getline(file, line)) {
                    if (!line.empty()) {
                        labels_.push_back(line);
                    }
                }
                if (!labels_.empty()) break;
            }
        }

        // Fallback to default labels
        if (labels_.empty()) {
            labels_ = IMAGENET_LABELS;
        }
    }
}

Classifier::Classifier(std::shared_ptr<Model> model)
    : model_(model)
{
    // Get input size from model (guard against dynamic dims: -1)
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto& shape = input_info[0].shape;
        if (shape.size() >= 4) {
            int w = static_cast<int>(shape[3]);
            int h = static_cast<int>(shape[2]);
            input_size_ = cv::Size(
                w > 0 ? w : 224,
                h > 0 ? h : 224
            );
        } else {
            input_size_ = cv::Size(224, 224);
        }
    } else {
        input_size_ = cv::Size(224, 224);
    }

    mean_ = {0.485f, 0.456f, 0.406f};
    std_ = {0.229f, 0.224f, 0.225f};
    labels_ = model_->labels();
}

Results Classifier::predict(const cv::Mat& image, int top_k) {
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
    // Replace dynamic dims (-1) with concrete values (batch=1)
    Shape input_shape = info.shape;
    for (auto& d : input_shape) { if (d < 0) d = 1; }
    Tensor input(input_shape, DataType::Float32);
    input.set_name(info.name);

    // Copy preprocessed data to tensor
    std::memcpy(input.data(), processed.data, input.byte_size());

    // Run inference
    std::map<std::string, Tensor> inputs;
    inputs[info.name] = std::move(input);

    auto outputs = model_->infer_raw(inputs);

    // Postprocess
    results = postprocess(outputs, top_k);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    results.inference_time_ms = duration.count() / 1000.0f;
    results.device_used = model_->device();
    results.image_size = image.size();

    return results;
}

Results Classifier::predict(const std::string& image_path, int top_k) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw IVITError("Failed to load image: " + image_path);
    }
    return predict(image, top_k);
}

std::vector<Results> Classifier::predict_batch(
    const std::vector<cv::Mat>& images,
    int top_k
) {
    if (images.empty()) {
        return {};
    }

    std::vector<Results> results_list;
    results_list.reserve(images.size());

    // Check if model supports dynamic batch
    auto input_info = model_->input_info();
    bool supports_batch = false;
    if (!input_info.empty()) {
        int64_t batch_dim = input_info[0].shape[0];
        supports_batch = (batch_dim == -1 || batch_dim == 0);
    }

    if (supports_batch && images.size() > 1) {
        // True batch inference
        auto start = std::chrono::high_resolution_clock::now();

        // Preprocess all images
        std::vector<cv::Mat> processed_images;
        processed_images.reserve(images.size());
        for (const auto& img : images) {
            processed_images.push_back(preprocess(img));
        }

        // Create batch input tensor
        const auto& info = input_info[0];
        auto shape = info.shape;
        shape[0] = static_cast<int64_t>(images.size());  // Set batch dimension

        Tensor batch_input(shape, DataType::Float32);
        batch_input.set_name(info.name);

        // Copy preprocessed data into batch tensor
        size_t single_size = batch_input.byte_size() / images.size();
        char* batch_ptr = static_cast<char*>(batch_input.data());

        for (size_t i = 0; i < processed_images.size(); i++) {
            std::memcpy(batch_ptr + i * single_size,
                       processed_images[i].data,
                       single_size);
        }

        // Run batch inference
        std::map<std::string, Tensor> inputs;
        inputs[info.name] = std::move(batch_input);
        auto outputs = model_->infer_raw(inputs);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float total_time = duration.count() / 1000.0f;
        float avg_time = total_time / images.size();

        // Batch postprocess: extract each sample's output
        if (!outputs.empty()) {
            const Tensor& output = outputs.begin()->second;
            const std::string& output_name = outputs.begin()->first;
            int64_t num_classes = output.shape().back();

            for (size_t i = 0; i < images.size(); i++) {
                // Create single sample output tensor
                std::vector<int64_t> sample_shape = output.shape();
                sample_shape[0] = 1;
                Tensor sample_output(sample_shape, output.dtype());
                sample_output.set_name(output_name);

                // Copy data for this sample
                size_t sample_size = sample_output.byte_size();
                const char* src = static_cast<const char*>(output.data()) + i * sample_size;
                std::memcpy(sample_output.data(), src, sample_size);

                // Postprocess
                std::map<std::string, Tensor> sample_outputs;
                sample_outputs[output_name] = std::move(sample_output);

                Results results = postprocess(sample_outputs, top_k);
                results.inference_time_ms = avg_time;
                results.device_used = model_->device();
                results.image_size = images[i].size();
                results_list.push_back(std::move(results));
            }
        }

        return results_list;
    }

    // Fallback: sequential processing for models with fixed batch size
    for (const auto& image : images) {
        results_list.push_back(predict(image, top_k));
    }

    return results_list;
}

cv::Mat Classifier::preprocess(const cv::Mat& image) {
    cv::Mat processed;

    // Resize to input size
    cv::resize(image, processed, input_size_);

    // Convert BGR to RGB
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    // Convert to float [0, 1]
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);

    // Apply normalization (ImageNet mean/std)
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);

    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }

    // Merge and convert to NCHW format
    cv::Mat normalized;
    cv::merge(channels, normalized);

    // Create NCHW tensor data
    int h = normalized.rows;
    int w = normalized.cols;
    int c = normalized.channels();

    cv::Mat nchw(1, c * h * w, CV_32F);
    float* ptr = nchw.ptr<float>();

    // Convert HWC to CHW
    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                ptr[ch * h * w + y * w + x] = normalized.at<cv::Vec3f>(y, x)[ch];
            }
        }
    }

    return nchw;
}

Results Classifier::postprocess(
    const std::map<std::string, Tensor>& outputs,
    int top_k
) {
    Results results;

    if (outputs.empty()) {
        return results;
    }

    // Get first output tensor
    const Tensor& output = outputs.begin()->second;
    const float* data = output.data_ptr<float>();
    int64_t num_classes = output.numel();

    // Apply softmax
    std::vector<float> probs(num_classes);
    float max_val = *std::max_element(data, data + num_classes);
    float sum = 0.0f;

    for (int64_t i = 0; i < num_classes; i++) {
        probs[i] = std::exp(data[i] - max_val);
        sum += probs[i];
    }

    for (int64_t i = 0; i < num_classes; i++) {
        probs[i] /= sum;
    }

    // Find top-k predictions
    std::vector<std::pair<float, int>> scores;
    scores.reserve(num_classes);

    for (int64_t i = 0; i < num_classes; i++) {
        scores.emplace_back(probs[i], static_cast<int>(i));
    }

    // Partial sort to get top-k
    int k = std::min(top_k, static_cast<int>(num_classes));
    std::partial_sort(
        scores.begin(),
        scores.begin() + k,
        scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Create classification results
    for (int i = 0; i < k; i++) {
        auto [score, class_id] = scores[i];

        ClassificationResult cls;
        cls.class_id = class_id;
        cls.score = score;

        if (class_id < static_cast<int>(labels_.size())) {
            cls.label = labels_[class_id];
        } else {
            cls.label = "class_" + std::to_string(class_id);
        }

        results.classifications.push_back(cls);
    }

    return results;
}

} // namespace vision
} // namespace ivit

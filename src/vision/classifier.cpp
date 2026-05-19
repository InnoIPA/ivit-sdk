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
#include <sstream>
#include <limits>

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

    // Hailo backend bakes ImageNet normalization into the HAR (via the
    // Hailo "normalization" model-script command at compile time), because
    // the chip only accepts UINT8 [0, 255] input — float mean/std cannot
    // survive the host-side → uint8 conversion. Skip host-side normalize
    // here so we don't double-apply it.
    if (model_ && model_->backend() == BackendType::HailoRT) {
        mean_ = {0.0f, 0.0f, 0.0f};
        std_ = {1.0f, 1.0f, 1.0f};
    }

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

    // Auto-load a CPU classifier head for Hailo backbone-only HEFs (e.g.
    // mambavision-t whose SSM/Transformer can't translate to Hailo). Same
    // pattern as Segmentor for DeepLabV3-MobileNetV3. See Bug #15.
    if (model_ && model_->backend() == BackendType::HailoRT) {
        namespace fs = std::filesystem;
        fs::path model_fs(model_path);
        fs::path head_path =
            model_fs.parent_path() /
            (model_fs.stem().string() + "_head.onnx");
        if (fs::exists(head_path)) {
            LoadConfig head_cfg;
            head_cfg.device = "cpu";
            head_cfg.task = "classification";
            cpu_head_ = ModelManager::instance().load(
                head_path.string(), head_cfg);
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

    // Hailo backend bakes ImageNet normalization into the HAR (via the
    // Hailo "normalization" model-script command at compile time), because
    // the chip only accepts UINT8 [0, 255] input — float mean/std cannot
    // survive the host-side → uint8 conversion. Skip host-side normalize
    // here so we don't double-apply it.
    if (model_ && model_->backend() == BackendType::HailoRT) {
        mean_ = {0.0f, 0.0f, 0.0f};
        std_ = {1.0f, 1.0f, 1.0f};
    }
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
    // Hailo backend may emit either a final logits vector OR a backbone
    // feature map (when SSM/Transformer head was cut from the HEF). Route
    // through the dedicated branch so it can run the CPU head if needed.
    if (model_ && model_->backend() == BackendType::HailoRT) {
        return postprocess_hailo(outputs, top_k);
    }

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

Results Classifier::postprocess_hailo(
    const std::map<std::string, Tensor>& outputs,
    int top_k
) {
    Results results;
    if (outputs.empty()) {
        return results;
    }

    const Tensor& output = outputs.begin()->second;
    const float* data = output.data_ptr<float>();
    const auto& shape = output.shape();
    int64_t numel = output.numel();

    // Three possibilities:
    //   (a) Final logits — shape is [N] or [1, N] with N == num_classes
    //       (e.g. ResNet50, EfficientNet-B0 full HEF). Run softmax + top-k.
    //   (b) Backbone feature map — multi-D shape, num elements != labels
    //       (e.g. mambavision_t with SSM/Transformer cut to CPU). Need to
    //       run cpu_head_ to produce final logits.
    //   (c) Backbone feature map but no cpu_head_ available -> hard error.
    const int n_labels = static_cast<int>(labels_.size());
    const bool looks_like_logits =
        (shape.size() == 1 && static_cast<int>(numel) == n_labels) ||
        (shape.size() == 2 && shape[0] == 1 &&
         static_cast<int>(shape[1]) == n_labels);

    if (looks_like_logits) {
        // Reuse the standard softmax + top-k path.
        std::vector<float> probs(n_labels);
        float max_val = *std::max_element(data, data + n_labels);
        float sum = 0.0f;
        for (int i = 0; i < n_labels; i++) {
            probs[i] = std::exp(data[i] - max_val);
            sum += probs[i];
        }
        for (int i = 0; i < n_labels; i++) probs[i] /= sum;

        std::vector<std::pair<float, int>> scores;
        scores.reserve(n_labels);
        for (int i = 0; i < n_labels; i++) {
            scores.emplace_back(probs[i], i);
        }
        int k = std::min(top_k, n_labels);
        std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int i = 0; i < k; i++) {
            auto [score, class_id] = scores[i];
            ClassificationResult cls;
            cls.class_id = class_id;
            cls.score = score;
            cls.label = (class_id < n_labels)
                ? labels_[class_id]
                : "class_" + std::to_string(class_id);
            results.classifications.push_back(cls);
        }
        return results;
    }

    // Backbone feature map path.
    if (!cpu_head_) {
        std::ostringstream msg;
        msg << "Hailo HEF returned a feature map of " << numel << " elems "
               "(num_classes=" << n_labels << ") instead of class logits. "
               "This looks like a backbone-only HEF and no CPU classifier "
               "head was found next to the model (expected "
               "<model_stem>_head.onnx). For mambavision-t use "
               "compile_mambavision_t.py + extract_mambavision_head.py. "
               "See docs/development/hailo-compile-bug-log.md Bug #15.";
        throw InferenceError(msg.str());
    }

    // Hailo NHWC [H, W, C] (or [1, H, W, C]) -> NCHW [1, C, H, W] for ONNX.
    int H, W, C;
    if (shape.size() == 4 && shape[0] == 1) {
        H = static_cast<int>(shape[1]);
        W = static_cast<int>(shape[2]);
        C = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        H = static_cast<int>(shape[0]);
        W = static_cast<int>(shape[1]);
        C = static_cast<int>(shape[2]);
    } else {
        throw InferenceError(
            "Unsupported Hailo backbone output rank for classifier hybrid");
    }

    auto head_in_info = cpu_head_->input_info();
    if (head_in_info.empty()) {
        throw InferenceError("CPU classifier head has no inputs");
    }
    Shape head_in_shape = {1, C, H, W};
    Tensor head_input(head_in_shape, DataType::Float32);
    head_input.set_name(head_in_info[0].name);
    float* dst = head_input.data_ptr<float>();
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                dst[c * H * W + y * W + x] = data[(y * W + x) * C + c];
            }
        }
    }

    std::map<std::string, Tensor> head_inputs;
    head_inputs[head_in_info[0].name] = std::move(head_input);
    auto head_outputs = cpu_head_->infer_raw(head_inputs);

    const Tensor& head_out = head_outputs.begin()->second;
    const float* hdata = head_out.data_ptr<float>();
    int64_t hn = head_out.numel();

    std::vector<float> probs(hn);
    float max_val = *std::max_element(hdata, hdata + hn);
    float sum = 0.0f;
    for (int64_t i = 0; i < hn; i++) {
        probs[i] = std::exp(hdata[i] - max_val);
        sum += probs[i];
    }
    for (int64_t i = 0; i < hn; i++) probs[i] /= sum;

    std::vector<std::pair<float, int>> scores;
    scores.reserve(hn);
    for (int64_t i = 0; i < hn; i++) {
        scores.emplace_back(probs[i], static_cast<int>(i));
    }
    int k = std::min(top_k, static_cast<int>(hn));
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int i = 0; i < k; i++) {
        auto [score, class_id] = scores[i];
        ClassificationResult cls;
        cls.class_id = class_id;
        cls.score = score;
        cls.label = (class_id < static_cast<int>(labels_.size()))
            ? labels_[class_id]
            : "class_" + std::to_string(class_id);
        results.classifications.push_back(cls);
    }
    return results;
}

} // namespace vision
} // namespace ivit

/**
 * @file model.cpp
 * @brief Model class implementation
 */

#include "ivit/core/model.hpp"
#include "ivit/core/runtime_config.hpp"
#include "ivit/core/video_source.hpp"
#include "ivit/runtime/runtime.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <thread>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace ivit {

// ============================================================================
// InferenceModel - Concrete model implementation using runtime backends
// ============================================================================

// Letterbox bookkeeping needed to map YOLOX grid-relative box coordinates
// back to original image pixels.
struct PreprocessInfo {
    float scale = 1.0f;
    int pad_w = 0;
    int pad_h = 0;
};

class InferenceModel : public Model {
public:
    InferenceModel(
        std::shared_ptr<IRuntime> runtime,
        void* handle,
        const std::string& name,
        TaskType task,
        const std::string& device
    )
        : runtime_(runtime)
        , handle_(handle)
        , name_(name)
        , task_(task)
        , device_(device)
    {
        input_info_ = runtime_->get_input_info(handle_);
        output_info_ = runtime_->get_output_info(handle_);

        // YOLOX uses a different preprocessing convention (BGR, [0,255],
        // letterbox) and output encoding (grid-relative, exp-scaled) than
        // the generic normalized-[0,1] assumption below; detect it from the
        // model name the same way vision::Detector does.
        std::string name_lower = name_;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        model_type_ = (name_lower.find("yolox") != std::string::npos) ? "yolox" : "";
    }

    ~InferenceModel() override {
        if (runtime_ && handle_) {
            runtime_->unload_model(handle_);
        }
    }

    // Properties
    std::string name() const override { return name_; }
    TaskType task() const override { return task_; }
    std::string device() const override { return device_; }
    BackendType backend() const override { return runtime_->type(); }
    std::vector<TensorInfo> input_info() const override { return input_info_; }
    std::vector<TensorInfo> output_info() const override { return output_info_; }

    size_t memory_usage() const override {
        // Estimate based on tensor sizes
        size_t total = 0;
        for (const auto& info : input_info_) {
            total += info.byte_size();
        }
        for (const auto& info : output_info_) {
            total += info.byte_size();
        }
        return total;
    }

    std::vector<std::string> labels() const override { return labels_; }

    void set_labels(const std::vector<std::string>& labels) {
        labels_ = labels;
    }

    // Inference
    Results predict(
        const cv::Mat& image,
        const InferConfig& config
    ) override {
        Results results;
        results.image_size = image.size();
        results.device_used = device_;

        auto start = std::chrono::high_resolution_clock::now();

        // Preprocess image
        PreprocessInfo pp;
        auto inputs = preprocess(image, pp);

        // Run inference
        auto outputs = runtime_->infer(handle_, inputs);

        // Postprocess results
        postprocess(outputs, image.size(), config, pp, results);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        results.inference_time_ms = duration.count() / 1000.0f;

        return results;
    }

    std::vector<Results> predict_batch(
        const std::vector<cv::Mat>& images,
        const InferConfig& config
    ) override {
        if (images.empty()) {
            return {};
        }

        std::vector<Results> results_list;
        results_list.reserve(images.size());

        // Check if model supports dynamic batch
        bool supports_batch = false;
        if (!input_info_.empty()) {
            int64_t batch_dim = input_info_[0].shape[0];
            supports_batch = (batch_dim == -1 || batch_dim == 0);
        }

        if (supports_batch && images.size() > 1) {
            // True batch inference
            auto start = std::chrono::high_resolution_clock::now();

            // Collect original sizes
            std::vector<cv::Size> orig_sizes;
            orig_sizes.reserve(images.size());
            for (const auto& img : images) {
                orig_sizes.push_back(img.size());
            }

            // Batch preprocess: preprocess each image and concatenate
            std::vector<std::map<std::string, Tensor>> all_inputs;
            std::vector<PreprocessInfo> all_pp;
            all_inputs.reserve(images.size());
            all_pp.reserve(images.size());
            for (const auto& img : images) {
                PreprocessInfo pp;
                all_inputs.push_back(preprocess(img, pp));
                all_pp.push_back(pp);
            }

            // Merge tensors into batch
            const auto& info = input_info_[0];
            auto shape = info.shape;
            shape[0] = static_cast<int64_t>(images.size());  // Set batch dimension

            Tensor batched_input(shape, info.dtype);
            batched_input.set_name(info.name);
            batched_input.set_layout(Layout::NCHW);

            // Copy each preprocessed image into the batch tensor
            size_t single_size = batched_input.byte_size() / images.size();
            char* batch_ptr = static_cast<char*>(batched_input.data());

            for (size_t i = 0; i < all_inputs.size(); i++) {
                const auto& single_tensor = all_inputs[i].begin()->second;
                std::memcpy(batch_ptr + i * single_size, single_tensor.data(), single_size);
            }

            // Single batch inference call
            std::map<std::string, Tensor> batch_inputs;
            batch_inputs[info.name] = std::move(batched_input);
            auto outputs = runtime_->infer(handle_, batch_inputs);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            float total_time = duration.count() / 1000.0f;
            float avg_time = total_time / images.size();

            // Batch postprocess: extract each sample's output and process
            for (size_t i = 0; i < images.size(); i++) {
                Results results;
                results.image_size = orig_sizes[i];
                results.device_used = device_;

                // Extract single sample outputs from batch
                std::map<std::string, Tensor> sample_outputs;
                for (const auto& [name, tensor] : outputs) {
                    // Create tensor for single sample
                    auto out_shape = tensor.shape();
                    out_shape[0] = 1;  // Single sample
                    Tensor sample_tensor(out_shape, tensor.dtype());
                    sample_tensor.set_name(name);

                    // Copy data for this sample
                    size_t sample_size = sample_tensor.byte_size();
                    const char* src = static_cast<const char*>(tensor.data()) + i * sample_size;
                    std::memcpy(sample_tensor.data(), src, sample_size);

                    sample_outputs[name] = std::move(sample_tensor);
                }

                postprocess(sample_outputs, orig_sizes[i], config, all_pp[i], results);
                results.inference_time_ms = avg_time;
                results_list.push_back(std::move(results));
            }

            return results_list;
        }

        // Fallback: sequential processing for models with fixed batch size
        for (const auto& image : images) {
            results_list.push_back(predict(image, config));
        }

        return results_list;
    }

    std::map<std::string, Tensor> infer_raw(
        const std::map<std::string, Tensor>& inputs
    ) override {
        return runtime_->infer(handle_, inputs);
    }

    void warmup(int iterations) override {
        if (input_info_.empty()) return;

        // Create dummy input
        const auto& info = input_info_[0];
        Tensor dummy(info.shape, info.dtype);
        dummy.set_name(info.name);

        std::map<std::string, Tensor> inputs;
        inputs[info.name] = std::move(dummy);

        for (int i = 0; i < iterations; i++) {
            runtime_->infer(handle_, inputs);
        }
    }

private:
    std::map<std::string, Tensor> preprocess(const cv::Mat& image, PreprocessInfo& pp) {
        if (input_info_.empty()) {
            throw InferenceError("Model has no inputs");
        }

        const auto& info = input_info_[0];
        const auto& shape = info.shape;

        // Assume NCHW layout — use int to match cv::Size and avoid
        // int64_t→int narrowing issues across OpenCV ABI boundaries
        int channels = static_cast<int>(shape.size() > 1 ? shape[1] : 3);
        int height   = static_cast<int>(shape.size() > 2 ? shape[2] : 224);
        int width    = static_cast<int>(shape.size() > 3 ? shape[3] : 224);

        // Guard against dynamic dimensions (-1) or invalid values
        if (height <= 0) height = 224;
        if (width  <= 0) width  = 224;
        if (channels <= 0) channels = 3;

        bool is_yolox = (model_type_ == "yolox");
        cv::Mat float_img;

        if (is_yolox) {
            // Letterbox resize (preserve aspect ratio, pad with gray 114)
            int orig_w = image.cols;
            int orig_h = image.rows;
            pp.scale = std::min(
                static_cast<float>(width) / orig_w,
                static_cast<float>(height) / orig_h
            );
            int new_w = static_cast<int>(orig_w * pp.scale);
            int new_h = static_cast<int>(orig_h * pp.scale);
            pp.pad_w = (width - new_w) / 2;
            pp.pad_h = (height - new_h) / 2;

            cv::Mat resized;
            cv::resize(image, resized, cv::Size(new_w, new_h));

            cv::Mat letterbox(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
            resized.copyTo(letterbox(cv::Rect(pp.pad_w, pp.pad_h, new_w, new_h)));

            // YOLOX expects BGR, [0, 255] — no color conversion, no normalize
            letterbox.convertTo(float_img, CV_32F);
        } else {
            pp.scale = 1.0f;
            pp.pad_w = 0;
            pp.pad_h = 0;

            // Resize image
            cv::Mat resized;
            cv::Size target_size(width, height);
            cv::resize(image, resized, target_size);

            // Convert to float and normalize
            resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

            // Convert BGR to RGB if needed
            if (channels == 3 && float_img.channels() == 3) {
                cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);
            }
        }

        // Create tensor
        Tensor input(shape, DataType::Float32);
        input.set_name(info.name);
        input.set_layout(Layout::NCHW);

        // Copy data in NCHW format
        float* ptr = input.data_ptr<float>();
        std::vector<cv::Mat> channels_vec;
        cv::split(float_img, channels_vec);

        for (int c = 0; c < channels; c++) {
            cv::Mat channel = channels_vec[c];
            std::memcpy(
                ptr + c * height * width,
                channel.data,
                static_cast<size_t>(height) * width * sizeof(float)
            );
        }

        std::map<std::string, Tensor> inputs;
        inputs[info.name] = std::move(input);
        return inputs;
    }

    void postprocess(
        const std::map<std::string, Tensor>& outputs,
        cv::Size image_size,
        const InferConfig& config,
        const PreprocessInfo& pp,
        Results& results
    ) {
        // Route to appropriate postprocessor based on task type
        switch (task_) {
            case TaskType::Classification:
                postprocess_classification(outputs, config, results);
                break;
            case TaskType::Detection:
                postprocess_detection(outputs, image_size, config, pp, results);
                break;
            case TaskType::Segmentation:
                postprocess_segmentation(outputs, image_size, config, results);
                break;
            default:
                // Just store raw outputs
                break;
        }
    }

    void postprocess_classification(
        const std::map<std::string, Tensor>& outputs,
        const InferConfig& config,
        Results& results
    ) {
        if (outputs.empty()) return;

        // Get first output (assuming single output)
        const Tensor& output = outputs.begin()->second;
        const float* data = output.data_ptr<float>();
        int64_t num_classes = output.numel();

        // Find top-k predictions
        std::vector<std::pair<float, int>> scores;
        scores.reserve(num_classes);

        for (int i = 0; i < num_classes; i++) {
            scores.emplace_back(data[i], i);
        }

        // Sort by score descending
        std::sort(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        // Apply softmax normalization if needed
        float max_score = scores[0].first;
        float sum = 0;
        for (auto& [score, idx] : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        for (auto& [score, idx] : scores) {
            score /= sum;
        }

        // Store results
        for (size_t i = 0; i < std::min(scores.size(), size_t(10)); i++) {
            auto [score, class_id] = scores[i];

            if (score < config.conf_threshold) break;

            ClassificationResult cls;
            cls.class_id = class_id;
            cls.score = score;
            cls.label = (class_id < labels_.size()) ?
                        labels_[class_id] :
                        "class_" + std::to_string(class_id);

            results.classifications.push_back(cls);
        }
    }

    void postprocess_detection(
        const std::map<std::string, Tensor>& outputs,
        cv::Size image_size,
        const InferConfig& config,
        const PreprocessInfo& pp,
        Results& results
    ) {
        if (outputs.empty()) return;

        // This is a simplified YOLO-style postprocessor
        // Real implementation would need model-specific handling

        const Tensor& output = outputs.begin()->second;
        const float* data = output.data_ptr<float>();
        const auto& shape = output.shape();

        // YOLO output typically: [batch, num_anchors, 5+num_classes] or similar
        // This is a simplified version for demonstration

        int num_predictions = (shape.size() > 1) ? shape[1] : shape[0];
        int prediction_size = (shape.size() > 2) ? shape[2] : 6;  // x,y,w,h,conf,class

        bool is_yolox = (model_type_ == "yolox");

        // YOLOX grid decode table: strides [8, 16, 32], grid sizes derived
        // from the actual model input resolution (not hardcoded to 640).
        struct StrideLevel { int grid_w; int offset; int stride; };
        std::vector<StrideLevel> stride_levels;
        if (is_yolox) {
            int in_h = 416, in_w = 416;
            if (!input_info_.empty()) {
                const auto& in_shape = input_info_[0].shape;
                in_h = static_cast<int>(in_shape.size() > 2 ? in_shape[2] : in_h);
                in_w = static_cast<int>(in_shape.size() > 3 ? in_shape[3] : in_w);
            }
            static const int strides[] = {8, 16, 32};
            int cumulative = 0;
            for (int s : strides) {
                int gw = in_w / s;
                int gh = in_h / s;
                stride_levels.push_back({gw, cumulative, s});
                cumulative += gh * gw;
            }
        }

        std::vector<Detection> detections;

        for (int i = 0; i < num_predictions; i++) {
            const float* pred = data + i * prediction_size;

            float cx = pred[0];
            float cy = pred[1];
            float w  = pred[2];
            float h  = pred[3];
            float confidence = pred[4];

            // YOLOX: decode grid-relative predictions to pixel coordinates
            if (is_yolox) {
                int stride = stride_levels.back().stride;
                int grid_w = stride_levels.back().grid_w;
                int local_idx = i;
                for (size_t s = 0; s < stride_levels.size(); s++) {
                    int next_offset = (s + 1 < stride_levels.size())
                        ? stride_levels[s + 1].offset : num_predictions;
                    if (i < next_offset) {
                        stride = stride_levels[s].stride;
                        grid_w = stride_levels[s].grid_w;
                        local_idx = i - stride_levels[s].offset;
                        break;
                    }
                }
                int gx = local_idx % grid_w;
                int gy = local_idx / grid_w;
                cx = (cx + gx) * stride;
                cy = (cy + gy) * stride;
                w = std::exp(w) * stride;
                h = std::exp(h) * stride;
            }

            if (confidence < config.conf_threshold) continue;

            // Find best class
            int best_class = 0;
            float best_class_score = 0;
            for (int c = 5; c < prediction_size; c++) {
                if (pred[c] > best_class_score) {
                    best_class_score = pred[c];
                    best_class = c - 5;
                }
            }

            float final_score = confidence * best_class_score;
            if (final_score < config.conf_threshold) continue;

            // Check class filter
            if (!config.classes.empty()) {
                bool found = false;
                for (int cls : config.classes) {
                    if (cls == best_class) {
                        found = true;
                        break;
                    }
                }
                if (!found) continue;
            }

            float x1, y1, x2, y2;
            if (is_yolox) {
                // Remove letterbox padding and undo the preprocessing scale
                x1 = (cx - w / 2 - pp.pad_w) / pp.scale;
                y1 = (cy - h / 2 - pp.pad_h) / pp.scale;
                x2 = (cx + w / 2 - pp.pad_w) / pp.scale;
                y2 = (cy + h / 2 - pp.pad_h) / pp.scale;
            } else {
                // Normalized [0,1] xywh (generic assumption)
                float ncx = cx * image_size.width;
                float ncy = cy * image_size.height;
                float nw  = w  * image_size.width;
                float nh  = h  * image_size.height;
                x1 = ncx - nw / 2;
                y1 = ncy - nh / 2;
                x2 = ncx + nw / 2;
                y2 = ncy + nh / 2;
            }

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(image_size.width)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(image_size.height)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(image_size.width)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(image_size.height)));

            Detection det;
            det.bbox = BBox(x1, y1, x2, y2);
            det.class_id = best_class;
            det.confidence = final_score;
            det.label = (best_class < labels_.size()) ?
                        labels_[best_class] :
                        "class_" + std::to_string(best_class);

            detections.push_back(det);
        }

        // Apply NMS
        results.detections = apply_nms(detections, config.iou_threshold);

        // Limit to max detections
        if (results.detections.size() > config.max_detections) {
            results.detections.resize(config.max_detections);
        }
    }

    void postprocess_segmentation(
        const std::map<std::string, Tensor>& outputs,
        cv::Size image_size,
        const InferConfig& config,
        Results& results
    ) {
        if (outputs.empty()) return;

        const Tensor& output = outputs.begin()->second;
        const auto& shape = output.shape();

        // Assume output shape: [batch, num_classes, height, width]
        int num_classes = (shape.size() > 1) ? shape[1] : 1;
        int height = (shape.size() > 2) ? shape[2] : image_size.height;
        int width = (shape.size() > 3) ? shape[3] : image_size.width;

        // Create segmentation mask (argmax over classes)
        cv::Mat mask(height, width, CV_32S);
        const float* data = output.data_ptr<float>();

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

        // Resize to original image size
        if (mask.size() != image_size) {
            cv::resize(mask, mask, image_size, 0, 0, cv::INTER_NEAREST);
        }

        results.segmentation_mask = mask;
    }

    std::vector<Detection> apply_nms(
        std::vector<Detection>& detections,
        float iou_threshold
    ) {
        if (detections.empty()) {
            return {};
        }

        if (detections.size() == 1) {
            return detections;
        }

        // Group detections by class for per-class NMS
        std::unordered_map<int, std::vector<size_t>> class_indices;
        for (size_t i = 0; i < detections.size(); i++) {
            class_indices[detections[i].class_id].push_back(i);
        }

        std::vector<Detection> result;
        result.reserve(detections.size());

        // Process each class separately using OpenCV NMS
        for (auto& [class_id, indices] : class_indices) {
            if (indices.empty()) continue;

            // Prepare boxes and scores for this class
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            boxes.reserve(indices.size());
            scores.reserve(indices.size());

            for (size_t idx : indices) {
                const auto& det = detections[idx];
                boxes.emplace_back(
                    static_cast<int>(det.bbox.x1),
                    static_cast<int>(det.bbox.y1),
                    static_cast<int>(det.bbox.x2 - det.bbox.x1),
                    static_cast<int>(det.bbox.y2 - det.bbox.y1)
                );
                scores.push_back(det.confidence);
            }

            // Use OpenCV's optimized NMS
            std::vector<int> keep_indices;
            cv::dnn::NMSBoxes(boxes, scores, 0.0f, iou_threshold, keep_indices);

            // Collect kept detections
            for (int keep_idx : keep_indices) {
                result.push_back(detections[indices[keep_idx]]);
            }
        }

        // Sort final results by confidence
        std::sort(result.begin(), result.end(),
            [](const Detection& a, const Detection& b) {
                return a.confidence > b.confidence;
            });

        return result;
    }

    std::shared_ptr<IRuntime> runtime_;
    void* handle_;
    std::string name_;
    TaskType task_;
    std::string device_;
    std::vector<TensorInfo> input_info_;
    std::vector<TensorInfo> output_info_;
    std::vector<std::string> labels_;
    std::string model_type_;
};

// ============================================================================
// Model path-based predict
// ============================================================================

Results Model::predict(
    const std::string& image_path,
    const InferConfig& config
) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw IVITError("Failed to load image: " + image_path);
    }
    return predict(image, config);
}

// ============================================================================
// Async Inference implementations
// ============================================================================

std::future<Results> Model::predict_async(
    const cv::Mat& image,
    const InferConfig& config
) {
    // Capture image by value to ensure it remains valid
    cv::Mat image_copy = image.clone();

    return std::async(std::launch::async, [this, image_copy, config]() {
        return this->predict(image_copy, config);
    });
}

std::future<std::vector<Results>> Model::predict_batch_async(
    const std::vector<cv::Mat>& images,
    const InferConfig& config
) {
    // Copy images to ensure they remain valid
    std::vector<cv::Mat> images_copy;
    images_copy.reserve(images.size());
    for (const auto& img : images) {
        images_copy.push_back(img.clone());
    }

    return std::async(std::launch::async, [this, images_copy = std::move(images_copy), config]() {
        return this->predict_batch(images_copy, config);
    });
}

void Model::submit_inference(
    const cv::Mat& image,
    std::function<void(Results)> callback,
    const InferConfig& config
) {
    // Fire-and-forget: launch async task that calls callback
    cv::Mat image_copy = image.clone();

    ensure_executor();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push([this, image_copy, callback, config]() {
            try {
                Results results = this->predict(image_copy, config);
                if (callback) {
                    callback(std::move(results));
                }
            } catch (const std::exception& e) {
                std::cerr << "[iVIT] Async inference error: " << e.what() << std::endl;
            }
        });
    }
    queue_cv_.notify_one();
}

// ============================================================================
// Model destructor
// ============================================================================

Model::~Model() {
    shutdown_async();
}

// ============================================================================
// Concurrent Inference
// ============================================================================

std::vector<Results> Model::predict_concurrent(
    const std::vector<cv::Mat>& images,
    int max_concurrent,
    const InferConfig& config
) {
    if (images.empty()) return {};

    trigger_callback(CallbackEvent::BatchStart, 0.0, static_cast<int>(images.size()));

    std::vector<std::future<Results>> futures;
    futures.reserve(images.size());

    // Use mutex + condition variable to limit concurrency (C++17 compatible)
    std::mutex sem_mutex;
    std::condition_variable sem_cv;
    int active_count = 0;

    for (const auto& image : images) {
        cv::Mat img_copy = image.clone();
        futures.push_back(std::async(std::launch::async,
            [this, img_copy, &config, &sem_mutex, &sem_cv, &active_count, max_concurrent]() {
                {
                    std::unique_lock<std::mutex> lock(sem_mutex);
                    sem_cv.wait(lock, [&]() { return active_count < max_concurrent; });
                    ++active_count;
                }
                auto result = this->predict(img_copy, config);
                {
                    std::lock_guard<std::mutex> lock(sem_mutex);
                    --active_count;
                }
                sem_cv.notify_one();
                return result;
            }
        ));
    }

    std::vector<Results> results;
    results.reserve(images.size());
    for (auto& f : futures) {
        results.push_back(f.get());
    }

    trigger_callback(CallbackEvent::BatchEnd, 0.0, static_cast<int>(images.size()));
    return results;
}

// ============================================================================
// Async Executor
// ============================================================================

void Model::ensure_executor() {
    bool expected = false;
    if (!executor_running_.compare_exchange_strong(expected, true)) return;

    executor_thread_ = std::make_unique<std::thread>([this]() {
        while (executor_running_.load()) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this]() {
                    return !task_queue_.empty() || !executor_running_.load();
                });
                if (!executor_running_.load() && task_queue_.empty()) return;
                if (task_queue_.empty()) continue;
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
            task();
        }
    });
}

void Model::shutdown_async() {
    if (!executor_running_.load()) return;

    executor_running_.store(false);
    queue_cv_.notify_all();

    if (executor_thread_ && executor_thread_->joinable()) {
        executor_thread_->join();
    }
    executor_thread_.reset();
}

// ============================================================================
// Callback helpers
// ============================================================================

int Model::on(
    const std::string& event,
    std::function<void(const CallbackContext&)> callback,
    int priority
) {
    return callbacks_.register_callback(event, std::move(callback), priority);
}

bool Model::remove_callback(const std::string& event, int callback_id) {
    return callbacks_.unregister_callback(
        callback_event_from_string(event), callback_id);
}

int Model::remove_all_callbacks(const std::string& event) {
    return callbacks_.unregister_all(event);
}

void Model::trigger_callback(
    CallbackEvent event,
    double latency_ms,
    int batch_size,
    int frame_index,
    const Results* results
) {
    if (!callbacks_.has_callbacks(event)) return;

    CallbackContext ctx;
    ctx.event = event;
    ctx.model_name = name();
    ctx.device = device();
    ctx.latency_ms = latency_ms;
    ctx.batch_size = batch_size;
    ctx.frame_index = frame_index;
    ctx.results = results;

    callbacks_.trigger(ctx);
}

// ============================================================================
// Hardware Configuration (default implementations - overridden by backends)
// ============================================================================

void Model::configure_openvino(const OpenVINOConfig& /*config*/) {
    throw IVITError("OpenVINO configuration not supported by this model backend");
}

void Model::configure_tensorrt(const TensorRTConfig& /*config*/) {
    throw IVITError("TensorRT configuration not supported by this model backend");
}

void Model::configure_qnn(const QNNConfig& /*config*/) {
    throw IVITError("QNN configuration not supported by this model backend");
}

// ============================================================================
// TTA (Test-Time Augmentation)
// ============================================================================

static cv::Mat apply_augmentation(const cv::Mat& image, const std::string& aug) {
    if (aug == "original") {
        return image.clone();
    } else if (aug == "hflip") {
        cv::Mat flipped;
        cv::flip(image, flipped, 1);
        return flipped;
    } else if (aug == "vflip") {
        cv::Mat flipped;
        cv::flip(image, flipped, 0);
        return flipped;
    } else if (aug == "rotate90") {
        cv::Mat rotated;
        cv::rotate(image, rotated, cv::ROTATE_90_CLOCKWISE);
        return rotated;
    } else if (aug == "rotate180") {
        cv::Mat rotated;
        cv::rotate(image, rotated, cv::ROTATE_180);
        return rotated;
    } else if (aug == "rotate270") {
        cv::Mat rotated;
        cv::rotate(image, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        return rotated;
    } else if (aug == "scale_up") {
        cv::Mat scaled;
        cv::resize(image, scaled, cv::Size(), 1.2, 1.2);
        return scaled;
    } else if (aug == "scale_down") {
        cv::Mat scaled;
        cv::resize(image, scaled, cv::Size(), 0.8, 0.8);
        return scaled;
    }
    return image.clone();
}

static void reverse_bbox_augmentation(
    Detection& det,
    const std::string& aug,
    int orig_w, int orig_h
) {
    auto& b = det.bbox;
    if (aug == "hflip") {
        float new_x1 = orig_w - b.x2;
        float new_x2 = orig_w - b.x1;
        b.x1 = new_x1;
        b.x2 = new_x2;
    } else if (aug == "vflip") {
        float new_y1 = orig_h - b.y2;
        float new_y2 = orig_h - b.y1;
        b.y1 = new_y1;
        b.y2 = new_y2;
    }
    // For rotations and scaling, more complex reverse transforms would be needed
}

Results Model::predict_tta(
    const cv::Mat& image,
    const std::vector<std::string>& augmentations,
    const InferConfig& config
) {
    std::vector<Results> all_results;
    all_results.reserve(augmentations.size());

    int orig_w = image.cols;
    int orig_h = image.rows;

    for (const auto& aug : augmentations) {
        cv::Mat augmented = apply_augmentation(image, aug);
        Results result = predict(augmented, config);

        // Reverse augmentation on detections
        for (auto& det : result.detections) {
            reverse_bbox_augmentation(det, aug, orig_w, orig_h);
        }

        all_results.push_back(std::move(result));
    }

    // Merge results using WBF-style approach
    Results merged;
    merged.image_size = image.size();
    merged.device_used = device();

    // For classifications: average scores
    if (!all_results.empty() && !all_results[0].classifications.empty()) {
        std::map<int, std::pair<float, std::string>> score_map;
        for (const auto& r : all_results) {
            for (const auto& cls : r.classifications) {
                auto& entry = score_map[cls.class_id];
                entry.first += cls.score;
                entry.second = cls.label;
            }
        }
        for (auto& [class_id, pair] : score_map) {
            ClassificationResult cls;
            cls.class_id = class_id;
            cls.score = pair.first / static_cast<float>(all_results.size());
            cls.label = pair.second;
            merged.classifications.push_back(cls);
        }
        std::sort(merged.classifications.begin(), merged.classifications.end(),
            [](const ClassificationResult& a, const ClassificationResult& b) {
                return a.score > b.score;
            });
    }

    // For detections: collect all and apply NMS
    for (const auto& r : all_results) {
        for (const auto& det : r.detections) {
            merged.detections.push_back(det);
        }
    }
    if (!merged.detections.empty()) {
        // Re-apply NMS to merged detections
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        for (const auto& det : merged.detections) {
            boxes.emplace_back(
                static_cast<int>(det.bbox.x1),
                static_cast<int>(det.bbox.y1),
                static_cast<int>(det.bbox.x2 - det.bbox.x1),
                static_cast<int>(det.bbox.y2 - det.bbox.y1)
            );
            scores.push_back(det.confidence);
            class_ids.push_back(det.class_id);
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, config.conf_threshold, config.iou_threshold, keep);

        std::vector<Detection> kept;
        for (int idx : keep) {
            kept.push_back(merged.detections[idx]);
        }
        merged.detections = std::move(kept);
    }

    return merged;
}

// ============================================================================
// Video Streaming
// ============================================================================

Model::StreamIterator::StreamIterator(
    Model* model, VideoSource* source, const InferConfig& config
) : model_(model), source_(source), config_(config) {
    start_time_ = std::chrono::steady_clock::now();
}

Model::StreamIterator Model::stream(
    const std::string& source,
    const InferConfig& config
) {
    stream_source_ = std::make_unique<VideoSource>(source);

    trigger_callback(CallbackEvent::StreamStart);

    return StreamIterator(this, stream_source_.get(), config);
}

Model::StreamResult Model::StreamIterator::next() {
    StreamResult sr;

    if (!source_ || !source_->is_opened()) {
        sr.end_of_stream = true;
        return sr;
    }

    cv::Mat frame = source_->read();
    if (frame.empty()) {
        sr.end_of_stream = true;
        if (model_) {
            model_->trigger_callback(CallbackEvent::StreamEnd);
        }
        return sr;
    }

    sr.frame = frame;
    sr.results = model_->predict(frame, config_);
    sr.frame_index = frame_index_++;

    auto now = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(now - start_time_).count();
    sr.fps = (elapsed_s > 0) ? frame_index_ / elapsed_s : 0.0;

    model_->trigger_callback(
        CallbackEvent::StreamFrame,
        sr.results.inference_time_ms,
        1, sr.frame_index, &sr.results);

    return sr;
}

bool Model::StreamIterator::has_next() const {
    return source_ && source_->is_opened();
}

// ============================================================================
// ModelManager implementation
// ============================================================================

ModelManager& ModelManager::instance() {
    static ModelManager instance;
    return instance;
}

std::shared_ptr<Model> ModelManager::load(
    const std::string& path,
    const LoadConfig& config
) {
    std::string cache_key = path + "_" + config.device + "_" + config.backend;

    // Thread-safe cache lookup
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(cache_key);
        if (it != cache_.end() && config.use_cache) {
            return it->second;
        }
    }

    // Get appropriate runtime (outside lock to avoid holding mutex during load)
    auto& factory = RuntimeFactory::instance();
    auto runtime = factory.get_best_runtime(path, config.device);

    if (!runtime) {
        throw IVITError("No suitable runtime found for model: " + path);
    }

    // Create runtime config
    RuntimeConfig rt_config;
    rt_config.device = config.device;
    rt_config.num_threads = 0;  // Auto

    if (config.precision == "fp16") {
        rt_config.precision = Precision::FP16;
    } else if (config.precision == "int8") {
        rt_config.precision = Precision::INT8;
    } else {
        rt_config.precision = Precision::FP32;
    }

    rt_config.cache_dir = config.cache_dir.empty() ? cache_dir_ : config.cache_dir;

    // Load model (potentially time-consuming, done outside lock)
    void* handle = runtime->load_model(path, rt_config);

    // Determine task type
    TaskType task = TaskType::Unknown;
    if (config.task == "classification") {
        task = TaskType::Classification;
    } else if (config.task == "detection") {
        task = TaskType::Detection;
    } else if (config.task == "segmentation") {
        task = TaskType::Segmentation;
    }

    // Create model wrapper
    auto model = std::make_shared<InferenceModel>(
        runtime,
        handle,
        path,
        task,
        config.device
    );

    // Thread-safe cache insertion
    if (config.use_cache) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Double-check if another thread already inserted
        auto it = cache_.find(cache_key);
        if (it == cache_.end()) {
            cache_[cache_key] = model;
        } else {
            // Another thread loaded first, return that one instead
            return it->second;
        }
    }

    return model;
}

void ModelManager::convert(
    const std::string& src_path,
    const std::string& dst_path,
    BackendType target_backend,
    Precision precision
) {
    auto& factory = RuntimeFactory::instance();
    auto runtime = factory.get_runtime(target_backend);

    if (!runtime) {
        throw IVITError("Target runtime not available");
    }

    RuntimeConfig config;
    config.precision = precision;

    runtime->convert_model(src_path, dst_path, config);
}

void ModelManager::clear_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

void ModelManager::set_cache_dir(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_dir_ = path;
}

} // namespace ivit

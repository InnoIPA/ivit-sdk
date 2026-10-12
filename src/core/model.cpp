/**
 * @file model.cpp
 * @brief Model class implementation
 */

#include "ivit/core/log.hpp"
#include "ivit/core/model.hpp"
#include "ivit/core/video_source.hpp"
#include "ivit/runtime/runtime.hpp"
#include "utils/fingerprint.hpp"
#include "vision/processing.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <unordered_map>

namespace ivit {

// ============================================================================
// InferenceModel - Concrete model implementation using runtime backends
// ============================================================================

class InferenceModel : public Model {
  public:
    InferenceModel(std::shared_ptr<IRuntime> runtime, void *handle, const std::string &name,
                   TaskType task, const std::string &device, const LoadConfig &contract)
        : runtime_(runtime), handle_(handle), name_(name), task_(task), device_(device),
          contract_(contract) {
        input_info_ = runtime_->get_input_info(handle_);
        output_info_ = runtime_->get_output_info(handle_);
    }

    ~InferenceModel() override {
        if (runtime_ && handle_) {
            try {
                runtime_->unload_model(handle_);
            } catch (const std::exception &error) {
                detail::log(LogLevel::ERROR, std::string("Model cleanup: ") + error.what());
            }
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
        for (const auto &info : input_info_) {
            total += info.byte_size();
        }
        for (const auto &info : output_info_) {
            total += info.byte_size();
        }
        return total;
    }

    std::vector<std::string> labels() const override { return contract_.labels; }
    LoadConfig load_config() const override { return contract_; }

    // Inference
    Results predict(const cv::Mat &image, const InferConfig &config) override {
        return detail::predict_image(*this, contract_, image, config);
    }

    std::vector<Results> predict_batch(const std::vector<cv::Mat> &images,
                                       const InferConfig &config) override {
        if (images.empty()) {
            return {};
        }

        std::vector<Results> results_list;
        results_list.reserve(images.size());

        // Check if model supports dynamic batch
        bool supports_batch = false;
        if (!input_info_.empty() && !input_info_[0].shape.empty()) {
            int64_t batch_dim = input_info_[0].shape[0];
            supports_batch = (batch_dim == -1 || batch_dim == 0);
        }

        if (supports_batch && images.size() > 1) {
            // True batch inference
            auto start = std::chrono::high_resolution_clock::now();

            // Collect original sizes
            std::vector<cv::Size> orig_sizes;
            orig_sizes.reserve(images.size());
            for (const auto &img : images) {
                orig_sizes.push_back(img.size());
            }

            // Batch preprocess: preprocess each image and concatenate
            std::vector<std::map<std::string, Tensor>> all_inputs;
            for (const auto &img : images) {
                all_inputs.push_back(preprocess(img));
            }

            // Merge tensors into batch
            const auto &info = input_info_[0];
            // Take the concrete (already resolved) shape from a preprocessed
            // sample; the declared shape still has dynamic dimensions.
            auto shape = all_inputs[0].begin()->second.shape();
            shape[0] = static_cast<int64_t>(images.size()); // Set batch dimension

            Tensor batched_input(shape, info.dtype);
            batched_input.set_name(info.name);
            batched_input.set_layout(contract_.input_layout == "nhwc" ? Layout::NHWC
                                                                      : Layout::NCHW);

            // Copy each preprocessed image into the batch tensor
            size_t single_size = batched_input.byte_size() / images.size();
            char *batch_ptr = static_cast<char *>(batched_input.data());

            for (size_t i = 0; i < all_inputs.size(); i++) {
                const auto &single_tensor = all_inputs[i].begin()->second;
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
                for (const auto &[name, tensor] : outputs) {
                    // Create tensor for single sample
                    auto out_shape = tensor.shape();
                    if (out_shape.empty() || out_shape[0] != static_cast<int64_t>(images.size()))
                        throw InferenceError("Batch output must preserve the batch dimension");
                    out_shape[0] = 1; // Single sample
                    Tensor sample_tensor(out_shape, tensor.dtype());
                    sample_tensor.set_name(name);

                    // Copy data for this sample
                    size_t sample_size = sample_tensor.byte_size();
                    const char *src = static_cast<const char *>(tensor.data()) + i * sample_size;
                    std::memcpy(sample_tensor.data(), src, sample_size);

                    sample_outputs[name] = std::move(sample_tensor);
                }

                postprocess(sample_outputs, orig_sizes[i], config, results);
                results.inference_time_ms = avg_time;
                results_list.push_back(std::move(results));
            }

            return results_list;
        }

        // Fallback: sequential processing for models with fixed batch size
        for (const auto &image : images) {
            results_list.push_back(predict(image, config));
        }

        return results_list;
    }

    std::map<std::string, Tensor> infer_raw(const std::map<std::string, Tensor> &inputs) override {
        return runtime_->infer(handle_, inputs);
    }

    void warmup(int iterations) override {
        if (input_info_.empty())
            return;

        // Create dummy input
        const auto &info = input_info_[0];
        const int64_t size = default_spatial_size();
        auto shape = info.shape;
        if (shape.size() == 4 && contract_.input_layout == "nhwc") {
            if (shape[0] < 0)
                shape[0] = 1;
            if (shape[1] < 0)
                shape[1] = size;
            if (shape[2] < 0)
                shape[2] = size;
            if (shape[3] < 0)
                throw InferenceError("Warmup needs fixed channel count");
        } else
            shape = resolve_dynamic_shape(shape, 1, size, size);
        Tensor dummy(shape, info.dtype);
        dummy.set_name(info.name);

        std::map<std::string, Tensor> inputs;
        inputs[info.name] = std::move(dummy);

        for (int i = 0; i < iterations; i++) {
            runtime_->infer(handle_, inputs);
        }
    }

  private:
    int64_t default_spatial_size() const { return task_ == TaskType::Detection ? 640 : 224; }
    std::map<std::string, Tensor> preprocess(const cv::Mat &image) {
        if (input_info_.size() != 1)
            throw InferenceError("Image API requires one input; use infer_raw");
        auto tensor = detail::preprocess_image(image, input_info_[0], contract_, task_);
        return {{input_info_[0].name, std::move(tensor)}};
    }
    void postprocess(const std::map<std::string, Tensor> &outputs, cv::Size image_size,
                     const InferConfig &config, Results &results) {
        results =
            detail::decode_outputs(outputs, image_size, input_info_[0], contract_, task_, config);
        results.device_used = device_;
    }

    std::shared_ptr<IRuntime> runtime_;
    void *handle_;
    std::string name_;
    TaskType task_;
    std::string device_;
    std::vector<TensorInfo> input_info_;
    std::vector<TensorInfo> output_info_;
    LoadConfig contract_;
};

// ============================================================================
// Model path-based predict
// ============================================================================

Results Model::predict(const std::string &image_path, const InferConfig &config) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw IVITError("Failed to load image: " + image_path);
    }
    return predict(image, config);
}

// ============================================================================
// Async Inference implementations
// ============================================================================

std::future<Results> Model::predict_async(const cv::Mat &image, const InferConfig &config) {
    // Capture image by value to ensure it remains valid
    cv::Mat image_copy = image.clone();

    // `self` keeps the model alive until the task finishes, even if the caller
    // drops its last reference while the future is still pending.
    return std::async(std::launch::async, [this, self = keep_alive(), image_copy, config]() {
        return this->predict(image_copy, config);
    });
}

std::future<std::vector<Results>> Model::predict_batch_async(const std::vector<cv::Mat> &images,
                                                             const InferConfig &config) {
    // Copy images to ensure they remain valid
    std::vector<cv::Mat> images_copy;
    images_copy.reserve(images.size());
    for (const auto &img : images) {
        images_copy.push_back(img.clone());
    }

    return std::async(std::launch::async,
                      [this, self = keep_alive(), images_copy = std::move(images_copy), config]() {
                          return this->predict_batch(images_copy, config);
                      });
}

void Model::submit_inference(const cv::Mat &image, std::function<void(Results)> callback,
                             const InferConfig &config) {
    // Fire-and-forget: launch async task that calls callback
    cv::Mat image_copy = image.clone();

    // The thread is detached, so it must own a reference to the model: a raw
    // `this` would dangle if the caller released the model right after submitting.
    std::thread([this, self = keep_alive(), image_copy, callback, config]() {
        try {
            Results results = this->predict(image_copy, config);
            if (callback) {
                callback(std::move(results));
            }
        } catch (const std::exception &e) {
            // Log error but don't propagate - this is fire-and-forget
            detail::log(LogLevel::ERROR, std::string("Async inference error: ") + e.what());
        }
    }).detach();
}

// ============================================================================
// Model destructor
// ============================================================================

Model::Model() = default;

Model::~Model() = default;

// ============================================================================
// Concurrent Inference
// ============================================================================

std::vector<Results> Model::predict_concurrent(const std::vector<cv::Mat> &images,
                                               int max_concurrent, const InferConfig &config) {
    if (max_concurrent <= 0) {
        throw IVITError("max_concurrent must be positive");
    }
    if (images.empty())
        return {};

    trigger_callback(CallbackEvent::BatchStart, 0.0, static_cast<int>(images.size()));

    // A bounded worker set avoids one thread per image. Each worker owns a
    // distinct result slot; all workers finish before an exception propagates.
    std::vector<Results> results(images.size());
    std::atomic<size_t> next{0};
    std::mutex error_mutex;
    std::exception_ptr first_error;
    const size_t workers = std::min(images.size(), static_cast<size_t>(max_concurrent));
    std::vector<std::future<void>> futures;
    futures.reserve(workers);
    for (size_t i = 0; i < workers; ++i) {
        futures.push_back(std::async(std::launch::async, [&]() {
            for (;;) {
                const size_t index = next.fetch_add(1);
                if (index >= images.size())
                    return;
                try {
                    results[index] = predict(images[index].clone(), config);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_error)
                        first_error = std::current_exception();
                }
            }
        }));
    }
    for (auto &future : futures)
        future.get();
    if (first_error)
        std::rethrow_exception(first_error);

    trigger_callback(CallbackEvent::BatchEnd, 0.0, static_cast<int>(images.size()));
    return results;
}

// ============================================================================
// Callback helpers
// ============================================================================

int Model::on(const std::string &event, std::function<void(const CallbackContext &)> callback,
              int priority) {
    return callbacks_.register_callback(event, std::move(callback), priority);
}

bool Model::remove_callback(const std::string &event, int callback_id) {
    return callbacks_.unregister_callback(callback_event_from_string(event), callback_id);
}

int Model::remove_all_callbacks(const std::string &event) {
    return callbacks_.unregister_all(event);
}

void Model::trigger_callback(CallbackEvent event, double latency_ms, int batch_size,
                             int frame_index, const Results *results) {
    if (!callbacks_.has_callbacks(event))
        return;

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
// TTA (Test-Time Augmentation)
// ============================================================================

static cv::Mat apply_augmentation(const cv::Mat &image, const std::string &aug) {
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

static void reverse_bbox_augmentation(Detection &det, const std::string &aug, int orig_w,
                                      int orig_h) {
    auto &b = det.bbox;
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

Results Model::predict_tta(const cv::Mat &image, const std::vector<std::string> &augmentations,
                           const InferConfig &config) {
    std::vector<Results> all_results;
    all_results.reserve(augmentations.size());

    int orig_w = image.cols;
    int orig_h = image.rows;

    for (const auto &aug : augmentations) {
        cv::Mat augmented = apply_augmentation(image, aug);
        Results result = predict(augmented, config);

        // Reverse augmentation on detections
        for (auto &det : result.detections) {
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
        for (const auto &r : all_results) {
            for (const auto &cls : r.classifications) {
                auto &entry = score_map[cls.class_id];
                entry.first += cls.score;
                entry.second = cls.label;
            }
        }
        for (auto &[class_id, pair] : score_map) {
            ClassificationResult cls;
            cls.class_id = class_id;
            cls.score = pair.first / static_cast<float>(all_results.size());
            cls.label = pair.second;
            merged.classifications.push_back(cls);
        }
        std::sort(merged.classifications.begin(), merged.classifications.end(),
                  [](const ClassificationResult &a, const ClassificationResult &b) {
                      return a.score > b.score;
                  });
    }

    // For detections: collect all and apply NMS
    for (const auto &r : all_results) {
        for (const auto &det : r.detections) {
            merged.detections.push_back(det);
        }
    }
    if (!merged.detections.empty()) {
        // Re-apply NMS to merged detections
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        for (const auto &det : merged.detections) {
            boxes.emplace_back(static_cast<int>(det.bbox.x1), static_cast<int>(det.bbox.y1),
                               static_cast<int>(det.bbox.x2 - det.bbox.x1),
                               static_cast<int>(det.bbox.y2 - det.bbox.y1));
            scores.push_back(det.confidence);
            class_ids.push_back(det.class_id);
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, detail::effective_confidence(config, TaskType::Detection),
                          config.iou_threshold, keep);

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

Model::StreamIterator::StreamIterator(std::shared_ptr<Model> model,
                                      std::shared_ptr<VideoSource> source,
                                      const InferConfig &config)
    : model_(std::move(model)), source_(std::move(source)), config_(config) {
    start_time_ = std::chrono::steady_clock::now();
}

Model::StreamIterator Model::stream(const std::string &source, const InferConfig &config) {
    auto owned_model = shared_from_this();
    auto owned_source = std::make_shared<VideoSource>(source);

    trigger_callback(CallbackEvent::StreamStart);

    return StreamIterator(std::move(owned_model), std::move(owned_source), config);
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

    model_->trigger_callback(CallbackEvent::StreamFrame, sr.results.inference_time_ms, 1,
                             sr.frame_index, &sr.results);

    return sr;
}

bool Model::StreamIterator::has_next() const { return source_ && source_->is_opened(); }

// ============================================================================
// ModelManager implementation
// ============================================================================

ModelManager &ModelManager::instance() {
    static ModelManager instance;
    return instance;
}

std::shared_ptr<Model> ModelManager::load(const std::string &path,
                                          const LoadConfig &requested_config) {
    LoadConfig config = requested_config;
    detail::validate_contract(config);
    const BackendType requested_backend = backend_from_string(config.backend);
    if (config.precision != "fp32" && config.precision != "fp16" && config.precision != "int8")
        throw ModelLoadError("precision must be fp32, fp16 or int8");
    if (config.batch_size < 1 || config.max_batch_size < config.batch_size ||
        config.num_threads < 0 || config.workspace_size == 0)
        throw ModelLoadError("Invalid batch profile, thread count or workspace size");
    if (config.num_instances < 1 || config.num_instances > 16)
        throw ModelLoadError("num_instances must be between 1 and 16");
    if (config.performance_mode != "LATENCY" && config.performance_mode != "THROUGHPUT")
        throw ModelLoadError("performance_mode must be LATENCY or THROUGHPUT");
    if (!config.task.empty() && config.task != "classification" && config.task != "detection" &&
        config.task != "segmentation")
        throw ModelLoadError("No image decoder for task '" + config.task +
                             "'; use task='' and infer_raw");
    // Explicit labels win; prefer a per-model sidecar over shared labels.txt.
    if (config.labels.empty()) {
        namespace fs = std::filesystem;
        fs::path labels_file = config.labels_path;
        if (labels_file.empty()) {
            labels_file = fs::path(path).replace_extension(".labels.txt");
            if (!fs::exists(labels_file))
                labels_file = fs::path(path).parent_path() / "labels.txt";
        }
        if (!config.labels_path.empty() || fs::exists(labels_file)) {
            std::ifstream file(labels_file);
            if (!file)
                throw ModelLoadError("Cannot read labels: " + labels_file.string());
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                if (line.empty())
                    throw ModelLoadError("Empty label in " + labels_file.string());
                config.labels.push_back(line);
            }
        }
    }
    // Get appropriate runtime (outside lock to avoid holding mutex during load)
    auto &factory = RuntimeFactory::instance();
    std::shared_ptr<IRuntime> runtime;

    if (requested_backend != BackendType::Auto) {
        // An explicit backend wins over extension/device based selection
        runtime = factory.get_runtime(requested_backend);
        if (!runtime) {
            throw IVITError("Backend '" + to_string(requested_backend) +
                            "' is not available in this build or on this machine");
        }
    } else {
        runtime = factory.get_best_runtime(path, config.device);
        if (!runtime) {
            throw IVITError("No suitable runtime found for model: " + path);
        }
    }

    if (config.device == "auto") {
        auto devices = runtime->get_devices();
        if (devices.empty())
            throw DeviceNotFoundError("No devices available for " + runtime->name());
        auto rank = [](const DeviceInfo &device) {
            return device.type == "gpu" ? 0 : device.type == "npu" ? 1 : 2;
        };
        auto selected = std::min_element(
            devices.begin(), devices.end(),
            [&](const auto &left, const auto &right) { return rank(left) < rank(right); });
        config.device = selected->id;
    }
    config.backend = to_string(runtime->type());
    if (runtime->type() == BackendType::TensorRT) {
        if (config.device != "auto" && config.device.find("cuda:") != 0)
            throw ModelLoadError("TensorRT requires device='auto' or 'cuda:N'");
        if (config.num_threads != 0 || config.enable_profiling ||
            config.performance_mode != "LATENCY")
            throw ModelLoadError(
                "num_threads, profiling and performance_mode are OpenVINO options");
    } else if (runtime->type() == BackendType::OpenVINO) {
        if (config.device.find("cuda:") == 0)
            throw ModelLoadError("OpenVINO does not use CUDA devices; select cpu, gpu:N or npu");
        if (config.batch_size != 1 || config.max_batch_size != 8 ||
            config.workspace_size != (1ULL << 30) || config.num_instances != 1)
            throw ModelLoadError("batch profiles, workspace_size and num_instances are TensorRT "
                                 "options; OpenVINO uses the model's shape");
        if (config.num_threads != 0 && config.device != "cpu")
            throw ModelLoadError("num_threads requires an explicit OpenVINO CPU device");
    }

    // Every setting that changes what gets loaded must be part of the key,
    // otherwise a later load with a different task/precision gets the wrong model.
    const std::string cache_key = detail::model_fingerprint(path) + ":" +
                                  detail::contract_fingerprint(config) + ":" +
                                  runtime->get_version();

    // Validate the backend name before doing any work (throws on unknown names)

    // Thread-safe cache lookup
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(cache_key);
        if (it != cache_.end() && config.use_cache) {
            return it->second;
        }
    }

    // Create runtime config
    RuntimeConfig rt_config;
    rt_config.use_cache = config.use_cache;
    rt_config.device = config.device;
    rt_config.num_threads = config.num_threads;
    rt_config.num_instances = config.num_instances;
    rt_config.enable_profiling = config.enable_profiling;
    rt_config.performance_mode = config.performance_mode;
    rt_config.batch_size = config.batch_size;
    rt_config.max_batch_size = config.max_batch_size;
    rt_config.workspace_size = config.workspace_size;

    if (config.precision == "fp16") {
        rt_config.precision = Precision::FP16;
    } else if (config.precision == "int8") {
        rt_config.precision = Precision::INT8;
    } else {
        rt_config.precision = Precision::FP32;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        rt_config.cache_dir = config.cache_dir.empty() ? cache_dir_ : config.cache_dir;
    }

    // Load model (potentially time-consuming, done outside lock)
    void *handle = runtime->load_model(path, rt_config);

    // Determine task type
    TaskType task = TaskType::Unknown;
    if (config.task == "classification") {
        task = TaskType::Classification;
    } else if (config.task == "detection") {
        task = TaskType::Detection;
    } else if (config.task == "segmentation") {
        task = TaskType::Segmentation;
    }

    // A wrong labels list is a configuration mistake: report it now, with the numbers, rather than
    // at the first prediction. Only checked where the class count follows from the output shape.
    if (!config.labels.empty()) {
        try {
            const int classes = detail::expected_class_count(runtime->get_output_info(handle),
                                                             config, task);
            if (classes >= 0 && static_cast<size_t>(classes) != config.labels.size())
                throw ModelLoadError("labels has " + std::to_string(config.labels.size()) +
                                     " entries but the model outputs " + std::to_string(classes) +
                                     " classes");
        } catch (...) {
            runtime->unload_model(handle);
            throw;
        }
    }

    // Create model wrapper
    std::shared_ptr<InferenceModel> model;
    try {
        model =
            std::make_shared<InferenceModel>(runtime, handle, path, task, config.device, config);
    } catch (...) {
        runtime->unload_model(handle);
        throw;
    }
    // CUDA/TRT may register shutdown handlers while the first engine builds.
    // Register cache cleanup afterwards so engines are freed before those
    // handlers, rather than waiting for the earlier singleton destructor.
    static std::once_flag trt_cleanup_registered, other_cleanup_registered;
    auto &cleanup_registered = runtime->type() == BackendType::TensorRT ? trt_cleanup_registered
                                                                        : other_cleanup_registered;
    std::call_once(cleanup_registered,
                   [] { std::atexit([] { ModelManager::instance().clear_cache(); }); });

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

void ModelManager::convert(const std::string &src_path, const std::string &dst_path,
                           BackendType target_backend, Precision precision) {
    auto &factory = RuntimeFactory::instance();
    auto runtime = factory.get_runtime(target_backend);

    if (!runtime) {
        throw IVITError("Target runtime not available");
    }

    RuntimeConfig config;
    config.precision = precision;

    runtime->convert_model(src_path, dst_path, config);
}

void ModelManager::clear_cache() {
    decltype(cache_) retired;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        retired.swap(cache_);
    }
}

void ModelManager::set_cache_dir(const std::string &path) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_dir_ = path;
}

} // namespace ivit

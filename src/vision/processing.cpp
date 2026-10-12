#include "vision/processing.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <opencv2/imgproc.hpp>

namespace ivit::detail {
namespace {
struct Geometry {
    int width, height, channels;
    int pad_x = 0, pad_y = 0;
    float scale_x = 1, scale_y = 1;
};

bool letterbox(const LoadConfig &c, TaskType task) {
    return c.resize_mode == "letterbox" || c.resize_mode == "letterbox_top_left" ||
           (c.resize_mode == "auto" && task == TaskType::Detection);
}

Geometry geometry(const TensorInfo &input, const LoadConfig &c, TaskType task, cv::Size original) {
    if (input.shape.size() != 4)
        throw InferenceError("Image API requires rank-4 input; use infer_raw");
    const bool nhwc = c.input_layout == "nhwc";
    const auto h = input.shape[nhwc ? 1 : 2];
    const auto w = input.shape[nhwc ? 2 : 3];
    const auto channels = input.shape[nhwc ? 3 : 1];
    const int fallback = task == TaskType::Detection ? 640 : 224;
    if (h > std::numeric_limits<int>::max() || w > std::numeric_limits<int>::max() ||
        (channels != 1 && channels != 3))
        throw InferenceError("Unsupported image input dimensions");
    Geometry g{w > 0 ? static_cast<int>(w) : fallback, h > 0 ? static_cast<int>(h) : fallback,
               static_cast<int>(channels)};
    if (original.width <= 0 || original.height <= 0)
        throw InferenceError("Image is empty");
    if (letterbox(c, task)) {
        const float ratio =
            std::min(float(g.width) / original.width, float(g.height) / original.height);
        const int resized_w = std::max(1, static_cast<int>(std::round(original.width * ratio)));
        const int resized_h = std::max(1, static_cast<int>(std::round(original.height * ratio)));
        g.pad_x = c.resize_mode == "letterbox_top_left" ? 0 : (g.width - resized_w) / 2;
        g.pad_y = c.resize_mode == "letterbox_top_left" ? 0 : (g.height - resized_h) / 2;
        g.scale_x = float(resized_w) / original.width;
        g.scale_y = float(resized_h) / original.height;
    } else {
        g.scale_x = float(g.width) / original.width;
        g.scale_y = float(g.height) / original.height;
    }
    return g;
}

std::string label(const LoadConfig &c, int id) {
    return id >= 0 && static_cast<size_t>(id) < c.labels.size() ? c.labels[id]
                                                                : "class_" + std::to_string(id);
}

const Tensor &output_tensor(const std::map<std::string, Tensor> &outputs, const LoadConfig &c) {
    if (!c.output_name.empty()) {
        auto found = outputs.find(c.output_name);
        if (found == outputs.end())
            throw InferenceError("Configured output_name is absent: " + c.output_name);
        return found->second;
    }
    if (outputs.size() != 1)
        throw InferenceError("Decoder requires one output; specify output_name or use infer_raw");
    return outputs.begin()->second;
}

// Read-only float32 view of an output tensor; float16 is widened into owned storage.
struct FloatView {
    const float *data = nullptr;
    size_t size = 0;
    std::vector<float> storage;
    bool empty() const { return size == 0; }
};

FloatView floats(const Tensor &tensor) {
    if (tensor.numel() <= 0 || tensor.numel() > std::numeric_limits<int>::max())
        throw InferenceError("Invalid output size");
    FloatView view;
    view.size = static_cast<size_t>(tensor.numel());
    if (tensor.dtype() == DataType::Float32) {
        view.data = static_cast<const float *>(tensor.data());
    } else if (tensor.dtype() == DataType::Float16) {
        view.storage.resize(view.size);
        cv::Mat source(1, static_cast<int>(tensor.numel()), CV_16F,
                       const_cast<void *>(tensor.data()));
        cv::Mat target(1, static_cast<int>(tensor.numel()), CV_32F, view.storage.data());
        source.convertTo(target, CV_32F);
        view.data = view.storage.data();
    } else {
        throw InferenceError(
            "Decoder expects float32/float16 output; use infer_raw for other types");
    }
    // Branch-free so the compiler can vectorise: an all-ones exponent means NaN or infinity.
    uint32_t bad = 0;
    for (size_t i = 0; i < view.size; ++i) {
        uint32_t bits;
        std::memcpy(&bits, view.data + i, sizeof(bits));
        bad |= static_cast<uint32_t>((bits & 0x7f800000u) == 0x7f800000u);
    }
    if (bad)
        throw InferenceError("Output contains NaN or infinity");
    return view;
}

// Greedy NMS: highest confidence first; a box is dropped when it overlaps a kept box of the
// same class (or of any class when config.agnostic_nms) above the IoU threshold.
void non_max_suppression(std::vector<Detection> &detections, const InferConfig &config) {
    std::stable_sort(detections.begin(), detections.end(),
                     [](const auto &a, const auto &b) { return a.confidence > b.confidence; });
    std::vector<Detection> kept;
    for (const auto &detection : detections) {
        bool suppressed = false;
        for (const auto &previous : kept)
            if ((config.agnostic_nms || previous.class_id == detection.class_id) &&
                previous.bbox.iou(detection.bbox) > config.iou_threshold) {
                suppressed = true;
                break;
            }
        if (!suppressed)
            kept.push_back(detection);
        if (static_cast<int>(kept.size()) == config.max_detections)
            break;
    }
    detections = std::move(kept);
}

void validate_inference(const InferConfig &c) {
    const bool unset = c.conf_threshold == -1.0f; // "task default"
    if ((!unset && (!std::isfinite(c.conf_threshold) || c.conf_threshold < 0 ||
                    c.conf_threshold > 1)) ||
        !std::isfinite(c.iou_threshold) || c.iou_threshold < 0 || c.iou_threshold > 1 ||
        c.max_detections <= 0)
        throw InferenceError("Invalid inference thresholds or max_detections");
}
} // namespace

float effective_confidence(const InferConfig &config, TaskType task) {
    if (config.conf_threshold != -1.0f)
        return config.conf_threshold;
    return task == TaskType::Detection ? 0.5f : 0.f;
}

void validate_contract(const LoadConfig &c) {
    if (c.input_color != "rgb" && c.input_color != "bgr")
        throw IVITError("input_color must be rgb or bgr");
    if (c.input_layout != "nchw" && c.input_layout != "nhwc")
        throw IVITError("input_layout must be nchw or nhwc");
    if (c.resize_mode != "auto" && c.resize_mode != "stretch" && c.resize_mode != "letterbox" &&
        c.resize_mode != "letterbox_top_left")
        throw IVITError("resize_mode must be auto, stretch or letterbox");
    const std::vector<std::string> decoders{"auto",   "logits", "probabilities", "yolov5", "yolov8",
                                            "legacy", "raw",    "yolox",         "yolox_decoded"};
    if (std::find(decoders.begin(), decoders.end(), c.decoder) == decoders.end())
        throw IVITError("Unknown decoder: " + c.decoder);
    if (c.decoder != "auto" && c.decoder != "raw") {
        if (c.task == "classification" && c.decoder != "logits" && c.decoder != "probabilities")
            throw IVITError("Classification decoder must be logits or probabilities");
        if (c.task == "segmentation" && c.decoder != "logits")
            throw IVITError("Segmentation decoder must be logits");
        if (c.task == "detection" && (c.decoder == "logits" || c.decoder == "probabilities"))
            throw IVITError("Detection requires an explicit detection decoder");
    }
    if (!std::isfinite(c.input_scale))
        throw IVITError("input_scale must be finite");
    if ((c.mean.size() != 1 && c.mean.size() != 3) || (c.std.size() != 1 && c.std.size() != 3))
        throw IVITError("mean/std must contain one or three values");
    for (float value : c.mean)
        if (!std::isfinite(value))
            throw IVITError("mean must be finite");
    for (float value : c.std)
        if (!std::isfinite(value) || value <= 0)
            throw IVITError("std must be finite and positive");
}

int expected_class_count(const std::vector<TensorInfo> &outputs, const LoadConfig &c,
                         TaskType task) {
    if (c.decoder == "raw" || task == TaskType::Unknown)
        return -1;
    const TensorInfo *output = nullptr;
    if (!c.output_name.empty()) {
        for (const auto &candidate : outputs)
            if (candidate.name == c.output_name)
                output = &candidate;
    } else if (outputs.size() == 1) {
        output = &outputs[0];
    }
    if (!output)
        return -1;
    const auto &shape = output->shape;
    auto known = [](int64_t value) { return value > 0 && value <= std::numeric_limits<int>::max(); };

    if (task == TaskType::Classification) {
        if (shape.size() == 1 && known(shape[0]))
            return static_cast<int>(shape[0]);
        if (shape.size() == 2 && (shape[0] == 1 || shape[0] <= 0) && known(shape[1]))
            return static_cast<int>(shape[1]);
    } else if (task == TaskType::Segmentation) {
        if (shape.size() == 4 && known(shape[1]))
            return static_cast<int>(shape[1]);
    } else if (task == TaskType::Detection) {
        if (shape.size() != 3)
            return -1;
        bool v8;
        if (c.decoder == "yolov8") {
            v8 = true;
        } else if (c.decoder == "auto") {
            // Same orientation rule as decode_outputs; dynamic dimensions leave it undecided.
            if (!known(shape[1]) || !known(shape[2]))
                return -1;
            if (shape[1] >= 5 && shape[1] < shape[2])
                v8 = true;
            else if (shape[2] >= 6 && shape[2] < shape[1])
                v8 = false;
            else
                return -1;
        } else {
            v8 = false; // yolov5, legacy, yolox, yolox_decoded: [1, anchors, features]
        }
        const int64_t features = v8 ? shape[1] : shape[2];
        const int64_t classes = features - (v8 ? 4 : 5);
        if (known(features) && classes > 0)
            return static_cast<int>(classes);
    }
    return -1;
}

Tensor preprocess_image(const cv::Mat &image, const TensorInfo &input, const LoadConfig &c,
                        TaskType task) {
    if (image.empty() || image.depth() != CV_8U || (image.channels() != 1 && image.channels() != 3))
        throw InferenceError("Image API expects nonempty uint8 grayscale or BGR image");
    auto g = geometry(input, c, task, image.size());
    if (input.shape[0] > 1)
        throw InferenceError(
            "Image API requires batch=1 or dynamic batch; use infer_raw for fixed batches");
    if (input.dtype != DataType::Float32 && input.dtype != DataType::Float16)
        throw InferenceError("Image API requires float32/float16 model input; use infer_raw");
    cv::Mat color;
    if (g.channels == 1 && image.channels() == 3)
        cv::cvtColor(image, color, cv::COLOR_BGR2GRAY);
    else if (g.channels == 3 && image.channels() == 1)
        cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
    else
        color = image;
    // Resize straight into the (padded) network-sized canvas: no intermediate copies.
    cv::Mat canvas(g.height, g.width, color.type());
    if (letterbox(c, task)) {
        const int rw = static_cast<int>(std::round(image.cols * g.scale_x));
        const int rh = static_cast<int>(std::round(image.rows * g.scale_y));
        canvas.setTo(cv::Scalar::all(114));
        cv::Mat target = canvas(cv::Rect(g.pad_x, g.pad_y, rw, rh));
        cv::resize(color, target, target.size());
    } else
        cv::resize(color, canvas, canvas.size());

    // (pixel * input_scale - mean) / std, in this exact order so results stay bit-identical to a
    // per-pixel loop. Each step is a vectorised OpenCV primitive writing into the tensor itself,
    // and the mean/std steps are skipped when they are the identity (the default contract).
    const bool nhwc = c.input_layout == "nhwc";
    Shape shape =
        nhwc ? Shape{1, g.height, g.width, g.channels} : Shape{1, g.channels, g.height, g.width};
    Tensor floating(shape, DataType::Float32);
    float *values = floating.data_ptr<float>();
    float mean[3], stddev[3];
    bool has_mean = false, has_std = false;
    for (int ch = 0; ch < g.channels; ++ch) {
        mean[ch] = c.mean[c.mean.size() == 1 ? 0 : ch];
        stddev[ch] = c.std[c.std.size() == 1 ? 0 : ch];
        has_mean |= mean[ch] != 0.f;
        has_std |= stddev[ch] != 1.f;
    }
    auto normalize = [&](cv::Mat &plane, int first, int count) {
        if (has_mean)
            cv::subtract(plane, cv::Scalar(mean[first], count > 1 ? mean[first + 1] : 0,
                                           count > 2 ? mean[first + 2] : 0),
                         plane);
        if (has_std)
            cv::divide(plane, cv::Scalar(stddev[first], count > 1 ? stddev[first + 1] : 1,
                                         count > 2 ? stddev[first + 2] : 1),
                       plane);
    };
    const bool rgb = g.channels == 3 && c.input_color == "rgb";
    if (g.channels == 1) {
        cv::Mat plane(g.height, g.width, CV_32F, values);
        canvas.convertTo(plane, CV_32F, c.input_scale);
        normalize(plane, 0, 1);
    } else if (!nhwc) {
        cv::Mat source[3];
        cv::split(canvas, source); // B, G, R
        const size_t plane_size = static_cast<size_t>(g.height) * g.width;
        for (int ch = 0; ch < 3; ++ch) {
            cv::Mat plane(g.height, g.width, CV_32F, values + ch * plane_size);
            source[rgb ? 2 - ch : ch].convertTo(plane, CV_32F, c.input_scale);
            normalize(plane, ch, 1);
        }
    } else {
        cv::Mat ordered = canvas;
        if (rgb)
            cv::cvtColor(canvas, ordered, cv::COLOR_BGR2RGB);
        cv::Mat interleaved(g.height, g.width, CV_32FC3, values);
        ordered.convertTo(interleaved, CV_32F, c.input_scale);
        normalize(interleaved, 0, 3);
    }
    Tensor result = floating;
    if (input.dtype == DataType::Float16) {
        result = Tensor(shape, DataType::Float16);
        cv::Mat source(1, static_cast<int>(floating.numel()), CV_32F, floating.data());
        cv::Mat target(1, static_cast<int>(floating.numel()), CV_16F, result.data());
        source.convertTo(target, CV_16F);
    }
    result.set_name(input.name);
    result.set_layout(nhwc ? Layout::NHWC : Layout::NCHW);
    return result;
}

Results decode_outputs(const std::map<std::string, Tensor> &outputs, cv::Size image_size,
                       const TensorInfo &input, const LoadConfig &c, TaskType task,
                       const InferConfig &requested) {
    validate_inference(requested);
    InferConfig config = requested;
    config.conf_threshold = effective_confidence(requested, task);
    Results result;
    result.image_size = image_size;
    result.raw_outputs = outputs;
    if (c.decoder == "raw" || task == TaskType::Unknown)
        return result;
    const auto &tensor = output_tensor(outputs, c);
    const auto &shape = tensor.shape();
    const auto view = floats(tensor);
    const float *data = view.data;
    if (view.empty())
        throw InferenceError("Decoder requires nonempty output");
    if (task == TaskType::Classification) {
        if (shape.size() != 1 && !(shape.size() == 2 && shape[0] == 1))
            throw InferenceError("Classification expects [C] or [1,C]");
        if (!c.labels.empty() && c.labels.size() != view.size)
            throw InferenceError("labels count does not match classes");
        std::vector<float> scores(data, data + view.size); // softmax works in place
        if (c.decoder != "probabilities") {
            const float maximum = *std::max_element(scores.begin(), scores.end());
            float total = 0;
            for (auto &value : scores) {
                value = std::exp(value - maximum);
                total += value;
            }
            for (auto &value : scores)
                value /= total;
        } else {
            float total = 0;
            for (float value : scores) {
                if (value < 0 || value > 1)
                    throw InferenceError("Invalid probabilities");
                total += value;
            }
            if (std::abs(total - 1) > 0.01)
                throw InferenceError("Probabilities must sum to one");
        }
        for (size_t id = 0; id < scores.size(); ++id)
            if (scores[id] >= config.conf_threshold)
                result.classifications.emplace_back(static_cast<int>(id),
                                                    label(c, static_cast<int>(id)), scores[id]);
        std::stable_sort(result.classifications.begin(), result.classifications.end(),
                         [](const auto &a, const auto &b) { return a.score > b.score; });
    } else if (task == TaskType::Detection) {
        if (shape.size() != 3 || shape[0] != 1)
            throw InferenceError("YOLO decoder expects rank-3 [1,*,*] output");
        std::string decoder = c.decoder;
        if (decoder == "auto") {
            // Only infer orientation when it is unambiguous; explicit decoder
            // is required for small/custom heads with overlapping dimensions.
            if (shape[1] >= 5 && shape[1] < shape[2])
                decoder = "yolov8";
            else if (shape[2] >= 6 && shape[2] < shape[1])
                decoder = "yolov5";
            else
                throw InferenceError(
                    "Ambiguous detection output; specify decoder=yolov5 or yolov8");
        }
        const bool v8 = decoder == "yolov8";
        if (!v8 && decoder != "yolov5" && decoder != "legacy" && decoder != "yolox" &&
            decoder != "yolox_decoded")
            throw InferenceError("Unsupported detection decoder");
        const int count = static_cast<int>(shape[v8 ? 2 : 1]);
        const int features = static_cast<int>(shape[v8 ? 1 : 2]);
        const int offset = v8 ? 4 : 5;
        const int classes = features - offset;
        if (classes <= 0 || (!c.labels.empty() && static_cast<int>(c.labels.size()) != classes))
            throw InferenceError("Detection feature/labels count does not match decoder");
        const auto g = geometry(input, c, task, image_size);
        std::vector<std::array<int, 3>> grids;
        if (decoder == "yolox") {
            for (int stride : {8, 16, 32})
                for (int y = 0; y < g.height / stride; ++y)
                    for (int x = 0; x < g.width / stride; ++x)
                        grids.push_back({x, y, stride});
            if (grids.size() != static_cast<size_t>(count))
                throw InferenceError("YOLOX anchor count does not match strides 8/16/32; use "
                                     "yolox_decoded or infer_raw for other exports");
        }
        // YOLOv8 is [features, anchors]: scanning class-major keeps every read contiguous.
        std::vector<float> best_score;
        std::vector<int> best_id;
        if (v8) {
            best_score.assign(data + static_cast<size_t>(offset) * count,
                              data + static_cast<size_t>(offset + 1) * count);
            best_id.assign(count, 0);
            for (int cls = 1; cls < classes; ++cls) {
                const float *row = data + static_cast<size_t>(offset + cls) * count;
                for (int i = 0; i < count; ++i)
                    if (row[i] > best_score[i]) {
                        best_score[i] = row[i];
                        best_id[i] = cls;
                    }
            }
        }
        for (int i = 0; i < count; ++i) {
            auto value = [&](int f) {
                return data[v8 ? static_cast<size_t>(f) * count + i
                               : static_cast<size_t>(i) * features + f];
            };
            int id = 0;
            float score;
            if (v8) {
                id = best_id[i];
                score = best_score[i];
            } else {
                // score = class * objectness; sigmoid outputs are <= 1, so an anchor whose
                // objectness is already below the threshold can never qualify.
                const float objectness = value(4);
                if (config.conf_threshold > 0 && objectness < config.conf_threshold)
                    continue;
                for (int cls = 1; cls < classes; ++cls)
                    if (value(offset + cls) > value(offset + id))
                        id = cls;
                score = value(offset + id) * objectness;
            }
            if (score < config.conf_threshold || score < 0 || score > 1)
                continue;
            if (!config.classes.empty() &&
                std::find(config.classes.begin(), config.classes.end(), id) == config.classes.end())
                continue;
            float cx = value(0), cy = value(1), w = value(2), h = value(3);
            if (decoder == "yolox") {
                const auto &grid = grids[i];
                cx = (cx + grid[0]) * grid[2];
                cy = (cy + grid[1]) * grid[2];
                w = std::exp(w) * grid[2];
                h = std::exp(h) * grid[2];
                if (!std::isfinite(w) || !std::isfinite(h))
                    throw InferenceError("YOLOX box size overflow");
            }
            if (decoder == "legacy") {
                cx *= g.width;
                w *= g.width;
                cy *= g.height;
                h *= g.height;
            }
            BBox box((cx - w / 2 - g.pad_x) / g.scale_x, (cy - h / 2 - g.pad_y) / g.scale_y,
                     (cx + w / 2 - g.pad_x) / g.scale_x, (cy + h / 2 - g.pad_y) / g.scale_y);
            box.x1 = std::clamp(box.x1, 0.0f, float(image_size.width));
            box.x2 = std::clamp(box.x2, 0.0f, float(image_size.width));
            box.y1 = std::clamp(box.y1, 0.0f, float(image_size.height));
            box.y2 = std::clamp(box.y2, 0.0f, float(image_size.height));
            if (box.width() > 0 && box.height() > 0)
                result.detections.emplace_back(box, id, label(c, id), score);
        }
        non_max_suppression(result.detections, config);
    } else if (task == TaskType::Segmentation) {
        if (shape.size() != 4 || shape[0] != 1)
            throw InferenceError("Segmentation expects [1,C,H,W]");
        const int classes = static_cast<int>(shape[1]), h = static_cast<int>(shape[2]),
                  w = static_cast<int>(shape[3]);
        if (!c.labels.empty() && static_cast<int>(c.labels.size()) != classes)
            throw InferenceError("labels count does not match classes");
        cv::Mat mask(h, w, CV_32S);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int best = 0;
                for (int cls = 1; cls < classes; ++cls)
                    if (data[(static_cast<size_t>(cls) * h + y) * w + x] >
                        data[(static_cast<size_t>(best) * h + y) * w + x])
                        best = cls;
                mask.at<int>(y, x) = best;
            }
        if (letterbox(c, task)) {
            const auto g = geometry(input, c, task, image_size);
            const int x = static_cast<int>(std::round(float(g.pad_x) * w / g.width));
            const int y = static_cast<int>(std::round(float(g.pad_y) * h / g.height));
            const int rw = std::max(1, w - 2 * x), rh = std::max(1, h - 2 * y);
            mask = mask(cv::Rect(x, y, rw, rh));
        }
        cv::resize(mask, result.segmentation_mask, image_size, 0, 0, cv::INTER_NEAREST);
    }
    return result;
}

Results predict_image(Model &model, const LoadConfig &c, const cv::Mat &image,
                      const InferConfig &config, const StageNotifier &notify) {
    validate_inference(config);
    const auto inputs = model.input_info();
    if (inputs.size() != 1)
        throw InferenceError("Image API requires one input; use infer_raw");
    using clock = std::chrono::steady_clock;
    const auto ms_since = [](clock::time_point from) {
        return std::chrono::duration<double, std::milli>(clock::now() - from).count();
    };
    const auto started = clock::now();
    if (notify)
        notify(CallbackEvent::PreProcess, 0.0, nullptr);
    auto tensor = preprocess_image(image, inputs[0], c, model.task());
    const auto infer_started = clock::now();
    if (notify)
        notify(CallbackEvent::InferStart, 0.0, nullptr);
    auto outputs = model.infer_raw({{inputs[0].name, std::move(tensor)}});
    if (notify)
        notify(CallbackEvent::InferEnd, ms_since(infer_started), nullptr);
    auto result = decode_outputs(outputs, image.size(), inputs[0], c, model.task(), config);
    result.device_used = model.device();
    result.inference_time_ms = static_cast<float>(ms_since(started));
    if (notify)
        notify(CallbackEvent::PostProcess, result.inference_time_ms, &result);
    return result;
}
} // namespace ivit::detail

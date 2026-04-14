/**
 * @file detector.cpp
 * @brief Object detector implementation
 */

#include "ivit/vision/detector.hpp"
#include "ivit/runtime/runtime.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>

namespace ivit {
namespace vision {

// ============================================================================
// COCO class labels (80 classes)
// ============================================================================
static const std::vector<std::string> COCO_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

// ============================================================================
// Detector implementation
// ============================================================================

Detector::Detector(
    const std::string& model_path,
    const std::string& device,
    const LoadConfig& config
) {
    // Create load config with detection task hint
    LoadConfig load_config = config;
    load_config.device = device;
    load_config.task = "detection";

    // Load model
    model_ = ModelManager::instance().load(model_path, load_config);

    // Determine model type from name
    std::string name_lower = model_path;
    std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

    if (name_lower.find("yolov8") != std::string::npos) {
        model_type_ = "yolov8";
    } else if (name_lower.find("yolov5") != std::string::npos) {
        model_type_ = "yolov5";
    } else if (name_lower.find("yolov7") != std::string::npos) {
        model_type_ = "yolov7";
    } else if (name_lower.find("yolox") != std::string::npos) {
        model_type_ = "yolox";
    } else if (name_lower.find("ssd") != std::string::npos) {
        model_type_ = "ssd";
    } else if (name_lower.find("fasterrcnn") != std::string::npos ||
               name_lower.find("faster_rcnn") != std::string::npos) {
        model_type_ = "fasterrcnn";
    } else {
        model_type_ = "yolov8";  // Default
    }

    // Get input size from model (guard against dynamic dims: -1)
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto& shape = input_info[0].shape;
        if (shape.size() >= 4) {
            int w = static_cast<int>(shape[3]);
            int h = static_cast<int>(shape[2]);
            input_size_ = cv::Size(
                w > 0 ? w : 640,
                h > 0 ? h : 640
            );
        } else {
            input_size_ = cv::Size(640, 640);
        }
    } else {
        input_size_ = cv::Size(640, 640);
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

        if (labels_.empty()) {
            labels_ = COCO_LABELS;
        }
    }
}

Detector::Detector(std::shared_ptr<Model> model)
    : model_(model)
    , model_type_("yolov8")
{
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto& shape = input_info[0].shape;
        if (shape.size() >= 4) {
            int w = static_cast<int>(shape[3]);
            int h = static_cast<int>(shape[2]);
            input_size_ = cv::Size(
                w > 0 ? w : 640,
                h > 0 ? h : 640
            );
        } else {
            input_size_ = cv::Size(640, 640);
        }
    } else {
        input_size_ = cv::Size(640, 640);
    }

    labels_ = model_->labels();
    if (labels_.empty()) {
        labels_ = COCO_LABELS;
    }
}

Results Detector::predict(
    const cv::Mat& image,
    float conf_threshold,
    float iou_threshold
) {
    InferConfig config;
    config.conf_threshold = conf_threshold;
    config.iou_threshold = iou_threshold;
    return predict(image, config);
}

Results Detector::predict(
    const std::string& image_path,
    float conf_threshold,
    float iou_threshold
) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw IVITError("Failed to load image: " + image_path);
    }
    return predict(image, conf_threshold, iou_threshold);
}

Results Detector::predict(
    const cv::Mat& image,
    const InferConfig& config
) {
    Results results;
    results.image_size = image.size();

    auto start = std::chrono::high_resolution_clock::now();

    // Preprocess with letterbox
    float scale;
    int pad_w, pad_h;
    cv::Mat processed = preprocess(image, scale, pad_w, pad_h);

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
    results = postprocess(outputs, image.size(), scale, pad_w, pad_h, config);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    results.inference_time_ms = duration.count() / 1000.0f;
    results.device_used = model_->device();
    results.image_size = image.size();

    return results;
}

std::vector<Results> Detector::predict_batch(
    const std::vector<cv::Mat>& images,
    const InferConfig& config
) {
    std::vector<Results> results_list;
    results_list.reserve(images.size());

    for (const auto& image : images) {
        results_list.push_back(predict(image, config));
    }

    return results_list;
}

void Detector::predict_video(
    const std::string& source,
    std::function<void(const Results&, const cv::Mat&)> callback,
    const InferConfig& config
) {
    cv::VideoCapture cap;

    // Try to open as camera index or file
    bool is_camera = false;
    try {
        int cam_id = std::stoi(source);
        cap.open(cam_id);
        is_camera = true;
    } catch (...) {
        cap.open(source);
    }

    if (!cap.isOpened()) {
        throw IVITError("Failed to open video source: " + source);
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        auto results = predict(frame, config);
        callback(results, frame);
    }

    cap.release();
}

cv::Mat Detector::preprocess(
    const cv::Mat& image,
    float& scale,
    int& pad_w,
    int& pad_h
) {
    int orig_h = image.rows;
    int orig_w = image.cols;
    int target_h = input_size_.height;
    int target_w = input_size_.width;

    // Calculate scale (letterbox)
    scale = std::min(
        static_cast<float>(target_w) / orig_w,
        static_cast<float>(target_h) / orig_h
    );

    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);

    pad_w = (target_w - new_w) / 2;
    pad_h = (target_h - new_h) / 2;

    // Resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    // Create letterbox image (gray padding)
    cv::Mat letterbox(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterbox(cv::Rect(pad_w, pad_h, new_w, new_h)));

    // YOLOX expects BGR [0, 255]; other models expect RGB [0, 1]
    cv::Mat float_img;
    if (model_type_ == "yolox") {
        letterbox.convertTo(float_img, CV_32F);
    } else {
        cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
        letterbox.convertTo(float_img, CV_32F, 1.0 / 255.0);
    }

    // Convert to NCHW format
    int h = float_img.rows;
    int w = float_img.cols;
    int c = float_img.channels();

    cv::Mat nchw(1, c * h * w, CV_32F);
    float* ptr = nchw.ptr<float>();

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    for (int ch = 0; ch < c; ch++) {
        cv::Mat channel = channels[ch];
        std::memcpy(
            ptr + ch * h * w,
            channel.data,
            h * w * sizeof(float)
        );
    }

    return nchw;
}

Results Detector::postprocess(
    const std::map<std::string, Tensor>& outputs,
    const cv::Size& orig_size,
    float scale,
    int pad_w,
    int pad_h,
    const InferConfig& config
) {
    if (outputs.empty()) {
        return Results();
    }

    const Tensor& output = outputs.begin()->second;

    if (model_type_.find("yolo") != std::string::npos) {
        return postprocess_yolo(output, orig_size, scale, pad_w, pad_h, config);
    }

    // Generic postprocessing
    return postprocess_yolo(output, orig_size, scale, pad_w, pad_h, config);
}

Results Detector::postprocess_yolo(
    const Tensor& output,
    const cv::Size& orig_size,
    float scale,
    int pad_w,
    int pad_h,
    const InferConfig& config
) {
    Results results;
    const float* data = output.data_ptr<float>();
    const auto& shape = output.shape();

    std::vector<Detection> detections;

    // YOLOv8 output format: [1, 84, 8400] (transposed) or [1, 8400, 84]
    // YOLOv5 output format: [1, 25200, 85]

    bool is_yolov8_format = false;
    int num_predictions, num_values;

    if (shape.size() == 3) {
        if (shape[1] < shape[2]) {
            // YOLOv8 format: [1, 84, 8400]
            is_yolov8_format = true;
            num_values = static_cast<int>(shape[1]);
            num_predictions = static_cast<int>(shape[2]);
        } else {
            // YOLOv5 format: [1, 8400, 85]
            num_predictions = static_cast<int>(shape[1]);
            num_values = static_cast<int>(shape[2]);
        }
    } else if (shape.size() == 2) {
        num_predictions = static_cast<int>(shape[0]);
        num_values = static_cast<int>(shape[1]);
    } else {
        return results;
    }

    int num_classes = num_values - 4;  // YOLOv8: 4 bbox + num_classes
    if (num_values >= 5 && !is_yolov8_format) {
        num_classes = num_values - 5;  // YOLOv5: 4 bbox + 1 obj_conf + num_classes
    }

    // YOLOX grid decode table: strides [8, 16, 32] → grids [80, 40, 20] for 640
    bool is_yolox = (model_type_ == "yolox" && !is_yolov8_format);
    struct StrideLevel { int grid_w; int offset; int stride; };
    std::vector<StrideLevel> stride_levels;
    if (is_yolox) {
        static const int strides[] = {8, 16, 32};
        int cumulative = 0;
        for (int s : strides) {
            int gw = input_size_.width / s;
            int gh = input_size_.height / s;
            stride_levels.push_back({gw, cumulative, s});
            cumulative += gh * gw;
        }
    }

    for (int i = 0; i < num_predictions; i++) {
        float cx, cy, w, h;
        float obj_conf = 1.0f;
        const float* class_scores;
        int class_offset;

        if (is_yolov8_format) {
            // YOLOv8: transposed format [84, 8400]
            cx = data[0 * num_predictions + i];
            cy = data[1 * num_predictions + i];
            w = data[2 * num_predictions + i];
            h = data[3 * num_predictions + i];
            class_offset = 4;
        } else {
            // YOLOv5 / YOLOX: [8400, 85]
            const float* pred = data + i * num_values;
            cx = pred[0];
            cy = pred[1];
            w = pred[2];
            h = pred[3];
            obj_conf = pred[4];
            class_offset = 5;

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

            if (obj_conf < config.conf_threshold) continue;
        }

        // Find best class
        int best_class = 0;
        float best_score = 0;

        for (int c = 0; c < num_classes; c++) {
            float score;
            if (is_yolov8_format) {
                score = data[(class_offset + c) * num_predictions + i];
            } else {
                score = data[i * num_values + class_offset + c];
            }

            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        float confidence = obj_conf * best_score;
        if (confidence < config.conf_threshold) continue;

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

        // Convert from center format to corner format
        // And remove letterbox padding
        float x1 = (cx - w / 2 - pad_w) / scale;
        float y1 = (cy - h / 2 - pad_h) / scale;
        float x2 = (cx + w / 2 - pad_w) / scale;
        float y2 = (cy + h / 2 - pad_h) / scale;

        // Clip to image bounds
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_size.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_size.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_size.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_size.height)));

        Detection det;
        det.bbox = BBox(x1, y1, x2, y2);
        det.class_id = best_class;
        det.confidence = confidence;

        if (best_class < static_cast<int>(labels_.size())) {
            det.label = labels_[best_class];
        } else {
            det.label = "class_" + std::to_string(best_class);
        }

        detections.push_back(det);
    }

    // Apply NMS
    results.detections = nms(detections, config.iou_threshold);

    // Limit to max detections
    if (static_cast<int>(results.detections.size()) > config.max_detections) {
        results.detections.resize(config.max_detections);
    }

    return results;
}

std::vector<Detection> Detector::nms(
    std::vector<Detection>& detections,
    float iou_threshold
) {
    // Sort by confidence
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;

            // Only suppress same class
            if (detections[i].class_id != detections[j].class_id) continue;

            float iou = detections[i].bbox.iou(detections[j].bbox);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

} // namespace vision
} // namespace ivit

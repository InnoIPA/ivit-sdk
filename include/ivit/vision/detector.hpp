/**
 * @file detector.hpp
 * @brief Object detection
 */

#ifndef IVIT_VISION_DETECTOR_HPP
#define IVIT_VISION_DETECTOR_HPP

#include "ivit/core/common.hpp"
#include "ivit/core/model.hpp"
#include "ivit/core/result.hpp"
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace ivit {
namespace vision {

/**
 * @brief Object detector
 *
 * High-level API for object detection tasks.
 * Supports YOLO, SSD, Faster R-CNN and other detection models.
 *
 * @example
 * @code
 * // Create detector
 * auto detector = Detector("yolov8n", "cuda:0");
 *
 * // Detect objects
 * auto results = detector.predict("image.jpg");
 * for (const auto& det : results.detections) {
 *     std::cout << det.label << ": " << det.confidence
 *               << " @ " << det.bbox.x1 << "," << det.bbox.y1 << std::endl;
 * }
 * @endcode
 */
class Detector {
public:
    /**
     * @brief Construct detector
     *
     * @param model Model name or path
     * @param device Target device
     * @param config Additional configuration
     */
    explicit Detector(
        const std::string& model,
        const std::string& device = "auto",
        const LoadConfig& config = LoadConfig{}
    );

    /**
     * @brief Construct from existing model
     */
    explicit Detector(std::shared_ptr<Model> model);

    ~Detector() = default;

    // ========================================================================
    // Inference
    // ========================================================================

    /**
     * @brief Detect objects in image
     *
     * @param image Input image (BGR format)
     * @param conf_threshold Confidence threshold
     * @param iou_threshold NMS IoU threshold
     * @return Detection results
     */
    Results predict(
        const cv::Mat& image,
        float conf_threshold = 0.5f,
        float iou_threshold = 0.45f
    );

    /**
     * @brief Detect objects from file
     */
    Results predict(
        const std::string& image_path,
        float conf_threshold = 0.5f,
        float iou_threshold = 0.45f
    );

    /**
     * @brief Detect with configuration
     */
    Results predict(
        const cv::Mat& image,
        const InferConfig& config
    );

    /**
     * @brief Batch detection
     */
    std::vector<Results> predict_batch(
        const std::vector<cv::Mat>& images,
        const InferConfig& config = InferConfig{}
    );

    /**
     * @brief Video/stream detection
     *
     * @param source Video path or camera ID
     * @param callback Callback for each frame
     * @param config Inference configuration
     */
    void predict_video(
        const std::string& source,
        std::function<void(const Results&, const cv::Mat&)> callback,
        const InferConfig& config = InferConfig{}
    );

    // ========================================================================
    // Properties
    // ========================================================================

    /**
     * @brief Get class labels
     */
    const std::vector<std::string>& classes() const { return labels_; }

    /**
     * @brief Get number of classes
     */
    int num_classes() const { return static_cast<int>(labels_.size()); }

    /**
     * @brief Get input size
     */
    cv::Size input_size() const { return input_size_; }

    /**
     * @brief Get underlying model
     */
    std::shared_ptr<Model> model() const { return model_; }

private:
    std::shared_ptr<Model> model_;
    std::vector<std::string> labels_;
    cv::Size input_size_;
    std::string model_type_;  // "yolov8", "yolov5", "ssd", etc.

    cv::Mat preprocess(const cv::Mat& image, float& scale, int& pad_w, int& pad_h);
    Results postprocess(
        const std::map<std::string, Tensor>& outputs,
        const cv::Size& orig_size,
        float scale,
        int pad_w,
        int pad_h,
        const InferConfig& config
    );

    // YOLO-specific postprocessing
    Results postprocess_yolo(
        const Tensor& output,
        const cv::Size& orig_size,
        float scale,
        int pad_w,
        int pad_h,
        const InferConfig& config
    );

    // Hailo on-device NMS postprocessing
    Results postprocess_hailo_nms(
        const Tensor& output,
        const cv::Size& orig_size,
        float scale,
        int pad_w,
        int pad_h,
        const InferConfig& config
    );

    // NMS
    std::vector<Detection> nms(
        std::vector<Detection>& detections,
        float iou_threshold
    );
};

} // namespace vision
} // namespace ivit

#endif // IVIT_VISION_DETECTOR_HPP

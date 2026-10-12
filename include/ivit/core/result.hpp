/**
 * @file result.hpp
 * @brief Inference result classes
 */

#ifndef IVIT_CORE_RESULT_HPP
#define IVIT_CORE_RESULT_HPP

#include "ivit/core/common.hpp"
#include "ivit/core/tensor.hpp"
#include <opencv2/core.hpp>
#include <vector>
#include <map>
#include <string>
#include <optional>

namespace ivit {

// ============================================================================
// Bounding Box
// ============================================================================

/**
 * @brief Bounding box class
 */
struct BBox {
    float x1 = 0;  ///< Left x coordinate
    float y1 = 0;  ///< Top y coordinate
    float x2 = 0;  ///< Right x coordinate
    float y2 = 0;  ///< Bottom y coordinate

    // Constructors
    BBox() = default;
    BBox(float x1, float y1, float x2, float y2)
        : x1(x1), y1(y1), x2(x2), y2(y2) {}

    // Properties
    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() * height(); }
    float center_x() const { return (x1 + x2) / 2; }
    float center_y() const { return (y1 + y2) / 2; }

    /**
     * @brief Calculate IoU with another box
     */
    float iou(const BBox& other) const {
        float inter_x1 = std::max(x1, other.x1);
        float inter_y1 = std::max(y1, other.y1);
        float inter_x2 = std::min(x2, other.x2);
        float inter_y2 = std::min(y2, other.y2);

        float inter_w = std::max(0.0f, inter_x2 - inter_x1);
        float inter_h = std::max(0.0f, inter_y2 - inter_y1);
        float inter_area = inter_w * inter_h;

        float union_area = area() + other.area() - inter_area;
        return union_area > 0 ? inter_area / union_area : 0;
    }

    /**
     * @brief Convert to [x, y, w, h] format
     */
    std::array<float, 4> to_xywh() const {
        return {x1, y1, width(), height()};
    }

    /**
     * @brief Convert to [cx, cy, w, h] format
     */
    std::array<float, 4> to_cxcywh() const {
        return {center_x(), center_y(), width(), height()};
    }

    /**
     * @brief Convert to cv::Rect
     */
    cv::Rect to_rect() const {
        return cv::Rect(
            static_cast<int>(x1),
            static_cast<int>(y1),
            static_cast<int>(width()),
            static_cast<int>(height())
        );
    }

    /**
     * @brief Create from [x, y, w, h] format
     */
    static BBox from_xywh(float x, float y, float w, float h) {
        return BBox(x, y, x + w, y + h);
    }

    /**
     * @brief Create from [cx, cy, w, h] format
     */
    static BBox from_cxcywh(float cx, float cy, float w, float h) {
        return BBox(cx - w/2, cy - h/2, cx + w/2, cy + h/2);
    }
};

// ============================================================================
// Detection Result
// ============================================================================

/**
 * @brief Single detection result
 */
struct Detection {
    BBox bbox;                          ///< Bounding box
    int class_id = -1;                  ///< Class ID
    std::string label;                  ///< Class label
    float confidence = 0;               ///< Confidence score
    std::optional<cv::Mat> mask;        ///< Instance mask (optional)

    Detection() = default;
    Detection(const BBox& bbox, int class_id, const std::string& label, float conf)
        : bbox(bbox), class_id(class_id), label(label), confidence(conf) {}
};

// ============================================================================
// Classification Result
// ============================================================================

/**
 * @brief Single classification result
 */
struct ClassificationResult {
    int class_id = -1;                  ///< Class ID
    std::string label;                  ///< Class label
    float score = 0;                    ///< Confidence score

    ClassificationResult() = default;
    ClassificationResult(int class_id, const std::string& label, float score)
        : class_id(class_id), label(label), score(score) {}
};

// ============================================================================
// Results Container
// ============================================================================

/**
 * @brief Unified results container
 *
 * Contains inference results for all task types.
 * Only relevant fields are populated based on task.
 */
class Results {
public:
    Results() = default;

    // ========================================================================
    // Classification
    // ========================================================================

    std::vector<ClassificationResult> classifications;

    /**
     * @brief Get top-1 classification result
     */
    const ClassificationResult& top1() const {
        if (classifications.empty()) {
            throw IVITError("No classification results");
        }
        return classifications[0];
    }

    /**
     * @brief Get top-K classification results
     */
    std::vector<ClassificationResult> topk(int k) const {
        int n = std::min(k, static_cast<int>(classifications.size()));
        return std::vector<ClassificationResult>(
            classifications.begin(),
            classifications.begin() + n
        );
    }

    // ========================================================================
    // Detection
    // ========================================================================

    std::vector<Detection> detections;

    /**
     * @brief Get number of detections
     */
    size_t num_detections() const { return detections.size(); }

    /**
     * @brief Filter detections by class
     */
    std::vector<Detection> filter_by_class(const std::vector<int>& class_ids) const;

    /**
     * @brief Filter detections by confidence
     */
    std::vector<Detection> filter_by_confidence(float min_conf) const;

    // ========================================================================
    // Segmentation
    // ========================================================================

    cv::Mat segmentation_mask;          ///< Segmentation mask (H, W)

    /**
     * @brief Colorize segmentation mask
     */
    cv::Mat colorize_mask(const std::map<int, cv::Vec3b>& colormap = {}) const;

    /**
     * @brief Overlay mask on image
     */
    cv::Mat overlay_mask(const cv::Mat& image, float alpha = 0.5) const;

    // ========================================================================
    // Raw Outputs
    // ========================================================================

    std::map<std::string, Tensor> raw_outputs;

    // ========================================================================
    // Metadata
    // ========================================================================

    float inference_time_ms = 0;        ///< Inference time in milliseconds
    std::string device_used;            ///< Device used for inference
    cv::Size image_size;                ///< Original image size

    // ========================================================================
    // Visualization
    // ========================================================================

    /**
     * @brief Visualize results on image
     *
     * @param image Input image (will be modified if not provided)
     * @param show_labels Show class labels
     * @param show_confidence Show confidence scores
     * @param show_boxes Show bounding boxes
     * @param show_masks Show segmentation masks
     * @return Visualized image
     */
    cv::Mat visualize(
        const cv::Mat& image,
        bool show_labels = true,
        bool show_confidence = true,
        bool show_boxes = true,
        bool show_masks = true
    ) const;

    // ========================================================================
    // Serialization
    // ========================================================================

    /**
     * @brief Convert to JSON string
     */
    std::string to_json() const;

    /**
     * @brief Save to file
     */
    void save(const std::string& path, const std::string& format = "json") const;

    // ========================================================================
    // Iteration
    // ========================================================================

    /**
     * @brief Get total number of results
     */
    size_t size() const {
        return std::max(classifications.size(), detections.size());
    }

    bool empty() const { return size() == 0; }
};

} // namespace ivit

#endif // IVIT_CORE_RESULT_HPP

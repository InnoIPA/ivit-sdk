/**
 * @file visualizer.hpp
 * @brief Visualization utilities
 */

#ifndef IVIT_UTILS_VISUALIZER_HPP
#define IVIT_UTILS_VISUALIZER_HPP

#include "ivit/core/result.hpp"
#include <opencv2/core.hpp>
#include <map>
#include <vector>
#include <string>

namespace ivit {
namespace utils {

/**
 * @brief Visualization configuration
 */
struct VisConfig {
    bool show_labels = true;
    bool show_confidence = true;
    bool show_boxes = true;
    bool show_masks = true;
    float font_scale = 0.5f;
    int thickness = 2;
    float mask_alpha = 0.5f;
};

/**
 * @brief Visualizer class for drawing results on images
 */
class Visualizer {
public:
    Visualizer() = default;

    /**
     * @brief Draw detection results
     *
     * @param image Input image (will be modified)
     * @param detections Detection results
     * @param labels Class labels (optional)
     * @param config Visualization config
     * @return Image with drawn detections
     */
    static cv::Mat draw_detections(
        const cv::Mat& image,
        const std::vector<Detection>& detections,
        const std::vector<std::string>& labels = {},
        const VisConfig& config = VisConfig{}
    );

    /**
     * @brief Draw segmentation mask
     *
     * @param image Input image
     * @param mask Segmentation mask (H, W)
     * @param colormap Color mapping
     * @param alpha Transparency
     * @return Image with overlaid mask
     */
    static cv::Mat draw_segmentation(
        const cv::Mat& image,
        const cv::Mat& mask,
        const std::map<int, cv::Vec3b>& colormap = {},
        float alpha = 0.5f
    );

    /**
     * @brief Draw classification results as text
     *
     * @param image Input image
     * @param results Classification results
     * @param top_k Number of results to show
     * @return Image with classification text
     */
    static cv::Mat draw_classification(
        const cv::Mat& image,
        const std::vector<ClassificationResult>& results,
        int top_k = 5
    );

    /**
     * @brief Create side-by-side comparison
     *
     * @param images Vector of images
     * @param titles Titles for each image
     * @param cols Number of columns
     * @return Combined image
     */
    static cv::Mat create_comparison(
        const std::vector<cv::Mat>& images,
        const std::vector<std::string>& titles = {},
        int cols = 2
    );

    /**
     * @brief Get color for class ID
     */
    static cv::Scalar get_color(int class_id);

    /**
     * @brief Generate default colormap
     */
    static std::map<int, cv::Vec3b> generate_colormap(int num_classes);

private:
    static const std::vector<cv::Scalar> COLORS;
};

} // namespace utils
} // namespace ivit

#endif // IVIT_UTILS_VISUALIZER_HPP

/**
 * @file segmentor.hpp
 * @brief Semantic segmentation
 */

#ifndef IVIT_VISION_SEGMENTOR_HPP
#define IVIT_VISION_SEGMENTOR_HPP

#include "ivit/core/common.hpp"
#include "ivit/core/model.hpp"
#include "ivit/core/result.hpp"
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ivit {
namespace vision {

/**
 * @brief Semantic segmentor
 *
 * High-level API for semantic segmentation tasks.
 *
 * @example
 * @code
 * // Create segmentor
 * auto segmentor = Segmentor("deeplabv3_resnet50");
 *
 * // Segment image
 * auto results = segmentor.predict("image.jpg");
 *
 * // Get colorized mask
 * cv::Mat colored = results.colorize_mask();
 * cv::imwrite("segmentation.png", colored);
 *
 * // Overlay on original image
 * cv::Mat overlay = results.overlay_mask(image, 0.5);
 * @endcode
 */
class Segmentor {
  public:
    /**
     * @brief Construct segmentor
     *
     * @param model Model name or path
     * @param device Target device
     * @param config Additional configuration
     */
    explicit Segmentor(const std::string &model, const std::string &device = "auto",
                       const LoadConfig &config = LoadConfig{});

    /**
     * @brief Construct from existing model
     */
    explicit Segmentor(std::shared_ptr<Model> model);

    ~Segmentor() = default;

    // ========================================================================
    // Inference
    // ========================================================================

    /**
     * @brief Segment image
     *
     * @param image Input image (BGR format)
     * @return Segmentation results with mask
     */
    Results predict(const cv::Mat &image);

    /**
     * @brief Segment image from file
     */
    Results predict(const std::string &image_path);

    /**
     * @brief Batch segmentation
     */
    std::vector<Results> predict_batch(const std::vector<cv::Mat> &images);

    // ========================================================================
    // Properties
    // ========================================================================

    /**
     * @brief Get class labels
     */
    const std::vector<std::string> &classes() const { return labels_; }

    /**
     * @brief Get number of classes
     */
    int num_classes() const { return static_cast<int>(labels_.size()); }

    /**
     * @brief Get input size
     */
    cv::Size input_size() const { return input_size_; }

    /**
     * @brief Get default colormap
     */
    const std::map<int, cv::Vec3b> &colormap() const { return colormap_; }

    /**
     * @brief Set custom colormap
     */
    void set_colormap(const std::map<int, cv::Vec3b> &colormap) { colormap_ = colormap; }

    /**
     * @brief Get underlying model
     */
    std::shared_ptr<Model> model() const { return model_; }

  private:
    std::shared_ptr<Model> model_;
    std::vector<std::string> labels_;
    cv::Size input_size_;
    std::map<int, cv::Vec3b> colormap_;

    void init_default_colormap();
};

} // namespace vision
} // namespace ivit

#endif // IVIT_VISION_SEGMENTOR_HPP

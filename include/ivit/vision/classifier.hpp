/**
 * @file classifier.hpp
 * @brief Image classification
 */

#ifndef IVIT_VISION_CLASSIFIER_HPP
#define IVIT_VISION_CLASSIFIER_HPP

#include "ivit/core/common.hpp"
#include "ivit/core/model.hpp"
#include "ivit/core/result.hpp"
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ivit {
namespace vision {

/**
 * @brief Image classifier
 *
 * High-level API for image classification tasks.
 *
 * @example
 * @code
 * // Create classifier
 * auto classifier = Classifier("efficientnet_b0");
 *
 * // Classify image
 * auto results = classifier.predict("image.jpg");
 * std::cout << "Top-1: " << results.top1().label
 *           << " (" << results.top1().score << ")" << std::endl;
 * @endcode
 */
class Classifier {
public:
    /**
     * @brief Construct classifier
     *
     * @param model Model name (from Model Zoo) or path to model file
     * @param device Target device ("auto", "cpu", "cuda:0", etc.)
     * @param config Additional configuration
     */
    explicit Classifier(
        const std::string& model,
        const std::string& device = "auto",
        const LoadConfig& config = LoadConfig{}
    );

    /**
     * @brief Construct from existing model
     */
    explicit Classifier(std::shared_ptr<Model> model);

    ~Classifier() = default;

    // ========================================================================
    // Inference
    // ========================================================================

    /**
     * @brief Classify a single image
     *
     * @param image Input image (BGR format)
     * @param top_k Return top K results
     * @return Classification results
     */
    Results predict(const cv::Mat& image, int top_k = 5);

    /**
     * @brief Classify image from file
     *
     * @param image_path Path to image file
     * @param top_k Return top K results
     * @return Classification results
     */
    Results predict(const std::string& image_path, int top_k = 5);

    /**
     * @brief Batch classification
     *
     * @param images Vector of input images
     * @param top_k Return top K results
     * @return Vector of classification results
     */
    std::vector<Results> predict_batch(
        const std::vector<cv::Mat>& images,
        int top_k = 5
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
    // Optional CPU-side head for Hailo backbone-only HEFs (e.g.
    // mambavision-t whose SSM/Transformer can't translate to Hailo).
    // Auto-loaded from "<model_stem>_head.onnx" next to the main model.
    // See Bug #15 in docs/development/hailo-compile-bug-log.md.
    std::shared_ptr<Model> cpu_head_;
    std::vector<std::string> labels_;
    cv::Size input_size_;
    std::vector<float> mean_;
    std::vector<float> std_;

    cv::Mat preprocess(const cv::Mat& image);
    Results postprocess(const std::map<std::string, Tensor>& outputs, int top_k);
    Results postprocess_hailo(
        const std::map<std::string, Tensor>& outputs, int top_k);
};

} // namespace vision
} // namespace ivit

#endif // IVIT_VISION_CLASSIFIER_HPP

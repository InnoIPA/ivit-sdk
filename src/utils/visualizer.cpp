/**
 * @file visualizer.cpp
 * @brief Visualization utilities implementation
 */

#include "ivit/utils/visualizer.hpp"
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace ivit {
namespace utils {

// ============================================================================
// Color palette (20 distinct colors for visualization)
// ============================================================================

const std::vector<cv::Scalar> Visualizer::COLORS = {
    {56, 56, 255},     // Red
    {50, 205, 50},     // Lime Green
    {255, 144, 30},    // Dodger Blue
    {0, 215, 255},     // Gold
    {255, 0, 255},     // Magenta
    {255, 255, 0},     // Cyan
    {0, 128, 128},     // Olive
    {128, 0, 128},     // Purple
    {0, 165, 255},     // Orange
    {203, 192, 255},   // Pink
    {128, 128, 0},     // Teal
    {60, 20, 220},     // Crimson
    {230, 216, 173},   // Light Steel Blue
    {144, 238, 144},   // Light Green
    {147, 112, 219},   // Medium Purple
    {255, 182, 193},   // Light Pink
    {0, 100, 0},       // Dark Green
    {139, 0, 0},       // Dark Blue
    {0, 139, 139},     // Dark Cyan
    {169, 169, 169},   // Dark Gray
};

// ============================================================================
// Visualizer implementation
// ============================================================================

cv::Scalar Visualizer::get_color(int class_id) {
    return COLORS[class_id % COLORS.size()];
}

std::map<int, cv::Vec3b> Visualizer::generate_colormap(int num_classes) {
    std::map<int, cv::Vec3b> colormap;

    // Background is always black
    colormap[0] = cv::Vec3b(0, 0, 0);

    for (int i = 1; i < num_classes; i++) {
        // Generate distinct colors using HSV color space
        float hue = static_cast<float>(i - 1) / static_cast<float>(num_classes - 1) * 180.0f;
        float saturation = 200.0f + (i % 3) * 20.0f;  // Vary saturation slightly
        float value = 180.0f + (i % 5) * 15.0f;       // Vary brightness slightly

        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, saturation, value));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        colormap[i] = bgr.at<cv::Vec3b>(0, 0);
    }

    return colormap;
}

cv::Mat Visualizer::draw_detections(
    const cv::Mat& image,
    const std::vector<Detection>& detections,
    const std::vector<std::string>& labels,
    const VisConfig& config
) {
    cv::Mat output = image.clone();

    for (const auto& det : detections) {
        cv::Scalar color = get_color(det.class_id);

        // Get label text
        std::string label_text;
        if (config.show_labels) {
            if (!det.label.empty()) {
                label_text = det.label;
            } else if (det.class_id >= 0 &&
                       static_cast<size_t>(det.class_id) < labels.size()) {
                label_text = labels[det.class_id];
            } else {
                label_text = "class_" + std::to_string(det.class_id);
            }
        }

        if (config.show_confidence) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << det.confidence;
            if (!label_text.empty()) {
                label_text += " " + ss.str();
            } else {
                label_text = ss.str();
            }
        }

        // Draw bounding box
        if (config.show_boxes) {
            cv::Point pt1(static_cast<int>(det.bbox.x1), static_cast<int>(det.bbox.y1));
            cv::Point pt2(static_cast<int>(det.bbox.x2), static_cast<int>(det.bbox.y2));
            cv::rectangle(output, pt1, pt2, color, config.thickness);
        }

        // Draw label background and text
        if (!label_text.empty()) {
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(
                label_text,
                cv::FONT_HERSHEY_SIMPLEX,
                config.font_scale,
                1,
                &baseline
            );

            int label_x = static_cast<int>(det.bbox.x1);
            int label_y = static_cast<int>(det.bbox.y1) - text_size.height - 4;

            // Ensure label stays within image bounds
            label_y = std::max(text_size.height + 4, label_y);
            label_x = std::max(0, label_x);

            // Draw label background
            cv::rectangle(
                output,
                cv::Point(label_x, label_y - text_size.height - 4),
                cv::Point(label_x + text_size.width + 4, label_y + 4),
                color,
                -1  // Filled
            );

            // Draw label text
            cv::putText(
                output,
                label_text,
                cv::Point(label_x + 2, label_y),
                cv::FONT_HERSHEY_SIMPLEX,
                config.font_scale,
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA
            );
        }

        // Draw instance mask if available
        if (det.mask.has_value() && config.show_masks) {
            cv::Mat mask = det.mask.value();
            cv::Mat mask_resized;

            // Resize mask to match bounding box if needed
            int box_w = static_cast<int>(det.bbox.width());
            int box_h = static_cast<int>(det.bbox.height());

            if (mask.size() != cv::Size(box_w, box_h)) {
                cv::resize(mask, mask_resized, cv::Size(box_w, box_h), 0, 0, cv::INTER_LINEAR);
            } else {
                mask_resized = mask;
            }

            // Create colored mask
            cv::Mat colored_mask(output.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            cv::Rect roi(
                static_cast<int>(det.bbox.x1),
                static_cast<int>(det.bbox.y1),
                box_w,
                box_h
            );

            // Clip ROI to image bounds
            roi &= cv::Rect(0, 0, output.cols, output.rows);
            if (roi.width > 0 && roi.height > 0) {
                cv::Mat roi_mask = mask_resized(cv::Rect(0, 0, roi.width, roi.height));
                cv::Mat roi_colored = colored_mask(roi);

                for (int y = 0; y < roi.height; y++) {
                    for (int x = 0; x < roi.width; x++) {
                        if (roi_mask.at<uchar>(y, x) > 127) {
                            roi_colored.at<cv::Vec3b>(y, x) = cv::Vec3b(
                                static_cast<uchar>(color[0]),
                                static_cast<uchar>(color[1]),
                                static_cast<uchar>(color[2])
                            );
                        }
                    }
                }

                cv::addWeighted(output, 1.0, colored_mask, config.mask_alpha, 0, output);
            }
        }
    }

    return output;
}

cv::Mat Visualizer::draw_segmentation(
    const cv::Mat& image,
    const cv::Mat& mask,
    const std::map<int, cv::Vec3b>& colormap,
    float alpha
) {
    if (mask.empty()) {
        return image.clone();
    }

    cv::Mat output = image.clone();
    cv::Mat colored_mask(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    // Resize mask if needed
    cv::Mat resized_mask;
    if (mask.size() != image.size()) {
        cv::resize(mask, resized_mask, image.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        resized_mask = mask;
    }

    // Generate colormap if not provided
    std::map<int, cv::Vec3b> cmap = colormap;
    if (cmap.empty()) {
        double min_val, max_val;
        cv::minMaxLoc(resized_mask, &min_val, &max_val);
        cmap = generate_colormap(static_cast<int>(max_val) + 1);
    }

    // Apply colormap
    for (int y = 0; y < resized_mask.rows; y++) {
        for (int x = 0; x < resized_mask.cols; x++) {
            int class_id;
            if (resized_mask.type() == CV_32S) {
                class_id = resized_mask.at<int>(y, x);
            } else if (resized_mask.type() == CV_8U) {
                class_id = resized_mask.at<uchar>(y, x);
            } else {
                class_id = static_cast<int>(resized_mask.at<float>(y, x));
            }

            if (cmap.count(class_id)) {
                colored_mask.at<cv::Vec3b>(y, x) = cmap.at(class_id);
            }
        }
    }

    // Blend with original image
    cv::addWeighted(output, 1.0 - alpha, colored_mask, alpha, 0, output);

    return output;
}

cv::Mat Visualizer::draw_classification(
    const cv::Mat& image,
    const std::vector<ClassificationResult>& results,
    int top_k
) {
    cv::Mat output = image.clone();

    int n = std::min(top_k, static_cast<int>(results.size()));
    int y_offset = 30;
    int line_height = 25;

    // Draw semi-transparent background
    int bg_height = n * line_height + 20;
    int bg_width = 300;
    cv::Rect bg_rect(10, 10, bg_width, bg_height);
    bg_rect &= cv::Rect(0, 0, output.cols, output.rows);

    if (bg_rect.width > 0 && bg_rect.height > 0) {
        cv::Mat roi = output(bg_rect);
        cv::Mat bg(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
        cv::addWeighted(roi, 0.5, bg, 0.5, 0, roi);
    }

    // Draw results
    for (int i = 0; i < n; i++) {
        const auto& result = results[i];

        // Format text
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1);
        ss << (i + 1) << ". ";
        if (!result.label.empty()) {
            ss << result.label;
        } else {
            ss << "class_" << result.class_id;
        }
        ss << " (" << (result.score * 100.0f) << "%)";

        // Color based on rank
        cv::Scalar color;
        if (i == 0) {
            color = cv::Scalar(0, 255, 0);   // Green for top-1
        } else if (i == 1) {
            color = cv::Scalar(0, 200, 255); // Orange for top-2
        } else {
            color = cv::Scalar(200, 200, 200); // Gray for others
        }

        cv::putText(
            output,
            ss.str(),
            cv::Point(20, y_offset + i * line_height),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv::LINE_AA
        );

        // Draw confidence bar
        int bar_x = 20;
        int bar_y = y_offset + i * line_height + 5;
        int bar_width = static_cast<int>(result.score * 200);
        int bar_height = 4;

        cv::rectangle(
            output,
            cv::Point(bar_x, bar_y),
            cv::Point(bar_x + bar_width, bar_y + bar_height),
            color,
            -1
        );
    }

    return output;
}

cv::Mat Visualizer::create_comparison(
    const std::vector<cv::Mat>& images,
    const std::vector<std::string>& titles,
    int cols
) {
    if (images.empty()) {
        return cv::Mat();
    }

    // Determine grid size
    int n = static_cast<int>(images.size());
    int rows = (n + cols - 1) / cols;

    // Find max size
    int max_w = 0, max_h = 0;
    for (const auto& img : images) {
        max_w = std::max(max_w, img.cols);
        max_h = std::max(max_h, img.rows);
    }

    // Add space for titles
    int title_height = 30;
    int cell_h = max_h + title_height;
    int cell_w = max_w;

    // Create output canvas
    cv::Mat output(rows * cell_h, cols * cell_w, CV_8UC3, cv::Scalar(40, 40, 40));

    for (int i = 0; i < n; i++) {
        int row = i / cols;
        int col = i % cols;

        int x = col * cell_w;
        int y = row * cell_h;

        // Draw title
        if (static_cast<size_t>(i) < titles.size() && !titles[i].empty()) {
            cv::putText(
                output,
                titles[i],
                cv::Point(x + 10, y + 20),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA
            );
        }

        // Draw image
        cv::Mat img = images[i];
        if (img.channels() == 1) {
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        }

        // Resize if needed
        if (img.cols != max_w || img.rows != max_h) {
            float scale = std::min(
                static_cast<float>(max_w) / img.cols,
                static_cast<float>(max_h) / img.rows
            );
            cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_LINEAR);
        }

        // Copy to output
        cv::Rect roi(
            x + (cell_w - img.cols) / 2,
            y + title_height + (max_h - img.rows) / 2,
            img.cols,
            img.rows
        );
        img.copyTo(output(roi));

        // Draw border
        cv::rectangle(
            output,
            cv::Point(x, y),
            cv::Point(x + cell_w - 1, y + cell_h - 1),
            cv::Scalar(80, 80, 80),
            1
        );
    }

    return output;
}

} // namespace utils
} // namespace ivit

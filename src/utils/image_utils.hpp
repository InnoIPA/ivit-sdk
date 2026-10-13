#pragma once
#include "ivit/core/common.hpp"
#include <opencv2/imgproc.hpp>

namespace ivit::detail {

/// Drawing helpers blend colour layers into the image, which fails (an OpenCV exception) for the
/// 1-channel and 4-channel images that predict() accepts. Return a 3-channel BGR view/copy.
inline cv::Mat ensure_bgr(const cv::Mat &image) {
    if (image.channels() == 3)
        return image;
    cv::Mat bgr;
    if (image.channels() == 1)
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    else if (image.channels() == 4)
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    else
        throw IVITError("Cannot draw on an image with " + std::to_string(image.channels()) +
                        " channels");
    return bgr;
}

} // namespace ivit::detail

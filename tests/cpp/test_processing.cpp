#include "utils/fingerprint.hpp"
#include "vision/processing.hpp"
#include "core/device_ranking.hpp"
#include <gtest/gtest.h>

using namespace ivit;

TEST(CacheFingerprint, Sha256MatchesPublishedTestVectors) {
    EXPECT_EQ(detail::fingerprint(""),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(detail::fingerprint("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(detail::fingerprint(std::string(1000000, 'a')),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(ModelContract, YoloxRawGridAndTopLeftPadding) {
    LoadConfig contract;
    contract.decoder = "yolox";
    contract.resize_mode = "letterbox_top_left";
    TensorInfo input{"input", {1, 3, 64, 64}, DataType::Float32, Layout::NCHW};
    Tensor output({1, 84, 6}, DataType::Float32);
    std::fill_n(output.data_ptr<float>(), output.numel(), 0.f);
    auto box = output.data_ptr<float>() + 9 * 6; // stride8 grid (1,1)
    box[0] = box[1] = 2;
    box[2] = box[3] = std::log(2.f);
    box[4] = box[5] = .9f;
    auto result = detail::decode_outputs({{"output", output}}, {64, 32}, input, contract,
                                         TaskType::Detection, {});
    ASSERT_EQ(result.detections.size(), 1);
    EXPECT_FLOAT_EQ(result.detections[0].bbox.x1, 16);
    EXPECT_FLOAT_EQ(result.detections[0].bbox.y1, 16);
    EXPECT_FLOAT_EQ(result.detections[0].bbox.x2, 32);
    EXPECT_FLOAT_EQ(result.detections[0].bbox.y2, 32);
}

TEST(ModelContract, NormalizationColorAndLayoutHaveKnownValues) {
    LoadConfig config;
    config.input_scale = 0.1f;
    config.mean = {1, 2, 3};
    config.std = {2, 2, 2};
    cv::Mat image(2, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    TensorInfo input{"input", {1, 3, 2, 4}, DataType::Float32, Layout::NCHW};
    auto result = detail::preprocess_image(image, input, config, TaskType::Classification);
    EXPECT_FLOAT_EQ(result.data_ptr<float>()[0], 1);
    EXPECT_FLOAT_EQ(result.data_ptr<float>()[8], 0);
    EXPECT_FLOAT_EQ(result.data_ptr<float>()[16], -1);
    config.input_layout = "nhwc";
    input.shape = {1, 2, 4, 3};
    result = detail::preprocess_image(image, input, config, TaskType::Classification);
    EXPECT_FLOAT_EQ(result.data_ptr<float>()[0], 1);
    EXPECT_FLOAT_EQ(result.data_ptr<float>()[1], 0);
    EXPECT_FLOAT_EQ(result.data_ptr<float>()[2], -1);
}

TEST(ModelContract, LetterboxCoordinatesAndClassAwareNms) {
    LoadConfig config;
    config.decoder = "yolov8";
    config.labels = {"first", "second"};
    TensorInfo input{"input", {1, 3, 64, 64}, DataType::Float32, Layout::NCHW};
    Tensor output({1, 6, 2}, DataType::Float32);
    const float values[] = {32, 32, 32, 32, 32, 32, 16, 16, .9f, .1f, .1f, .8f};
    std::memcpy(output.data(), values, sizeof(values));
    InferConfig infer;
    infer.conf_threshold = .2f;
    auto result = detail::decode_outputs({{"output", output}}, {40, 20}, input, config,
                                         TaskType::Detection, infer);
    ASSERT_EQ(result.detections.size(), 2);
    const auto &box = result.detections[0].bbox;
    EXPECT_NEAR(box.x1, 10, 1e-5);
    EXPECT_NEAR(box.y1, 5, 1e-5);
    EXPECT_NEAR(box.x2, 30, 1e-5);
    EXPECT_NEAR(box.y2, 15, 1e-5);
    infer.classes = {1};
    result = detail::decode_outputs({{"output", output}}, {40, 20}, input, config,
                                    TaskType::Detection, infer);
    ASSERT_EQ(result.detections.size(), 1);
    EXPECT_EQ(result.detections[0].label, "second");
}

TEST(ModelContract, ProbabilitiesAreNotSoftmaxedAgain) {
    LoadConfig config;
    config.decoder = "probabilities";
    config.labels = {"a\"b", "line\nback\\slash"};
    TensorInfo input{"input", {1, 3, 2, 2}, DataType::Float32, Layout::NCHW};
    Tensor output({1, 2}, DataType::Float32);
    output.data_ptr<float>()[0] = .2f;
    output.data_ptr<float>()[1] = .8f;
    InferConfig infer;
    infer.conf_threshold = 0;
    auto result = detail::decode_outputs({{"output", output}}, {2, 2}, input, config,
                                         TaskType::Classification, infer);
    ASSERT_EQ(result.classifications.size(), 2);
    EXPECT_FLOAT_EQ(result.top1().score, .8f);
    auto json = result.to_json();
    EXPECT_NE(json.find("a\\\"b"), std::string::npos);
    EXPECT_NE(json.find("line\\u000aback\\\\slash"), std::string::npos);
    config.labels.push_back("extra");
    EXPECT_THROW(detail::decode_outputs({{"output", output}}, {2, 2}, input, config,
                                        TaskType::Classification, infer),
                 InferenceError);
}

TEST(ModelContract, SemanticMaskArgmax) {
    LoadConfig config;
    TensorInfo input{"input", {1, 3, 2, 2}, DataType::Float32, Layout::NCHW};
    Tensor output({1, 2, 2, 2}, DataType::Float32);
    const float values[] = {0, 2, 0, 2, 1, 0, 1, 0};
    std::memcpy(output.data(), values, sizeof(values));
    auto result = detail::decode_outputs({{"output", output}}, {2, 2}, input, config,
                                         TaskType::Segmentation, InferConfig{});
    EXPECT_EQ(result.segmentation_mask.at<int>(0, 0), 1);
    EXPECT_EQ(result.segmentation_mask.at<int>(0, 1), 0);
    EXPECT_EQ(result.segmentation_mask.at<int>(1, 0), 1);
}

TEST(ModelContract, RejectsAmbiguousOrMalformedOutputs) {
    LoadConfig config;
    TensorInfo input{"input", {1, 3, 2, 2}, DataType::Float32, Layout::NCHW};
    Tensor bad({1, 3}, DataType::Float32);
    EXPECT_THROW(detail::decode_outputs({{"x", bad}}, {2, 2}, input, config, TaskType::Detection,
                                        InferConfig{}),
                 InferenceError);
    EXPECT_THROW(detail::decode_outputs({{"x", bad}, {"y", bad}}, {2, 2}, input, config,
                                        TaskType::Classification, InferConfig{}),
                 InferenceError);
    input.shape.clear();
    EXPECT_THROW(
        detail::preprocess_image(cv::Mat(2, 2, CV_8UC3), input, config, TaskType::Classification),
        InferenceError);
    config.std = {0};
    EXPECT_THROW(detail::validate_contract(config), IVITError);
}

// ---- preprocessing / decoding fast paths: pinned to values computed by hand ----

namespace {
// Straightforward per-pixel reference for (pixel * scale - mean) / std.
float normalized(float pixel, const LoadConfig &c, int channel) {
    return (pixel * c.input_scale - c.mean[c.mean.size() == 1 ? 0 : channel]) /
           c.std[c.std.size() == 1 ? 0 : channel];
}
} // namespace

TEST(Preprocess, DistinctChannelsLandInTheRightPlaneForEveryColorAndLayout) {
    cv::Mat image(2, 3, CV_8UC3);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 3; ++x)
            image.at<cv::Vec3b>(y, x) = {uint8_t(10 + x), uint8_t(50 + y), uint8_t(200 - x - y)};
    for (const char *color : {"rgb", "bgr"})
        for (const char *layout : {"nchw", "nhwc"}) {
            LoadConfig c;
            c.input_color = color;
            c.input_layout = layout;
            c.input_scale = 0.5f;
            c.mean = {1, 2, 3};
            c.std = {2, 4, 5};
            const bool nhwc = c.input_layout == "nhwc";
            TensorInfo input{"x", nhwc ? Shape{1, 2, 3, 3} : Shape{1, 3, 2, 3}, DataType::Float32,
                             nhwc ? Layout::NHWC : Layout::NCHW};
            auto tensor = detail::preprocess_image(image, input, c, TaskType::Classification);
            const float *out = tensor.data_ptr<float>();
            for (int ch = 0; ch < 3; ++ch)
                for (int y = 0; y < 2; ++y)
                    for (int x = 0; x < 3; ++x) {
                        const int source = c.input_color == "rgb" ? 2 - ch : ch; // BGR image
                        const float expected =
                            normalized(image.at<cv::Vec3b>(y, x)[source], c, ch);
                        const size_t index = nhwc ? (y * 3 + x) * 3 + ch : (ch * 2 + y) * 3 + x;
                        EXPECT_NEAR(out[index], expected, 1e-5)
                            << color << ' ' << layout << " ch" << ch << " (" << x << ',' << y << ')';
                    }
        }
}

TEST(Preprocess, GrayscaleModelAcceptsColorAndGrayImages) {
    LoadConfig c;
    c.input_scale = 1.f / 255.f;
    c.mean = {0.5f};
    c.std = {0.25f};
    TensorInfo input{"x", {1, 1, 2, 2}, DataType::Float32, Layout::NCHW};
    cv::Mat gray(2, 2, CV_8UC1, cv::Scalar(255));
    auto from_gray = detail::preprocess_image(gray, input, c, TaskType::Classification);
    EXPECT_NEAR(from_gray.data_ptr<float>()[3], (1.f - 0.5f) / 0.25f, 1e-5);
    cv::Mat color(2, 2, CV_8UC3, cv::Scalar(255, 255, 255));
    auto from_color = detail::preprocess_image(color, input, c, TaskType::Classification);
    EXPECT_NEAR(from_color.data_ptr<float>()[0], (1.f - 0.5f) / 0.25f, 1e-5);
}

TEST(Preprocess, LetterboxPaddingUsesGray114AndKeepsImageInPlace) {
    for (const char *mode : {"letterbox", "letterbox_top_left"}) {
        LoadConfig c;
        c.resize_mode = mode;
        c.input_color = "bgr";
        c.input_scale = 1.f;
        c.mean = {0};
        c.std = {1};
        TensorInfo input{"x", {1, 3, 8, 8}, DataType::Float32, Layout::NCHW};
        cv::Mat wide(2, 8, CV_8UC3, cv::Scalar(10, 20, 30)); // scaled x1 -> 8x2, pad 6 rows
        auto tensor = detail::preprocess_image(wide, input, c, TaskType::Detection);
        const float *blue = tensor.data_ptr<float>();
        const int top = std::string(mode) == "letterbox" ? 3 : 0;
        EXPECT_FLOAT_EQ(blue[top * 8], 10) << mode;                // inside the image
        EXPECT_FLOAT_EQ(blue[(top + 1) * 8 + 7], 10) << mode;
        EXPECT_FLOAT_EQ(blue[7 * 8], 114) << mode;                 // bottom padding
        EXPECT_FLOAT_EQ(blue[top == 0 ? 2 * 8 : 0], 114) << mode;  // row outside the image
    }
}

TEST(Preprocess, Float16InputIsConvertedFromTheNormalizedValues) {
    LoadConfig c;
    c.input_scale = 1.f / 255.f;
    TensorInfo input{"x", {1, 3, 2, 2}, DataType::Float16, Layout::NCHW};
    cv::Mat image(2, 2, CV_8UC3, cv::Scalar(255, 0, 51));
    auto tensor = detail::preprocess_image(image, input, c, TaskType::Classification);
    ASSERT_EQ(tensor.dtype(), DataType::Float16);
    cv::Mat half(1, 12, CV_16F, tensor.data()), wide;
    half.convertTo(wide, CV_32F);
    EXPECT_NEAR(wide.at<float>(0, 0), 51.f / 255.f, 1e-3); // R plane first
    EXPECT_NEAR(wide.at<float>(0, 4), 0.f, 1e-3);          // G
    EXPECT_NEAR(wide.at<float>(0, 8), 1.f, 1e-3);          // B
}

TEST(Decode, ObjectnessEarlyRejectMatchesAFullScan) {
    // 3 anchors: low objectness/high class, high objectness, borderline objectness.
    LoadConfig c;
    c.decoder = "yolov5";
    c.resize_mode = "stretch";
    TensorInfo input{"x", {1, 3, 64, 64}, DataType::Float32, Layout::NCHW};
    Tensor output({1, 3, 7}, DataType::Float32); // 2 classes
    const float rows[3][7] = {{20, 20, 10, 10, 0.1f, 1.0f, 0.0f},
                              {40, 40, 10, 10, 0.9f, 0.2f, 0.95f},
                              {10, 50, 8, 8, 0.5f, 0.7f, 0.1f}};
    std::memcpy(output.data(), rows, sizeof(rows));
    InferConfig infer;
    infer.conf_threshold = 0.3f;
    auto result =
        detail::decode_outputs({{"o", output}}, {64, 64}, input, c, TaskType::Detection, infer);
    ASSERT_EQ(result.detections.size(), 2);
    EXPECT_EQ(result.detections[0].class_id, 1);
    EXPECT_NEAR(result.detections[0].confidence, 0.9f * 0.95f, 1e-6);
    EXPECT_NEAR(result.detections[1].confidence, 0.5f * 0.7f, 1e-6);
    infer.conf_threshold = 0; // threshold 0 disables early rejection: all 3 are scored
    result = detail::decode_outputs({{"o", output}}, {64, 64}, input, c, TaskType::Detection, infer);
    EXPECT_EQ(result.detections.size(), 3);
}

TEST(Decode, Yolov8ClassMajorScanKeepsFirstClassOnTies) {
    LoadConfig c;
    c.decoder = "yolov8";
    c.resize_mode = "stretch";
    TensorInfo input{"x", {1, 3, 64, 64}, DataType::Float32, Layout::NCHW};
    Tensor output({1, 7, 2}, DataType::Float32); // 3 classes, 2 anchors, [features, anchors]
    const float values[] = {20, 50, 20, 50, 10, 10, 10, 10, // cx cy w h
                            0.8f, 0.1f,                      // class 0
                            0.8f, 0.9f,                      // class 1: ties anchor 0, wins anchor 1
                            0.2f, 0.3f};                     // class 2
    std::memcpy(output.data(), values, sizeof(values));
    InferConfig infer;
    infer.conf_threshold = 0.5f;
    auto result =
        detail::decode_outputs({{"o", output}}, {64, 64}, input, c, TaskType::Detection, infer);
    ASSERT_EQ(result.detections.size(), 2);
    EXPECT_EQ(result.detections[0].class_id, 1);  // anchor 1, 0.9
    EXPECT_EQ(result.detections[1].class_id, 0);  // anchor 0: tie resolves to the first class
}

TEST(Decode, NonFiniteOutputsAreRejectedForBothPrecisions) {
    LoadConfig c;
    c.decoder = "probabilities";
    TensorInfo input{"x", {1, 3, 2, 2}, DataType::Float32, Layout::NCHW};
    for (float bad : {NAN, INFINITY, -INFINITY}) {
        Tensor output({1, 2}, DataType::Float32);
        output.data_ptr<float>()[0] = 0.5f;
        output.data_ptr<float>()[1] = bad;
        EXPECT_THROW(detail::decode_outputs({{"o", output}}, {2, 2}, input, c,
                                            TaskType::Classification, InferConfig{}),
                     InferenceError);
    }
}

TEST(Decode, NmsIsPerClassByDefaultAndCrossClassWhenAgnostic) {
    LoadConfig c;
    c.decoder = "yolov8";
    c.resize_mode = "stretch";
    TensorInfo input{"x", {1, 3, 64, 64}, DataType::Float32, Layout::NCHW};
    // Two almost identical boxes with different classes, plus one far away.
    Tensor output({1, 6, 3}, DataType::Float32);
    const float values[] = {20, 21, 50,   // cx
                            20, 20, 50,   // cy
                            10, 10, 8,    // w
                            10, 10, 8,    // h
                            0.9f, 0.1f, 0.8f,   // class 0
                            0.1f, 0.8f, 0.1f};  // class 1
    std::memcpy(output.data(), values, sizeof(values));
    InferConfig infer;
    infer.conf_threshold = 0.5f;
    auto per_class =
        detail::decode_outputs({{"o", output}}, {64, 64}, input, c, TaskType::Detection, infer);
    EXPECT_EQ(per_class.detections.size(), 3); // overlap survives across classes
    infer.agnostic_nms = true;
    auto agnostic =
        detail::decode_outputs({{"o", output}}, {64, 64}, input, c, TaskType::Detection, infer);
    ASSERT_EQ(agnostic.detections.size(), 2);
    EXPECT_EQ(agnostic.detections[0].class_id, 0); // the stronger duplicate wins
    EXPECT_FLOAT_EQ(agnostic.detections[0].confidence, 0.9f);
}

TEST(Labels, ExpectedClassCountFollowsTheDecoderAndOutputShape) {
    auto output = [](Shape shape, Layout layout = Layout::Unknown, std::string name = "o") {
        return std::vector<TensorInfo>{{name, shape, DataType::Float32, layout}};
    };
    LoadConfig c;
    // Classification: [C] or [1,C], also with a dynamic batch.
    EXPECT_EQ(detail::expected_class_count(output({1, 5}), c, TaskType::Classification), 5);
    EXPECT_EQ(detail::expected_class_count(output({7}), c, TaskType::Classification), 7);
    EXPECT_EQ(detail::expected_class_count(output({-1, 9}), c, TaskType::Classification), 9);
    EXPECT_EQ(detail::expected_class_count(output({1, -1}), c, TaskType::Classification), -1);
    // Segmentation: [1,C,H,W].
    EXPECT_EQ(detail::expected_class_count(output({1, 21, 8, 8}), c, TaskType::Segmentation), 21);
    // Detection by decoder.
    c.decoder = "yolov8";
    EXPECT_EQ(detail::expected_class_count(output({1, 84, 8400}), c, TaskType::Detection), 80);
    c.decoder = "yolox_decoded";
    EXPECT_EQ(detail::expected_class_count(output({1, 8400, 85}), c, TaskType::Detection), 80);
    c.decoder = "auto";
    EXPECT_EQ(detail::expected_class_count(output({1, 84, 8400}), c, TaskType::Detection), 80);
    EXPECT_EQ(detail::expected_class_count(output({1, 8400, 85}), c, TaskType::Detection), 80);
    EXPECT_EQ(detail::expected_class_count(output({1, 8, 8}), c, TaskType::Detection), -1);
    EXPECT_EQ(detail::expected_class_count(output({1, -1, 85}), c, TaskType::Detection), -1);
    // Unknown: raw decoder, untyped task, several outputs, or a named output that is absent.
    c.decoder = "raw";
    EXPECT_EQ(detail::expected_class_count(output({1, 5}), c, TaskType::Classification), -1);
    c.decoder = "auto";
    EXPECT_EQ(detail::expected_class_count(output({1, 5}), c, TaskType::Unknown), -1);
    std::vector<TensorInfo> two = {{"a", {1, 3}, DataType::Float32, Layout::NC},
                                   {"b", {1, 4}, DataType::Float32, Layout::NC}};
    EXPECT_EQ(detail::expected_class_count(two, c, TaskType::Classification), -1);
    c.output_name = "b";
    EXPECT_EQ(detail::expected_class_count(two, c, TaskType::Classification), 4);
    c.output_name = "missing";
    EXPECT_EQ(detail::expected_class_count(two, c, TaskType::Classification), -1);
}

TEST(Confidence, UnsetMeansTheTaskDefault) {
    InferConfig unset; // conf_threshold defaults to -1
    EXPECT_EQ(unset.conf_threshold, -1.0f);
    EXPECT_FLOAT_EQ(detail::effective_confidence(unset, TaskType::Detection), 0.5f);
    EXPECT_FLOAT_EQ(detail::effective_confidence(unset, TaskType::Classification), 0.0f);
    EXPECT_FLOAT_EQ(detail::effective_confidence(unset, TaskType::Segmentation), 0.0f);
    InferConfig explicit_value;
    explicit_value.conf_threshold = 0.25f;
    EXPECT_FLOAT_EQ(detail::effective_confidence(explicit_value, TaskType::Detection), 0.25f);
    EXPECT_FLOAT_EQ(detail::effective_confidence(explicit_value, TaskType::Classification), 0.25f);
    explicit_value.conf_threshold = 0.f; // zero is a real threshold, not "unset"
    EXPECT_FLOAT_EQ(detail::effective_confidence(explicit_value, TaskType::Detection), 0.f);
}

TEST(Confidence, ClassificationDefaultReturnsAllClassesButAnExplicitThresholdFilters) {
    LoadConfig c;
    c.decoder = "probabilities";
    TensorInfo input{"x", {1, 3, 2, 2}, DataType::Float32, Layout::NCHW};
    Tensor output({1, 3}, DataType::Float32);
    const float values[] = {0.3f, 0.3f, 0.4f};
    std::memcpy(output.data(), values, sizeof(values));
    auto all = detail::decode_outputs({{"o", output}}, {2, 2}, input, c, TaskType::Classification,
                                      InferConfig{});
    EXPECT_EQ(all.classifications.size(), 3);
    InferConfig strict;
    strict.conf_threshold = 0.35f;
    auto some = detail::decode_outputs({{"o", output}}, {2, 2}, input, c, TaskType::Classification,
                                       strict);
    ASSERT_EQ(some.classifications.size(), 1);
    EXPECT_EQ(some.classifications[0].class_id, 2);
}

TEST(Confidence, DetectionDefaultIsStillHalf) {
    LoadConfig c;
    c.decoder = "yolov8";
    c.resize_mode = "stretch";
    TensorInfo input{"x", {1, 3, 64, 64}, DataType::Float32, Layout::NCHW};
    Tensor output({1, 5, 2}, DataType::Float32); // 1 class, 2 anchors
    const float values[] = {20, 50, 20, 50, 10, 10, 10, 10, 0.4f, 0.6f};
    std::memcpy(output.data(), values, sizeof(values));
    auto result = detail::decode_outputs({{"o", output}}, {64, 64}, input, c, TaskType::Detection,
                                         InferConfig{});
    ASSERT_EQ(result.detections.size(), 1); // 0.6 kept, 0.4 dropped: same as before this change
    EXPECT_NEAR(result.detections[0].confidence, 0.6f, 1e-6);
}

TEST(Confidence, OtherNegativeAndNonFiniteThresholdsAreRejected) {
    LoadConfig c;
    c.decoder = "probabilities";
    TensorInfo input{"x", {1, 3, 2, 2}, DataType::Float32, Layout::NCHW};
    Tensor output({1, 2}, DataType::Float32);
    output.data_ptr<float>()[0] = output.data_ptr<float>()[1] = 0.5f;
    for (float bad : {-0.3f, -2.0f, 1.5f, std::nanf("")}) {
        InferConfig config;
        config.conf_threshold = bad;
        EXPECT_THROW(detail::decode_outputs({{"o", output}}, {2, 2}, input, c,
                                            TaskType::Classification, config),
                     InferenceError)
            << bad;
    }
}

namespace {
DeviceInfo device(std::string id, std::string type, std::string backend, std::string name) {
    DeviceInfo info;
    info.id = std::move(id);
    info.type = std::move(type);
    info.backend = std::move(backend);
    info.name = std::move(name);
    return info;
}

std::string best(std::vector<DeviceInfo> devices, const std::string &priority) {
    std::stable_sort(devices.begin(), devices.end(), [&](const auto &a, const auto &b) {
        return detail::device_score(a, priority) > detail::device_score(b, priority);
    });
    return devices.front().id;
}
} // namespace

TEST(DeviceRanking, NvidiaGpuBeatsIntelIgpuWhicheverIsListedFirst) {
    // This machine's real list: Intel iGPU and NVIDIA GPU both report type "gpu".
    auto cpu = device("cpu", "cpu", "openvino", "13th Gen Intel(R) Core(TM) i5-13500");
    auto igpu = device("gpu:0", "gpu", "openvino", "Intel(R) UHD Graphics 770 (iGPU)");
    auto nvidia = device("cuda:0", "gpu", "tensorrt", "NVIDIA GeForce RTX 2070 SUPER");
    EXPECT_EQ(best({cpu, igpu, nvidia}, "performance"), "cuda:0");
    EXPECT_EQ(best({nvidia, igpu, cpu}, "performance"), "cuda:0");
    EXPECT_EQ(best({cpu, igpu, nvidia}, "memory"), "cuda:0");
    EXPECT_EQ(best({cpu, nvidia, igpu}, "efficiency"), "gpu:0"); // no NPU: iGPU is most efficient
}

TEST(DeviceRanking, EfficiencyPrefersNpuAndPerformancePrefersAnyGpuOverNpu) {
    auto npu = device("npu", "npu", "openvino", "Intel(R) AI Boost");
    auto igpu = device("gpu:0", "gpu", "openvino", "Intel(R) Graphics");
    auto cpu = device("cpu", "cpu", "openvino", "Intel CPU");
    EXPECT_EQ(best({cpu, igpu, npu}, "efficiency"), "npu");
    EXPECT_EQ(best({cpu, npu, igpu}, "performance"), "gpu:0");
    EXPECT_EQ(best({cpu, npu}, "performance"), "npu");
    EXPECT_EQ(best({cpu, npu}, "memory"), "cpu");
}

TEST(DeviceRanking, IntelDiscreteGpuBeatsIntegratedAndUnknownPriorityMeansPerformance) {
    auto igpu = device("gpu:0", "gpu", "openvino", "Intel(R) UHD Graphics 770 (iGPU)");
    auto arc = device("gpu:1", "gpu", "openvino", "Intel(R) Arc(TM) A770 Graphics (dGPU)");
    EXPECT_TRUE(detail::is_discrete_gpu(arc));
    EXPECT_FALSE(detail::is_discrete_gpu(igpu));
    EXPECT_EQ(best({igpu, arc}, "performance"), "gpu:1");
    EXPECT_EQ(best({igpu, arc}, "efficiency"), "gpu:0");
    EXPECT_EQ(best({igpu, arc}, "something-else"), "gpu:1");
}

/**
 * @file test_model.cpp
 * @brief Unit tests for Model class
 */

#include <gtest/gtest.h>
#include "ivit/core/model.hpp"
#include "ivit/core/tensor.hpp"

using namespace ivit;

class ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code
    }

    void TearDown() override {
        // Teardown code
    }
};

TEST_F(ModelTest, ModelManagerSingleton) {
    auto& manager1 = ModelManager::instance();
    auto& manager2 = ModelManager::instance();
    EXPECT_EQ(&manager1, &manager2);
}

TEST_F(ModelTest, LoadConfigDefaults) {
    LoadConfig config;
    EXPECT_EQ(config.device, "auto");
    EXPECT_EQ(config.precision, "fp32");
    EXPECT_TRUE(config.use_cache);
}

TEST_F(ModelTest, TensorCreation) {
    std::vector<int64_t> shape = {1, 3, 224, 224};
    Tensor tensor(shape, DataType::Float32);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.dtype(), DataType::Float32);
    EXPECT_EQ(tensor.numel(), 1 * 3 * 224 * 224);
}

TEST_F(ModelTest, TensorByteSize) {
    std::vector<int64_t> shape = {1, 3, 224, 224};

    Tensor fp32_tensor(shape, DataType::Float32);
    EXPECT_EQ(fp32_tensor.byte_size(), 1 * 3 * 224 * 224 * sizeof(float));

    Tensor fp16_tensor(shape, DataType::Float16);
    EXPECT_EQ(fp16_tensor.byte_size(), 1 * 3 * 224 * 224 * 2);

    Tensor int8_tensor(shape, DataType::Int8);
    EXPECT_EQ(int8_tensor.byte_size(), 1 * 3 * 224 * 224 * 1);
}

TEST_F(ModelTest, TensorDataAccess) {
    std::vector<int64_t> shape = {2, 3};
    Tensor tensor(shape, DataType::Float32);

    float* data = tensor.data_ptr<float>();
    EXPECT_NE(data, nullptr);

    // Write and read data
    data[0] = 1.0f;
    data[1] = 2.0f;

    EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[1], 2.0f);
}

TEST_F(ModelTest, TensorByteSizeCoversAllDataTypes) {
    auto bytes = [](DataType dtype) {
        return TensorInfo{"", {10}, dtype, Layout::NC}.byte_size();
    };
    EXPECT_EQ(bytes(DataType::Float32), 40u);
    EXPECT_EQ(bytes(DataType::Float16), 20u);
    EXPECT_EQ(bytes(DataType::Int8), 10u);
    EXPECT_EQ(bytes(DataType::UInt8), 10u);
    EXPECT_EQ(bytes(DataType::Bool), 10u);
    EXPECT_EQ(bytes(DataType::Int32), 40u);
    EXPECT_EQ(bytes(DataType::Int64), 80u);
}

TEST_F(ModelTest, DynamicShapeIsDetectedNotMultiplied) {
    TensorInfo dynamic{"x", {-1, 3, 224, 224}, DataType::Float32, Layout::NCHW};
    EXPECT_TRUE(dynamic.is_dynamic());
    EXPECT_EQ(dynamic.numel(), -1);
    EXPECT_EQ(dynamic.byte_size(), 0u);

    // Two negative dimensions used to multiply into a positive element count
    TensorInfo two_dynamic{"x", {-1, -1}, DataType::Float32, Layout::NC};
    EXPECT_EQ(two_dynamic.numel(), -1);
}

TEST_F(ModelTest, TensorRejectsDynamicShape) {
    EXPECT_THROW(Tensor({-1, 3, 224, 224}, DataType::Float32), IVITError);
    EXPECT_THROW(Tensor({1, -1}, DataType::Float32), IVITError);
}

TEST_F(ModelTest, ResolveDynamicShape) {
    EXPECT_EQ(resolve_dynamic_shape({-1, 3, 640, 640}, 1, 224, 224), (Shape{1, 3, 640, 640}));
    EXPECT_EQ(resolve_dynamic_shape({-1, -1, -1, -1}, 4, 480, 320), (Shape{4, 3, 480, 320}));
    EXPECT_EQ(resolve_dynamic_shape({0, 3, 32, 32}, 2, 1, 1), (Shape{2, 3, 32, 32}));
    EXPECT_EQ(resolve_dynamic_shape({-1, 1000}, 1, 224, 224), (Shape{1, 1000}));

    // The result can always be allocated
    EXPECT_NO_THROW(Tensor(resolve_dynamic_shape({-1, 3, -1, -1}, 1, 8, 8), DataType::Float32));
}

TEST_F(ModelTest, BackendFromString) {
    EXPECT_EQ(backend_from_string(""), BackendType::Auto);
    EXPECT_EQ(backend_from_string("auto"), BackendType::Auto);
    EXPECT_EQ(backend_from_string("OpenVINO"), BackendType::OpenVINO);
    EXPECT_EQ(backend_from_string("TENSORRT"), BackendType::TensorRT);
    EXPECT_THROW(backend_from_string("qnn"), IVITError);  // removed backend
    EXPECT_THROW(backend_from_string("bogus"), IVITError);
}

TEST_F(ModelTest, LoadRejectsUnknownBackendBeforeTouchingTheFile) {
    LoadConfig config;
    config.backend = "bogus";
    EXPECT_THROW(ModelManager::instance().load("does_not_exist.onnx", config), IVITError);
}

TEST_F(ModelTest, BBoxIoU) {
    BBox box1(0, 0, 10, 10);
    BBox box2(5, 5, 15, 15);
    BBox box3(20, 20, 30, 30);

    // Overlapping boxes
    float iou12 = box1.iou(box2);
    EXPECT_GT(iou12, 0.0f);
    EXPECT_LT(iou12, 1.0f);

    // Non-overlapping boxes
    float iou13 = box1.iou(box3);
    EXPECT_FLOAT_EQ(iou13, 0.0f);

    // Same box
    float iou11 = box1.iou(box1);
    EXPECT_FLOAT_EQ(iou11, 1.0f);
}

TEST_F(ModelTest, BBoxArea) {
    BBox box(0, 0, 10, 20);
    EXPECT_FLOAT_EQ(box.area(), 200.0f);

    BBox zero_box(5, 5, 5, 5);
    EXPECT_FLOAT_EQ(zero_box.area(), 0.0f);
}

namespace {
class FakeModel : public Model {
public:
    std::atomic<int> active{0};
    std::atomic<int> peak{0};
    std::atomic<int> completed{0};
    bool fail = false;
    std::string name() const override { return "fake"; }
    TaskType task() const override { return TaskType::Classification; }
    std::string device() const override { return "fake"; }
    BackendType backend() const override { return BackendType::Auto; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    size_t memory_usage() const override { return 0; }
    std::vector<std::string> labels() const override { return {}; }
    Results predict(const cv::Mat& image, const InferConfig&) override {
        const int running = ++active;
        int previous = peak.load();
        while (previous < running && !peak.compare_exchange_weak(previous, running)) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        --active;
        ++completed;
        if (fail) throw InferenceError("injected failure");
        Results result;
        result.inference_time_ms = image.at<unsigned char>(0, 0);
        return result;
    }
    std::vector<Results> predict_batch(const std::vector<cv::Mat>&, const InferConfig&) override { return {}; }
    std::map<std::string, Tensor> infer_raw(const std::map<std::string, Tensor>&) override { return {}; }
    void warmup(int) override {}
};
}

TEST(ConcurrentModel, RejectsInvalidLimitEvenForEmptyInput) {
    FakeModel model;
    EXPECT_THROW(model.predict_concurrent({}, 0), IVITError);
    EXPECT_THROW(model.predict_concurrent({}, -1), IVITError);
    EXPECT_TRUE(model.predict_concurrent({}, 1).empty());
}

TEST(ConcurrentModel, BoundsWorkersAndPreservesOrder) {
    FakeModel model;
    std::vector<cv::Mat> images;
    for (int i = 0; i < 20; ++i) images.emplace_back(1, 1, CV_8U, cv::Scalar(i));
    const auto results = model.predict_concurrent(images, 3);
    ASSERT_EQ(results.size(), images.size());
    for (size_t i = 0; i < results.size(); ++i) EXPECT_EQ(results[i].inference_time_ms, i);
    EXPECT_LE(model.peak.load(), 3);
    EXPECT_EQ(model.completed.load(), 20);
}

TEST(ConcurrentModel, ErrorsDrainWorkersAndAllowReuse) {
    FakeModel model;
    const std::vector<cv::Mat> images(8, cv::Mat(1, 1, CV_8U, cv::Scalar(1)));
    model.fail = true;
    EXPECT_THROW(model.predict_concurrent(images, 2), InferenceError);
    EXPECT_EQ(model.completed.load(), 8);
    EXPECT_EQ(model.active.load(), 0);
    model.fail = false;
    EXPECT_EQ(model.predict_concurrent(images, 2).size(), 8u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

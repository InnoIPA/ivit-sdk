/**
 * @file test_inference.cpp
 * @brief Unit tests for inference functionality
 */

#include <gtest/gtest.h>
#include "ivit/core/model.hpp"
#include "ivit/core/tensor.hpp"
#include "ivit/core/device.hpp"
#include "ivit/runtime/runtime.hpp"

using namespace ivit;

class InferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code
    }

    void TearDown() override {
        // Teardown code
    }
};

TEST_F(InferenceTest, RuntimeFactorySingleton) {
    auto& factory1 = RuntimeFactory::instance();
    auto& factory2 = RuntimeFactory::instance();
    EXPECT_EQ(&factory1, &factory2);
}

TEST_F(InferenceTest, DeviceManagerSingleton) {
    auto& manager1 = DeviceManager::instance();
    auto& manager2 = DeviceManager::instance();
    EXPECT_EQ(&manager1, &manager2);
}

TEST_F(InferenceTest, DeviceManagerListDevices) {
    auto& manager = DeviceManager::instance();
    auto devices = manager.list_devices();

    // Should have at least CPU
    bool has_cpu = false;
    for (const auto& device : devices) {
        if (device.type == "cpu") {
            has_cpu = true;
            break;
        }
    }
    EXPECT_TRUE(has_cpu);
}

TEST_F(InferenceTest, InferConfigDefaults) {
    InferConfig config;
    // -1 = "task default": 0.5 for detection, 0 for classification (see effective_confidence).
    EXPECT_FLOAT_EQ(config.conf_threshold, -1.0f);
    EXPECT_FLOAT_EQ(config.iou_threshold, 0.45f);
    EXPECT_EQ(config.max_detections, 100);
    EXPECT_TRUE(config.classes.empty());
}

TEST_F(InferenceTest, ResultsStructure) {
    Results results;

    EXPECT_TRUE(results.detections.empty());
    EXPECT_TRUE(results.classifications.empty());
    EXPECT_TRUE(results.segmentation_mask.empty());
    EXPECT_FLOAT_EQ(results.inference_time_ms, 0.0f);
}

TEST_F(InferenceTest, DetectionStructure) {
    Detection det;
    det.bbox = BBox(10, 20, 100, 200);
    det.class_id = 0;
    det.label = "person";
    det.confidence = 0.95f;

    EXPECT_EQ(det.class_id, 0);
    EXPECT_EQ(det.label, "person");
    EXPECT_FLOAT_EQ(det.confidence, 0.95f);
    EXPECT_FLOAT_EQ(det.bbox.x1, 10.0f);
    EXPECT_FLOAT_EQ(det.bbox.y1, 20.0f);
    EXPECT_FLOAT_EQ(det.bbox.x2, 100.0f);
    EXPECT_FLOAT_EQ(det.bbox.y2, 200.0f);
}

TEST_F(InferenceTest, ClassificationResultStructure) {
    ClassificationResult cls;
    cls.class_id = 281;
    cls.label = "tabby_cat";
    cls.score = 0.89f;

    EXPECT_EQ(cls.class_id, 281);
    EXPECT_EQ(cls.label, "tabby_cat");
    EXPECT_FLOAT_EQ(cls.score, 0.89f);
}

TEST_F(InferenceTest, TensorInfoStructure) {
    TensorInfo info;
    info.name = "input";
    info.shape = {1, 3, 224, 224};
    info.dtype = DataType::Float32;
    info.layout = Layout::NCHW;

    EXPECT_EQ(info.name, "input");
    EXPECT_EQ(info.shape.size(), 4);
    EXPECT_EQ(info.dtype, DataType::Float32);
    EXPECT_EQ(info.layout, Layout::NCHW);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

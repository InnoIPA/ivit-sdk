// Manual acceptance helper, derived from SDK manual sections 4.2 and 7.1-7.3.
#include <ivit/ivit.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    try {
        if (argc != 8)
            throw std::invalid_argument("matrix_app MODEL IMAGE TASK BACKEND DEVICE PRECISION OUTPUT");
        ivit::LoadConfig config;
        config.task = argv[3];
        config.backend = argv[4];
        config.device = argv[5];
        config.precision = argv[6];
        config.use_cache = false;
        if (config.task == "detection") {
            // This branch is specifically for the Zoo's pre-converted YOLOX-S.
            config.decoder = "yolox_decoded";
            config.input_color = "bgr";
            config.input_scale = 1.0f;
            config.resize_mode = "letterbox_top_left";
        } else if (config.task == "classification" || config.task == "segmentation") {
            config.decoder = "logits";
            config.mean = {0.485f, 0.456f, 0.406f};
            config.std = {0.229f, 0.224f, 0.225f};
        } else {
            throw std::invalid_argument("Unsupported task");
        }
        auto model = ivit::load_model(argv[1], config);
        auto image = cv::imread(argv[2]);
        if (image.empty())
            throw std::runtime_error("Cannot read image");
        model->warmup(3);
        auto result = model->predict(image);
        if (result.device_used != config.device)
            throw std::runtime_error("Unexpected inference device");
        if (config.task == "classification" && result.classifications.size() != 1000)
            throw std::runtime_error("Expected 1000 classification scores");
        if (config.task == "detection" && result.detections.empty())
            throw std::runtime_error("Expected detections on bus.jpg");
        if (config.task == "segmentation" &&
            (result.segmentation_mask.empty() || result.segmentation_mask.size() != image.size()))
            throw std::runtime_error("Unexpected segmentation mask");
        result.save(std::string(argv[7]) + ".json");
        if (!cv::imwrite(std::string(argv[7]) + ".jpg", result.visualize(image)))
            throw std::runtime_error("Cannot save visualization");
        if (config.task == "segmentation") {
            cv::imwrite(std::string(argv[7]) + "-mask.png", result.colorize_mask());
            cv::Mat classes;
            result.segmentation_mask.convertTo(classes, CV_8U);
            cv::imwrite(std::string(argv[7]) + "-classes.png", classes);
        }
        std::cout << "backend=" << config.backend << " device=" << result.device_used
                  << " precision=" << config.precision << " ms=" << result.inference_time_ms << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

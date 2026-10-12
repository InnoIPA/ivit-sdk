#pragma once
#include "ivit/core/model.hpp"

namespace ivit::detail {
Tensor preprocess_image(const cv::Mat &image, const TensorInfo &input, const LoadConfig &contract,
                        TaskType task);
Results decode_outputs(const std::map<std::string, Tensor> &outputs, cv::Size image_size,
                       const TensorInfo &input, const LoadConfig &contract, TaskType task,
                       const InferConfig &config);
Results predict_image(Model &model, const LoadConfig &contract, const cv::Mat &image,
                      const InferConfig &config);
void validate_contract(const LoadConfig &contract);
/// Number of classes a decoder will read from the model's declared outputs, or -1 when it cannot
/// be known before running (decoder "raw", several outputs without output_name, dynamic or
/// ambiguous shapes). Used to reject a wrong labels list at load time instead of at first use.
/// The threshold a task actually uses: the configured value, or the task default when it is unset.
float effective_confidence(const InferConfig &config, TaskType task);
int expected_class_count(const std::vector<TensorInfo> &outputs, const LoadConfig &contract,
                         TaskType task);
} // namespace ivit::detail

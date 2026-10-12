#include "ivit/c_api.h"
#include "ivit/ivit.hpp"
#include "utils/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

// The C structs are part of the ABI: any padding or field change must be deliberate.
#if defined(__LP64__) || defined(_WIN64)
static_assert(sizeof(ivit_load_options_v1) == 128, "ivit_load_options_v1 ABI layout changed");
static_assert(sizeof(ivit_load_options_v2) == 176, "ivit_load_options_v2 ABI layout changed");
static_assert(sizeof(ivit_image_v1) == 40, "ivit_image_v1 ABI layout changed");
static_assert(sizeof(ivit_infer_options_v1) == 40, "ivit_infer_options_v1 ABI layout changed");
#endif

namespace {
thread_local std::array<char, 2048> error_message{};
thread_local int32_t error_category = IVIT_C_ERR_NONE;
std::mutex handles_mutex;
uint64_t next_handle = 1;
std::unordered_map<uint64_t, std::shared_ptr<ivit::Model>> models;
std::unordered_map<uint64_t, std::shared_ptr<ivit::Results>> results;

struct InvalidHandle : std::exception {};

ivit_status fail(int32_t category, const char *message) noexcept {
    error_category = category;
    std::snprintf(error_message.data(), error_message.size(), "%s", message);
    return IVIT_C_RUNTIME_ERROR;
}

template <class Function> ivit_status guard(Function function) noexcept {
    error_message[0] = '\0';
    error_category = IVIT_C_ERR_NONE;
    try {
        return function();
    } catch (const InvalidHandle &) {
        std::snprintf(error_message.data(), error_message.size(), "Invalid or released handle");
        return IVIT_C_INVALID_HANDLE;
    } catch (const std::invalid_argument &error) {
        std::snprintf(error_message.data(), error_message.size(), "%s", error.what());
        return IVIT_C_INVALID_ARGUMENT;
    } catch (const ivit::ModelLoadError &error) {
        return fail(IVIT_C_ERR_MODEL_LOAD, error.what());
    } catch (const ivit::UnsupportedFormatError &error) {
        return fail(IVIT_C_ERR_UNSUPPORTED_FORMAT, error.what());
    } catch (const ivit::DeviceNotFoundError &error) {
        return fail(IVIT_C_ERR_DEVICE_NOT_FOUND, error.what());
    } catch (const ivit::InferenceError &error) {
        return fail(IVIT_C_ERR_INFERENCE, error.what());
    } catch (const std::exception &error) {
        return fail(IVIT_C_ERR_GENERIC, error.what());
    } catch (...) {
        std::snprintf(error_message.data(), error_message.size(), "Unknown native failure");
        return IVIT_C_RUNTIME_ERROR;
    }
}

template <class T>
std::shared_ptr<T> lookup(const std::unordered_map<uint64_t, std::shared_ptr<T>> &table,
                          uint64_t handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    const auto found = table.find(handle);
    if (found == table.end())
        throw InvalidHandle{};
    return found->second;
}

template <class T>
uint64_t insert(std::unordered_map<uint64_t, std::shared_ptr<T>> &table, std::shared_ptr<T> value) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    if (next_handle == 0)
        throw std::runtime_error("Handle space exhausted");
    const auto id = next_handle++;
    table.emplace(id, std::move(value));
    return id;
}

template <class T>
void release(std::unordered_map<uint64_t, std::shared_ptr<T>> &table, uint64_t handle) {
    std::shared_ptr<T> removed;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        const auto found = table.find(handle);
        if (found == table.end())
            throw InvalidHandle{};
        removed = std::move(found->second);
        table.erase(found);
    }
    // Destruction can block; never hold the registry lock during teardown.
}

ivit_status copy_text(const char *text, char *buffer, uint64_t capacity, uint64_t *required) {
    if (!required || (!buffer && capacity))
        return IVIT_C_INVALID_ARGUMENT;
    *required = std::strlen(text) + 1;
    if (capacity < *required)
        return IVIT_C_BUFFER_TOO_SMALL;
    std::memcpy(buffer, text, static_cast<size_t>(*required));
    return IVIT_C_OK;
}

// Validates and wraps caller memory; the Mat does not own the pixels.
cv::Mat wrap_image(const uint8_t *pixels, uint64_t bytes, int32_t width, int32_t height,
                   uint64_t stride, int32_t channels) {
    if (!pixels || width <= 0 || height <= 0 || (channels != 1 && channels != 3))
        throw std::invalid_argument("Invalid image arguments");
    const uint64_t row_bytes = static_cast<uint64_t>(width) * channels;
    if (stride < row_bytes || stride > std::numeric_limits<size_t>::max() || bytes < row_bytes ||
        static_cast<uint64_t>(height - 1) > (bytes - row_bytes) / stride)
        throw std::invalid_argument("Image buffer is too short or stride is invalid");
    return cv::Mat(height, width, channels == 1 ? CV_8UC1 : CV_8UC3, const_cast<uint8_t *>(pixels),
                   static_cast<size_t>(stride));
}

cv::Mat wrap_image(const ivit_image_v1 &image) {
    if (image.reserved != 0)
        throw std::invalid_argument("ivit_image_v1.reserved must be 0");
    return wrap_image(image.pixels, image.bytes, image.width, image.height, image.stride,
                      image.channels);
}

ivit::InferConfig infer_config(const ivit_infer_options_v1 *options) {
    ivit::InferConfig config;
    if (!options)
        return config;
    if (options->struct_size != sizeof(*options) || options->abi_version != 1 ||
        options->reserved != 0)
        throw std::invalid_argument("Invalid infer options version/size");
    const bool task_default = options->confidence == -1.0f;
    if ((!task_default && (!std::isfinite(options->confidence) || options->confidence < 0 ||
                           options->confidence > 1)) ||
        !std::isfinite(options->iou) || options->iou < 0 ||
        options->iou > 1 || options->max_detections <= 0 ||
        (options->agnostic_nms != 0 && options->agnostic_nms != 1))
        throw std::invalid_argument("Invalid inference option values");
    if (options->class_count > 100000 || (options->class_count && !options->classes))
        throw std::invalid_argument("Invalid class filter");
    config.conf_threshold = options->confidence; // -1 stays "task default"
    config.iou_threshold = options->iou;
    config.max_detections = options->max_detections;
    config.agnostic_nms = options->agnostic_nms != 0;
    config.classes.assign(options->classes, options->classes + options->class_count);
    return config;
}

const char *layout_name(ivit::Layout layout) {
    switch (layout) {
    case ivit::Layout::NCHW:
        return "nchw";
    case ivit::Layout::NHWC:
        return "nhwc";
    case ivit::Layout::NC:
        return "nc";
    case ivit::Layout::CHW:
        return "chw";
    case ivit::Layout::HWC:
        return "hwc";
    default:
        return "unknown";
    }
}

std::string tensors_json(const std::vector<ivit::TensorInfo> &tensors) {
    std::string json = "[";
    for (const auto &tensor : tensors) {
        if (json.size() > 1)
            json += ',';
        json += "{\"name\":" + ivit::detail::json_string(tensor.name) + ",\"shape\":[";
        for (size_t i = 0; i < tensor.shape.size(); ++i)
            json += (i ? "," : "") + std::to_string(tensor.shape[i]);
        json += "],\"dtype\":" + ivit::detail::json_string(ivit::to_string(tensor.dtype)) +
                ",\"layout\":" + ivit::detail::json_string(layout_name(tensor.layout)) + "}";
    }
    return json + "]";
}

std::string number_array_json(const std::vector<float> &values) {
    std::string json = "[";
    for (size_t i = 0; i < values.size(); ++i)
        json += (i ? "," : "") + ivit::detail::json_number(values[i]);
    return json + "]";
}
} // namespace

uint32_t IVIT_C_CALL ivit_abi_version(void) { return 1; }

const char *IVIT_C_CALL ivit_version_string(void) { return IVIT_VERSION_STRING; }

ivit_status IVIT_C_CALL ivit_set_log_callback(ivit_log_callback callback, void *user_data) {
    return guard([&]() -> ivit_status {
        if (!callback) {
            ivit::set_log_callback(nullptr);
            return IVIT_C_OK;
        }
        ivit::set_log_callback([callback, user_data](ivit::LogLevel level,
                                                     const std::string &message) {
            callback(static_cast<int32_t>(level), message.c_str(), user_data);
        });
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_set_log_level(const char *level) {
    return guard([&]() -> ivit_status {
        const std::string name = level ? level : "";
        if (name != "debug" && name != "info" && name != "warning" && name != "error" &&
            name != "off")
            throw std::invalid_argument("Log level must be debug, info, warning, error or off");
        ivit::set_log_level(name);
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_last_error(char *buffer, uint64_t capacity, uint64_t *required) {
    return copy_text(error_message.data(), buffer, capacity, required);
}

ivit_status IVIT_C_CALL ivit_devices_json(char *buffer, uint64_t capacity, uint64_t *required) {
    return guard([&]() -> ivit_status {
        std::string json = "[";
        for (const auto &device : ivit::list_devices()) {
            if (json.size() > 1)
                json += ',';
            json += "{\"id\":" + ivit::detail::json_string(device.id) +
                    ",\"name\":" + ivit::detail::json_string(device.name) +
                    ",\"backend\":" + ivit::detail::json_string(device.backend) + "}";
        }
        json += ']';
        return copy_text(json.c_str(), buffer, capacity, required);
    });
}

ivit_status IVIT_C_CALL ivit_model_load(const char *path, const char *device, const char *backend,
                                        const char *task, const char *precision, int32_t use_cache,
                                        ivit_model_handle *model) {
    if (model)
        *model = 0;
    return guard([&]() -> ivit_status {
        if (!model || !path || !device || !backend || !task || !precision ||
            (use_cache != 0 && use_cache != 1))
            throw std::invalid_argument("Invalid load arguments");
        ivit::LoadConfig config;
        config.device = device;
        config.backend = backend;
        config.task = task;
        config.precision = precision;
        config.use_cache = use_cache != 0;
        *model = insert(models, ivit::load_model(path, config));
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_model_release(ivit_model_handle model) {
    return guard([&]() -> ivit_status {
        release(models, model);
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_load_options_init(ivit_load_options_v1 *options) {
    return guard([&]() -> ivit_status {
        if (!options || options->struct_size != sizeof(*options))
            throw std::invalid_argument("Invalid load options size");
        *options = {};
        options->struct_size = sizeof(*options);
        options->abi_version = 1;
        options->input_scale = 1.0f / 255.0f;
        options->stddev[0] = options->stddev[1] = options->stddev[2] = 1;
        options->batch_size = 1;
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_model_load_ex(const char *path, const ivit_load_options_v1 *options,
                                           ivit_model_handle *model) {
    if (model)
        *model = 0;
    return guard([&]() -> ivit_status {
        if (!path || !model || !options || options->struct_size != sizeof(*options) ||
            options->abi_version != 1 || (options->use_cache != 0 && options->use_cache != 1))
            throw std::invalid_argument("Invalid load options version/size/arguments");
        ivit::LoadConfig config;
        if (options->device)
            config.device = options->device;
        if (options->backend)
            config.backend = options->backend;
        if (options->task)
            config.task = options->task;
        if (options->precision)
            config.precision = options->precision;
        if (options->decoder)
            config.decoder = options->decoder;
        if (options->input_color)
            config.input_color = options->input_color;
        if (options->input_layout)
            config.input_layout = options->input_layout;
        if (options->resize_mode)
            config.resize_mode = options->resize_mode;
        if (options->labels_path)
            config.labels_path = options->labels_path;
        if (options->output_name)
            config.output_name = options->output_name;
        config.input_scale = options->input_scale;
        config.mean.assign(options->mean, options->mean + 3);
        config.std.assign(options->stddev, options->stddev + 3);
        config.use_cache = options->use_cache != 0;
        config.batch_size = options->batch_size;
        *model = insert(models, ivit::load_model(path, config));
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_model_predict_bgr(ivit_model_handle model, const uint8_t *pixels,
                                               uint64_t bytes, int32_t width, int32_t height,
                                               uint64_t stride, float confidence, float iou,
                                               int32_t max_detections, ivit_result_handle *result) {
    if (result)
        *result = 0;
    return guard([&]() -> ivit_status {
        if (!pixels || !result || width <= 0 || height <= 0 || !std::isfinite(confidence) ||
            confidence < 0 || confidence > 1 || !std::isfinite(iou) || iou < 0 || iou > 1 ||
            max_detections <= 0)
            throw std::invalid_argument("Invalid image or inference arguments");
        cv::Mat image = wrap_image(pixels, bytes, width, height, stride, 3);
        auto owned = lookup(models, model);
        ivit::InferConfig config;
        config.conf_threshold = confidence;
        config.iou_threshold = iou;
        config.max_detections = max_detections;
        *result = insert(results, std::make_shared<ivit::Results>(owned->predict(image, config)));
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_result_release(ivit_result_handle result) {
    return guard([&]() -> ivit_status {
        release(results, result);
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_result_json(ivit_result_handle result, char *buffer, uint64_t capacity,
                                         uint64_t *required) {
    return guard([&]() -> ivit_status {
        const auto owned = lookup(results, result);
        return copy_text(owned->to_json().c_str(), buffer, capacity, required);
    });
}

ivit_status IVIT_C_CALL ivit_result_mask_i32(ivit_result_handle result, int32_t *buffer,
                                             uint64_t capacity, int32_t *width, int32_t *height,
                                             uint64_t *required) {
    return guard([&]() -> ivit_status {
        if (!width || !height || !required || (!buffer && capacity))
            throw std::invalid_argument("Invalid mask output arguments");
        const auto owned = lookup(results, result);
        const auto &mask = owned->segmentation_mask;
        *width = mask.cols;
        *height = mask.rows;
        *required = mask.total();
        if (capacity < *required)
            return IVIT_C_BUFFER_TOO_SMALL;
        if (*required) {
            cv::Mat converted;
            mask.convertTo(converted, CV_32S);
            if (converted.channels() != 1)
                throw std::runtime_error("Expected a single-channel class mask");
            for (int row = 0; row < converted.rows; ++row)
                std::memcpy(buffer + static_cast<uint64_t>(row) * converted.cols,
                            converted.ptr<int32_t>(row),
                            static_cast<size_t>(converted.cols) * sizeof(int32_t));
        }
        return IVIT_C_OK;
    });
}

// ---------------------------------------------------------------------------
// Additions in SDK 1.1
// ---------------------------------------------------------------------------

int32_t IVIT_C_CALL ivit_last_error_category(void) { return error_category; }

ivit_status IVIT_C_CALL ivit_model_info_json(ivit_model_handle model, char *buffer,
                                             uint64_t capacity, uint64_t *required) {
    return guard([&]() -> ivit_status {
        const auto owned = lookup(models, model);
        const auto contract = owned->load_config();
        std::string json = "{\"name\":" + ivit::detail::json_string(owned->name()) +
                           ",\"task\":" + ivit::detail::json_string(ivit::to_string(owned->task())) +
                           ",\"device\":" + ivit::detail::json_string(owned->device()) +
                           ",\"backend\":" +
                           ivit::detail::json_string(ivit::to_string(owned->backend())) +
                           ",\"inputs\":" + tensors_json(owned->input_info()) +
                           ",\"outputs\":" + tensors_json(owned->output_info()) + ",\"labels\":[";
        const auto labels = owned->labels();
        for (size_t i = 0; i < labels.size(); ++i)
            json += (i ? "," : "") + ivit::detail::json_string(labels[i]);
        json += "],\"contract\":{\"decoder\":" + ivit::detail::json_string(contract.decoder) +
                ",\"input_color\":" + ivit::detail::json_string(contract.input_color) +
                ",\"input_layout\":" + ivit::detail::json_string(contract.input_layout) +
                ",\"resize_mode\":" + ivit::detail::json_string(contract.resize_mode) +
                ",\"input_scale\":" + ivit::detail::json_number(contract.input_scale) +
                ",\"mean\":" + number_array_json(contract.mean) +
                ",\"std\":" + number_array_json(contract.std) + "}}";
        return copy_text(json.c_str(), buffer, capacity, required);
    });
}

ivit_status IVIT_C_CALL ivit_load_options_v2_init(ivit_load_options_v2 *options) {
    return guard([&]() -> ivit_status {
        if (!options || options->struct_size != sizeof(*options))
            throw std::invalid_argument("Invalid load options size");
        *options = {};
        options->struct_size = sizeof(*options);
        options->abi_version = 2;
        options->num_instances = 1;
        options->workspace_bytes = 1ULL << 30;
        options->use_cache = 1;
        options->batch_size = 1;
        options->max_batch_size = 8;
        options->input_scale = 1.0f / 255.0f;
        options->stddev[0] = options->stddev[1] = options->stddev[2] = 1;
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_model_load_v2(const char *path, const ivit_load_options_v2 *options,
                                           ivit_model_handle *model) {
    if (model)
        *model = 0;
    return guard([&]() -> ivit_status {
        if (!path || !model || !options || options->struct_size != sizeof(*options) ||
            options->abi_version != 2 || options->num_instances < 1 ||
            options->num_instances > 16 ||
            (options->use_cache != 0 && options->use_cache != 1) ||
            (options->enable_profiling != 0 && options->enable_profiling != 1) ||
            options->num_threads < 0 || options->label_count > 100000 ||
            (options->label_count && !options->labels))
            throw std::invalid_argument("Invalid load options version/size/arguments");
        ivit::LoadConfig config;
        auto text = [](const char *value, std::string &target) {
            if (value)
                target = value;
        };
        text(options->device, config.device);
        text(options->backend, config.backend);
        text(options->task, config.task);
        text(options->precision, config.precision);
        text(options->decoder, config.decoder);
        text(options->input_color, config.input_color);
        text(options->input_layout, config.input_layout);
        text(options->resize_mode, config.resize_mode);
        text(options->labels_path, config.labels_path);
        text(options->output_name, config.output_name);
        text(options->cache_dir, config.cache_dir);
        text(options->performance_mode, config.performance_mode);
        for (uint32_t i = 0; i < options->label_count; ++i) {
            if (!options->labels[i])
                throw std::invalid_argument("labels must not contain NULL");
            config.labels.emplace_back(options->labels[i]);
        }
        config.input_scale = options->input_scale;
        config.mean.assign(options->mean, options->mean + 3);
        config.std.assign(options->stddev, options->stddev + 3);
        config.use_cache = options->use_cache != 0;
        config.batch_size = options->batch_size;
        config.max_batch_size = options->max_batch_size;
        config.num_threads = options->num_threads;
        config.num_instances = options->num_instances;
        config.enable_profiling = options->enable_profiling != 0;
        config.workspace_size = static_cast<size_t>(options->workspace_bytes);
        *model = insert(models, ivit::load_model(path, config));
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_infer_options_init(ivit_infer_options_v1 *options) {
    return guard([&]() -> ivit_status {
        if (!options || options->struct_size != sizeof(*options))
            throw std::invalid_argument("Invalid infer options size");
        *options = {};
        options->struct_size = sizeof(*options);
        options->abi_version = 1;
        options->confidence = -1.0f; // task default: 0.5 detection, 0 classification
        options->iou = 0.45f;
        options->max_detections = 100;
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_model_predict_ex(ivit_model_handle model, const ivit_image_v1 *image,
                                              const ivit_infer_options_v1 *options,
                                              ivit_result_handle *result) {
    if (result)
        *result = 0;
    return guard([&]() -> ivit_status {
        if (!image || !result)
            throw std::invalid_argument("Invalid image or result arguments");
        cv::Mat frame = wrap_image(*image);
        const auto config = infer_config(options);
        auto owned = lookup(models, model);
        *result = insert(results, std::make_shared<ivit::Results>(owned->predict(frame, config)));
        return IVIT_C_OK;
    });
}

ivit_status IVIT_C_CALL ivit_model_predict_batch(ivit_model_handle model,
                                                 const ivit_image_v1 *images, uint32_t count,
                                                 const ivit_infer_options_v1 *options,
                                                 ivit_result_handle *out) {
    if (out && count > 0 && count <= 1024)
        std::fill(out, out + count, ivit_result_handle{0});
    return guard([&]() -> ivit_status {
        if (!images || !out || count == 0 || count > 1024)
            throw std::invalid_argument("Batch needs 1..1024 images and an output array");
        std::vector<cv::Mat> frames;
        frames.reserve(count);
        for (uint32_t i = 0; i < count; ++i)
            frames.push_back(wrap_image(images[i]));
        const auto config = infer_config(options);
        auto owned = lookup(models, model);
        auto batch = owned->predict_batch(frames, config);
        if (batch.size() != count)
            throw std::runtime_error("Batch result count does not match the input count");
        // Register every handle under one lock so a failure cannot leave a partial batch behind.
        std::lock_guard<std::mutex> lock(handles_mutex);
        if (next_handle > std::numeric_limits<uint64_t>::max() - count)
            throw std::runtime_error("Handle space exhausted");
        for (uint32_t i = 0; i < count; ++i) {
            const auto id = next_handle++;
            results.emplace(id, std::make_shared<ivit::Results>(std::move(batch[i])));
            out[i] = id;
        }
        return IVIT_C_OK;
    });
}

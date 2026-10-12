/**
 * @file _binding.cpp
 * @brief Python bindings for iVIT-SDK using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <opencv2/opencv.hpp>

// Include iVIT headers
#include "ivit/ivit.hpp"
#include "ivit/core/callback.hpp"
#include "ivit/core/video_source.hpp"
#include "ivit/vision/classifier.hpp"
#include "ivit/vision/detector.hpp"
#include "ivit/vision/segmentor.hpp"

namespace py = pybind11;

// ============================================================================
// OpenCV Mat <-> NumPy conversion helpers
// ============================================================================

/**
 * @brief Convert cv::Mat to numpy array (creates a copy)
 */
py::array mat_to_numpy(const cv::Mat& mat) {
    if (mat.empty()) {
        return py::array();
    }

    // Determine numpy dtype based on OpenCV type
    std::string format;
    size_t elem_size;

    int depth = mat.depth();
    switch (depth) {
        case CV_8U:  format = py::format_descriptor<uint8_t>::format(); elem_size = 1; break;
        case CV_8S:  format = py::format_descriptor<int8_t>::format(); elem_size = 1; break;
        case CV_16U: format = py::format_descriptor<uint16_t>::format(); elem_size = 2; break;
        case CV_16S: format = py::format_descriptor<int16_t>::format(); elem_size = 2; break;
        case CV_32S: format = py::format_descriptor<int32_t>::format(); elem_size = 4; break;
        case CV_32F: format = py::format_descriptor<float>::format(); elem_size = 4; break;
        case CV_64F: format = py::format_descriptor<double>::format(); elem_size = 8; break;
        default:
            throw std::runtime_error("Unsupported cv::Mat type");
    }

    int channels = mat.channels();

    // Create buffer info
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (channels == 1) {
        shape = {mat.rows, mat.cols};
        strides = {static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(elem_size)};
    } else {
        shape = {mat.rows, mat.cols, channels};
        strides = {static_cast<ssize_t>(mat.step[0]),
                   static_cast<ssize_t>(channels * elem_size),
                   static_cast<ssize_t>(elem_size)};
    }

    // Create numpy array (copy data for safety)
    py::array result(py::buffer_info(
        mat.data,
        elem_size,
        format,
        shape.size(),
        shape,
        strides
    ));

    return result.attr("copy")();
}

/**
 * @brief Convert numpy array to cv::Mat
 */
cv::Mat numpy_to_mat(py::array array) {
    // cv::Mat below is built from a raw pointer with default (packed) strides,
    // so views such as img[:, :, ::-1] or img[::2] would be read as garbage.
    // ensure() returns the array itself when it is already C-contiguous and a
    // contiguous copy otherwise.
    py::array contiguous = py::array::ensure(array, py::array::c_style);
    if (!contiguous) {
        throw std::runtime_error("Cannot convert input to a C-contiguous array");
    }

    py::buffer_info buf = contiguous.request();

    // Determine OpenCV type
    int cv_type;
    if (buf.format == py::format_descriptor<uint8_t>::format()) {
        cv_type = CV_8U;
    } else if (buf.format == py::format_descriptor<float>::format()) {
        cv_type = CV_32F;
    } else if (buf.format == py::format_descriptor<double>::format()) {
        cv_type = CV_64F;
    } else if (buf.format == py::format_descriptor<int32_t>::format()) {
        cv_type = CV_32S;
    } else {
        throw std::runtime_error("Unsupported numpy dtype for cv::Mat conversion");
    }

    int rows, cols, channels;

    if (buf.ndim == 2) {
        rows = buf.shape[0];
        cols = buf.shape[1];
        channels = 1;
    } else if (buf.ndim == 3) {
        rows = buf.shape[0];
        cols = buf.shape[1];
        channels = buf.shape[2];
    } else {
        throw std::runtime_error("Array must be 2D or 3D");
    }

    cv_type = CV_MAKETYPE(cv_type, channels);

    // Create Mat (clone to own the data)
    cv::Mat mat(rows, cols, cv_type, buf.ptr);
    return mat.clone();
}

/**
 * @brief Python callable that can be copied and destroyed on any thread
 *
 * A bare py::function captured in a C++ lambda is decref'd wherever that lambda
 * dies. A model may now be released by a background inference thread (it keeps
 * the model alive until its work is done), and that thread does not hold the
 * GIL, so the reference is dropped under the GIL via the shared_ptr deleter.
 */
struct GilSafeCallable {
    std::shared_ptr<py::function> fn;

    explicit GilSafeCallable(py::function f)
        : fn(new py::function(std::move(f)), [](py::function* p) {
              py::gil_scoped_acquire gil;
              delete p;
          }) {}
};

// ============================================================================
// Python Module Definition
// ============================================================================

namespace {
// Py_IsFinalizing() is public from 3.13; earlier versions only expose the underscore name.
inline bool ivit_py_finalizing() {
#if PY_VERSION_HEX >= 0x030D0000
    return Py_IsFinalizing();
#else
    return _Py_IsFinalizing();
#endif
}
} // namespace

PYBIND11_MODULE(_ivit_core, m) {
    m.doc() = "iVIT-SDK C++ bindings - Unified Computer Vision SDK";

    // Version
    m.attr("__version__") = IVIT_VERSION_STRING;

    // ========================================================================
    // Enums
    // ========================================================================

    py::enum_<ivit::DataType>(m, "DataType", "Tensor data type")
        .value("Float32", ivit::DataType::Float32)
        .value("Float16", ivit::DataType::Float16)
        .value("Int8", ivit::DataType::Int8)
        .value("UInt8", ivit::DataType::UInt8)
        .value("Int32", ivit::DataType::Int32)
        .value("Int64", ivit::DataType::Int64)
        .value("Bool", ivit::DataType::Bool)
        .value("Unknown", ivit::DataType::Unknown)
        .export_values();

    py::enum_<ivit::Precision>(m, "Precision", "Model precision")
        .value("FP32", ivit::Precision::FP32)
        .value("FP16", ivit::Precision::FP16)
        .value("INT8", ivit::Precision::INT8)
        .value("INT4", ivit::Precision::INT4)
        .value("Unknown", ivit::Precision::Unknown)
        .export_values();

    py::enum_<ivit::Layout>(m, "Layout", "Tensor layout")
        .value("NCHW", ivit::Layout::NCHW)
        .value("NHWC", ivit::Layout::NHWC)
        .value("NC", ivit::Layout::NC)
        .value("CHW", ivit::Layout::CHW)
        .value("HWC", ivit::Layout::HWC)
        .value("Unknown", ivit::Layout::Unknown)
        .export_values();

    // Model.task and Model.backend return these; without a registration pybind11
    // cannot convert them and raises TypeError on access. No export_values():
    // their "Unknown"/"Auto" members would shadow the enums above.
    py::enum_<ivit::TaskType>(m, "TaskType", "Model task type")
        .value("Classification", ivit::TaskType::Classification)
        .value("Detection", ivit::TaskType::Detection)
        .value("Segmentation", ivit::TaskType::Segmentation)
        .value("Unknown", ivit::TaskType::Unknown);

    py::enum_<ivit::BackendType>(m, "BackendType", "Inference backend")
        .value("OpenVINO", ivit::BackendType::OpenVINO)
        .value("TensorRT", ivit::BackendType::TensorRT)
        .value("Auto", ivit::BackendType::Auto)
        .value("Unknown", ivit::BackendType::Unknown);

    // ========================================================================
    // Configuration Classes
    // ========================================================================

    py::class_<ivit::LoadConfig>(m, "LoadConfig", "Model loading configuration")
        .def(py::init<>())
        .def_readwrite("device", &ivit::LoadConfig::device, "Target device (e.g., 'cpu', 'cuda:0')")
        .def_readwrite("backend", &ivit::LoadConfig::backend,
                       "Backend: 'auto', 'openvino' or 'tensorrt'")
        .def_readwrite("task", &ivit::LoadConfig::task,
                       "Task hint: 'classification', 'detection' or 'segmentation'")
        .def_readwrite("precision", &ivit::LoadConfig::precision, "Model precision")
        .def_readwrite("batch_size", &ivit::LoadConfig::batch_size, "Batch size")
        .def_readwrite("max_batch_size", &ivit::LoadConfig::max_batch_size)
        .def_readwrite("num_threads", &ivit::LoadConfig::num_threads)
        .def_readwrite("num_instances", &ivit::LoadConfig::num_instances,
                       "TensorRT: execution contexts for concurrent callers (1-16)")
        .def_readwrite("enable_profiling", &ivit::LoadConfig::enable_profiling)
        .def_readwrite("performance_mode", &ivit::LoadConfig::performance_mode)
        .def_readwrite("workspace_size", &ivit::LoadConfig::workspace_size)
        .def_readwrite("cache_dir", &ivit::LoadConfig::cache_dir, "Cache directory")
        .def_readwrite("use_cache", &ivit::LoadConfig::use_cache, "Use cached engine")
        .def_readwrite("decoder", &ivit::LoadConfig::decoder)
        .def_readwrite("input_color", &ivit::LoadConfig::input_color)
        .def_readwrite("input_layout", &ivit::LoadConfig::input_layout)
        .def_readwrite("resize_mode", &ivit::LoadConfig::resize_mode)
        .def_readwrite("input_scale", &ivit::LoadConfig::input_scale)
        .def_readwrite("mean", &ivit::LoadConfig::mean)
        .def_readwrite("std", &ivit::LoadConfig::std)
        .def_readwrite("labels", &ivit::LoadConfig::labels)
        .def_readwrite("labels_path", &ivit::LoadConfig::labels_path)
        .def_readwrite("output_name", &ivit::LoadConfig::output_name)
        .def("__repr__", [](const ivit::LoadConfig& c) {
            return "<LoadConfig device='" + c.device + "'>";
        });

    py::class_<ivit::InferConfig>(m, "InferConfig", "Inference configuration")
        .def(py::init<>())
        .def_readwrite("conf_threshold", &ivit::InferConfig::conf_threshold, "Confidence threshold")
        .def_readwrite("iou_threshold", &ivit::InferConfig::iou_threshold, "NMS IoU threshold")
        .def_readwrite("max_detections", &ivit::InferConfig::max_detections, "Maximum detections")
        .def_readwrite("classes", &ivit::InferConfig::classes, "Filter classes")
        .def_readwrite("agnostic_nms", &ivit::InferConfig::agnostic_nms,
                       "Suppress overlapping boxes across classes (default: per class)")
        .def("__repr__", [](const ivit::InferConfig& c) {
            return "<InferConfig conf=" + std::to_string(c.conf_threshold) +
                   " iou=" + std::to_string(c.iou_threshold) + ">";
        });

    // ========================================================================
    // Data Structures
    // ========================================================================

    py::class_<ivit::TensorInfo>(m, "TensorInfo", "Tensor information")
        .def(py::init<>())
        .def_readonly("name", &ivit::TensorInfo::name)
        .def_readonly("shape", &ivit::TensorInfo::shape)
        .def_readonly("dtype", &ivit::TensorInfo::dtype)
        .def_readonly("layout", &ivit::TensorInfo::layout)
        .def("numel", &ivit::TensorInfo::numel, "Get total number of elements")
        .def("byte_size", &ivit::TensorInfo::byte_size, "Get size in bytes")
        .def("__repr__", [](const ivit::TensorInfo& t) {
            std::string shape_str = "[";
            for (size_t i = 0; i < t.shape.size(); i++) {
                shape_str += std::to_string(t.shape[i]);
                if (i < t.shape.size() - 1) shape_str += ", ";
            }
            shape_str += "]";
            return "<TensorInfo name='" + t.name + "' shape=" + shape_str + ">";
        });

    py::class_<ivit::DeviceInfo>(m, "DeviceInfo", "Device information")
        .def(py::init<>())
        .def_readonly("id", &ivit::DeviceInfo::id)
        .def_readonly("name", &ivit::DeviceInfo::name)
        .def_readonly("backend", &ivit::DeviceInfo::backend)
        .def_readonly("type", &ivit::DeviceInfo::type)
        .def_readonly("memory_total", &ivit::DeviceInfo::memory_total)
        .def_readonly("memory_available", &ivit::DeviceInfo::memory_available)
        .def_readonly("is_available", &ivit::DeviceInfo::is_available)
        .def("__repr__", [](const ivit::DeviceInfo& d) {
            return "<DeviceInfo id='" + d.id + "' name='" + d.name +
                   "' backend='" + d.backend + "'>";
        });

    // ========================================================================
    // BBox
    // ========================================================================

    py::class_<ivit::BBox>(m, "BBox", "Bounding box (x1, y1, x2, y2)")
        .def(py::init<>())
        .def(py::init<float, float, float, float>(),
             py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"))
        .def_readwrite("x1", &ivit::BBox::x1)
        .def_readwrite("y1", &ivit::BBox::y1)
        .def_readwrite("x2", &ivit::BBox::x2)
        .def_readwrite("y2", &ivit::BBox::y2)
        .def_property_readonly("width", &ivit::BBox::width)
        .def_property_readonly("height", &ivit::BBox::height)
        .def_property_readonly("area", &ivit::BBox::area)
        .def_property_readonly("center_x", &ivit::BBox::center_x)
        .def_property_readonly("center_y", &ivit::BBox::center_y)
        .def("iou", &ivit::BBox::iou, py::arg("other"), "Calculate IoU with another box")
        .def("to_xywh", &ivit::BBox::to_xywh, "Convert to [x, y, w, h]")
        .def("to_cxcywh", &ivit::BBox::to_cxcywh, "Convert to [cx, cy, w, h]")
        .def_static("from_xywh", &ivit::BBox::from_xywh, "Create from [x, y, w, h]")
        .def_static("from_cxcywh", &ivit::BBox::from_cxcywh, "Create from [cx, cy, w, h]")
        .def("__repr__", [](const ivit::BBox& b) {
            return "<BBox x1=" + std::to_string(b.x1) + " y1=" + std::to_string(b.y1) +
                   " x2=" + std::to_string(b.x2) + " y2=" + std::to_string(b.y2) + ">";
        });

    // ========================================================================
    // Detection
    // ========================================================================

    py::class_<ivit::Detection>(m, "Detection", "Detection result")
        .def(py::init<>())
        .def_readwrite("bbox", &ivit::Detection::bbox)
        .def_readwrite("class_id", &ivit::Detection::class_id)
        .def_readwrite("label", &ivit::Detection::label)
        .def_readwrite("confidence", &ivit::Detection::confidence)
        .def("__repr__", [](const ivit::Detection& d) {
            return "<Detection '" + d.label + "' conf=" +
                   std::to_string(d.confidence) + ">";
        });

    // ========================================================================
    // ClassificationResult
    // ========================================================================

    py::class_<ivit::ClassificationResult>(m, "ClassificationResult", "Classification result")
        .def(py::init<>())
        .def_readwrite("class_id", &ivit::ClassificationResult::class_id)
        .def_readwrite("label", &ivit::ClassificationResult::label)
        .def_readwrite("score", &ivit::ClassificationResult::score)
        .def("__repr__", [](const ivit::ClassificationResult& c) {
            return "<ClassificationResult '" + c.label + "' score=" +
                   std::to_string(c.score) + ">";
        });

    // ========================================================================
    // Results
    // ========================================================================

    py::class_<ivit::Results>(m, "Results", "Unified inference results")
        .def(py::init<>())

        // Classification
        .def_readwrite("classifications", &ivit::Results::classifications)
        .def("top1", &ivit::Results::top1, "Get top-1 classification")
        .def("topk", &ivit::Results::topk, py::arg("k"), "Get top-k classifications")

        // Detection
        .def_readwrite("detections", &ivit::Results::detections)
        .def("num_detections", &ivit::Results::num_detections, "Get number of detections")
        .def("filter_by_class", &ivit::Results::filter_by_class,
             py::arg("class_ids"), "Filter detections by class IDs")
        .def("filter_by_confidence", &ivit::Results::filter_by_confidence,
             py::arg("min_conf"), "Filter detections by minimum confidence")

        // Segmentation
        .def_property("segmentation_mask",
            [](const ivit::Results& r) { return mat_to_numpy(r.segmentation_mask); },
            [](ivit::Results& r, py::array arr) { r.segmentation_mask = numpy_to_mat(arr); },
            "Segmentation mask as numpy array")
        .def("colorize_mask", [](const ivit::Results& r) {
            return mat_to_numpy(r.colorize_mask());
        }, "Get colorized segmentation mask")
        .def("overlay_mask", [](const ivit::Results& r, py::array image, float alpha) {
            cv::Mat mat = numpy_to_mat(image);
            return mat_to_numpy(r.overlay_mask(mat, alpha));
        }, py::arg("image"), py::arg("alpha") = 0.5f, "Overlay mask on image")

        // Metadata
        .def_readwrite("inference_time_ms", &ivit::Results::inference_time_ms)
        .def_readwrite("device_used", &ivit::Results::device_used)
        .def_property("image_size",
            [](const ivit::Results& r) {
                return py::make_tuple(r.image_size.width, r.image_size.height);
            },
            [](ivit::Results& r, py::tuple t) {
                r.image_size = cv::Size(t[0].cast<int>(), t[1].cast<int>());
            })

        // Visualization
        .def("visualize", [](const ivit::Results& r, py::array image,
                            bool show_labels, bool show_confidence,
                            bool show_boxes, bool show_masks) {
            cv::Mat mat = numpy_to_mat(image);
            cv::Mat vis = r.visualize(mat, show_labels, show_confidence, show_boxes, show_masks);
            return mat_to_numpy(vis);
        }, py::arg("image"), py::arg("show_labels") = true,
           py::arg("show_confidence") = true, py::arg("show_boxes") = true,
           py::arg("show_masks") = true, "Visualize results on image")

        // Serialization
        .def("to_json", &ivit::Results::to_json, "Convert to JSON string")
        .def("save", &ivit::Results::save, py::arg("path"),
             py::arg("format") = "json", "Save results to file")

        // Container methods
        .def("__len__", &ivit::Results::size)
        .def("__bool__", [](const ivit::Results& r) { return !r.empty(); })

        .def("__repr__", [](const ivit::Results& r) {
            std::string s = "<Results ";
            if (!r.classifications.empty()) {
                s += "classifications=" + std::to_string(r.classifications.size()) + " ";
            }
            if (!r.detections.empty()) {
                s += "detections=" + std::to_string(r.detections.size()) + " ";
            }
            if (!r.segmentation_mask.empty()) {
                s += "segmentation ";
            }
            s += "time=" + std::to_string(r.inference_time_ms) + "ms>";
            return s;
        });

    // ========================================================================
    // Classifier
    // ========================================================================

    py::class_<ivit::vision::Classifier>(m, "Classifier", "Image classifier")
        .def(py::init<const std::string&, const std::string&, const ivit::LoadConfig&>(),
             py::arg("model"), py::arg("device") = "auto",
             py::arg("config") = ivit::LoadConfig{},
             "Create classifier from model path")

        // Predict with numpy array
        .def("predict", [](ivit::vision::Classifier& clf, py::array image, int top_k) {
            cv::Mat mat = numpy_to_mat(image);
            return clf.predict(mat, top_k);
        }, py::arg("image"), py::arg("top_k") = 5, "Classify image (numpy array)")

        // Predict with file path
        .def("predict", py::overload_cast<const std::string&, int>(
            &ivit::vision::Classifier::predict),
             py::arg("image_path"), py::arg("top_k") = 5, "Classify image from file")

        // Batch predict
        .def("predict_batch", [](ivit::vision::Classifier& clf,
                                 std::vector<py::array>& images, int top_k) {
            std::vector<cv::Mat> mats;
            mats.reserve(images.size());
            for (auto& img : images) {
                mats.push_back(numpy_to_mat(img));
            }
            return clf.predict_batch(mats, top_k);
        }, py::arg("images"), py::arg("top_k") = 5, "Batch classify images")

        // Properties
        .def_property_readonly("classes", &ivit::vision::Classifier::classes, "Get class labels")
        .def_property_readonly("num_classes", &ivit::vision::Classifier::num_classes, "Get number of classes")
        .def_property_readonly("input_size", [](const ivit::vision::Classifier& clf) {
            auto size = clf.input_size();
            return py::make_tuple(size.width, size.height);
        }, "Get input size (width, height)")

        .def("__repr__", [](const ivit::vision::Classifier& clf) {
            return "<Classifier classes=" + std::to_string(clf.num_classes()) + ">";
        })

        // Allow calling with __call__
        .def("__call__", [](ivit::vision::Classifier& clf, py::array image, int top_k) {
            cv::Mat mat = numpy_to_mat(image);
            return clf.predict(mat, top_k);
        }, py::arg("image"), py::arg("top_k") = 5);

    // ========================================================================
    // Detector
    // ========================================================================

    py::class_<ivit::vision::Detector>(m, "Detector", "Object detector")
        .def(py::init<const std::string&, const std::string&, const ivit::LoadConfig&>(),
             py::arg("model"), py::arg("device") = "auto",
             py::arg("config") = ivit::LoadConfig{},
             "Create detector from model path")

        // Predict with numpy array
        .def("predict", [](ivit::vision::Detector& det, py::array image,
                          float conf_threshold, float iou_threshold) {
            cv::Mat mat = numpy_to_mat(image);
            return det.predict(mat, conf_threshold, iou_threshold);
        }, py::arg("image"), py::arg("conf_threshold") = 0.5f,
           py::arg("iou_threshold") = 0.45f, "Detect objects in image")

        // Predict with file path
        .def("predict", py::overload_cast<const std::string&, float, float>(
            &ivit::vision::Detector::predict),
             py::arg("image_path"), py::arg("conf_threshold") = 0.5f,
             py::arg("iou_threshold") = 0.45f, "Detect objects from file")

        // Predict with InferConfig
        .def("predict_config", [](ivit::vision::Detector& det, py::array image,
                                  const ivit::InferConfig& config) {
            cv::Mat mat = numpy_to_mat(image);
            return det.predict(mat, config);
        }, py::arg("image"), py::arg("config"), "Detect with config")

        // Batch predict
        .def("predict_batch", [](ivit::vision::Detector& det,
                                 std::vector<py::array>& images,
                                 const ivit::InferConfig& config) {
            std::vector<cv::Mat> mats;
            mats.reserve(images.size());
            for (auto& img : images) {
                mats.push_back(numpy_to_mat(img));
            }
            return det.predict_batch(mats, config);
        }, py::arg("images"), py::arg("config") = ivit::InferConfig{},
           "Batch detect objects")

        // Properties
        .def_property_readonly("classes", &ivit::vision::Detector::classes, "Get class labels")
        .def_property_readonly("num_classes", &ivit::vision::Detector::num_classes, "Get number of classes")
        .def_property_readonly("input_size", [](const ivit::vision::Detector& det) {
            auto size = det.input_size();
            return py::make_tuple(size.width, size.height);
        }, "Get input size (width, height)")

        .def("__repr__", [](const ivit::vision::Detector& det) {
            return "<Detector classes=" + std::to_string(det.num_classes()) + ">";
        })

        // Allow calling with __call__
        .def("__call__", [](ivit::vision::Detector& det, py::array image,
                           float conf_threshold, float iou_threshold) {
            cv::Mat mat = numpy_to_mat(image);
            return det.predict(mat, conf_threshold, iou_threshold);
        }, py::arg("image"), py::arg("conf_threshold") = 0.5f,
           py::arg("iou_threshold") = 0.45f);

    // ========================================================================
    // Segmentor
    // ========================================================================

    py::class_<ivit::vision::Segmentor>(m, "Segmentor", "Semantic segmentor")
        .def(py::init<const std::string&, const std::string&, const ivit::LoadConfig&>(),
             py::arg("model"), py::arg("device") = "auto",
             py::arg("config") = ivit::LoadConfig{},
             "Create segmentor from model path")

        // Predict with numpy array
        .def("predict", [](ivit::vision::Segmentor& seg, py::array image) {
            cv::Mat mat = numpy_to_mat(image);
            return seg.predict(mat);
        }, py::arg("image"), "Segment image")

        // Predict with file path
        .def("predict", py::overload_cast<const std::string&>(
            &ivit::vision::Segmentor::predict),
             py::arg("image_path"), "Segment image from file")

        // Properties
        .def_property_readonly("classes", &ivit::vision::Segmentor::classes, "Get class labels")
        .def_property_readonly("num_classes", &ivit::vision::Segmentor::num_classes, "Get number of classes")
        .def_property_readonly("input_size", [](const ivit::vision::Segmentor& seg) {
            auto size = seg.input_size();
            return py::make_tuple(size.width, size.height);
        }, "Get input size (width, height)")

        .def("__repr__", [](const ivit::vision::Segmentor& seg) {
            return "<Segmentor classes=" + std::to_string(seg.num_classes()) + ">";
        })

        // Allow calling with __call__
        .def("__call__", [](ivit::vision::Segmentor& seg, py::array image) {
            cv::Mat mat = numpy_to_mat(image);
            return seg.predict(mat);
        }, py::arg("image"));

    // ========================================================================
    // Callback System
    // ========================================================================

    py::enum_<ivit::CallbackEvent>(m, "CallbackEvent", "Callback event types")
        .value("PreProcess", ivit::CallbackEvent::PreProcess)
        .value("PostProcess", ivit::CallbackEvent::PostProcess)
        .value("InferStart", ivit::CallbackEvent::InferStart)
        .value("InferEnd", ivit::CallbackEvent::InferEnd)
        .value("BatchStart", ivit::CallbackEvent::BatchStart)
        .value("BatchEnd", ivit::CallbackEvent::BatchEnd)
        .value("StreamStart", ivit::CallbackEvent::StreamStart)
        .value("StreamFrame", ivit::CallbackEvent::StreamFrame)
        .value("StreamEnd", ivit::CallbackEvent::StreamEnd)
        .export_values();

    py::class_<ivit::CallbackContext>(m, "CallbackContext", "Context passed to callbacks")
        .def(py::init<>())
        .def_readonly("event", &ivit::CallbackContext::event)
        .def_readonly("model_name", &ivit::CallbackContext::model_name)
        .def_readonly("device", &ivit::CallbackContext::device)
        .def_readonly("latency_ms", &ivit::CallbackContext::latency_ms)
        .def_readonly("batch_size", &ivit::CallbackContext::batch_size)
        .def_readonly("frame_index", &ivit::CallbackContext::frame_index)
        .def_readonly("metadata", &ivit::CallbackContext::metadata)
        .def("__repr__", [](const ivit::CallbackContext& ctx) {
            return "<CallbackContext event='" + ivit::to_string(ctx.event) +
                   "' latency=" + std::to_string(ctx.latency_ms) + "ms>";
        });

    // ========================================================================
    // Model (C++ core model with full API)
    // ========================================================================

    py::class_<ivit::Model, std::shared_ptr<ivit::Model>>(m, "Model", "Inference model")
        // Properties
        .def_property_readonly("name", &ivit::Model::name)
        .def_property_readonly("task", &ivit::Model::task)
        .def_property_readonly("device", &ivit::Model::device)
        .def_property_readonly("backend", &ivit::Model::backend)
        .def_property_readonly("load_config", &ivit::Model::load_config)
        .def_property_readonly("input_info", &ivit::Model::input_info)
        .def_property_readonly("output_info", &ivit::Model::output_info)
        .def_property_readonly("memory_usage", &ivit::Model::memory_usage)
        .def_property_readonly("labels", &ivit::Model::labels)

        // Predict with numpy
        .def("predict", [](ivit::Model& model, py::array image, const ivit::InferConfig& config) {
            cv::Mat mat = numpy_to_mat(image);
            // The GIL is not needed once the data is copied into a cv::Mat
            py::gil_scoped_release release;
            return model.predict(mat, config);
        }, py::arg("image"), py::arg("config") = ivit::InferConfig{},
           "Run inference on image")

        // Predict with file path
        .def("predict", py::overload_cast<const std::string&, const ivit::InferConfig&>(
            &ivit::Model::predict),
             py::arg("image_path"), py::arg("config") = ivit::InferConfig{},
             py::call_guard<py::gil_scoped_release>())

        // model(image) is shorthand for model.predict(image)
        .def("__call__", [](ivit::Model& model, py::array image, const ivit::InferConfig& config) {
            cv::Mat mat = numpy_to_mat(image);
            py::gil_scoped_release release;
            return model.predict(mat, config);
        }, py::arg("image"), py::arg("config") = ivit::InferConfig{},
           "Run inference on image (same as predict)")

        .def("__call__", [](ivit::Model& model, const std::string& image_path,
                             const ivit::InferConfig& config) {
            py::gil_scoped_release release;
            return model.predict(image_path, config);
        }, py::arg("image_path"), py::arg("config") = ivit::InferConfig{})

        // Batch predict
        .def("predict_batch", [](ivit::Model& model, std::vector<py::array>& images,
                                  const ivit::InferConfig& config) {
            std::vector<cv::Mat> mats;
            mats.reserve(images.size());
            for (auto& img : images) {
                mats.push_back(numpy_to_mat(img));
            }
            py::gil_scoped_release release;
            return model.predict_batch(mats, config);
        }, py::arg("images"), py::arg("config") = ivit::InferConfig{})

        // Async predict
        .def("predict_async", [](py::object model, py::array image,
                                  const ivit::InferConfig& config) {
            return py::module_::import("ivit._async").attr("submit")(model, image, config);
        }, py::arg("image"), py::arg("config") = ivit::InferConfig{},
           "Return a concurrent.futures.Future; queued work is cancellable")
        .def("predict_batch_async", [](py::object model, py::list images,
                                       const ivit::InferConfig& config) {
            return py::module_::import("ivit._async").attr("submit")(model, images, config, true);
        }, py::arg("images"), py::arg("config") = ivit::InferConfig{},
           "Return a Future for batch inference")

        // Concurrent predict
        .def("predict_concurrent", [](ivit::Model& model, std::vector<py::array>& images,
                                       int max_concurrent, const ivit::InferConfig& config) {
            std::vector<cv::Mat> mats;
            mats.reserve(images.size());
            for (auto& img : images) {
                mats.push_back(numpy_to_mat(img));
            }
            py::gil_scoped_release release;
            return model.predict_concurrent(mats, max_concurrent, config);
        }, py::arg("images"), py::arg("max_concurrent") = 4,
           py::arg("config") = ivit::InferConfig{})

        // Submit inference (fire-and-forget with Python callback)
        .def("submit_inference", [](ivit::Model& model, py::array image,
                                     py::function callback, const ivit::InferConfig& config) {
            cv::Mat mat = numpy_to_mat(image);
            GilSafeCallable cb(std::move(callback));
            model.submit_inference(mat, [cb](ivit::Results results) {
                py::gil_scoped_acquire acquire;
                (*cb.fn)(results);
            }, config);
        }, py::arg("image"), py::arg("callback"),
           py::arg("config") = ivit::InferConfig{})

        // Warmup
        .def("warmup", &ivit::Model::warmup, py::arg("iterations") = 3)

        // Callbacks
        .def("on", [](ivit::Model& model, const std::string& event,
                       py::function callback, int priority) {
            GilSafeCallable cb(std::move(callback));
            return model.on(event, [cb](const ivit::CallbackContext& ctx) {
                py::gil_scoped_acquire acquire;
                (*cb.fn)(ctx);
            }, priority);
        }, py::arg("event"), py::arg("callback"), py::arg("priority") = 0,
           "Register event callback")

        .def("remove_callback", &ivit::Model::remove_callback,
             py::arg("event"), py::arg("callback_id"))
        .def("remove_all_callbacks", &ivit::Model::remove_all_callbacks,
             py::arg("event"))

        // TTA
        .def("predict_tta", [](ivit::Model& model, py::array image,
                                const std::vector<std::string>& augmentations,
                                const ivit::InferConfig& config) {
            cv::Mat mat = numpy_to_mat(image);
            return model.predict_tta(mat, augmentations, config);
        }, py::arg("image"),
           py::arg("augmentations") = std::vector<std::string>{"original", "hflip"},
           py::arg("config") = ivit::InferConfig{},
           "Run inference with test-time augmentation")

        // Stream - returns StreamIterator
        .def("stream", [](ivit::Model& model, const std::string& source,
                           const ivit::InferConfig& config) {
            return model.stream(source, config);
        }, py::arg("source"), py::arg("config") = ivit::InferConfig{})

        .def("__repr__", [](const ivit::Model& m) {
            return "<Model '" + m.name() + "' device='" + m.device() + "'>";
        });

    // StreamResult
    py::class_<ivit::Model::StreamResult>(m, "StreamResult", "Streaming inference result")
        .def(py::init<>())
        .def_readonly("results", &ivit::Model::StreamResult::results)
        .def_property_readonly("frame", [](const ivit::Model::StreamResult& sr) {
            return mat_to_numpy(sr.frame);
        })
        .def_readonly("frame_index", &ivit::Model::StreamResult::frame_index)
        .def_readonly("fps", &ivit::Model::StreamResult::fps)
        .def_readonly("end_of_stream", &ivit::Model::StreamResult::end_of_stream);

    // StreamIterator
    py::class_<ivit::Model::StreamIterator>(m, "StreamIterator", "Video stream iterator")
        .def("__iter__", [](ivit::Model::StreamIterator& it) -> ivit::Model::StreamIterator& {
            return it;
        })
        .def("__next__", [](ivit::Model::StreamIterator& it) {
            if (!it.has_next()) {
                throw py::stop_iteration();
            }
            auto result = it.next();
            if (result.end_of_stream) {
                throw py::stop_iteration();
            }
            return result;
        });

    // ModelManager - load_model function
    m.def("load_model", [](const std::string& path, const ivit::LoadConfig& config) {
        return ivit::ModelManager::instance().load(path, config);
    }, py::arg("path"), py::arg("config") = ivit::LoadConfig{},
       "Load a model for inference");

    // ========================================================================
    // Module-level Functions
    // ========================================================================

    m.def("version", &ivit::version, "Get iVIT-SDK version");

    // For `ivit info`: the backends this SDK build registered (compiled in and usable here) and the
    // runtime version each one reports. Not part of the public API.
    m.def("_backend_versions", []() {
        std::map<std::string, std::string> versions;
        auto &factory = ivit::RuntimeFactory::instance();
        for (auto type : factory.list_available_runtimes()) {
            if (auto runtime = factory.get_runtime(type))
                versions[ivit::to_string(type)] = runtime->get_version();
        }
        return versions;
    });

    m.def("list_devices", &ivit::list_devices, "List available inference devices");

    m.def("get_best_device", &ivit::get_best_device,
          py::arg("priority") = "performance",
          "Get the best available device");

    m.def("set_log_level", &ivit::set_log_level,
          py::arg("level"), "Set log level ('debug', 'info', 'warning', 'error')");

    m.def(
        "set_log_callback",
        [](py::object callback) {
            if (callback.is_none()) {
                ivit::set_log_callback(nullptr);
                return;
            }
            if (!PyCallable_Check(callback.ptr()))
                throw py::type_error("callback must be callable or None");
            // Leaked on purpose: the SDK can log during interpreter shutdown,
            // when destroying a Python object here would be unsafe.
            auto *held = new py::object(std::move(callback));
            ivit::set_log_callback([held](ivit::LogLevel level, const std::string &message) {
                if (!Py_IsInitialized() || ivit_py_finalizing())
                    return;
                py::gil_scoped_acquire gil;
                try {
                    (*held)(static_cast<int>(level), message);
                } catch (py::error_already_set &) {
                    // Python errors in a log sink must not break inference.
                }
            });
        },
        py::arg("callback"),
        R"doc(Route SDK log messages to callback(level, message); None restores stderr.

level is 0=debug, 1=info, 2=warning, 3=error. The callback may be called from any
SDK thread. Clear it (None) before interpreter exit; ivit does this automatically.)doc");

    m.def("set_cache_dir", &ivit::set_cache_dir,
          py::arg("path"), "Set cache directory for converted models");

    m.def("convert_model", &ivit::convert_model,
          py::arg("src_path"), py::arg("dst_path"),
          py::arg("device") = "auto", py::arg("precision") = "fp16",
          R"doc(
Convert model to optimized format.

Args:
    src_path: Source model path (ONNX)
    dst_path: Destination path (.engine for TensorRT, .xml for OpenVINO)
    device: Target device (determines output format)
    precision: Target precision ("fp32", "fp16", "int8")

Examples:
    # Convert to TensorRT engine
    ivit.convert_model("yolov8n.onnx", "yolov8n.engine", "cuda:0", "fp16")

    # Convert to OpenVINO IR
    ivit.convert_model("yolov8n.onnx", "yolov8n.xml", "cpu", "fp16")
)doc");

    m.def("clear_cache", &ivit::clear_cache,
          py::arg("cache_dir") = "",
          "Release memory cache; an explicit directory also removes engine/blob files");

    // ========================================================================
    // Exception Translation
    // ========================================================================

    // Mirror the C++ hierarchy so `except ivit.IVITError` catches every SDK error.
    // A base must be registered before its subclasses: pybind11 tries the most
    // recently registered translator first, so the most derived type wins.
    auto ivit_error = py::register_exception<ivit::IVITError>(m, "IVITError", PyExc_RuntimeError);
    auto model_error = py::register_exception<ivit::ModelError>(m, "ModelError", ivit_error);
    py::register_exception<ivit::ModelLoadError>(m, "ModelLoadError", model_error);
    py::register_exception<ivit::UnsupportedFormatError>(m, "UnsupportedFormatError", model_error);
    auto device_error = py::register_exception<ivit::DeviceError>(m, "DeviceError", ivit_error);
    py::register_exception<ivit::DeviceNotFoundError>(m, "DeviceNotFoundError", device_error);
    py::register_exception<ivit::InferenceError>(m, "InferenceError", ivit_error);
}

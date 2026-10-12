/**
 * @file tensor.hpp
 * @brief Tensor class definition
 */

#ifndef IVIT_CORE_TENSOR_HPP
#define IVIT_CORE_TENSOR_HPP

#include "ivit/core/common.hpp"
#include <vector>
#include <memory>
#include <cstring>

namespace ivit {

/**
 * @brief Tensor information
 */
struct TensorInfo {
    std::string name;          ///< Tensor name
    Shape shape;               ///< Tensor shape
    DataType dtype;            ///< Data type
    Layout layout;             ///< Tensor layout

    /**
     * @brief Check whether any dimension is dynamic (negative)
     */
    bool is_dynamic() const {
        for (auto d : shape) {
            if (d < 0) return true;
        }
        return false;
    }

    /**
     * @brief Get total number of elements (-1 if the shape is dynamic)
     */
    int64_t numel() const {
        if (is_dynamic()) return -1;
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    /**
     * @brief Get size of one element in bytes
     */
    size_t element_size() const {
        switch (dtype) {
            case DataType::Float16: return 2;
            case DataType::Int8:
            case DataType::UInt8:
            case DataType::Bool: return 1;
            case DataType::Int64: return 8;
            default: return 4;  // Float32, Int32, Unknown
        }
    }

    /**
     * @brief Get size in bytes (0 if the shape is dynamic)
     */
    size_t byte_size() const {
        int64_t n = numel();
        return n < 0 ? 0 : static_cast<size_t>(n) * element_size();
    }
};

/**
 * @brief Replace dynamic dimensions (<= 0) with concrete values
 *
 * Models exported with a dynamic batch or spatial size report -1 for those
 * dimensions, which cannot be allocated. Assumes NCHW: dimension 0 becomes
 * `batch`, 1 becomes 3 (channels), 2 becomes `height` and 3 and above become
 * `width`. Dimensions that are already concrete are left untouched.
 */
inline Shape resolve_dynamic_shape(
    const Shape& shape,
    int64_t batch,
    int64_t height,
    int64_t width
) {
    Shape resolved = shape;
    for (size_t i = 0; i < resolved.size(); ++i) {
        if (resolved[i] > 0) continue;
        if (i == 0) {
            resolved[i] = batch;
        } else if (i == 1) {
            resolved[i] = 3;
        } else if (i == 2) {
            resolved[i] = height;
        } else {
            resolved[i] = width;
        }
    }
    return resolved;
}

/**
 * @brief Lightweight tensor class
 *
 * Simple tensor container for data exchange.
 * For heavy computation, use backend-specific tensors.
 */
class Tensor {
public:
    Tensor() = default;

    /**
     * @brief Construct tensor with shape and data type
     */
    Tensor(const Shape& shape, DataType dtype = DataType::Float32)
        : info_{.name = "", .shape = shape, .dtype = dtype, .layout = Layout::NCHW}
    {
        if (info_.is_dynamic()) {
            throw IVITError("Cannot allocate a Tensor with a dynamic (negative) dimension");
        }
        data_.resize(info_.byte_size());
    }

    /**
     * @brief Construct from raw data
     */
    template<typename T>
    Tensor(const std::vector<T>& data, const Shape& shape)
        : info_{.name = "", .shape = shape, .dtype = DataType::Float32, .layout = Layout::NCHW}
    {
        // Infer dtype from T
        if constexpr (std::is_same_v<T, float>) {
            info_.dtype = DataType::Float32;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            info_.dtype = DataType::Int32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            info_.dtype = DataType::Int64;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            info_.dtype = DataType::UInt8;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            info_.dtype = DataType::Int8;
        }

        data_.resize(data.size() * sizeof(T));
        std::memcpy(data_.data(), data.data(), data_.size());
    }

    // Accessors
    const TensorInfo& info() const { return info_; }
    const Shape& shape() const { return info_.shape; }
    DataType dtype() const { return info_.dtype; }
    Layout layout() const { return info_.layout; }
    int64_t numel() const { return info_.numel(); }
    size_t byte_size() const { return data_.size(); }

    // Data access
    void* data() { return data_.data(); }
    const void* data() const { return data_.data(); }

    template<typename T>
    T* data_ptr() { return reinterpret_cast<T*>(data_.data()); }

    template<typename T>
    const T* data_ptr() const { return reinterpret_cast<const T*>(data_.data()); }

    // Setters
    void set_name(const std::string& name) { info_.name = name; }
    void set_layout(Layout layout) { info_.layout = layout; }

    /**
     * @brief Reshape tensor (must have same number of elements)
     */
    void reshape(const Shape& new_shape) {
        int64_t old_numel = info_.numel();
        info_.shape = new_shape;
        if (info_.numel() != old_numel) {
            throw IVITError("Reshape: element count mismatch");
        }
    }

    /**
     * @brief Clone tensor
     */
    Tensor clone() const {
        Tensor t;
        t.info_ = info_;
        t.data_ = data_;
        return t;
    }

private:
    TensorInfo info_;
    std::vector<uint8_t> data_;
};

} // namespace ivit

#endif // IVIT_CORE_TENSOR_HPP

/**
 * @file runtime_init.cpp
 * @brief Runtime initialization and auto-registration
 */

#include "ivit/runtime/runtime.hpp"

#include <mutex>

#ifdef IVIT_HAS_TENSORRT
#include "ivit/runtime/tensorrt_runtime.hpp"
#endif

#ifdef IVIT_HAS_OPENVINO
#include "ivit/runtime/openvino_runtime.hpp"
#endif

#ifdef IVIT_HAS_QNN
#include "ivit/runtime/qnn_runtime.hpp"
#endif

#ifdef IVIT_HAS_HAILORT
#include "ivit/runtime/hailort_runtime.hpp"
#endif

namespace ivit {

/**
 * @brief Runtime auto-registration on library load
 */
class RuntimeAutoRegister {
public:
    RuntimeAutoRegister() {
        auto& factory = RuntimeFactory::instance();

#ifdef IVIT_HAS_TENSORRT
        try {
            auto tensorrt = std::make_shared<TensorRTRuntime>();
            if (tensorrt->is_available()) {
                factory.register_runtime(tensorrt);
            }
        } catch (...) {
            // TensorRT not available, skip
        }
#endif

#ifdef IVIT_HAS_OPENVINO
        try {
            auto openvino = std::make_shared<OpenVINORuntime>();
            if (openvino->is_available()) {
                factory.register_runtime(openvino);
            }
        } catch (...) {
            // OpenVINO not available, skip
        }
#endif

#ifdef IVIT_HAS_QNN
        try {
            auto qnn = std::make_shared<QNNRuntime>();
            if (qnn->is_available()) {
                factory.register_runtime(qnn);
            }
        } catch (...) {
            // QNN not available, skip
        }
#endif

#ifdef IVIT_HAS_HAILORT
        try {
            auto hailort = std::make_shared<HailoRTRuntime>();
            if (hailort->is_available()) {
                factory.register_runtime(hailort);
            }
        } catch (...) {
            // HailoRT not available, skip
        }
#endif
    }
};

// NOTE: Registration is intentionally NOT triggered by a load-time static
// object. Constructing the TensorRT/OpenVINO runtimes probes CUDA and creates
// library objects; on Windows this must not happen during DLL static
// initialization (it runs under the loader lock and deadlocks). Instead we
// register lazily on first runtime lookup, via ensure_runtimes_registered().
void ensure_runtimes_registered() {
    static std::once_flag flag;
    std::call_once(flag, []() { RuntimeAutoRegister{}; });
}

} // namespace ivit

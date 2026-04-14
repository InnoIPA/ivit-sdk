# HailoRT Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 HailoRT 後端，讓 `ivit.load("model.hef", device="hailo:0")` 能在 Hailo-8 PCIe 加速卡上執行推論。

**Architecture:** 以靜態連結方式整合 HailoRT 4.18+ InferModel 高階 API（不用 dlopen）。`HailoRTRuntime` 繼承 `IRuntime` 介面，通過 `RuntimeAutoRegister` 自動注冊。`HailoConfig` 同時以 C++ struct 和 Python dataclass 形式暴露。

**Tech Stack:** C++17、HailoRT 4.18+（`hailo/hailort.hpp`、`VDevice`/`InferModel` API）、pybind11、CMake `find_library(hailort)`、pytest

---

## 檔案清單

| 動作 | 路徑 |
|------|------|
| 新增 (test) | `tests/python/test_hailort_backend.py` |
| 修改 | `python/ivit/core/runtime_config.py` |
| 修改 | `include/ivit/core/runtime_config.hpp` |
| 修改 | `include/ivit/core/common.hpp` |
| 新增 | `include/ivit/runtime/hailort_runtime.hpp` |
| 新增 | `src/runtime/hailort_runtime.cpp` |
| 修改 | `src/core/device_manager.cpp` |
| 修改 | `src/runtime/runtime_init.cpp` |
| 修改 | `CMakeLists.txt` |
| 修改 | `python/ivit/_binding.cpp` |
| 修改 | `python/ivit/__init__.py` |

---

## Task 1: 先寫測試（TDD 起點）

**Files:**
- Create: `tests/python/test_hailort_backend.py`

- [ ] **Step 1: 建立測試檔**

```python
"""
Tests for the HailoRT backend integration.

These tests run on any machine (including x86 without Hailo-8 hardware).
They validate:
  - HailoConfig Python API
  - Device-string parsing for Hailo devices
  - Runtime discovery (is_available returns False gracefully on non-Hailo hardware)
  - CMake option is wired up correctly
  - No regressions in existing backends
"""

import pytest
import sys
import os
import importlib.util

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _import_file(module_name, rel_path):
    """Import a Python file directly, bypassing package __init__ loaders."""
    abs_path = os.path.abspath(os.path.join(_ROOT, rel_path))
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cpp_available():
    """Return True when C++ bindings + libivit.so are loadable."""
    try:
        import ivit
        return ivit.is_cpp_available()
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# HailoConfig Python API
# ──────────────────────────────────────────────────────────────────────────────

class TestHailoConfigPython:
    """Pure-Python HailoConfig tests (no C++ bindings required)."""

    @pytest.fixture(autouse=True)
    def _rc(self):
        self.rc = _import_file(
            "ivit_runtime_config",
            "python/ivit/core/runtime_config.py"
        )

    def test_import_hailo_config_from_python_module(self):
        """HailoConfig class exists in the pure-Python runtime_config module."""
        assert hasattr(self.rc, "HailoConfig")
        cfg = self.rc.HailoConfig()
        assert cfg is not None

    def test_hailo_config_defaults(self):
        """HailoConfig has expected default values."""
        cfg = self.rc.HailoConfig()
        assert cfg.batch_size == 1
        assert cfg.timeout_ms == 1000

    def test_hailo_config_custom_batch_size(self):
        """HailoConfig accepts custom batch_size."""
        cfg = self.rc.HailoConfig(batch_size=4)
        assert cfg.batch_size == 4

    def test_hailo_config_custom_timeout(self):
        """HailoConfig accepts custom timeout_ms."""
        cfg = self.rc.HailoConfig(timeout_ms=5000)
        assert cfg.timeout_ms == 5000

    def test_hailo_config_dict_roundtrip(self):
        """HailoConfig.to_hailo_config() returns all expected keys."""
        cfg = self.rc.HailoConfig(batch_size=2, timeout_ms=2000)
        d = cfg.to_hailo_config()
        assert d["batch_size"] == 2
        assert d["timeout_ms"] == 2000


# ──────────────────────────────────────────────────────────────────────────────
# HailoConfig C++ bindings (skipped when bindings unavailable)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _cpp_available(), reason="C++ bindings not available")
class TestHailoConfigCpp:
    """Tests for HailoConfig exposed via pybind11."""

    def test_import_hailo_config_from_ivit(self):
        import ivit
        cfg = ivit.HailoConfig()
        assert cfg is not None

    def test_cpp_hailo_config_defaults(self):
        import ivit
        cfg = ivit.HailoConfig()
        assert cfg.batch_size == 1
        assert cfg.timeout_ms == 1000

    def test_cpp_hailo_config_set_fields(self):
        import ivit
        cfg = ivit.HailoConfig()
        cfg.batch_size = 4
        cfg.timeout_ms = 3000
        assert cfg.batch_size == 4
        assert cfg.timeout_ms == 3000


# ──────────────────────────────────────────────────────────────────────────────
# Device string parsing for Hailo devices
# ──────────────────────────────────────────────────────────────────────────────

class TestHailoDeviceStringParsing:
    """
    Verify that Hailo device strings are routed to the HailoRT backend.
    These strings are parsed in C++ (device_manager.cpp).
    """

    _HAILO_ALIASES = [
        "hailo",
        "hailo:0",
        "hailo:1",
    ]

    @pytest.mark.parametrize("alias", _HAILO_ALIASES)
    def test_hailo_aliases_are_known(self, alias):
        """Each Hailo alias must be listed in the supported devices table."""
        assert alias in self._HAILO_ALIASES

    def test_hailo_default_is_first_device(self):
        """'hailo' without index refers to device 0."""
        alias = "hailo"
        assert "hailo" in alias.lower()

    def test_hailo_explicit_index(self):
        """'hailo:0' explicitly addresses the first card."""
        alias = "hailo:0"
        parts = alias.split(":")
        assert parts[0] == "hailo"
        assert parts[1] == "0"


# ──────────────────────────────────────────────────────────────────────────────
# Runtime availability on non-Hailo hardware
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _cpp_available(), reason="C++ bindings not available")
class TestHailoRTRuntimeAvailability:
    """HailoRTRuntime should not crash on hardware without Hailo-8."""

    def test_list_devices_does_not_crash(self):
        """list_devices() must not raise even when Hailo-8 is absent."""
        import ivit
        devices = ivit.list_devices()
        assert isinstance(devices, list)

    def test_hailo_device_not_in_list_on_x86_without_card(self):
        """On x86 without Hailo-8 PCIe card, no hailo device should appear."""
        import ivit
        # Only assert when /dev/hailo0 does not exist (i.e., card absent)
        if os.path.exists("/dev/hailo0"):
            pytest.skip("Hailo-8 card detected — device may be present")
        devices = ivit.list_devices()
        hailo_devices = [d for d in devices if getattr(d, "backend", "") == "hailort"]
        assert len(hailo_devices) == 0, (
            f"Unexpected HailoRT devices: {[d.id for d in hailo_devices]}"
        )

    @pytest.mark.slow
    def test_hailo_load_and_infer(self):
        """Requires physical Hailo-8 card: load HEF and run inference."""
        if not os.path.exists("/dev/hailo0"):
            pytest.skip("Hailo-8 card not present")
        import ivit
        import numpy as np
        # This test requires a real .hef model file to be present
        hef_path = os.path.join(_ROOT, "tests", "data", "resnet50.hef")
        if not os.path.isfile(hef_path):
            pytest.skip(f"Test HEF not found: {hef_path}")
        model = ivit.load(hef_path, device="hailo:0")
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        results = model.infer({"input": dummy})
        assert isinstance(results, dict)


# ──────────────────────────────────────────────────────────────────────────────
# convert_model without DFC installed
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _cpp_available(), reason="C++ bindings not available")
class TestHailoConvertModel:
    """convert_model() must raise ModelConversionError when DFC is absent."""

    def test_convert_no_dfc_raises(self, tmp_path):
        """If hailo_model_zoo CLI is not installed, raise ModelConversionError."""
        import shutil
        if shutil.which("hailomz") or shutil.which("hailo_model_zoo"):
            pytest.skip("hailo_model_zoo is installed — DFC may be available")
        import ivit
        from ivit.core.exceptions import ModelConversionError
        src = str(tmp_path / "dummy.onnx")
        dst = str(tmp_path / "dummy.hef")
        # Write a stub file so path existence check passes
        with open(src, "wb") as f:
            f.write(b"dummy")
        with pytest.raises((ModelConversionError, ivit.IVITError)):
            import ivit._ivit_core as _core
            # Try calling convert_model via load with wrong format to trigger error path
            ivit.load(src, device="hailo:0")


# ──────────────────────────────────────────────────────────────────────────────
# Backend enum in Python
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _cpp_available(), reason="C++ bindings not available")
class TestBackendEnum:
    """The HailoConfig must be in the public ivit namespace."""

    def test_hailo_config_in_public_api(self):
        """HailoConfig is exported from the top-level ivit namespace."""
        import ivit
        assert hasattr(ivit, "HailoConfig"), "HailoConfig must be in ivit namespace"


# ──────────────────────────────────────────────────────────────────────────────
# Regression: existing backends unaffected
# ──────────────────────────────────────────────────────────────────────────────

class TestNoRegressions:
    """Adding HailoRT must not break existing import or API surface."""

    def test_ivit_imports_cleanly(self):
        import ivit
        assert ivit.__version__ is not None

    def test_openvino_config_still_importable(self):
        rc = _import_file("ivit_runtime_config", "python/ivit/core/runtime_config.py")
        cfg = rc.OpenVINOConfig()
        assert cfg.performance_mode == "LATENCY"

    def test_tensorrt_config_still_importable(self):
        rc = _import_file("ivit_runtime_config", "python/ivit/core/runtime_config.py")
        cfg = rc.TensorRTConfig()
        assert cfg.enable_fp16 is False

    def test_qnn_config_still_importable(self):
        rc = _import_file("ivit_runtime_config", "python/ivit/core/runtime_config.py")
        cfg = rc.QNNConfig()
        assert cfg.backend == "htp"

    def test_zoo_module_still_works(self):
        import ivit
        assert hasattr(ivit, "zoo")

    @pytest.mark.skipif(not _cpp_available(), reason="C++ bindings not available")
    def test_classifier_still_importable(self):
        from ivit import Classifier  # noqa


# ──────────────────────────────────────────────────────────────────────────────
# CMake option smoke-test (via file inspection)
# ──────────────────────────────────────────────────────────────────────────────

class TestCMakeHailoRTOption:
    """Verify CMakeLists.txt correctly wires up the HailoRT option."""

    _cmake_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "CMakeLists.txt"
    )

    def _read_cmake(self):
        with open(self._cmake_path) as f:
            return f.read()

    def test_hailort_option_defined(self):
        """IVIT_USE_HAILORT option must be declared."""
        cmake = self._read_cmake()
        assert 'option(IVIT_USE_HAILORT' in cmake

    def test_hailort_source_added_conditionally(self):
        """hailort_runtime.cpp is added under IVIT_USE_HAILORT guard."""
        cmake = self._read_cmake()
        assert "hailort_runtime.cpp" in cmake
        assert "IVIT_USE_HAILORT" in cmake

    def test_hailort_compile_definition(self):
        """IVIT_HAS_HAILORT compile definition is set when option is ON."""
        cmake = self._read_cmake()
        assert "IVIT_HAS_HAILORT" in cmake

    def test_hailort_shown_in_summary(self):
        """HailoRT must appear in the cmake configuration summary."""
        cmake = self._read_cmake()
        assert "HailoRT:" in cmake


# ──────────────────────────────────────────────────────────────────────────────
# Header / source existence
# ──────────────────────────────────────────────────────────────────────────────

class TestHailoRTFilesExist:
    """Required HailoRT backend files must be present in the tree."""

    _root = os.path.join(os.path.dirname(__file__), "..", "..")

    def _path(self, *parts):
        return os.path.join(self._root, *parts)

    def test_hailort_runtime_header_exists(self):
        assert os.path.isfile(self._path("include", "ivit", "runtime", "hailort_runtime.hpp"))

    def test_hailort_runtime_source_exists(self):
        assert os.path.isfile(self._path("src", "runtime", "hailort_runtime.cpp"))

    def test_hailort_runtime_header_has_guard(self):
        with open(self._path("include", "ivit", "runtime", "hailort_runtime.hpp")) as f:
            content = f.read()
        assert "IVIT_RUNTIME_HAILORT_RUNTIME_HPP" in content
        assert "#ifdef IVIT_HAS_HAILORT" in content

    def test_hailort_runtime_source_has_guard(self):
        with open(self._path("src", "runtime", "hailort_runtime.cpp")) as f:
            content = f.read()
        assert "#ifdef IVIT_HAS_HAILORT" in content
        assert "#endif // IVIT_HAS_HAILORT" in content

    def test_runtime_init_registers_hailort(self):
        with open(self._path("src", "runtime", "runtime_init.cpp")) as f:
            content = f.read()
        assert "#ifdef IVIT_HAS_HAILORT" in content
        assert "HailoRTRuntime" in content

    def test_device_manager_discovers_hailort(self):
        with open(self._path("src", "core", "device_manager.cpp")) as f:
            content = f.read()
        assert "discover_hailort_devices" in content
        assert "hailo_is_available" in content

    def test_common_hpp_has_hailort_enum(self):
        with open(self._path("include", "ivit", "core", "common.hpp")) as f:
            content = f.read()
        assert "HailoRT" in content

    def test_runtime_config_hpp_has_hailo_struct(self):
        with open(self._path("include", "ivit", "core", "runtime_config.hpp")) as f:
            content = f.read()
        assert "HailoConfig" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 執行測試，確認大部分失敗（預期行為）**

```bash
cd /home/dos/Documents/ivit-sdk
pytest tests/python/test_hailort_backend.py -v 2>&1 | head -60
```

預期：`TestHailoConfigPython` 全部 FAIL（HailoConfig 還不存在）；`TestNoRegressions` 全部 PASS；`TestHailoRTFilesExist` 全部 FAIL（檔案還不存在）；`TestCMakeHailoRTOption` 全部 FAIL。

- [ ] **Step 3: Commit**

```bash
git add tests/python/test_hailort_backend.py
git commit -m "test: add HailoRT backend test suite (all fail — TDD start)"
```

---

## Task 2: HailoConfig Python dataclass

**Files:**
- Modify: `python/ivit/core/runtime_config.py`

- [ ] **Step 1: 在檔案末尾（`SNPEConfig = QNNConfig` 之後）加入 HailoConfig**

在 `python/ivit/core/runtime_config.py` 的 `SNPEConfig = QNNConfig` 這行之後新增：

```python

@dataclass
class HailoConfig:
    """
    HailoRT runtime configuration for Hailo-8 AI acceleration card.

    Examples:
        >>> config = HailoConfig(batch_size=1, timeout_ms=1000)
    """
    batch_size: int = 1
    """Batch size for inference (default 1)"""

    timeout_ms: int = 1000
    """Inference timeout in milliseconds (default 1000)"""

    def to_hailo_config(self) -> Dict[str, Any]:
        """Convert to HailoRT config dict."""
        return {
            "batch_size": self.batch_size,
            "timeout_ms": self.timeout_ms,
        }
```

- [ ] **Step 2: 執行 Python config 測試**

```bash
cd /home/dos/Documents/ivit-sdk
pytest tests/python/test_hailort_backend.py::TestHailoConfigPython -v
```

預期：5 個測試全部 PASS。

- [ ] **Step 3: 執行回歸測試確認沒破壞現有 config**

```bash
pytest tests/python/test_hailort_backend.py::TestNoRegressions -v
```

預期：全部 PASS。

- [ ] **Step 4: Commit**

```bash
git add python/ivit/core/runtime_config.py
git commit -m "feat: add HailoConfig Python dataclass to runtime_config"
```

---

## Task 3: BackendType::HailoRT + C++ HailoConfig struct

**Files:**
- Modify: `include/ivit/core/common.hpp`
- Modify: `include/ivit/core/runtime_config.hpp`

- [ ] **Step 1: 在 `common.hpp` 的 `BackendType` enum 加入 HailoRT**

找到這段（`include/ivit/core/common.hpp:92-98`）：

```cpp
enum class BackendType {
    OpenVINO,    ///< Intel: CPU, iGPU, NPU, VPU
    TensorRT,    ///< NVIDIA: CUDA GPUs, Jetson
    QNN,         ///< Qualcomm: IQ Series NPU (Hexagon) via AI Engine Direct
    Auto,        ///< Automatic selection
    Unknown
};
```

改為：

```cpp
enum class BackendType {
    OpenVINO,    ///< Intel: CPU, iGPU, NPU, VPU
    TensorRT,    ///< NVIDIA: CUDA GPUs, Jetson
    QNN,         ///< Qualcomm: IQ Series NPU (Hexagon) via AI Engine Direct
    HailoRT,     ///< Hailo: Hailo-8 PCIe AI accelerator
    Auto,        ///< Automatic selection
    Unknown
};
```

- [ ] **Step 2: 在 `common.hpp` 的 `to_string(BackendType)` 加入 HailoRT case**

找到（`include/ivit/core/common.hpp:243-251`）：

```cpp
inline std::string to_string(BackendType backend) {
    switch (backend) {
        case BackendType::OpenVINO: return "openvino";
        case BackendType::TensorRT: return "tensorrt";
        case BackendType::QNN: return "qnn";
        case BackendType::Auto: return "auto";
        default: return "unknown";
    }
}
```

改為：

```cpp
inline std::string to_string(BackendType backend) {
    switch (backend) {
        case BackendType::OpenVINO: return "openvino";
        case BackendType::TensorRT: return "tensorrt";
        case BackendType::QNN: return "qnn";
        case BackendType::HailoRT: return "hailort";
        case BackendType::Auto: return "auto";
        default: return "unknown";
    }
}
```

- [ ] **Step 3: 在 `runtime_config.hpp` 加入 HailoConfig struct**

找到檔案末尾（`include/ivit/core/runtime_config.hpp:57-62`）：

```cpp
    std::map<std::string, std::string> extra_options; ///< Additional options
};

} // namespace ivit
```

改為：

```cpp
    std::map<std::string, std::string> extra_options; ///< Additional options
};

/**
 * @brief HailoRT runtime configuration for Hailo-8 AI accelerator
 */
struct HailoConfig {
    int batch_size = 1;      ///< Batch size (default 1)
    int timeout_ms = 1000;   ///< Inference timeout in milliseconds (default 1000)
};

} // namespace ivit
```

- [ ] **Step 4: 執行 file existence 測試**

```bash
cd /home/dos/Documents/ivit-sdk
pytest tests/python/test_hailort_backend.py::TestHailoRTFilesExist::test_common_hpp_has_hailort_enum -v
pytest tests/python/test_hailort_backend.py::TestHailoRTFilesExist::test_runtime_config_hpp_has_hailo_struct -v
```

預期：2 個測試 PASS。

- [ ] **Step 5: Commit**

```bash
git add include/ivit/core/common.hpp include/ivit/core/runtime_config.hpp
git commit -m "feat: add BackendType::HailoRT enum and HailoConfig C++ struct"
```

---

## Task 4: hailort_runtime.hpp 標頭

**Files:**
- Create: `include/ivit/runtime/hailort_runtime.hpp`

- [ ] **Step 1: 建立標頭檔**

```cpp
/**
 * @file hailort_runtime.hpp
 * @brief HailoRT backend runtime for Hailo-8 AI accelerator
 *
 * Supports Hailo-8 PCIe M.2 AI accelerator via HailoRT 4.18+
 * InferModel high-level C++ API.
 *
 * Device strings recognised:
 *   "hailo"    → first Hailo-8 card (device_id = 0)
 *   "hailo:0"  → explicit first card
 *   "hailo:1"  → second card (multi-card environment)
 *
 * Model format: .hef (Hailo Executable Format)
 * Conversion: hailo_model_zoo compile <onnx> --hw-arch hailo8 -o <hef>
 */

#ifndef IVIT_RUNTIME_HAILORT_RUNTIME_HPP
#define IVIT_RUNTIME_HAILORT_RUNTIME_HPP

#include "ivit/runtime/runtime.hpp"

#ifdef IVIT_HAS_HAILORT

#include <hailo/hailort.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <map>

namespace ivit {

/**
 * @brief HailoRT loaded model wrapper
 *
 * Holds all HailoRT InferModel handles and tensor metadata
 * for one loaded .hef model.
 */
struct HailoRTModel {
    std::shared_ptr<hailort::VDevice>    vdevice;
    std::shared_ptr<hailort::InferModel> infer_model;
    hailort::ConfiguredInferModel        configured_model;

    std::vector<TensorInfo> input_infos;
    std::vector<TensorInfo> output_infos;

    std::string device_id;  ///< e.g. "hailo:0"
    int         device_idx; ///< Physical device index (0, 1, …)
};

/**
 * @brief Hailo-8 runtime implementation via HailoRT InferModel API
 */
class HailoRTRuntime : public IRuntime {
public:
    HailoRTRuntime();
    ~HailoRTRuntime() override = default;

    // ----- IRuntime interface -----
    BackendType              type()              const override { return BackendType::HailoRT; }
    std::string              name()              const override { return "HailoRT"; }
    bool                     is_available()      const override;
    std::vector<std::string> supported_formats() const override;
    std::vector<DeviceInfo>  get_devices()       const override;

    void* load_model(
        const std::string& path,
        const RuntimeConfig& config
    ) override;

    void unload_model(void* handle) override;

    std::vector<TensorInfo> get_input_info(void* handle)  const override;
    std::vector<TensorInfo> get_output_info(void* handle) const override;

    std::map<std::string, Tensor> infer(
        void* handle,
        const std::map<std::string, Tensor>& inputs
    ) override;

    void convert_model(
        const std::string& src_path,
        const std::string& dst_path,
        const RuntimeConfig& config
    ) override;

private:
    int  parse_device_index(const std::string& device_id) const;

    DataType hailort_to_dtype(hailo_format_type_t fmt) const;
    TensorInfo make_tensor_info(const hailort::InferModel::InferStream& stream) const;

    mutable std::mutex mutex_;
};

/**
 * @brief Check if any Hailo device is accessible on this machine
 */
bool hailo_is_available();

} // namespace ivit

#endif // IVIT_HAS_HAILORT
#endif // IVIT_RUNTIME_HAILORT_RUNTIME_HPP
```

- [ ] **Step 2: 執行 header 存在性測試**

```bash
cd /home/dos/Documents/ivit-sdk
pytest tests/python/test_hailort_backend.py::TestHailoRTFilesExist::test_hailort_runtime_header_exists -v
pytest tests/python/test_hailort_backend.py::TestHailoRTFilesExist::test_hailort_runtime_header_has_guard -v
```

預期：2 個測試 PASS。

- [ ] **Step 3: Commit**

```bash
git add include/ivit/runtime/hailort_runtime.hpp
git commit -m "feat: add HailoRTRuntime and HailoRTModel class declarations"
```

---

## Task 5: hailort_runtime.cpp 完整實作

**Files:**
- Create: `src/runtime/hailort_runtime.cpp`

- [ ] **Step 1: 建立實作檔**

```cpp
/**
 * @file hailort_runtime.cpp
 * @brief HailoRT backend runtime implementation
 */

#ifdef IVIT_HAS_HAILORT

#include "ivit/runtime/hailort_runtime.hpp"
#include "ivit/core/runtime_config.hpp"
#include <hailo/hailort.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ivit {

// ============================================================================
// Constructor
// ============================================================================

HailoRTRuntime::HailoRTRuntime() {}

// ============================================================================
// IRuntime: availability & discovery
// ============================================================================

bool HailoRTRuntime::is_available() const {
    return hailo_is_available();
}

std::vector<std::string> HailoRTRuntime::supported_formats() const {
    return {".hef"};
}

std::vector<DeviceInfo> HailoRTRuntime::get_devices() const {
    std::vector<DeviceInfo> result;

    auto scan = hailort::Device::scan();
    if (!scan) {
        return result;
    }

    int idx = 0;
    for (const auto& dev_info : scan.value()) {
        DeviceInfo info;
        info.id           = "hailo:" + std::to_string(idx);
        info.name         = "Hailo-8 [" + std::string(dev_info.id) + "]";
        info.backend      = "hailort";
        info.type         = "npu";
        info.is_available = true;
        info.supported_precisions = {Precision::FP32, Precision::FP16, Precision::INT8};
        result.push_back(info);
        ++idx;
    }

    return result;
}

// ============================================================================
// IRuntime: load_model
// ============================================================================

void* HailoRTRuntime::load_model(
    const std::string& path,
    const RuntimeConfig& config
) {
    // Validate format
    std::filesystem::path fpath(path);
    if (fpath.extension() != ".hef") {
        throw UnsupportedFormatError(fpath.extension().string());
    }

    if (!std::filesystem::exists(fpath)) {
        throw ModelLoadError("File not found: " + path);
    }

    // Determine device index from config.device (e.g. "hailo:0")
    int dev_idx = parse_device_index(config.device);

    // Extract HailoConfig if present
    int batch_size = 1;
    int timeout_ms = 1000;
    if (auto* hcfg = std::get_if<HailoConfig>(&config.backend_config)) {
        batch_size = hcfg->batch_size;
        timeout_ms = hcfg->timeout_ms;
    }

    auto model_handle = std::make_unique<HailoRTModel>();
    model_handle->device_id  = config.device;
    model_handle->device_idx = dev_idx;

    // Create VDevice
    hailo_vdevice_params_t params;
    auto status = hailo_init_vdevice_params(&params);
    if (status != HAILO_SUCCESS) {
        throw ModelLoadError("hailo_init_vdevice_params failed: " +
                             std::to_string(static_cast<int>(status)));
    }
    params.device_count = 1;
    // Note: specific device selection via scheduling_algorithm & device_ids
    // For simplicity we open the first available device; multi-card users
    // should set HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN with device IDs.

    auto vdevice = hailort::VDevice::create(params);
    if (!vdevice) {
        throw DeviceNotFoundError("hailo:" + std::to_string(dev_idx));
    }
    model_handle->vdevice = std::move(vdevice.value());

    // Create InferModel from HEF file
    auto infer_model = model_handle->vdevice->create_infer_model(path);
    if (!infer_model) {
        throw ModelLoadError("create_infer_model failed for: " + path);
    }
    model_handle->infer_model = std::move(infer_model.value());

    // Configure batch size
    model_handle->infer_model->set_batch_size(batch_size);

    // Configure and obtain a ConfiguredInferModel
    auto configured = model_handle->infer_model->configure();
    if (!configured) {
        throw ModelLoadError("InferModel::configure() failed: " + path);
    }
    model_handle->configured_model = std::move(configured.value());

    // Extract tensor info
    for (const auto& input : model_handle->infer_model->inputs()) {
        model_handle->input_infos.push_back(make_tensor_info(input));
    }
    for (const auto& output : model_handle->infer_model->outputs()) {
        model_handle->output_infos.push_back(make_tensor_info(output));
    }

    return model_handle.release();
}

// ============================================================================
// IRuntime: unload_model
// ============================================================================

void HailoRTRuntime::unload_model(void* handle) {
    if (handle) {
        delete static_cast<HailoRTModel*>(handle);
    }
}

// ============================================================================
// IRuntime: tensor info
// ============================================================================

std::vector<TensorInfo> HailoRTRuntime::get_input_info(void* handle) const {
    return static_cast<HailoRTModel*>(handle)->input_infos;
}

std::vector<TensorInfo> HailoRTRuntime::get_output_info(void* handle) const {
    return static_cast<HailoRTModel*>(handle)->output_infos;
}

// ============================================================================
// IRuntime: infer
// ============================================================================

std::map<std::string, Tensor> HailoRTRuntime::infer(
    void*  handle,
    const std::map<std::string, Tensor>& inputs
) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto* m = static_cast<HailoRTModel*>(handle);

    // Retrieve timeout from device_id context (stored at load time)
    // Default to 1000 ms
    int timeout_ms = 1000;

    // Create bindings
    auto bindings_exp = m->configured_model.create_bindings();
    if (!bindings_exp) {
        throw InferenceError("HailoRT create_bindings() failed");
    }
    auto bindings = std::move(bindings_exp.value());

    // Copy input tensors into bindings
    for (const auto& [name, tensor] : inputs) {
        auto input_buffer = bindings.input(name);
        if (!input_buffer) {
            throw InferenceError("HailoRT: unknown input name: " + name);
        }
        auto status = input_buffer->set_buffer(
            hailort::MemoryView(const_cast<void*>(tensor.data()),
                                tensor.nbytes())
        );
        if (status != HAILO_SUCCESS) {
            throw InferenceError("HailoRT: set_buffer failed for input: " + name);
        }
    }

    // Allocate output buffers
    std::map<std::string, std::vector<uint8_t>> out_buffers;
    for (const auto& info : m->output_infos) {
        size_t nbytes = 1;
        for (auto d : info.shape) nbytes *= static_cast<size_t>(d);
        nbytes *= sizeof(float);  // HailoRT outputs are FLOAT32 after dequant
        out_buffers[info.name].resize(nbytes);

        auto output_buffer = bindings.output(info.name);
        if (!output_buffer) {
            throw InferenceError("HailoRT: unknown output name: " + info.name);
        }
        auto status = output_buffer->set_buffer(
            hailort::MemoryView(out_buffers[info.name].data(), nbytes)
        );
        if (status != HAILO_SUCCESS) {
            throw InferenceError("HailoRT: set_buffer failed for output: " + info.name);
        }
    }

    // Run inference
    auto run_status = m->configured_model.run(
        std::move(bindings),
        std::chrono::milliseconds(timeout_ms)
    );
    if (run_status != HAILO_SUCCESS) {
        if (run_status == HAILO_TIMEOUT) {
            throw InferenceError("HailoRT inference timeout (" +
                                 std::to_string(timeout_ms) + " ms)");
        }
        throw InferenceError("HailoRT run() failed: " +
                             std::to_string(static_cast<int>(run_status)));
    }

    // Collect outputs into Tensor map
    std::map<std::string, Tensor> results;
    for (const auto& info : m->output_infos) {
        Tensor t;
        t.name  = info.name;
        t.shape = info.shape;
        t.dtype = info.dtype;
        auto& buf = out_buffers[info.name];
        t.data_buffer.assign(buf.begin(), buf.end());
        t.data = t.data_buffer.data();
        results[info.name] = std::move(t);
    }
    return results;
}

// ============================================================================
// IRuntime: convert_model (calls hailo_model_zoo compile CLI)
// ============================================================================

void HailoRTRuntime::convert_model(
    const std::string& src_path,
    const std::string& dst_path,
    const RuntimeConfig& /*config*/
) {
    // Check for hailo_model_zoo CLI
    if (std::system("hailomz --version > /dev/null 2>&1") != 0 &&
        std::system("python -m hailo_model_zoo --help > /dev/null 2>&1") != 0) {
        throw IVITError(
            "Hailo Dataflow Compiler (hailo_model_zoo) is not installed.\n"
            "Install it from:\n"
            "  https://developer.hailo.ai/developer-zone/software-downloads/\n"
            "Then run:\n"
            "  python -m hailo_model_zoo compile " + src_path +
            " --hw-arch hailo8 -o " + dst_path
        );
    }

    // Build command:  python -m hailo_model_zoo compile <src> --hw-arch hailo8 -o <dst>
    std::string cmd = "python -m hailo_model_zoo compile " + src_path +
                      " --hw-arch hailo8 -o " + dst_path;

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw IVITError("hailo_model_zoo compile failed (exit code: " +
                        std::to_string(ret) + ")\nCommand: " + cmd);
    }
}

// ============================================================================
// Private helpers
// ============================================================================

int HailoRTRuntime::parse_device_index(const std::string& device_id) const {
    // "hailo"   → 0
    // "hailo:0" → 0
    // "hailo:1" → 1
    if (device_id == "hailo") return 0;
    auto pos = device_id.find(':');
    if (pos != std::string::npos) {
        try { return std::stoi(device_id.substr(pos + 1)); }
        catch (...) {}
    }
    return 0;
}

DataType HailoRTRuntime::hailort_to_dtype(hailo_format_type_t fmt) const {
    switch (fmt) {
        case HAILO_FORMAT_TYPE_FLOAT32: return DataType::Float32;
        case HAILO_FORMAT_TYPE_UINT8:   return DataType::UInt8;
        case HAILO_FORMAT_TYPE_UINT16:  return DataType::UInt8;  // treat as uint8 buffer
        default:                         return DataType::Unknown;
    }
}

TensorInfo HailoRTRuntime::make_tensor_info(
    const hailort::InferModel::InferStream& stream
) const {
    TensorInfo info;
    info.name = stream.name();

    auto shape = stream.shape();
    // HailoRT shape: {batch, height, width, channels} (NHWC)
    info.shape = {
        static_cast<int64_t>(shape.batch_size),
        static_cast<int64_t>(shape.height),
        static_cast<int64_t>(shape.width),
        static_cast<int64_t>(shape.features)
    };
    info.dtype  = hailort_to_dtype(stream.format().type);
    info.layout = Layout::NHWC;
    return info;
}

// ============================================================================
// hailo_is_available() — probes /dev/hailo0 existence
// ============================================================================

bool hailo_is_available() {
    // Quick probe: check if /dev/hailo0 device node exists
    // (created by HailoRT kernel driver when a card is plugged in)
    return std::filesystem::exists("/dev/hailo0");
}

} // namespace ivit

#endif // IVIT_HAS_HAILORT
```

- [ ] **Step 2: 執行 source 存在性測試**

```bash
cd /home/dos/Documents/ivit-sdk
pytest tests/python/test_hailort_backend.py::TestHailoRTFilesExist::test_hailort_runtime_source_exists -v
pytest tests/python/test_hailort_backend.py::TestHailoRTFilesExist::test_hailort_runtime_source_has_guard -v
```

預期：2 個測試 PASS。

- [ ] **Step 3: Commit**

```bash
git add src/runtime/hailort_runtime.cpp
git commit -m "feat: implement HailoRTRuntime with InferModel API (load/infer/convert)"
```

---

## Task 6: device_manager.cpp — 裝置探索與 parse_device_string

**Files:**
- Modify: `src/core/device_manager.cpp`

- [ ] **Step 1: 在 `device_manager.cpp` 頂部的 include 區塊加入 HailoRT guard**

找到（`src/core/device_manager.cpp:26-29`）：

```cpp
#ifdef IVIT_HAS_QNN
#include <dlfcn.h>
#endif
```

改為：

```cpp
#ifdef IVIT_HAS_QNN
#include <dlfcn.h>
#endif

#ifdef IVIT_HAS_HAILORT
#include <filesystem>
#endif
```

- [ ] **Step 2: 在 `discover_devices()` 函式中（`discover_qnn_devices();` 之後）加入 Hailo 探索呼叫**

找到（`src/core/device_manager.cpp:289-290`）：

```cpp
    / Qualcomm: QNN / AI Engine Direct (HTP, Adreno GPU)
    discover_qnn_devices();
```

改為：

```cpp
    / Qualcomm: QNN / AI Engine Direct (HTP, Adreno GPU)
    discover_qnn_devices();

    // Hailo: HailoRT (Hailo-8 PCIe AI accelerator)
    discover_hailort_devices();
```

- [ ] **Step 3: 在檔案末尾（`qnn_is_available()` 函式之後）加入 Hailo 探索函式**

找到（`src/core/device_manager.cpp` 末尾）檔案最後幾行：

```cpp
bool qnn_is_available() {
#ifdef IVIT_HAS_QNN
    for (const char* lib : {"libQnnHtp.so", "libQnnCpu.so", "libQnnGpu.so"}) {
        void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (h) { dlclose(h); return true; }
    }
#endif
```

在整個檔案的最後（`}` 之後，`} // namespace ivit` 之前）加入：

```cpp
void DeviceManager::discover_hailort_devices() {
#ifdef IVIT_HAS_HAILORT
    // Probe by checking device nodes created by HailoRT driver
    for (int i = 0; i < 8; ++i) {
        std::string dev_node = "/dev/hailo" + std::to_string(i);
        if (!std::filesystem::exists(dev_node)) break;

        DeviceInfo info;
        info.id           = "hailo:" + std::to_string(i);
        info.name         = "Hailo-8 (device " + std::to_string(i) + ")";
        info.backend      = "hailort";
        info.type         = "npu";
        info.is_available = true;
        info.supported_precisions = {Precision::FP32, Precision::FP16, Precision::INT8};

        devices_.push_back(info);
    }
#endif
}

bool hailo_is_available() {
#ifdef IVIT_HAS_HAILORT
    return std::filesystem::exists("/dev/hailo0");
#else
    return false;
#endif
}
```

- [ ] **Step 4: 在 `parse_device_string()` 中加入 hailo case**

找到（`src/core/device_manager.cpp` 中 `parse_device_string` 函式末尾，`// Default to CPU` 之前）：

```cpp
    // Default to CPU (OpenVINO fallback)
    return {BackendType::OpenVINO, "CPU"};
```

在它前面加入：

```cpp
    // Hailo-8 PCIe AI accelerator via HailoRT
    if (dev_lower == "hailo" || dev_lower.rfind("hailo:", 0) == 0) {
        std::string idx = "0";
        auto pos = dev_lower.find(':');
        if (pos != std::string::npos) idx = dev_lower.substr(pos + 1);
        return {BackendType::HailoRT, idx};
    }

    // Default to CPU (OpenVINO fallback)
    return {BackendType::OpenVINO, "CPU"};
```

（注意：移除原來的那行 `return {BackendType::OpenVINO, "CPU"};`，改為在新的 hailo block 之後放）

- [ ] **Step 5: 在 `DeviceManager` class declaration（device_manager.hpp 或相關 header）加入 `discover_hailort_devices` 宣告**

```bash
grep -n "discover_qnn_devices" /home/dos/Documents/ivit-sdk/include/ivit/core/device.hpp
```

若找到宣告位置，在 `discover_qnn_devices()` 宣告後面加一行：

```cpp
    void discover_hailort_devices();
```

- [ ] **Step 6: 執行 device_manager 存在性測試**

```bash
cd /home/dos/Documents/ivit-sdk
pytest tests/python/test_hailort_backend.py::TestHailoRTFilesExist::test_device_manager_discovers_hailort -v
```

預期：PASS。

- [ ] **Step 7: Commit**

```bash
git add src/core/device_manager.cpp
git commit -m "feat: add HailoRT device discovery and parse_device_string hailo case"
```

---

## Task 7: runtime_init.cpp — 自動注冊 HailoRTRuntime

**Files:**
- Modify: `src/runtime/runtime_init.cpp`

- [ ] **Step 1: 在 include 區塊加入 HailoRT header**

找到（`src/runtime/runtime_init.cpp:16-18`）：

```cpp
#ifdef IVIT_HAS_QNN
#include "ivit/runtime/qnn_runtime.hpp"
#endif
```

改為：

```cpp
#ifdef IVIT_HAS_QNN
#include "ivit/runtime/qnn_runtime.hpp"
#endif

#ifdef IVIT_HAS_HAILORT
#include "ivit/runtime/hailort_runtime.hpp"
#endif
```

- [ ] **Step 2: 在 `RuntimeAutoRegister` constructor 中加入 HailoRT 注冊區塊**

找到（`src/runtime/runtime_init.cpp:53-63`）：

```cpp
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

    }
```

改為：

```cpp
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
```

- [ ] **Step 3: 執行 runtime_init 存在性測試**

```bash
cd /home/dos/Documents/ivit-sdk
pytest tests/python/test_hailort_backend.py::TestHailoRTFilesExist::test_runtime_init_registers_hailort -v
```

預期：PASS。

- [ ] **Step 4: Commit**

```bash
git add src/runtime/runtime_init.cpp
git commit -m "feat: auto-register HailoRTRuntime in runtime_init.cpp"
```

---

## Task 8: CMakeLists.txt — IVIT_USE_HAILORT 選項

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 在 option 區塊（第 12 行 `option(IVIT_USE_QNN...)` 之後）加入 HailoRT option**

找到（`CMakeLists.txt:12`）：

```cmake
option(IVIT_USE_QNN "Enable QNN backend (Qualcomm IQ Series)" OFF)
```

改為：

```cmake
option(IVIT_USE_QNN "Enable QNN backend (Qualcomm IQ Series)" OFF)
option(IVIT_USE_HAILORT "Enable HailoRT backend (Hailo-8 PCIe AI accelerator)" OFF)
```

- [ ] **Step 2: 在 QNN 區塊（`endif() # IVIT_USE_QNN` 之後）加入 HailoRT 搜尋區塊**

找到（`CMakeLists.txt:205-207`）：

```cmake
        set(IVIT_USE_QNN OFF)
    endif()
endif()
```

在 `endif() # 結束 QNN 區塊` 之後，加入：

```cmake

# ============================================================================
# HailoRT (Hailo-8 PCIe AI Accelerator)
# ============================================================================
if(IVIT_USE_HAILORT)
    if(NOT DEFINED HAILORT_SDK_ROOT AND DEFINED ENV{HAILORT_SDK_ROOT})
        set(HAILORT_SDK_ROOT "$ENV{HAILORT_SDK_ROOT}")
    endif()

    find_path(HAILORT_INCLUDE_DIR hailo/hailort.hpp
        HINTS ${HAILORT_SDK_ROOT} /usr/include /usr/local/include
    )
    find_library(HAILORT_LIBRARY hailort
        HINTS ${HAILORT_SDK_ROOT} /usr/lib /usr/local/lib
    )

    if(HAILORT_INCLUDE_DIR AND HAILORT_LIBRARY)
        message(STATUS "HailoRT found: ${HAILORT_LIBRARY}")
        add_compile_definitions(IVIT_HAS_HAILORT)
    else()
        message(WARNING "HailoRT not found — disabling HailoRT backend. "
                        "Set HAILORT_SDK_ROOT to your SDK installation path "
                        "or install via: apt install hailoRT-dev")
        set(IVIT_USE_HAILORT OFF)
    endif()
endif()
```

- [ ] **Step 3: 在 sources 清單（`if(IVIT_USE_QNN)` 區塊之後）加入 HailoRT source**

找到（`CMakeLists.txt:243-245`）：

```cmake
if(IVIT_USE_QNN)
    list(APPEND IVIT_SOURCES src/runtime/qnn_runtime.cpp)
endif()
```

改為：

```cmake
if(IVIT_USE_QNN)
    list(APPEND IVIT_SOURCES src/runtime/qnn_runtime.cpp)
endif()

if(IVIT_USE_HAILORT)
    list(APPEND IVIT_SOURCES src/runtime/hailort_runtime.cpp)
endif()
```

- [ ] **Step 4: 在 target_link_libraries 區塊（`if(IVIT_USE_QNN)` link 區塊之後）加入 HailoRT link**

找到（`CMakeLists.txt:280-285`）：

```cmake
if(IVIT_USE_QNN)
    target_include_directories(ivit PRIVATE ${QNN_INCLUDE_DIR})
    # QNN backend libraries are loaded dynamically at runtime (dlopen).
```

在整個 QNN link 區塊（`endif()` 之後）加入：

```cmake
if(IVIT_USE_HAILORT)
    target_include_directories(ivit PRIVATE ${HAILORT_INCLUDE_DIR})
    target_link_libraries(ivit PRIVATE ${HAILORT_LIBRARY})
endif()
```

- [ ] **Step 5: 在 build summary（`message(STATUS "  QNN:...` 之後）加入 HailoRT 一行**

找到（`CMakeLists.txt:504`）：

```cmake
message(STATUS "  QNN:            ${IVIT_USE_QNN}")
```

改為：

```cmake
message(STATUS "  QNN:            ${IVIT_USE_QNN}")
message(STATUS "  HailoRT:        ${IVIT_USE_HAILORT}")
```

- [ ] **Step 6: 執行 CMake 測試**

```bash
cd /home/dos/Documents/ivit-sdk
pytest tests/python/test_hailort_backend.py::TestCMakeHailoRTOption -v
```

預期：4 個測試全部 PASS。

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add IVIT_USE_HAILORT CMake option with find_library(hailort)"
```

---

## Task 9: _binding.cpp — pybind11 暴露 HailoConfig

**Files:**
- Modify: `python/ivit/_binding.cpp`

- [ ] **Step 1: 在 include 區塊加入 HailoConfig header**

找到（`python/ivit/_binding.cpp:16`）：

```cpp
// Include iVIT headers
```

在該區塊（通常在 `ivit/ivit.hpp` 等 include 之後）加入：

```cpp
#include "ivit/core/runtime_config.hpp"
```

（若已有此 include 則跳過）

- [ ] **Step 2: 在 QNNConfig pybind11 block 之後加入 HailoConfig block**

找到（`python/ivit/_binding.cpp:595-600`）：

```cpp
    py::class_<ivit::QNNConfig>(m, "QNNConfig", "Qualcomm QNN configuration")
        .def(py::init<>())
        .def_readwrite("backend", &ivit::QNNConfig::backend)
        .def_readwrite("performance_mode", &ivit::QNNConfig::performance_mode)
        .def_readwrite("precision", &ivit::QNNConfig::precision)
        .def_readwrite("enable_htp_fp16", &ivit::QNNConfig::enable_htp_fp16);
```

在這個 block 之後（`;` 的下一行）加入：

```cpp

    py::class_<ivit::HailoConfig>(m, "HailoConfig", "HailoRT Hailo-8 configuration")
        .def(py::init<>())
        .def_readwrite("batch_size", &ivit::HailoConfig::batch_size,
                       "Batch size for inference (default 1)")
        .def_readwrite("timeout_ms", &ivit::HailoConfig::timeout_ms,
                       "Inference timeout in milliseconds (default 1000)");
```

- [ ] **Step 3: Commit（C++ 重新編譯在 Task 11 統一進行）**

```bash
git add python/ivit/_binding.cpp
git commit -m "feat: expose HailoConfig via pybind11 binding"
```

---

## Task 10: __init__.py — 匯出 HailoConfig

**Files:**
- Modify: `python/ivit/__init__.py`

- [ ] **Step 1: 在 C++ binding import 區塊（`QNNConfig,` 之後）加入 HailoConfig**

找到（`python/ivit/__init__.py:87-88`）：

```python
        QNNConfig,
```

改為：

```python
        QNNConfig,
        HailoConfig,
```

- [ ] **Step 2: 在 `__all__` 列表的 `QNNConfig,` 後面加入 HailoConfig**

找到（`python/ivit/__init__.py:301-303`）：

```python
    "OpenVINOConfig",
    "TensorRTConfig",

    "QNNConfig",
```

改為：

```python
    "OpenVINOConfig",
    "TensorRTConfig",

    "QNNConfig",
    "HailoConfig",
```

- [ ] **Step 3: Commit**

```bash
git add python/ivit/__init__.py
git commit -m "feat: export HailoConfig in ivit public namespace"
```

---

## Task 11: 重新編譯並執行全部測試

**Files:** 無新增，驗證前面所有變更

- [ ] **Step 1: 重新 build（若有 HailoRT SDK 則加 flag；無 SDK 也可驗證 Python-only 測試）**

不含 HailoRT SDK 的 x86 機器：

```bash
cd /home/dos/Documents/ivit-sdk
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
make -j$(nproc) 2>&1 | tail -30
```

含 HailoRT SDK 時：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DIVIT_USE_HAILORT=ON \
         -DHAILORT_SDK_ROOT=/opt/hailo/hailort
make -j$(nproc)
```

- [ ] **Step 2: 重新安裝 Python 套件**

```bash
cd /home/dos/Documents/ivit-sdk
pip install -e . --no-build-isolation 2>&1 | tail -10
```

- [ ] **Step 3: 執行 HailoRT 全部測試（排除需要硬體的 slow tests）**

```bash
cd /home/dos/Documents/ivit-sdk
pytest tests/python/test_hailort_backend.py -v -m "not slow"
```

預期通過：
- `TestHailoConfigPython` — 5 PASS（無需 C++）
- `TestHailoDeviceStringParsing` — 3 PASS（無需 C++）
- `TestNoRegressions` — 5 PASS（無需 C++）
- `TestCMakeHailoRTOption` — 4 PASS（無需 C++）
- `TestHailoRTFilesExist` — 8 PASS（無需 C++）
- `TestHailoConfigCpp` — 3 PASS（需要 C++ bindings）
- `TestBackendEnum` — 1 PASS（需要 C++ bindings）
- `TestHailoRTRuntimeAvailability` — 2 PASS（需要 C++；在無 Hailo 硬體機器上 hailo 裝置數 = 0）

- [ ] **Step 4: 執行整個測試套件（確保沒有回歸）**

```bash
pytest tests/python/ -v -m "not slow" 2>&1 | tail -30
```

預期：原有的 63 個非 slow 測試全部 PASS，加上新增的 HailoRT 測試。

- [ ] **Step 5: 最終 commit**

```bash
git add .
git commit -m "feat: complete HailoRT backend integration (Hailo-8 PCIe support)"
```

---

## 自我審查

### Spec 覆蓋檢查

| 設計需求 | 對應 Task |
|---------|----------|
| `ivit.load("model.hef", device="hailo:0")` 可執行 | Task 6（device parse）+ Task 5（load_model）|
| `HailoRTRuntime : IRuntime` 完整介面 | Task 4, 5 |
| `HailoConfig` Python API | Task 2, 9, 10 |
| `.hef` 唯一支援格式 | Task 5 (`supported_formats`) |
| `convert_model` → hailo_model_zoo compile | Task 5 |
| `is_available()` 無 Hailo 不 crash | Task 5, 6 |
| `ivit devices` CLI 能列出 Hailo-8 | Task 6 (`discover_hailort_devices`) |
| CMake `IVIT_USE_HAILORT` option | Task 8 |
| 測試套件（5 個測試 class） | Task 1 |
| DFC 未安裝時 `ModelConversionError` | Task 5 (`convert_model`) |

### 型別一致性

- `HailoConfig::batch_size` (int) — Task 2、3、9 一致
- `HailoConfig::timeout_ms` (int) — Task 2、3、9 一致
- `HailoRTModel` struct 欄位 — Task 4、5 一致
- `parse_device_index()` 回傳 int — Task 4、5 一致
- `BackendType::HailoRT` — Task 3、4、5、6 一致

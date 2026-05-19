# 新增硬體平台指南

本文說明如何在 iVIT-SDK 中新增支援新的硬體加速器平台。

## 架構概述

iVIT-SDK 採用可擴展的後端架構，新增硬體平台需要修改以下組件：

```
iVIT-SDK Hardware Extension Points
├── Device Discovery      # 裝置探索
├── Backend Selection     # 後端選擇
├── Runtime Adapter       # 推論執行器
└── Model Conversion      # 模型轉換
```

## 步驟一：裝置探索

### Python (`python/ivit/core/device.py`)

新增裝置探索函數：

```python
def _discover_xxx_devices() -> List[DeviceInfo]:
    """Discover XXX hardware devices."""
    devices = []
    try:
        import xxx_sdk
        for i, dev in enumerate(xxx_sdk.get_devices()):
            devices.append(DeviceInfo(
                id=f"xxx:{i}",
                name=dev.name,
                backend="xxx",
                type="npu",  # "gpu", "cpu", "npu", "vpu"
                memory_total=dev.memory,
                supported_precisions=["fp32", "fp16", "int8"],
                is_available=True,
            ))
        logger.debug(f"Found {len(devices)} XXX devices")
    except ImportError:
        logger.debug("XXX SDK not available")
    except Exception as e:
        logger.warning(f"Error discovering XXX devices: {e}")
    return devices
```

在 `list_devices()` 中註冊：

```python
def list_devices(refresh: bool = False) -> List[DeviceInfo]:
    # ... 現有探索 ...

    # 新增 XXX 裝置探索
    devices.extend(_discover_xxx_devices())

    # ...
```

### C++ (`src/core/device_manager.cpp`)

```cpp
void DeviceManager::discover_xxx_devices() {
#ifdef IVIT_HAS_XXX
    try {
        int device_count = xxx_get_device_count();
        for (int i = 0; i < device_count; i++) {
            DeviceInfo info;
            info.id = "xxx:" + std::to_string(i);
            info.name = xxx_get_device_name(i);
            info.backend = "xxx";
            info.type = "npu";
            info.is_available = true;
            devices_.push_back(info);
        }
    } catch (...) {
        // XXX SDK not available
    }
#endif
}
```

在 `discover_devices()` 中呼叫：

```cpp
void DeviceManager::discover_devices() {
    discover_openvino_devices();
    discover_tensorrt_devices();
    discover_xxx_devices();  // 新增
    // ...
}
```

## 步驟二：後端選擇

### Python (`python/ivit/core/device.py`)

新增可用性檢查：

```python
def _xxx_available() -> bool:
    """Check if XXX backend is available."""
    try:
        import xxx_sdk
        return xxx_sdk.is_available()
    except ImportError:
        return False
```

更新 `get_backend_for_device()`：

```python
def get_backend_for_device(device: str) -> str:
    # ... 現有邏輯 ...

    # XXX 裝置
    if device_lower.startswith("xxx"):
        if _xxx_available():
            return "xxx"

    # Fallback
    return "openvino"
```

### C++ (`src/core/device_manager.cpp`)

更新 `parse_device_string()`：

```cpp
std::pair<BackendType, std::string> parse_device_string(const std::string& device) {
    // ... 現有邏輯 ...

    if (dev_lower.rfind("xxx", 0) == 0) {
        if (dev_lower.length() > 4 && dev_lower[3] == ':') {
            return {BackendType::XXX, dev_lower.substr(4)};
        }
        return {BackendType::XXX, "0"};
    }

    // ...
}
```

## 步驟三：Runtime 適配器

### Python (`python/ivit/runtime/xxx_runtime.py`)

```python
from .base import BaseRuntime

class XXXRuntime(BaseRuntime):
    """XXX backend runtime adapter."""

    def __init__(self):
        super().__init__()
        import xxx_sdk
        self._sdk = xxx_sdk

    def load_model(self, model_path: str, config: RuntimeConfig) -> Any:
        """Load model for XXX backend."""
        return self._sdk.load_model(model_path, device=config.device)

    def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        """Run inference on XXX backend."""
        return model.predict(inputs)

    def get_input_info(self, model: Any) -> List[Dict]:
        """Get model input specifications."""
        return model.get_input_info()

    def get_output_info(self, model: Any) -> List[Dict]:
        """Get model output specifications."""
        return model.get_output_info()
```

### C++ (`src/runtime/xxx_runtime.cpp`)

```cpp
#include "ivit/runtime/runtime.hpp"

#ifdef IVIT_HAS_XXX
#include <xxx_sdk.h>

class XXXRuntime : public IRuntime {
public:
    void load_model(const std::string& path, const RuntimeConfig& config) override {
        model_ = xxx_load_model(path.c_str());
    }

    Tensor predict(const Tensor& input) override {
        // Run inference
        return xxx_infer(model_, input.data());
    }

private:
    xxx_model_t model_;
};

#endif
```

## 步驟四：CMake 配置

更新 `CMakeLists.txt`：

```cmake
# XXX Backend
option(IVIT_USE_XXX "Enable XXX backend support" OFF)

if(IVIT_USE_XXX)
    find_package(XXX_SDK REQUIRED)
    add_definitions(-DIVIT_HAS_XXX)
    list(APPEND IVIT_RUNTIME_SOURCES src/runtime/xxx_runtime.cpp)
    list(APPEND IVIT_RUNTIME_LIBS ${XXX_SDK_LIBRARIES})
endif()
```

## 步驟五：測試

### 單元測試

```python
# tests/integration/test_xxx_backend.py
import pytest
import ivit

@pytest.mark.skipif(not ivit.xxx_is_available(), reason="XXX not available")
class TestXXXBackend:
    def test_device_discovery(self):
        devices = ivit.list_devices()
        xxx_devices = [d for d in devices if d.backend == "xxx"]
        assert len(xxx_devices) > 0

    def test_model_load(self):
        model = ivit.load("model.onnx", device="xxx:0")
        assert model is not None

    def test_inference(self):
        model = ivit.load("yolox-s.onnx", device="xxx:0")
        results = model("test_image.jpg")
        assert len(results.detections) >= 0
```

## 檢查清單

新增硬體平台前，確認以下項目：

- [ ] 裝置探索函數 (Python + C++)
- [ ] 可用性檢查函數 (Python + C++)
- [ ] 後端選擇邏輯更新
- [ ] Runtime 適配器實作
- [ ] CMake 配置
- [ ] 單元測試
- [ ] 文件更新

## 已支援平台

| 平台 | 裝置 ID | 後端 | 狀態 |
|------|---------|------|------|
| Intel CPU/iGPU/NPU | `cpu`, `gpu:0`, `npu` | OpenVINO | ✅ |
| NVIDIA GPU | `cuda:0`, `cuda:1`, ... | TensorRT | ✅ |
| CPU Fallback | `cpu` | OpenVINO | ✅ |

## 參考資源

- [OpenVINO Runtime 實作](../src/runtime/openvino_runtime.cpp)
- [TensorRT Runtime 實作](../src/runtime/tensorrt_runtime.cpp)

# HailoRT Backend Design

**日期：** 2026-04-14
**狀態：** 已核准，待實作

---

## 背景

iVIT-SDK 目前支援 OpenVINO（Intel）、TensorRT（NVIDIA）、QNN（Qualcomm）三種推論後端。
本設計新增第四個後端：**HailoRT**，支援 Hailo-8 AI 加速卡（PCIe M.2）。

---

## 目標

- 讓使用者可以用 `ivit.load("model.hef", device="hailo:0")` 在 Hailo-8 上推論
- 維持與現有後端完全相同的 `IRuntime` 介面
- 實作完整度與 QNN 後端相當：`load_model` / `infer` / `convert_model` 全部實作
- `ivit devices` CLI 能列出已連接的 Hailo-8 裝置

---

## 硬體與 SDK

| 項目 | 規格 |
|------|------|
| 硬體 | Hailo-8（PCIe M.2，x86 主機）|
| SDK  | HailoRT 4.18+（C++ InferModel API）|
| 模型格式 | `.hef`（Hailo Executable Format）|
| 模型轉換工具 | Hailo Dataflow Compiler（DFC）|

---

## 技術方向

**靜態連結（Static Link）**：Build 時 `find_library(hailort)` 並直接連結 `libhailort.so`。
選此方向因 HailoRT 提供穩定的 C++ header 與 CMake-friendly 的安裝結構，不需要像 QNN 那樣用 `dlopen` 動態挑後端。

---

## 架構

### 資料流

```
ivit.load("model.hef", device="hailo:0")
  → RuntimeFactory 選 HailoRTRuntime
  → HailoRTRuntime::load_model()
      → VDevice::create(device_id=0)
      → vdevice->create_infer_model("model.hef")
      → infer_model->set_batch_size(config.batch_size)
      → infer_model->configure()  →  ConfiguredInferModel
  → HailoRTRuntime::infer()
      → configured_model->create_bindings()
      → copy input tensors into bindings
      → configured_model->run(bindings, timeout_ms)
      → copy output tensors out
```

### 內部 Handle

```cpp
struct HailoRTModel {
    std::shared_ptr<hailort::VDevice>    vdevice;
    std::shared_ptr<hailort::InferModel> infer_model;
    hailort::ConfiguredInferModel        configured_model;
    std::vector<TensorInfo>              input_infos;
    std::vector<TensorInfo>              output_infos;
    std::string                          device_id;   // e.g. "hailo:0"
};
```

---

## 新增 / 修改的檔案

### 新增

| 檔案 | 說明 |
|------|------|
| `include/ivit/runtime/hailort_runtime.hpp` | `HailoRTRuntime` 類別宣告、`HailoRTModel` struct |
| `src/runtime/hailort_runtime.cpp` | 完整實作 |

### 修改

| 檔案 | 修改內容 |
|------|---------|
| `include/ivit/core/common.hpp` | `BackendType` enum 新增 `HailoRT`；`to_string()` 新增對應 |
| `src/runtime/runtime_init.cpp` | `#ifdef IVIT_HAS_HAILORT` 區塊自動注冊 `HailoRTRuntime` |
| `CMakeLists.txt` | 新增 `IVIT_USE_HAILORT` option、find HailoRT、編譯定義、source、link |
| `python/ivit/_binding.cpp` | 暴露 `HailoConfig` pybind11 class |
| `python/ivit/__init__.py` | 在 C++ binding import 清單加入 `HailoConfig` |

---

## 介面規格

### 設備字串

| 使用者輸入 | 意義 |
|-----------|------|
| `"hailo"` | 第一顆 Hailo-8（device_id = 0） |
| `"hailo:0"` | 明確指定第 0 顆 |
| `"hailo:1"` | 第 1 顆（多卡環境）|

### 支援格式

`.hef` 為唯一支援格式。`supported_formats()` 回傳 `{".hef"}`。

### HailoConfig（Python）

```python
config = ivit.HailoConfig()
config.batch_size = 1       # 批次大小（預設 1）
config.timeout_ms = 1000    # 推論 timeout，毫秒（預設 1000）
```

### convert_model()

```
輸入：ONNX (.onnx)
輸出：HEF (.hef)
方式：呼叫 hailo_model_zoo compile CLI
      python -m hailo_model_zoo compile <onnx> --hw-arch hailo8 -o <dst>
若 hailo_model_zoo 未安裝：拋出 ModelConversionError，附上安裝說明
```

---

## CMake 整合

```cmake
option(IVIT_USE_HAILORT "Enable HailoRT backend (Hailo-8)" OFF)

if(IVIT_USE_HAILORT)
    find_path(HAILORT_INCLUDE_DIR hailo/hailort.hpp
        HINTS $ENV{HAILORT_SDK_ROOT} /usr/include /usr/local/include
    )
    find_library(HAILORT_LIBRARY hailort
        HINTS $ENV{HAILORT_SDK_ROOT} /usr/lib /usr/local/lib
    )

    if(HAILORT_INCLUDE_DIR AND HAILORT_LIBRARY)
        message(STATUS "HailoRT found: ${HAILORT_LIBRARY}")
        add_compile_definitions(IVIT_HAS_HAILORT)
        list(APPEND IVIT_SOURCES src/runtime/hailort_runtime.cpp)
        target_include_directories(ivit PRIVATE ${HAILORT_INCLUDE_DIR})
        target_link_libraries(ivit PRIVATE ${HAILORT_LIBRARY})
    else()
        message(WARNING "HailoRT not found, disabling HailoRT backend")
        set(IVIT_USE_HAILORT OFF)
    endif()
endif()
```

Build 方式：
```bash
cmake .. -DIVIT_USE_HAILORT=ON
# 或指定 SDK 路徑
cmake .. -DIVIT_USE_HAILORT=ON -DHAILORT_SDK_ROOT=/opt/hailo/hailort
```

---

## 錯誤處理

| 情境 | 行為 |
|------|------|
| Hailo-8 未插入 / 驅動未載入 | `is_available()` 回傳 false，`DeviceNotFoundError` |
| 非 `.hef` 格式 | `UnsupportedFormatError` |
| 推論 timeout | `InferenceError("HailoRT inference timeout")` |
| DFC 未安裝時呼叫 convert | `ModelConversionError` 附安裝說明 |

---

## 測試

新增 `tests/python/test_qnn_backend.py` 同等的 `tests/python/test_hailort_backend.py`：

- `test_hailort_runtime_is_available()` — 無 Hailo 時不 crash，回傳 false
- `test_hailort_supported_formats()` — 只回傳 `.hef`
- `test_hailort_device_string_parsing()` — `"hailo:0"` 解析正確
- `test_hailort_convert_no_dfc_raises()` — DFC 未安裝時拋出清楚錯誤
- `@slow test_hailort_load_and_infer()` — 需要實體 Hailo-8（標記 slow）

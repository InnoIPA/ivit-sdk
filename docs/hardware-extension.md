# 新增推論後端

推論與前後處理以 C++ 核心為準；Python bindings 與 C ABI 共用核心，不另建純 Python runtime。

## 實作位置

1. 在 [common.hpp](../include/ivit/core/common.hpp) 增加 `BackendType`，保留既有 enum 數值。
2. 在 `include/ivit/runtime/` 與 `src/runtime/` 實作 [IRuntime](../include/ivit/runtime/runtime.hpp)。必須提供可用性、裝置資訊、模型格式、載入／卸載、tensor 資訊及推論；`get_version()` 應回報實際 runtime 版本供快取使用。
3. 在 [runtime_init.cpp](../src/runtime/runtime_init.cpp) 註冊 factory，並在 [device_manager.cpp](../src/core/device_manager.cpp) 加入明確的裝置對應。
4. 在 [CMakeLists.txt](../CMakeLists.txt) 宣告可選後端、SDK headers／libraries 與來源；啟用卻缺依賴時明確失敗。更新 [ivitConfig.cmake.in](../cmake/ivitConfig.cmake.in) 的下游依賴。

## 契約與測試

後端只處理 tensors。影像 normalization、layout、resize、labels 與 decoder 由 [processing.cpp](../src/vision/processing.cpp) 管理。後端不支援的設定須拒絕，避免默默降級。

模型 handle 必須在失敗、正常卸載與程序退出時釋放；回傳 tensor 不得引用失效的 vendor buffers。驗證動態 shape、錯誤後重用、快取失效、併發、反覆載入／卸載及退出。

新後端需完成 SDK 編譯、實機推論、Python／C++／C ABI 與外部 CMake consumer 驗收，再更新 [支援範圍](sdk-manual.zh-TW.md)。目前支援的後端只有 OpenVINO 與 TensorRT。

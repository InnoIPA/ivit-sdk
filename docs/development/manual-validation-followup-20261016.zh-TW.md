# Checklist 修正後複驗（2026-10-16）

依據：[前次完整實跑結果](manual-validation-results-20261016.zh-TW.md)。本次重新驗證 C++ 裝置測試、.NET 前置說明及分割匯出修正，並補測 GPU 不可見的情境；沒有修改 SDK 實作來取得通過結果。

## 結論

前次第 9 步「純 TensorRT 仍要求 CPU」的測試問題，以及第 12 步缺少 .NET PATH 說明，均已修正並通過正常 GPU 環境驗收。分割匯出的 training-mode warning 也已消除。

**仍有一個本次補測確認的 SDK 問題：TensorRT 11.4 在沒有可見 GPU 時，runtime 初始化失敗後會崩潰。** 因此不能宣稱「沒有可用 GPU 時正常回傳空裝置列表」已通過；此現象不表示是本次修正才引入。

## 無 GPU 崩潰的修正（複驗後）

根因：沒有可見 GPU 時，`createInferRuntime` 仍回傳物件，`RuntimeAutoRegister` 發現 `is_available()` 為 false 而丟棄它，TensorRT 11.4 銷毀該 `IRuntime` 時 SIGSEGV。

修正：`TensorRTRuntime::initialize()`（`src/runtime/tensorrt_runtime.cpp`）在建立 runtime 之前先檢查 CUDA 裝置，沒有就丟 `IVITError`，由 `runtime_init.cpp` 既有的 `catch` 略過註冊。新增 CTest `test_inference_no_gpu`（以 `CUDA_VISIBLE_DEVICES=` 執行裝置測試）。

驗證（TensorRT-only build）：還原修正時該測試 SegFault，套用後通過；`CUDA_VISIBLE_DEVICES=` 下 `simple_inference devices` 回傳 0 個裝置、退出碼 0；正常 GPU 的 CTest 4/4 通過、裝置列舉仍為 `cuda:0`。OpenVINO＋TensorRT 的 `py-trt` 以清單路徑 B 重新安裝後：`CUDA_VISIBLE_DEVICES=` 下 `import ivit; list_devices()` 不再崩潰，回傳 `cpu`、`gpu:0`（OpenVINO）；有 GPU 時仍有 `cuda:0`（TensorRT）。Python 全套 366 passed／3 skipped，Flake8 與 `git diff --check` 通過。日誌：`/tmp/ivit-checklist-20261015/log/step04-reinstall.log`。

## 已通過的修正

| 項目 | 驗證方式 | 結果 |
| --- | --- | --- |
| OpenVINO C++ | 重新 build、CTest、單獨執行 DeviceManagerListDevices | CTest 3/3；指定 GoogleTest 1/1 通過 |
| TensorRT-only C++ | 重新 build、CTest、單獨執行 DeviceManagerListDevices | CTest 3/3；指定 GoogleTest 1/1 通過 |
| 測試條件編譯確實生效 | 檢查 test_inference 的編譯 flags | OpenVINO 為 `IVIT_HAS_OPENVINO`；TensorRT-only 為 `IVIT_HAS_TENSORRT`，不是落入無後端的 skip |
| .NET PATH 說明 | 按新增說明將實際 SDK 目錄加入 PATH，再使用字面 `dotnet` 命令 | 8.0.425；build 零警告、零錯誤 |
| C ABI／C# | 分別使用 OpenVINO／TensorRT native prefix | 兩後端均通過；C# 顯示 lifecycle/buffer checks passed |
| 分割 wrapper eval | 在新的模型 cache 首次匯出 DeepLabV3-MobileNetV3 | 成功，沒有前次 training-mode warning |
| 新匯出模型的推論 | CPU 載入新 ONNX，與前次 CPU class mask 比對 | 1080×810；class mask 100% 一致 |
| Python 回歸 | 兩個既有隔離 venv 重跑全套，含 C ABI 設定 | OpenVINO 359 passed／10 skipped；TensorRT 366 passed／3 skipped |
| 靜態檢查 | Flake8 指定規則與 git diff --check | 通過 |

Python 仍有既有 ONNX／TorchScript 等 warnings，分別 24／30 個；本次只確認分割首次匯出的 training-mode warning 消失，沒有宣稱全部 warnings 清除。

## 未通過：沒有可見 GPU 的初始化

在本機 TensorRT 11.4／CUDA 12.9、RTX 2070 SUPER 環境，以 `CUDA_VISIBLE_DEVICES=` 隱藏所有 CUDA GPU。這是無可見 GPU 的重現方式，不等同已驗證所有無 GPU／無驅動主機。

以下三種操作都以 **139（SIGSEGV）** 結束：

```bash
CUDA_VISIBLE_DEVICES= /tmp/ivit-checklist-20261015/cpp-trt/bin/test_inference \
  --gtest_filter=InferenceTest.DeviceManagerListDevices

CUDA_VISIBLE_DEVICES= /tmp/ivit-checklist-20261015/cpp-trt/bin/simple_inference devices

CUDA_VISIBLE_DEVICES= /tmp/ivit-checklist-20261015/py-trt/bin/python3 \
  -c 'import ivit; print(ivit.list_devices())'
```

重現時另有相依函式庫的 LD_LIBRARY_PATH，完整設定見驗證腳本。log 先出現 CUDA error 100（no CUDA-capable device），接著崩潰。Python 這份建置同時有 OpenVINO，仍在 import 階段崩潰，尚未能列出可用 CPU。

GDB 顯示崩潰位於：

```text
std::_Sp_counted_deleter<...TensorRTRuntime::initialize()...>::_M_dispose()
ivit::TensorRTRuntime::~TensorRTRuntime()
ivit::RuntimeAutoRegister::RuntimeAutoRegister()
call_init / _dl_init
```

所以不是新增 GoogleTest 斷言失敗，而是在共享函式庫的自動註冊、進入測試 main 之前就發生 SIGSEGV。GDB 批次命令自身退出碼是 0，不代表被除錯程式成功；應以記錄中的 SIGSEGV 與上述直接重現退出碼判讀。

### 建議修正方向與驗收條件

- 檢查 `src/runtime/tensorrt_runtime.cpp` 的 `TensorRTRuntime::initialize()`：runtime 建立失敗時應安全處理，不能在清理時崩潰。
- `src/runtime/runtime_init.cpp` 目前先建構 TensorRT runtime，之後才呼叫 `is_available()`。應考慮在建立 runtime 前先確認 CUDA 裝置可用，或將初始化延後至真正需要時。
- 對失敗後 shared_ptr／deleter 的資源清理路徑增加回歸覆蓋；不能僅靠 `catch (...)` 處理 SIGSEGV。
- TensorRT-only：沒有可見 GPU 時，`simple_inference devices` 與裝置測試應安全完成，回傳符合契約的空列表／明確錯誤。
- OpenVINO＋TensorRT：沒有可見 GPU 時，Python import 不應崩潰，仍應可用 OpenVINO CPU。
- 保留正常 GPU 的 CTest、裝置列舉與推論驗證，避免修正無 GPU 路徑時破壞正常流程。

## 範圍與證據

本次為增量複驗，沒有重做系統安裝、完整三模型 FP32／FP16 矩陣或 NVIDIA 自動安裝。正常推論完整矩陣見前次報告；本次另外驗證 C／C# 小型分類器與新匯出的分割模型。

產物：`/tmp/ivit-final-recheck-20261016/`。

- `ctest-ov.log`、`ctest-trt.log`、`device-test-*.log`：正常測試及無 GPU 補測。
- `flags-ov.log`、`flags-trt.log`：編譯條件證據。
- `dotnet-*.log`、`c-*.log`：兩後端整合。
- `segmentation-export.log`、`segmentation-reload.log`：分割匯出與結果比對。
- `pytest-ov.xml`、`pytest-trt.xml`：全套回歸。
- `no-gpu-backtrace.log`、`no-gpu-cpp-devices.log`、`no-gpu-python.log`：崩潰重現與堆疊。
- `results.tsv`：個別命令退出碼。

腳本：`/tmp/ivit-final-recheck-20261016.sh`。驗證使用含未提交修改的工作樹，尚非固定 commit 的發行認證。

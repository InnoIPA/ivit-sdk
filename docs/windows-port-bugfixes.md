# iVIT-SDK Windows 移植 — Bug 修正紀錄

本文件整理將 iVIT-SDK 在 Windows（MSVC / VS 2026 / Python 3.10）從「完全無法建置」修到「CPU + GPU 全功能可跑」過程中，發現並修正的所有問題。

- **驗證平台**：Windows 11、Intel Core i5-13500（+ iGPU）、NVIDIA RTX 3060、OpenVINO 2026.2.1、TensorRT 10.16（C++）、CUDA 12.6
- **分支**：`window_v2`
- **每個 bug 都含**：症狀 → 技術原因 → 技術白話 → 修法 → 對應 commit

> **嚴重度圖例**：🟥 阻斷（不修無法建置／匯入） ｜ 🟧 崩潰（正確建置下呼叫即崩） ｜ 🟨 靜默失敗（不崩但結果錯） ｜ 🟩 體驗（錯誤訊息／可攜性）

---

## 目錄

| # | 類別 | 問題 | 嚴重度 | Commit |
|---|------|------|:---:|--------|
| 1 | 建置 | pyproject 用了已移除的 `cmake.minimum-version` | 🟥 | `build:` |
| 2 | 建置 | C++17 卻用了 C++20 語法（designated initializers） | 🟥 | `build:` |
| 3 | 移植 | 無條件 `#include <unistd.h>` | 🟥 | `fix(windows):` |
| 4 | 移植 | POSIX 檔案描述子/環境變數 API（dup/setenv…） | 🟥 | `fix(windows):` |
| 5 | 移植 | `ssize_t` 型別在 MSVC 不存在 | 🟥 | `fix(windows):` |
| 6 | 建置 | 動態庫沒有匯出符號 → 不產生 `ivit.lib` | 🟥 | `build:` |
| 7 | 移植 | Python 載入原生 DLL 的邏輯是 Linux 專屬 | 🟥 | `fix(python):` |
| 8 | 建置 | CMake 找不到版本化的 TensorRT lib（`nvinfer_10`） | 🟥 | `build:` |
| 9 | 執行期 | DLL 載入期初始化 CUDA → loader-lock 死結 | 🟥 | `fix(runtime):` |
| 10 | API | CLI `devices` 用了不存在的 `DeviceInfo.available` | 🟧 | `fix(cli):` `fix(python):` |
| 11 | API | `Model.input_info[0]["shape"]` 把物件當 dict | 🟧 | `fix(cli):` |
| 12 | API | `Results.show()` 不存在 | 🟧 | `fix(cli):` |
| 13 | API | `predict --save` 把 JSON 寫進 `.jpg` | 🟨 | `fix(cli):` |
| 14 | API | video 串流 `model.stream()` 參數/回傳型別錯 | 🟧 | `fix(cli):` |
| 15 | API | `serve` 用了不存在的 `predict(conf=)` 參數 | 🟧 | `fix(cli):` |
| 16 | API | `load(task=)` → `LoadConfig` 無 `task` 欄位 | 🟧 | `fix(api):` |
| 17 | API | `devices.best()` C++ 評分路徑永遠失效 | 🟨 | `fix(api):` |
| 18 | Zoo | `download(format=openvino/tensorrt)` 靜默回傳假路徑 | 🟨 | `fix(zoo):` |
| 19 | Zoo | 未處理的 source（mambavision）靜默回傳假路徑 | 🟨 | `fix(zoo):` |
| 20 | 腳本 | `see_architecture.py` 第 1 行就 IndentationError | 🟧 | `fix(scripts):` |
| 21 | 腳本 | 多處硬編 `/home/dos/...` Linux 路徑 | 🟩 | `fix(scripts):` |
| 22 | 範例 | `si_quickstart.py` 預設圖片路徑不存在 | 🟩 | `fix(examples):` |
| 23 | 範例 | onnxruntime fallback 缺套件時 raw crash | 🟩 | `fix(vision):` |
| 24 | 推論 | 通用 `Model` 對偵測任務永遠 `TaskType::Unknown` → 後處理整段被跳過 | 🟨 | `fix(api):` |
| 25 | 推論 | 通用 `InferenceModel` 前處理/後處理不認得 YOLOX，偵測永遠 0 個框 | 🟨 | `fix(core):` |
| 26 | API | `model.task` 讀取即崩：`TaskType` enum 沒註冊到 pybind11 | 🟧 | 未修復 |
| 27 | API | `ivit devices` 純 Python 降級模式崩潰：fallback stub 用舊屬性名 `available` | 🟧 | 未修復 |

---

# 第一部分：建置 / 跨平台移植（讓它能編、能匯入）

這些問題讓 SDK 在 Windows 上**根本無法建置或無法 `import`**。共同根因：這份程式碼是為 GCC/Linux 寫的，許多「在 Linux 剛好能過」的寫法，到 MSVC/Windows 就不成立。

## 1. pyproject 用了已移除的 `cmake.minimum-version` 🟥

- **位置**：`pyproject.toml` `[tool.scikit-build]`
- **症狀**：`pip install -e .` 一開始就報錯 `Use cmake.version instead of cmake.minimum-version with scikit-build-core >= 0.8`。
- **技術原因**：建置後端 `scikit-build-core` 在 0.8 版把設定鍵 `cmake.minimum-version` 改名為 `cmake.version`。
- **技術白話**：PEP 517 的 build isolation 每次建置都會臨時抓「最新版」的 scikit-build-core（實測抓到 1.0.3）來跑，不會用你環境裡的舊版。1.0.3 已經不認得 `cmake.minimum-version` 這個鍵，於是在「解析 pyproject 設定」這個最早的階段就直接失敗，連 CMake 都還沒開始跑。
- **修法**：`cmake.minimum-version = "3.18"` → `cmake.version = ">=3.18"`。

## 2. 宣告 C++17，卻用了 C++20 語法 🟥

- **位置**：`CMakeLists.txt`（`CMAKE_CXX_STANDARD 17`）＋ `include/ivit/core/tensor.hpp:64`
- **症狀**：`error C7555: 使用指定的初始設定式需要 '/std:c++20'`。
- **技術原因**：程式用了 **designated initializers**（`Tensor{.name=..., .shape=...}`），這是 C++20 才正式納入標準的語法。
- **技術白話**：designated initializers（用 `.欄位=值` 初始化 struct）在 C++20 才進標準。GCC 很早就把它當延伸功能支援，所以即使把標準設成 C++17 也照樣能編過；MSVC 嚴格照標準走，只有在 `/std:c++20` 模式才接受。也就是說程式實際上依賴了 C++20 的功能，但建置設定只宣告 C++17——這個矛盾在 GCC 上被容忍、在 MSVC 上就直接報錯。
- **修法**：`CMAKE_CXX_STANDARD 17` → `20`。

## 3. 無條件 `#include <unistd.h>` 🟥

- **位置**：`src/core/device_manager.cpp:15`、`src/runtime/openvino_runtime.cpp:13`
- **症狀**：`error C1083: 無法開啟包含檔案: 'unistd.h'`。
- **技術原因**：`unistd.h` 是 POSIX 專屬標頭，Windows/MSVC 沒有這個檔。
- **技術白話**：`unistd.h` 只存在於 POSIX（Unix/Linux）系統，是 POSIX 系統呼叫的宣告集。MSVC 的標準函式庫裡沒有這個標頭檔，`#include` 到它時在前置處理階段就找不到檔案而中斷。
- **修法**：改用 `#ifndef _WIN32` 條件包起來（配合 #4 的統一 shim）。

## 4. POSIX 檔案描述子 / 環境變數 API 🟥

- **位置**：`device_manager.cpp`、`openvino_runtime.cpp`（`dup` / `dup2` / `open` / `close` / `STDERR_FILENO` / `setenv`）
- **症狀**：`error C3861: 'dup': 找不到識別項`、`'STDERR_FILENO' 未宣告`、`'setenv': 找不到`…
- **技術原因**：這些是 POSIX 標準函式，MSVC 提供的是加底線的對應版：`_dup`/`_dup2`/`_open`/`_close`/`_putenv_s`，空裝置路徑 `/dev/null` 對應 `"NUL"`。
- **技術白話**：程式用這些函式做「暫時把 stderr 導向空裝置、蓋掉 OpenVINO/OpenCL 列舉裝置時印的雜訊，做完再還原」。功能兩個平台都有，但 MSVC 把這些函式與常數放在**加底線的名稱**下（`_dup`、`_putenv_s` 等），而且「空裝置」的路徑字串也不同（Linux 是 `/dev/null`，Windows 是 `"NUL"`）。程式直接寫 POSIX 名稱，MSVC 連結不到這些符號。
- **修法**：新增 `include/ivit/core/platform_compat.hpp`，依平台把 `ivit_dup()` 等包裝對應到各自的正確名稱；兩個 .cpp 統一改呼叫包裝函式。

## 5. `ssize_t` 在 MSVC 不存在 🟥

- **位置**：`python/ivit/_binding.cpp:56-57`
- **症狀**：`error C2065: 'ssize_t': 未宣告的識別項`（連帶讓 `numpy_to_mat` 整段編不出來）。
- **技術原因**：`ssize_t`（可為負的 size 型別）是 POSIX 定義的型別，Windows 沒有內建。
- **技術白話**：`ssize_t` 是 POSIX 標頭裡定義的型別（用來表示可為負的大小，例如 numpy 的 stride/shape）。MSVC 的標頭沒有定義它，於是用到 `ssize_t` 的那幾行變成「用到未定義型別」，整個轉換函式編不過。pybind11 內部有可攜的 `py::ssize_t` 可直接替代。
- **修法**：`_binding.cpp` 頂端加 `#ifdef _WIN32 using ssize_t = py::ssize_t; #endif`。

## 6. 動態庫沒有匯出符號 → 不產生 `ivit.lib` 🟥

- **位置**：`CMakeLists.txt`（`add_library(ivit SHARED ...)`）
- **症狀**：C++ 全部編過，最後連結 `_ivit_core.pyd` 時 `LINK : fatal error LNK1181: 無法開啟輸入檔 'ivit.lib'`。
- **技術原因**：Windows 的 DLL 預設不匯出任何符號，只匯出明確標了 `__declspec(dllexport)` 的；這份程式沒有任何匯出標記。
- **技術白話**：兩個平台的符號可見性預設相反。Linux 的共享庫（`.so`）預設把所有非 static 的符號都對外可見；Windows 的 DLL 預設「什麼都不對外」，你得逐一用 `__declspec(dllexport)` 標記要匯出的符號。這份程式一個匯出標記都沒有，於是 MSVC 產出了 `ivit.dll`，卻因為「沒有任何符號要匯出」而**不生成對應的 import library `ivit.lib`**。而 `_ivit_core.pyd` 連結時需要 `ivit.lib` 才能綁定到 `ivit.dll` 的函式，找不到就連結失敗。
- **修法**：對 ivit target 開啟 `WINDOWS_EXPORT_ALL_SYMBOLS ON`，讓 CMake 自動匯出所有符號並產生 `ivit.lib`。

## 7. Python 載入原生 DLL 的邏輯是 Linux 專屬 🟥

- **位置**：`python/ivit/__init__.py`
- **症狀**：build 成功、但 `import ivit` 後 `is_cpp_available()` 是 `False`（靜默退回純 Python），`_CPP_ERROR` 顯示 `DLL load failed while importing _ivit_core: 找不到指定的模組`。
- **技術原因**：原本的載入邏輯只處理 Linux（找 `libivit.so`、靠 ELF RUNPATH 解析相依）。Windows 上 Python 3.8+ 不再用 `PATH` 解析副檔模組的相依 DLL，必須用 `os.add_dll_directory()`。
- **技術白話**：Python 3.8 起，Windows 為了安全性改了 DLL 搜尋規則：載入副檔模組（`.pyd`）時，**不再從 `PATH` 環境變數去找它相依的 DLL**，只會搜尋「有用 `os.add_dll_directory()` 明確註冊過的目錄」。原本的 `__init__.py` 只寫了 Linux 版（找 `.so`、靠 RUNPATH），在 Windows 完全沒做 `add_dll_directory`，所以 `ivit.dll` 與它相依的 OpenVINO/OpenCV/TensorRT DLL 全都搜不到，`import _ivit_core` 因缺 DLL 而失敗；這個失敗被 `try/except` 接住，程式就靜靜退回「純 Python」模式（`is_cpp_available()=False`）。
- **修法**：`__init__.py` 加 Windows 分支，用 `os.add_dll_directory()` 註冊套件的 `bin\` 目錄與目前 PATH 上每個目錄；Linux 維持原邏輯。

## 8. CMake 找不到版本化的 TensorRT lib 🟥（僅 GPU 路徑）

- **位置**：`CMakeLists.txt`（`find_library(TENSORRT_LIBRARY nvinfer ...)`）
- **症狀**：明明系統有 TensorRT C++ 開發包，CMake 卻回報 `TensorRT not found, disabling`。
- **技術原因**：`find_library` 找的是 `nvinfer`（展開成 `nvinfer.lib`），但 Windows 上 TensorRT 10.x 的 import lib 帶主版本號：`nvinfer_10.lib`、`nvonnxparser_10.lib`。
- **技術白話**：`find_library(<var> nvinfer ...)` 會把名稱 `nvinfer` 依平台命名規則展開去找檔案，Windows 上找的是 `nvinfer.lib`。但 TensorRT 10 在 Windows 的 import library 檔名把主版本號嵌進去了，實際檔名是 `nvinfer_10.lib`（Linux 則是 `libnvinfer.so` 這種不含版本號的形式）。名稱對不上就找不到，CMake 的偵測邏輯於是判定「TensorRT 不存在」並自動關掉整個後端。
- **修法**：`find_library(... NAMES nvinfer nvinfer_10 ...)` 同時列兩種名稱，並用 `TENSORRT_ROOT` 指向實際安裝位置。

## 9. DLL 載入期初始化 CUDA → loader-lock 死結 🟥（最棘手）

- **位置**：`src/runtime/runtime_init.cpp`（`static RuntimeAutoRegister g_runtime_auto_register;`）
- **症狀**：開啟 TensorRT 後端重新 build 後，`import ivit` **直接卡死 16 分鐘不動**（CPU 凍結、GPU 閒置），不是變慢而是完全卡住。
- **技術原因**：全域靜態物件 `RuntimeAutoRegister` 的建構子在 DLL 載入期就 `new TensorRTRuntime()` → `createInferRuntime()` + `cudaGetDeviceCount()`，觸發 CUDA/TensorRT 初始化。
- **技術白話**：Windows 在載入任何 DLL 期間（包含執行該 DLL 的 C++ 全域靜態物件建構子、DllMain）會持有一把**行程級的 loader lock**，同一時間只允許一條載入流程。這段程式的全域靜態物件在「DLL 正在載入的當下」就呼叫 CUDA/TensorRT 初始化，而 CUDA 初始化內部會**建立執行緒、動態載入其他 DLL**——這些動作又需要取得同一把 loader lock。結果：持鎖的那條流程在等 CUDA 初始化完成，CUDA 又在等 loader lock 被釋放，兩邊互等，永久卡住。Linux 沒有這種「載入期全域鎖」，所以同一份程式在 Linux 完全正常。通則：**不要在 DLL 載入期（靜態初始化 / DllMain）呼叫任何會建立執行緒或載入 DLL 的重量級函式庫。**
- **修法**：把 runtime 註冊從「載入期」改為「首次使用時」（延遲初始化）。移除載入期的靜態物件，改為 `ensure_runtimes_registered()`，用 `std::call_once` 在第一次「查詢 runtime」時才執行。那時已離開 loader lock、是正常的使用者執行緒，CUDA 可以安全初始化。

---

# 第二部分：執行期 API / CLI Bug（能編、能匯入，但一呼叫就崩）

第一部分修完後 SDK 能建置、匯入、推論。但一份稽核（4 個並行分析 + 動態實跑）發現一批 **Python 層呼叫 C++ 綁定時「名字/簽章對不上」** 的問題。共同根因：C++ 綁定（`_binding.cpp`）改過屬性/方法名或簽章，但 Python 端（CLI、部分 API）沒同步更新——這類問題只要有整合測試就能攔截，目前 CLI 幾乎沒有測試覆蓋是根本原因。

## 10. CLI `devices` 用了不存在的 `DeviceInfo.available` 🟧

- **位置**：`cli.py`（`cmd_devices`：表格與 `--json` 兩條路徑）
- **症狀**：`ivit devices` 與 `ivit devices --json` 皆 `AttributeError: 'DeviceInfo' object has no attribute 'available'. Did you mean: 'is_available'?`
- **技術原因**：綁定屬性名是 `is_available`，Python 寫成 `available`。
- **技術白話**：C++ 綁定把這個布林屬性命名為 `is_available`，但 CLI 存取時打成 `available`。pybind 物件不允許取用未定義的屬性，執行到那行就拋 AttributeError。
- **修法**：兩處 `d.available` → `d.is_available`。

## 11. `Model.input_info[0]["shape"]` 把物件當 dict 用 🟧

- **位置**：`cli.py`（`cmd_benchmark`、`cmd_serve` 的 `/info`）
- **症狀**：`ivit benchmark <model>` → `TypeError: 'ivit._ivit_core.TensorInfo' object is not subscriptable`。
- **技術原因**：`input_info` 回傳 `TensorInfo` 物件的 list，形狀是屬性 `.shape`，非字典鍵。
- **技術白話**：`model.input_info` 回傳的是一串 `TensorInfo` 物件，形狀要用屬性存取 `.shape`。程式用了字典取值語法 `["shape"]`，但 `TensorInfo` 這個 pybind 型別沒有實作 `__getitem__`，不能用中括號索引，於是一取就 TypeError。
- **修法**：`input_info[0]["shape"]` → `input_info[0].shape`。

## 12. `Results.show()` 不存在 🟧

- **位置**：`cli.py`（`cmd_predict`，`--show`）
- **症狀**：`ivit predict <img> --show` → `AttributeError: 'Results' object has no attribute 'show'`。
- **技術原因**：`Results` 只有 `visualize(image)`（回傳畫好標註的影像），沒有 `show()`。
- **技術白話**：綁定的 `Results` 型別沒有 `show()` 這個方法；它提供的是 `visualize(image)`，會回傳一張畫好偵測框的影像陣列。要把結果顯示出來，得拿 `visualize()` 的回傳值自己再用 OpenCV 開視窗。呼叫不存在的 `show()` 就是 AttributeError。
- **修法**：改為 `vis = results.visualize(image); cv2.imshow(...)`。

## 13. `predict --save` 把 JSON 寫進 `.jpg` 🟨

- **位置**：`cli.py`（`cmd_predict`，`--save`）
- **症狀**：不崩，但存出的 `xxx_result.jpg` 打開是 JSON 文字，不是圖。
- **技術原因**：`Results.save(path)` 預設 `format="json"`，程式只換副檔名沒改格式。
- **技術白話**：`Results.save()` 的預設輸出格式是 JSON。CLI 只是把輸出檔名的副檔名設成 `.jpg`，並沒有告訴它要輸出影像，所以實際寫入的是 JSON 內容——副檔名是 `.jpg`、內容卻是文字，兩者不符。
- **修法**：改為 `vis = results.visualize(image); cv2.imwrite(out.jpg, vis)`，真正輸出標註影像。

## 14. video 串流 `model.stream()` 參數與回傳型別都錯 🟧

- **位置**：`cli.py`（`cmd_predict` 的影片/攝影機分支）
- **症狀**：`ivit predict <video>` → `TypeError: stream(): incompatible function arguments`；即使修好參數，`len(results)` / `results.inference_time_ms` 仍會出錯。
- **技術原因**：(a) `stream()` 簽章是 `(source: str, config: InferConfig)`，程式傳了不存在的 `conf=/iou=/show=/save=`，且攝影機來源傳成 int（需要字串）；(b) `stream()` 產出的是 `StreamResult`，實際結果在其巢狀欄位 `.results`。
- **技術白話**：這段是照舊版介面寫的，但函式簽章與回傳結構都改過了。第一，`model.stream()` 現在只收兩個引數：來源字串和一個 `InferConfig` 設定物件；程式卻硬塞了 `conf/iou/show/save` 這些不存在的具名參數（且攝影機把來源當整數傳，但介面要字串）。第二，`stream()` 每次 yield 的是 `StreamResult`，這一幀真正的偵測結果與耗時放在它的巢狀欄位 `.results` 裡；程式卻直接對 `StreamResult` 本身做 `len()` 和取 `.inference_time_ms`，而這兩者在 `StreamResult` 上都不存在。
- **修法**：建立 `InferConfig` 傳給 `stream(str(source), cfg)`；迴圈改用 `stream_result.results` 再取 `len(res.detections)` / `res.inference_time_ms`。

## 15. `serve` 用了不存在的 `predict(conf=)` 參數 🟧

- **位置**：`cli.py`（`cmd_serve` 的 `/predict`）
- **症狀**：`ivit serve` 的推論端點 → `TypeError: predict(): incompatible function arguments`（另外 `serve` 本身還需 fastapi/uvicorn）。
- **技術原因**：`Model.predict` 只接受 `(image, config)`，沒有 `conf=` 具名參數。
- **技術白話**：C++ 的 `Model.predict` 綁定簽章是 `(image, config)`，信心閾值這類設定要包在 `InferConfig` 物件裡傳；並不存在 `conf=` 這個關鍵字參數。直接傳 `conf=` 會因引數不符而 TypeError。
- **修法**：建立 `InferConfig().conf_threshold = conf`，改呼叫 `model.predict(image, cfg)`。

## 16. `load(task=)` → `LoadConfig` 沒有 `task` 欄位 🟧

- **位置**：`python/ivit/__init__.py`（`load`）、`python/ivit/core/model.py`（`load_model`）
- **症狀**：`ivit.load("m.onnx", task="detect")` → `AttributeError: 'LoadConfig' object has no attribute 'task'`（`task=` 正是 docstring 示範的用法）。
- **技術原因**：C++ 的 `LoadConfig` 綁定沒有 `task` 欄位，也沒開 `py::dynamic_attr()`。
- **技術白話**：C++ 的 `LoadConfig` 綁定只註冊了固定幾個欄位，而且沒有開啟 pybind 的動態屬性功能（`py::dynamic_attr`）。在這種情況下，對它設定任何「沒被註冊的屬性」都會被 pybind 擋下並拋 AttributeError。`load()` 裡那行 `config.task = task` 沒有先判斷欄位是否存在，只要呼叫時帶了 `task=` 就會踩到。（下方處理 `kwargs` 的迴圈有用 `hasattr` 保護，唯獨這行漏了。）
- **修法**：改為 `if task and hasattr(config, "task"):`，讓帶 `task=` 不再崩（task 本就由模型推斷或由任務專屬類別處理）。

## 17. `devices.best()` 的 C++ 評分路徑永遠失效 🟨

- **位置**：`python/ivit/devices.py`（`best()`）
- **症狀**：不崩（被 `try/except` 吞掉），但「C++ 裝置評分」這條路永遠拋例外、永遠退回 Python 啟發式——是一段死碼。
- **技術原因**：(a) 呼叫 `get_best_device(strategy=...)`，但參數名是 `priority`；(b) 讀 `result.vendor`/`result.is_discrete`，但 C++ `DeviceInfo` 沒這兩欄位。
- **技術白話**：這段有兩個對不上的地方。其一，它呼叫核心函式時用了關鍵字 `strategy=`，但那個函式的參數名其實是 `priority`，傳未知關鍵字會 TypeError。其二，就算呼叫成功，它接著讀取 `result.vendor` 和 `result.is_discrete`，而 C++ 的 `DeviceInfo` 綁定根本沒有這兩個欄位，一讀就 AttributeError。兩個錯任一個都會拋例外，而整段被 `try/except Exception: pass` 包住，例外被靜靜吞掉、直接跳到 Python 的簡易挑法。結果是這條「用 C++ 評分挑最佳裝置」的路徑從頭到尾沒成功執行過，外部完全察覺不到。
- **修法**：`strategy=` → `priority=`；`result.vendor/is_discrete` 改用 `getattr(..., 預設值)`。C++ 評分路徑現在能真正運作。

---

# 第三部分：靜默失敗（不崩潰，但回傳錯誤結果）

這類最難追：程式不報錯，回傳一個看起來正常的路徑，但那個檔案根本不存在，要到下一步用檔案時才失敗，且錯誤點離真正的原因很遠。

## 18 & 19. `zoo.download` 對做不出來的格式/來源，靜默回傳假路徑 🟨

- **位置**：`python/ivit/zoo/registry.py`（`download()` 結尾的 `return model_path`）
- **症狀**：
  - `download("yolox-nano", format="openvino")` → 回傳 `.../yolox-nano_openvino/yolox-nano.xml`，但**檔案不存在**（所有模型的 openvino/tensorrt 格式皆如此）。
  - `download("mambavision-t")` → 回傳 `.../mambavision-t.onnx`，但**檔案不存在**（mambavision 這個 source 沒有對應的匯出分支）。
- **技術原因**：`download()` 只在「format 是 onnx 且該 source 有分支」時才真的產生檔案；其他情況整段路由不執行，最後仍 `return model_path`。
- **技術白話**：`download()` 內部依「格式 + 模型來源」分派到不同的下載/匯出分支，但這些分支全部只在 `format == "onnx"` 且來源被列入時才會執行。當呼叫方要求 openvino/tensorrt 格式，或來源沒有被任何分支處理（例如 mambavision）時，中間所有分支都跳過，函式結尾卻照樣 `return model_path`——那是流程一開始就先組好的路徑字串，但對應的檔案從沒被建立。呼叫端拿到一個「字串存在、檔案不存在」的結果，沒有任何例外提示。
- **修法**：`return` 前檢查檔案是否存在；不存在就丟明確錯誤——非 onnx 格式丟 `NotImplementedError`（提示改用 `ivit convert`），其餘丟 `RuntimeError`。

---

# 第四部分：腳本 / 範例 / 可攜性

## 20. `see_architecture.py` 第 1 行就 IndentationError 🟧

- **位置**：`scripts/tools/see_architecture.py`
- **症狀**：任何環境、任何平台都無法執行，第 1 行 `IndentationError: unexpected indent`。
- **技術原因**：整個檔案是從 Python REPL 直接貼出來的內容，每行都帶著 REPL 縮排。
- **技術白話**：這個檔案是把 Python 互動式直譯器（REPL）裡的輸入整段複製、原封不動存成 `.py`，所以每一行前面都保留了 REPL 的縮排。模組最外層的敘述不該有縮排，Python 在「語法解析」階段（還沒開始執行）就會因第一行的多餘縮排拋 IndentationError。
- **修法**：改寫為正常腳本——用 `argparse` 接收 ONNX 路徑、加 `if __name__ == "__main__"`、移除硬編 Linux 路徑。

## 21. 多處硬編 `/home/dos/...` Linux 路徑 🟩

- **位置**：`scripts/compile/*.py`（calib-dir 預設）、`compile_mambavision_t.py`、`scripts/tools/extract_mambavision_head.py`
- **症狀**：不帶參數執行時預設路徑指向某人 Linux 家目錄（`/home/dos/...`），Windows 上直接 `FileNotFoundError`。
- **技術原因**：argparse 的 `default=` 寫死了開發者本機的絕對路徑。
- **技術白話**：這些腳本把 argparse 參數的預設值直接填成開發者自己機器上的絕對路徑（`/home/dos/...`）。這種路徑只在那台機器成立，換任何其他機器、尤其是沒有 `/home` 結構的 Windows，不帶參數執行就找不到檔。
- **修法**：模型快取類改用 `os.path.expanduser("~/.cache/ivit/models/...")`（跨平台）；校準資料夾改用相對的 `"calib_images"`。

## 22. `si_quickstart.py` 預設圖片路徑不存在 🟩

- **位置**：`examples/python/advanced/si_quickstart.py`
- **症狀**：不帶 `--image` 執行 → 找不到 `test_images/bus.jpg`。
- **技術原因**：預設值 `test_images/bus.jpg` 在 repo 裡不存在；內建圖片實際在 `examples/data/images/bus.jpg`。
- **技術白話**：`--image` 的預設值字串指向的 `test_images/bus.jpg` 這個路徑在專案裡根本沒有；內建的測試圖實際位於 `examples/data/images/bus.jpg`。此外，相對路徑是相對於「執行時的工作目錄」而非腳本位置，即使路徑對了、從別的目錄執行也會找不到。
- **修法**：用 `os.path.dirname(__file__)` 算出相對於腳本自身的 `examples/data/images/bus.jpg`，不受執行時工作目錄影響。

## 23. onnxruntime fallback 缺套件時 raw crash 🟩

- **位置**：`python/ivit/vision/_classifier_ort.py`
- **症狀**：無 C++ 綁定時 `Classifier` 走純 Python（onnxruntime）後備路徑；若沒裝 onnxruntime → 直接噴 `ModuleNotFoundError` 的完整 traceback。
- **技術原因**：`import onnxruntime` 沒被 try/except 包起來給友善訊息。
- **技術白話**：當 C++ 綁定不可用時，`Classifier` 會退回一個用 onnxruntime 實作的純 Python 版本。那段的 `import onnxruntime` 沒有做例外處理，所以在沒裝這個選用套件的環境，會直接以一長串 `ModuleNotFoundError` traceback 中斷，而不是給一句「請 `pip install onnxruntime`」這種可以直接照做的提示。
- **修法**：包起 import，缺套件時改丟明確的 `ImportError`，指示安裝 onnxruntime 或改建置 C++ 綁定。

---

# 第五部分：偵測管線靜默失敗（本次追加，2026-07-17）

第一~四部分修完後，SDK 在 CPU/GPU 上都能正常 `import`、跑 `simple_inference.exe devices`、`ivit info`。但實際拿 Model Zoo 下載的 `yolox-nano.onnx` 跑偵測（`ivit predict`）時，不管圖片、裝置、`--conf` 門檻怎麼調，**永遠回報 `Detections: 0`**——不崩潰、不報錯，看起來像是跑完了，結果就是空的。用純 onnxruntime 直接跑同一顆 ONNX 驗證過，模型本身輸出的 objectness 最高到 0.987、對應類別分數到 0.92，證明問題不在模型或門檻，而在 SDK 的 Python 推論路徑本身。往下追出兩層疊加的 bug。

## 24. 通用 `Model` 對偵測任務永遠 `TaskType::Unknown` 🟨

- **位置**：`python/ivit/_binding.cpp`（`LoadConfig` 綁定）、`src/core/model.cpp`（`ModelManager::load`）
- **症狀**：`ivit.load(path, device=...)` 之後不管模型是什麼任務，`postprocess()` 一律落到 `switch(task_)` 的 `default: break;`，`results.detections` / `.classifications` 永遠是空的。
- **技術原因**：C++ 端 `struct LoadConfig` 本來就有 `std::string task` 欄位（`include/ivit/core/common.hpp:112`），但 pybind11 綁定（`_binding.cpp`）只註冊了 `device`/`precision`/`batch_size`/`cache_dir`/`use_cache`，**沒有把 `task` 暴露給 Python**。`ModelManager::load()` 只在 `config.task` 精確等於 `"classification"`/`"detection"`/`"segmentation"` 時才會設對應的 `TaskType`，收不到值就停在預設的 `TaskType::Unknown`。
- **技術白話**：這其實是 #16 那個 bug 的延伸真相。#16 當時只發現「傳 `task=` 會直接 `AttributeError` 崩潰」，修法是在 Python 端加 `hasattr(config, "task")` 判斷，讓它「不崩，但也什麼都沒發生」——問題被**消音**而不是**解決**，`task` 這個資訊實際上從來沒有傳到 C++ 那一側。而 `postprocess()` 是靠 `task_` 這個成員變數的值去分派該用分類、偵測還是分割的後處理邏輯；沒收到值就永遠停在 `Unknown`，什麼後處理都不執行，回傳的 `Results` 是空的。因為函式本身不拋例外、CLI 也只是印 `Detections: 0`，看起來像「跑完了，剛好沒偵測到東西」，而不是「程式邏輯根本沒執行」，非常容易被誤判成模型或門檻問題。
- **修法**：`_binding.cpp` 的 `LoadConfig` 補上 `.def_readwrite("task", &ivit::LoadConfig::task, ...)`；`cli.py` 的 `cmd_predict`/`cmd_benchmark`/`cmd_serve` 呼叫 `ivit.load()` 時明確傳 `task="detection"`（這三個指令的輸出格式本來就只處理偵測結果，補上任務提示是合理預設）。

## 25. 通用 `InferenceModel` 前處理/後處理不認得 YOLOX 🟨（本次主要 bug）

- **位置**：`src/core/model.cpp`（`InferenceModel::preprocess()`、`InferenceModel::postprocess_detection()`）
- **症狀**：即使 #24 修好、`task_` 正確變成 `Detection`，`ivit predict` 對 YOLOX 模型仍然回報 0 個框。
- **技術原因**：`InferenceModel`（Python `ivit.load()` 走的通用路徑）的前處理**不分模型類型**，一律做「BGR→RGB 色彩轉換 + 除以 255 正規化 + 單純 resize（無 letterbox）」；後處理 `postprocess_detection()` 也一律假設輸出座標是「已正規化到 [0,1] 的 xywh」，直接乘上圖片寬高還原成像素座標。但 YOLOX 的 ONNX 匯出慣例完全不同：輸入要求 **BGR、保持 [0,255] 不正規化、要 letterbox（等比縮放+灰邊填充）**；輸出座標是**相對格子（grid）的原始值**，中心點要加上格子座標再乘上對應 stride，寬高要先取 `exp()` 再乘 stride，而非任何形式的 [0,1] 正規化值。兩邊的假設從輸入到輸出全部對不上，餵進模型的張量本質上是「用錯誤方式重新編碼過的雜訊」，導致輸出的 objectness 全部趨近於零，即使門檻調到 0.1 也篩不出任何框。程式碼裡其實留了誠實的自白註解：`// This is a simplified YOLO-style postprocessor / Real implementation would need model-specific handling`——這條路徑從一開始就只是示範用的簡化版，從沒真正支援過 YOLOX。
- **技術白話**：真正「懂」YOLOX 的邏輯（BGR/[0,255]、letterbox、grid+stride 解碼）一直都存在，只是寫在另一個類別 `vision::Detector`（`src/vision/detector.cpp`，C++ 專用、`simple_inference.exe` 走的路徑）裡，且它會用模型檔名判斷（含 `"yolox"` 字串）該用哪套前後處理。Python 的 `ivit.load()` 拿到的是完全不同、沒有這層判斷的通用 `Model`/`InferenceModel`，所以會出現「C++ CLI 跑同一顆模型結果完全正確、Python 跑永遠 0 個框」的落差——不是 Python 綁定本身壞了，是通用 C++ 後端從設計上就沒實作 YOLOX 支援。
- **修法**：把 `Detector` 的判斷/解碼邏輯搬進 `InferenceModel`：建構子依模型檔名判斷 `model_type_`（含 `"yolox"` → `"yolox"`，其餘沿用原本行為，不影響非 YOLOX 模型）；`preprocess()` 新增 letterbox 分支（YOLOX 用 BGR/[0,255]/letterbox，其餘維持原本 RGB/[0,1]/單純 resize）並回傳 `PreprocessInfo{scale, pad_w, pad_h}`；`postprocess_detection()` 新增 grid+stride 解碼分支（YOLOX 用 grid 座標解碼 + 拿掉 letterbox padding還原像素座標，其餘維持原本 [0,1] 假設）。修復後 Python `ivit predict` 對 `yolox-nano.onnx` + `bus.jpg` 的結果（4 個框，87.66%/86.91%/78.04%/77.28%）與 C++ `simple_inference.exe detect` 的結果（87.7%/86.9%/78.0%/77.3%）幾乎一致，僅浮點誤差。

## 26. `model.task` 讀取即崩：`TaskType` 沒註冊到 pybind11 🟧（發現，未修復）

- **位置**：`python/ivit/_binding.cpp`（`py::class_<ivit::Model>`：`.def_property_readonly("task", &ivit::Model::task)`）
- **症狀**：修完 #24、#25 後想在 Python 驗證 `model.task` 的值，一讀就崩：`TypeError: Unable to convert function return value to a Python type! The signature was (arg0: ivit._ivit_core.Model) -> ivit::TaskType`。
- **技術原因**：`Model::task()` 回傳 C++ enum `ivit::TaskType`，但整個 `_binding.cpp` 從頭到尾沒有任何 `py::enum_<ivit::TaskType>(...)` 註冊。pybind11 不知道怎麼把這個型別轉成 Python 物件，於是在型別轉換這一步直接拋錯。
- **技術白話**：pybind11 要讓 C++ enum 能在 Python 那邊被讀取或比較，必須額外用 `py::enum_<T>(m, "T").value(...)...` 明確註冊每個列舉值；沒註冊的話，即使 C++ 函式本身完全正常執行、拿到了正確的值，*把這個值從 C++ 傳回 Python* 這一步就會失敗。這個屬性目前完全不能從 Python 讀，任何呼叫 `model.task` 的程式碼都會炸。
- **狀態**：本次會話中發現、確認可重現，尚未修復（不影響 #24/#25 的修復——那兩個 bug 全程都是設定/讀取 C++ 內部的 `task_`，不透過這個有問題的 Python 屬性）。之後要修的話：在 `_binding.cpp` 加 `py::enum_<ivit::TaskType>(m, "TaskType").value("Classification", ...).value("Detection", ...)...`。

## 27. `ivit devices` 純 Python 降級模式崩潰 🟧（發現，未修復）

- **位置**：`python/ivit/cli.py`（`cmd_devices`）與其在無 C++ 綁定時使用的純 Python `_DeviceInfo` fallback stub
- **症狀**：在 DLL 路徑沒設好、C++ 綁定不可用的情況下執行 `ivit devices`，崩潰：`AttributeError: '_DeviceInfo' object has no attribute 'is_available'`。C++ 可用時（DLL 路徑設好）則完全正常。
- **技術原因**：#10 把 `cli.py` 對 `available` 的存取全部改成 `is_available`，跟 pybind11 綁定的真正 `DeviceInfo` 對上了；但純 Python 降級模式另外自己造了一個 `_DeviceInfo` stub class，屬性名還停在舊的 `available`，沒有跟著 #10 一起改。
- **技術白話**：這是 #10 修復留下的副作用——同一個 CLI 函式其實服務兩種物件：C++ 可用時是 pybind11 綁定的 `DeviceInfo`（屬性叫 `is_available`），C++ 不可用時是 Python 自己寫的 `_DeviceInfo` 替代品（屬性還叫 `available`）。#10 只同步了前者，後者沒人記得要一起改，於是「降級模式」這個原本該最穩、最不該崩的路徑，反而成了唯一會崩的路徑。
- **狀態**：本次會話中發現、確認可重現（在還沒設定 OpenVINO/OpenCV DLL 路徑的那個 PowerShell session 裡踩到），尚未修復。修法應該是把 `_DeviceInfo` stub 的屬性名同步改成 `is_available`，或反過來讓 `cmd_devices` 用 `getattr(d, "is_available", getattr(d, "available", False))` 兩邊都吃。

---

# 第六部分：全庫掃描（2026-07-17，找剩下的 Windows-only bug）

前五部分修完後，對整個 repo（`src/`、`include/`、`python/`、`scripts/`、`examples/`、`CMakeLists.txt`）做了一次系統性掃描，找還沒被記錄過的「因為是 Windows 才會出問題」的程式碼。以下兩個是 Windows 預設編譯路徑（`IVIT_USE_QNN=OFF`、`IVIT_USE_HAILORT=OFF`）就會踩到的，已修復；其餘只在額外啟用 QNN / HailoRT（Linux 專用後端）才會出問題，先記錄不修。

## 28. C++ 測試目標用了 GNU ld 專屬的 linker flag 🟥

- **位置**：`CMakeLists.txt:370,375`（`target_link_options(test_model/test_inference PRIVATE -Wl,--unresolved-symbols=ignore-in-shared-libs)`）
- **症狀**：若裝了 GTest 且 `-DIVIT_BUILD_TESTS=ON`（本文件建置流程預設用 `OFF`，所以正常流程不會踩到），MSVC 的 `link.exe` 完全看不懂 `-Wl,...` 這種 GNU `ld` 的 passthrough 語法，直接編譯失敗。
- **技術白話**：`-Wl,` 是 GCC/Clang 用來把後面的參數原封不動轉交給底層連結器 `ld` 的慣例寫法，這裡用它告訴 `ld`「忽略共享庫裡沒解析到的符號」。這個慣例只有 GNU 系連結器認得；MSVC 的 `link.exe` 用的是完全不同的參數格式（例如 `/FORCE:UNRESOLVED`），看到 `-Wl,` 只會當成無法識別的選項報錯。原本的程式碼沒有依平台/編譯器判斷，MSVC 建置時一定會炸。
- **修法**：用 `if(NOT MSVC)` 包住這兩行 `target_link_options`。

## 29. `get_cache_dir()` 只認 `$HOME`，Windows 上會 fallback 到錯誤路徑 🟨

- **位置**：`src/core/device_manager.cpp:719-729`
- **症狀**：`clear_cache()` 沒帶明確路徑時，實際清除的快取目錄不是真正的使用者快取目錄。不會當掉，只是靜默清不到東西。
- **技術原因**：程式用 `std::getenv("HOME")` 取使用者家目錄，抓不到就 fallback 到 `"/tmp/ivit_cache"`。`HOME` 是 POSIX 慣例的環境變數，Windows 原生 cmd/PowerShell 通常沒有設定它（Windows 對應的是 `%USERPROFILE%`）。
- **技術白話**：抓不到 `HOME` 之後 fallback 的 `"/tmp/ivit_cache"` 這個字串，在 Windows 上經過 `std::filesystem` 解析會變成**目前所在磁碟根目錄**下的 `\tmp\ivit_cache`（例如 `C:\tmp\ivit_cache`），跟使用者實際的家目錄毫無關係。這跟 Python 端的行為不一致——`python/ivit/zoo/registry.py`、`python/ivit/core/config.py` 都是用 `Path.home()`/`os.path.expanduser("~/.cache/ivit")`，這兩者在 Windows 上會正確解析成 `C:\Users\<帳號>\.cache\ivit`。C++ 端漏了這層平台判斷。
- **修法**：在既有的 `include/ivit/core/platform_compat.hpp`（`unistd.h`/`ssize_t`/`setenv` 那些 Windows shim 所在的檔案）新增 `ivit_home_dir()`：Windows 上先試 `HOME` 再 fallback `USERPROFILE`，POSIX 維持原本的 `HOME`。`device_manager.cpp` 改呼叫這個函式取代直接呼叫 `std::getenv("HOME")`。

## 30~34. QNN / HailoRT 後端的 POSIX-only 程式碼 🟩（記錄，Windows 預設編譯不受影響，暫不修）

以下全部只在額外開啟 `IVIT_USE_QNN=ON` 或啟用 HailoRT 才會被編譯/執行到；Windows 預設建置（本文件的 `cmake` 指令都沒開這兩個選項）完全碰不到，先記錄供未來真要在 Windows 上啟用這兩個後端時參考：

- **`dlopen`/`dlsym`/`dlclose` 直接呼叫**（`src/core/device_manager.cpp:444,446,564,565`、`src/runtime/qnn_runtime.cpp:61-197`）：POSIX 動態載入 API，MSVC 沒有 `<dlfcn.h>`，要改用 `LoadLibrary`/`GetProcAddress`/`FreeLibrary`。
- **硬編 `.so` 後端庫名**（`device_manager.cpp:438-440,563`、`qnn_runtime.cpp:32-34,71,96-100,160`，例如 `"libQnnHtp.so"`）：即使改用 `LoadLibrary`，Windows 上的檔名也該是 `.dll`，否則探測永遠失敗、裝置永遠找不到。
- **`std::system("... > /dev/null 2>&1")`**（`src/runtime/hailort_runtime.cpp:334-335`）：`/dev/null` 在 Windows `cmd.exe` 不存在，應該用 `NUL`。
- **`/dev/hailo0` 裝置節點存在性檢查**（`device_manager.cpp:573,591`）：Windows 沒有 `/dev` 裝置節點，這個檢查在 Windows 上永遠回傳 false（無害的 no-op，但邏輯上是 Linux-only 假設）。
- **CMake `HINTS /opt/qnn /usr/local/qnn`**（`CMakeLists.txt:212,229,233,237`）：Unix-only 搜尋路徑，Windows 上永遠解析不到，無害但該補上 Windows 對應路徑（例如 QNN SDK 的安裝目錄）。

---

# 總結

| 部分 | 問題數 | 性質 |
|------|:---:|------|
| 一、建置/移植 | 9 | 讓 SDK 在 Windows 能編、能匯入、能跑 GPU |
| 二、執行期 API/CLI | 8 | Python 端與 C++ 綁定「名字/簽章對不上」 |
| 三、靜默失敗 | 2 | 不報錯但回傳假路徑（zoo 下載） |
| 四、腳本/範例/可攜性 | 4 | 語法、硬編路徑、錯誤訊息 |
| 五、偵測管線靜默失敗 | 4（2 已修 / 2 待修） | Python 通用推論路徑對 YOLOX 永遠回傳空結果 |
| 六、全庫掃描 | 5（2 已修 / 3 記錄不修） | 建置系統與 QNN/HailoRT 後端的 POSIX-only 殘留 |

**三大 pattern（供後續維護參考）**：

1. **跨平台假設**：程式碼預設 POSIX/Linux 環境——`unistd.h`、`ssize_t`、`setenv`、`$HOME`、共享庫預設匯出所有符號、DLL 靠 PATH 搜尋、`/home/dos` 絕對路徑、GNU ld 專屬 linker flag、`dlopen`/`.so`。移植到 Windows 就逐一浮現。其中最嚴重的是 **#9 loader-lock 死結**：它在 Linux 完全正常、只在 Windows 觸發，且症狀是「卡死」而非拋錯，最難定位。
2. **Python↔C++ 綁定漂移**：CLI 與部分 API 使用的屬性/方法名與簽章（`available`、`input_info["shape"]`、`show`、`stream(conf=)`、`LoadConfig.task`、`get_best_device(strategy=)`、`TaskType` 列舉未註冊）與實際 pybind 綁定對不上。這類**全部可由整合測試攔截**；CLI 幾乎沒有測試覆蓋是根本原因。
3. **能跑 ≠ 結果對**：#25 是目前最隱蔽的一類——程式不崩、不報錯，甚至印出「跑完了」的訊息，但核心邏輯（YOLOX 前後處理）從一開始就沒被實作、只是行為正確的另一個類別（`vision::Detector`）沒被通用路徑共用。這類 bug 只能靠**拿真模型、真圖片跑一次、檢查數字合理性**抓到，型別檢查/單元測試都測不出來。

> 第一~四部分已在 `window_v2` 分支以 Conventional Commits 提交，並經 CPU（OpenVINO）與 GPU（TensorRT / cuda:0）實跑回歸驗證。第五、六部分為 2026-07-17 追加驗證的發現：#24/#25/#28/#29 已修復並實測（#24/#25 的 `ivit predict` 結果與 C++ `simple_inference.exe` 一致；#28/#29 修復後重新 configure/build 確認無誤，且偵測結果無回歸）；#26/#27 已確認可重現、尚待修復；#30~#34 為 QNN/HailoRT 專用、Windows 預設建置不受影響，先記錄。

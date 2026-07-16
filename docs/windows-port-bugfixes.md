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

# 總結

| 部分 | 問題數 | 性質 |
|------|:---:|------|
| 一、建置/移植 | 9 | 讓 SDK 在 Windows 能編、能匯入、能跑 GPU |
| 二、執行期 API/CLI | 8 | Python 端與 C++ 綁定「名字/簽章對不上」 |
| 三、靜默失敗 | 2 | 不報錯但回傳假路徑（zoo 下載） |
| 四、腳本/範例/可攜性 | 4 | 語法、硬編路徑、錯誤訊息 |

**兩大 pattern（供後續維護參考）**：

1. **跨平台假設**：程式碼預設 POSIX/Linux 環境——`unistd.h`、`ssize_t`、`setenv`、共享庫預設匯出所有符號、DLL 靠 PATH 搜尋、`/home/dos` 絕對路徑。移植到 Windows 就逐一浮現。其中最嚴重的是 **#9 loader-lock 死結**：它在 Linux 完全正常、只在 Windows 觸發，且症狀是「卡死」而非拋錯，最難定位。
2. **Python↔C++ 綁定漂移**：CLI 與部分 API 使用的屬性/方法名與簽章（`available`、`input_info["shape"]`、`show`、`stream(conf=)`、`LoadConfig.task`、`get_best_device(strategy=)`）與實際 pybind 綁定對不上。這類**全部可由整合測試攔截**；CLI 幾乎沒有測試覆蓋是根本原因。

> 全部修正皆已在 `window_v2` 分支以 Conventional Commits 提交，並經 CPU（OpenVINO）與 GPU（TensorRT / cuda:0）實跑回歸驗證。

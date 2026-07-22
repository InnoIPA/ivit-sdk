# iVIT-SDK Getting Started Guide（Windows v2）

iVIT-SDK (Innodisk Vision Intelligence Toolkit) 是一個統一的電腦視覺推理 SDK，支援多種 AI 加速器後端，包括 Intel OpenVINO、NVIDIA TensorRT，以及 Qualcomm QNN (IQ Series)（規劃中）。

> **文件版本**：Windows v2 ｜ **驗證日期**：2026-07-14
> **驗證平台**：Windows 11 Pro、Intel Core i5-13500（含 iGPU）、NVIDIA GeForce RTX 3060（12 GB, driver 560.94）、Python 3.10

## 環境與功能驗證狀態

以下為本版於上述平台的實測結果：

| 項目                                  | 狀態    | 實測結論                                                                                                                                                                                                                                                                     |
| ------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OpenVINO Runtime 環境與功能** | ✅ 通過 | OpenVINO**2026.2.1** 已安裝並啟用；裝置列舉出 `CPU`、`GPU`；合成模型與真實 ONNX（yolox-nano，輸出 `(1,3549,85)`）CPU 推論皆成功。                                                                                                                                |
| **Model Zoo 環境與功能**        | ✅ 通過 | Registry 登錄**25 個模型**；`get_model_info` / `download` 正常，實際下載 `yolox-nano.onnx`（3.66 MB）成功。相依 `requests`、`tqdm`（隨 `.[zoo]` 安裝）。                                                                                                   |
| **TensorRT Runtime 環境與功能** | ✅ 通過 | TensorRT**11.1.0.106**（RTX 3060 + CUDA 12.6）；由 `yolox-nano.onnx` 解析 → 建置 engine（6.86 MB）→ 反序列化 → I/O bindings 正確（`images (1,3,416,416)` / `output (1,3549,85)`）皆成功。即時推論執行需另裝 `pycuda` 或 `cuda-python`（GPU 記憶體綁定）。 |

> **與 v1 的主要差異**：
>
> 1. OpenVINO 版本由 2025.4.1 提升為 **2026.2.1**（`openvino.runtime` 命名空間已於 2026 移除，改用 `import openvino.opset13` / `ov.Core` 新式 API）。
> 2. Model Zoo 純 Python 功能已實測通過。
> 3. TensorRT 改以 pip wheel 安裝於專案 venv（**11.1.0.106**），engine 建置/反序列化實測通過。

## 目錄

- [系統需求](#系統需求)
- [環境安裝](#環境安裝)
- [編譯 SDK](#編譯-sdk)
- [執行期環境設定（DLL 搜尋路徑）](#執行期環境設定dll-搜尋路徑)
- [快速開始](#快速開始)
- [核心概念](#核心概念)
- [API 使用指南](#api-使用指南)
- [裝置選擇](#裝置選擇)
- [模型格式](#模型格式)
- [Hailo 模型編譯（ONNX → HEF）](#hailo-模型編譯onnx--hef)
- [效能優化](#效能優化)
- [常見問題](#常見問題)

---

## 系統需求

### 作業系統

- Windows 10 64-bit（22H2 或更新）
- Windows 11 64-bit（推薦）

### 硬體支援

| 廠商     | 硬體             | 後端          |
| -------- | ---------------- | ------------- |
| Intel    | CPU / iGPU / NPU | OpenVINO      |
| NVIDIA   | dGPU             | TensorRT      |
| Qualcomm | IQ9 / IQ8 / IQ6  | QNN（規劃中） |

> **注意**：Windows 不支援 NVIDIA Jetson 裝置（Jetson 為 ARM Linux 平台）。

### 軟體依賴

| 套件                            | 用途                               | 安裝方式                |
| ------------------------------- | ---------------------------------- | ----------------------- |
| Visual Studio 2026 Build Tools  | C++ 編譯器 (MSVC 19.4+)            | 官網安裝程式            |
| CMake 3.18+                     | 建置系統                           | winget / 官網           |
| Ninja                           | Ninja 建置工具                     | winget / 官網           |
| OpenCV 4.5+（需 vc18）          | OpenCV（含 dnn 模組）              | 從原始碼編譯，見 Step 4 |
| Intel OpenVINO Toolkit 2026.2.1 | OpenVINO C++ Runtime               | Intel 官網安裝程式      |
| Intel GPU Driver（選用）        | Intel iGPU OpenCL Runtime          | Intel 官網              |
| CUDA Toolkit + TensorRT（選用） | NVIDIA GPU 推論                    | NVIDIA 官網             |
| Python 3.9–3.11                | Python 環境                        | python.org              |
| pybind11                        | Python/C++ 綁定（Python 綁定需要） | pip                     |

> **重要**：OpenVINO C++ 開發必須使用 **Intel 官網安裝的 C++ 版本**，不能單獨用 `pip install openvino`。
> pip 版本僅適用於純 Python 開發，不適合 C++ 編譯。

---

## 環境安裝

所有指令請以 **PowerShell（系統管理員）** 執行，除非另有說明。

### Step 1：安裝 Visual Studio Build Tools（C++ 編譯器）

前往 [Visual Studio 下載頁面](https://visualstudio.microsoft.com/downloads/) 下載 **Visual Studio 2026 Build Tools**，執行安裝程式後勾選 **「使用 C++ 的桌面開發」** 工作負載。

> **Windows IoT / LTSC 注意**：這些版本預設不含 winget，請一律使用官網下載安裝程式。

### Step 2：安裝 CMake

1. 前往 [cmake.org/download](https://cmake.org/download/)
2. 下載 **Windows x64 Installer**（`cmake-x.x.x-windows-x86_64.msi`）
3. 執行安裝程式，安裝時勾選 **「Add CMake to the system PATH for all users」**

### Step 3：安裝 Ninja

1. 前往 [github.com/ninja-build/ninja/releases](https://github.com/ninja-build/ninja/releases)
2. 下載 `ninja-win.zip`，解壓出 `ninja.exe`
3. 將 `ninja.exe` 複製至已在 PATH 中的目錄，例如 `C:\Windows\System32\`

安裝完成後，開啟新的 PowerShell 視窗確認版本：

```powershell
cmake --version
ninja --version
```

### Step 4：安裝 OpenCV

> **重要**：[OpenCV 官網](https://opencv.org/releases/) 提供的 Windows 預編譯安裝程式（`opencv-4.x.x-windows.exe`）目前只內附到 **vc16**（VS2019 工具集）的二進位檔，**不含 VS2026（vc18）工具集**。用 Visual Studio 2026 編譯 iVIT-SDK 時，MSVC 版本必須與連結的 OpenCV 二進位一致，因此**不能直接用預編譯版**，需自行從原始碼編譯出 vc18 版本。

**從原始碼編譯 OpenCV（vc18）**

先照原本方式從 [OpenCV 官網](https://opencv.org/releases/) 下載並執行 Windows 安裝程式（`opencv-4.x.x-windows.exe`），預設解壓至 `C:\Users\<你的帳號>\Documents\opencv\`。

> **這個安裝程式其實是自解壓縮檔**：解壓後會同時得到 `opencv\build\`（僅內附 vc16 預編譯二進位，**不含 vc18**）與 `opencv\sources\`（完整原始碼，含 `CMakeLists.txt`）。不需要另外 `git clone`，直接用 `sources\` 資料夾編譯 vc18 版本即可。

以下需在 **Developer PowerShell for VS 2026** 中執行（確保 CMake 選到 MSVC v143/VS2026 的 Generator）。**不要重用 `build\` 資料夾**（裡面是 vc16 預編譯產物），改建一個獨立的 `build_vc18\`：

```powershell
cd C:\Users\<你的帳號>\Documents\opencv
mkdir build_vc18
cd build_vc18
```

```powershell
# 配置（來源指向 ..\sources；安裝到 build_vc18\install，
# 產物會落在 build_vc18\install\x64\vc18\bin，與本文件其他路徑一致）
cmake ..\sources -G "Visual Studio 18 2026" -A x64 `
    -DCMAKE_INSTALL_PREFIX=install `
    -DBUILD_SHARED_LIBS=ON `
    -DBUILD_TESTS=OFF `
    -DBUILD_PERF_TESTS=OFF `
    -DBUILD_EXAMPLES=OFF
```

```powershell
# 編譯並安裝（Release，時間較長，視硬體約 20–40 分鐘）
cmake --build . --config Release --target INSTALL
```

安裝完成後設定環境變數，讓 CMake 能找到 OpenCV：

```powershell
# 將以下路徑加入系統環境變數 OpenCV_DIR（依實際編譯路徑調整）
[Environment]::SetEnvironmentVariable("OpenCV_DIR", "C:\Users\<你的帳號>\Documents\opencv\build_vc18\install", "Machine")

# 將 OpenCV DLL 路徑加入 PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
[Environment]::SetEnvironmentVariable("PATH", "$currentPath;C:\Users\<你的帳號>\Documents\opencv\build_vc18\install\x64\vc18\bin", "Machine")
```

> 安裝後需重新開啟 PowerShell 使環境變數生效。

### Step 5：安裝 Intel OpenVINO Toolkit

從 [Intel OpenVINO 下載頁面](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) 下載 Windows 封存版（Archive，ZIP，**2026.2.1**）並解壓至目標目錄，例如 `C:\Users\<使用者名稱>\Documents\openvino_2026\`。

> 安裝路徑會依下載方式不同而異，例如直接解壓下載的 ZIP 路徑會是：
> `C:\Users\<使用者名稱>\Downloads\openvino_toolkit_windows_2026.2.1.xxxxx_x86_64\`
> 請依實際路徑調整下方指令（本文件以 `C:\Users\<使用者名稱>\Documents\openvino_2026\` 為例）。

安裝後，每次開啟新 PowerShell 時需初始化 OpenVINO 環境。

**請使用 `.ps1` 版本**（需先解除一次 PowerShell 執行原則限制）：

```powershell
# 僅對目前使用者解除限制（執行一次即永久生效）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

將初始化加入 PowerShell Profile 以自動套用：

> **`notepad $PROFILE` 出現「找不到指定的路徑」**：新機器通常還沒有 Profile 資料夾／檔案，`notepad` 不會自動建立缺少的目錄。先建立好再開（下方為獨立指令區塊，複製時不會帶到 `>` 字元）：

```powershell
$profileDir = Split-Path $PROFILE -Parent
if (-not (Test-Path $profileDir)) { New-Item -ItemType Directory -Force -Path $profileDir | Out-Null }
if (-not (Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE | Out-Null }
```

```powershell
# 編輯 PowerShell Profile
notepad $PROFILE

# 在 Profile 中加入以下這一行（必須用 & 呼叫 .ps1，不能用 cmd /c 呼叫 .bat，理由見上方說明）：
# & "C:\Users\<使用者名稱>\Documents\openvino_2026\setupvars.ps1"
```

> **iGPU 支援**：需另外安裝 [Intel Arc &amp; Iris Xe Graphics Driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)。
>
> **NPU 支援**：Intel Core Ultra NPU 需安裝對應的 [NPU Driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)。

### Step 6：（選用）安裝 CUDA Toolkit + TensorRT（NVIDIA GPU）

#### 安裝 CUDA Toolkit

從 [NVIDIA CUDA 下載頁面](https://developer.nvidia.com/cuda-downloads) 選擇 Windows 版本，下載並執行安裝程式（`.exe`）。

安裝完成後確認：

```powershell
nvcc --version
```

#### 安裝 TensorRT
TensorRt 版本會隨著本機的cuda版本不同而不同, 需要自行對應版本

1. 前往 [NVIDIA TensorRT 下載頁面](https://developer.nvidia.com/tensorrt)（需登入 NVIDIA 帳號）
2. 下載 Windows ZIP 版本（`TensorRT-10.x.x.x.Windows.win10.cuda-12.x.zip`）
3. 解壓至 `C:\TensorRT\`

設定環境變數：

```powershell
[Environment]::SetEnvironmentVariable("TensorRT_DIR", "C:\TensorRT", "Machine")

$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
[Environment]::SetEnvironmentVariable("PATH", "$currentPath;C:\TensorRT\bin", "Machine")
```

> 安裝後需重新開啟 PowerShell 使環境變數生效。

### Step 7：裝ivit-sdk, 安裝 Pyenv-win 及 virtualenv

取得原始碼

```powershell
git clone https://github.com/innodisk-mannywang/ivit-sdk.git
cd ivit-sdk
```

進入[pyenv](https://github.com/pyenv-win/pyenv-win/releases) 下載zip檔（解壓縮並加入進環境變數）

```powershell
pyenv --version
pyenv install 3.11.9
pyenv global 3.11.9
```

安裝virtualenv

```powershell
pip3 install virtualenv
#建立虛擬環境
python -m venv venv
#啟動虛擬環境
venv\Scripts\Activate.ps1
```

### Step 8：安裝 Python 與 pybind11

從 [python.org](https://www.python.org/downloads/) 下載並安裝 Python 3.9–3.11（建議 **勾選「Add Python to PATH」**）。

或使用 winget：

```powershell
winget install Python.Python.3.11
```

安裝 pybind11：

```powershell
pip install pybind11
```

### 驗證安裝

```powershell
# 確認 CMake 版本
cmake --version

# 確認已執行 setupvars.ps1、OpenVINO 環境變數確實有值（不是空字串）
$env:INTEL_OPENVINO_DIR

# （選用，在任一 CMake 專案的 build 目錄下執行）確認 CMake 找得到 C++ SDK：
# cma7ke .. -G "Visual Studio 18 2026" -A x64 -DOpenVINO_DIR="C:\Users\<使用者名稱>\Documents\openvino_2026\runtime\cmake" -DCMAKE_BUILD_TYPE=Release
# 完整指令（含 OpenCV_DIR）請見下方「編譯 SDK」章節

# （選用）確認 CUDA
nvcc --version

# （選用）確認 TensorRT
pip install tensorrt
python -c "import tensorrt; print(tensorrt.__version__)"
```

---

## 編譯 SDK

### 開啟 Developer PowerShell

### C++ 編譯（不含 Python 綁定）

此編譯後以下的功能將無法使用：

* ivit cli指令
* 訓練
* 模型匯出
* int8量化
* model zoo 下載

逐行執行，每行確認成功後再執行下一行：

```powershell
mkdir build
```

```powershell
cd build
```

```powershell
cmake .. -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release -DIVIT_BUILD_EXAMPLES=ON -DIVIT_BUILD_TESTS=OFF -DIVIT_BUILD_PYTHON=OFF
```

```powershell
cmake --build . --config Release
```

### C++ + Python 綁定編譯

```powershell
mkdir build
```

```powershell
cd build
```

```powershell
cmake .. -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release -DIVIT_BUILD_EXAMPLES=ON -DIVIT_BUILD_TESTS=OFF -DIVIT_BUILD_PYTHON=ON
```

```powershell
cmake --build . --config Release
```

### 安裝至 site-packages

> **重要**：`python -m build` 會用 scikit-build-core 在**全新的隔離環境**裡重新執行一次 CMake configure，
> 跟你手動建立的 `build\` 資料夾是分開的、互不相干，**不會**沿用你先前設定的 `OpenCV_DIR` / `OpenVINO_DIR`
> 環境變數。必須用 `-C cmake.define.<NAME>=<VALUE>` 把這兩個路徑明確傳進去，每次執行都要帶，
> 不是一次性的權宜之計。

```powershell
cd ..
pip install build
python -m build 

# 檔名依你實際的 Python 版本而定（例如 Python 3.10 產出的是 cp310-cp310，3.11 是 cp311-cp311）
# 用 dir dist\ 看實際檔名，或直接用萬用字元安裝：
pip install (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName
```

### 安裝 Python 套件

編譯完成後，安裝 Python 套件以使用 `ivit` CLI 和 Python API。**`pip install -e .` 同樣會觸發 CMake，需要一樣的 `-C cmake.define.` 參數**：

```powershell
pip install -e ".[zoo]" 
```

> **PATH 設定**：若安裝後出現 `WARNING: The script ivit is installed in ... which is not on PATH`，
> 請將 Python Scripts 目錄加入 PATH：
>
> ```powershell
> # 查詢 Scripts 路徑
> python -c "import site; print(site.getusersitepackages())"
>
> # 手動加入 PATH（範例，依實際路徑調整）
> $scriptsPath = "$env:USERPROFILE\AppData\Roaming\Python\Python311\Scripts"
> [Environment]::SetEnvironmentVariable("PATH", "$env:PATH;$scriptsPath", "User")
> ```
>
> 使用 venv 的情況下不會有此問題。

### CMake 選項

| 選項                    | 預設值 | 說明                 |
| ----------------------- | ------ | -------------------- |
| `IVIT_BUILD_PYTHON`   | ON     | 編譯 Python bindings |
| `IVIT_BUILD_TESTS`    | ON     | 編譯單元測試         |
| `IVIT_BUILD_EXAMPLES` | ON     | 編譯範例程式         |
| `IVIT_USE_OPENVINO`   | ON     | 啟用 OpenVINO 後端   |
| `IVIT_USE_TENSORRT`   | ON     | 啟用 TensorRT 後端   |

### 預期輸出

CMake 正確配置後應顯示：

```
-- OpenCV version: 4.5.4
-- OpenVINO found: 2026.2.1
--
-- Backends:
--   OpenVINO:       ON
--   TensorRT:       OFF
--   QNN:            OFF
```

### 編譯產物

```powershell
dir build\Release\          # ivit.dll、ivit.lib
dir build\bin\Release\      # simple_inference.exe、detection_demo.exe、...
```

---

## 執行期環境設定（DLL 搜尋路徑）

安裝完成後，**首次 `import ivit` 或執行 `ivit` CLI 前**，必須讓 Python 找得到原生 DLL，否則會靜默退回「Python only」模式（`ivit.is_cpp_available()` 回傳 `False`，Detector/Classifier/Segmentor 無法使用）。


### 需要在 PATH 中的三個目錄

| 目錄                                 | 內容                                                                                                                              | 如何加入                                           |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| OpenVINO runtime bin                 | `openvino.dll`、TBB 等                                                                                                          | 執行`setupvars.bat`（自動加入）                  |
| OpenCV bin                           | `opencv_core4130.dll`、`opencv_imgproc4130.dll` 等模組化 DLL（未啟用 `BUILD_opencv_world`，故非單一 `opencv_world*.dll`） | 手動加入`...\build_vc18\install\x64\vc18\bin`    |
| **`site-packages\ivit\bin`** | **`ivit.dll`（SDK 核心）**                                                                                                | **僅 editable 安裝需手動加入**（見下方說明） |

### 一鍵設定（依實際安裝路徑調整）

以下以本文件驗證平台的實際路徑為例，請改成你自己的位置：

```powershell
# 1. OpenVINO 環境（自動把 runtime bin 加入 PATH；務必用 & 呼叫 .ps1，見 Step 5 說明）
& "C:\Users\<你的帳號>\Documents\openvino_2026\setupvars.ps1"

# 2. OpenCV DLL 目錄
$env:PATH = "C:\Users\<你的帳號>\Documents\opencv\build_vc18\install\x64\vc18\bin;$env:PATH"

# 3. SDK 核心 ivit.dll 所在目錄（editable 安裝才需要，路徑依 venv 位置調整）
$env:PATH = "C:\Users\<你的帳號>\Documents\ivit-sdk\venv\Lib\site-packages\ivit\bin;$env:PATH"
```

設定後驗證：

```powershell
python -c "import ivit; print('cpp:', ivit.is_cpp_available())"
# 預期輸出：cpp: True
```

### editable 安裝 vs 正式安裝

| 安裝方式                                         | `ivit.dll` 位置                                                      | 是否需手動加`site-packages\ivit\bin`                         |
| ------------------------------------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------- |
| **正式安裝**（`pip install .` 或 wheel） | 與套件同目錄（`site-packages\ivit\`）                                | **不需要**，`__init__.py` 會自動 `add_dll_directory` |
| **editable**（`pip install -e .`）       | `site-packages\ivit\bin\`，但 `__init__.py` 由原始碼載入，兩者分離 | **需要**（加入上方第 3 項）                              |

> **替代方案（手動 C++ build）**：若是用 `cmake` 手動編譯（非 pip 安裝），也可改為把 `ivit.dll` 複製到系統路徑，效果相同：
>
> ```powershell
> Copy-Item build\Release\ivit.dll "C:\Windows\System32\"
> ```

---

## 快速開始

### 快速驗證指令

編譯與安裝完成後，用下面這組指令快速確認整條路徑（OpenVINO + OpenCV + ivit）都通：

```powershell
#若環境未正常啟動的話, 需要手動加入, 若正常啟動的話，直接跳到3.
# 1. 初始化 OpenVINO 環境（缺這步會導致載入 openvino.dll 失敗，見 Step 5 說明）
& "C:\Users\<使用者名稱>\openvino_2026\setupvars.ps1"

# 2. 補上 OpenCV / ivit.dll 的 DLL 搜尋路徑（editable 安裝才需要第二個路徑）
$env:PATH = "C:\Users\<使用者名稱>opencv\build_vc18\install\x64\vc18\bin;C:\Users\<使用者名稱>\Documents\ivit-sdk\venv\Lib\site-packages\ivit\bin;$env:PATH"

# 3. 下載一個小模型（若已下載過會直接跳過）
ivit zoo download yolox-nano

# 4. 實際跑一次偵測
ivit predict "$env:USERPROFILE\.cache\ivit\models\yolox-nano.onnx" "C:\Users\<使用者名稱>\ivit-sdk\examples\data\images\bus.jpg" --device cpu --conf 0.3
```

預期會看到 `Detections: 4`（bus + 3 個 person，信心值約 77~88%）。

### 確認裝置

```powershell
cd build
.\bin\Release\simple_inference.exe devices
```

### C++ 物件偵測

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ivit/vision/detector.hpp"

using namespace ivit;
using namespace ivit::vision;

int main() {
    // 載入模型（使用 NPU）
    Detector detector("yolox-s.onnx", "npu");

    // 讀取影像
    cv::Mat image = cv::imread("image.jpg");

    // 執行推理
    auto results = detector.predict(image, 0.5f);

    // 處理結果
    for (const auto& det : results.detections) {
        std::cout << det.label << ": " << det.confidence * 100 << "%" << std::endl;
        cv::rectangle(image, det.bbox.to_rect(), cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("output.jpg", image);
    return 0;
}
```

### C++ 影像分類

```cpp
#include "ivit/vision/classifier.hpp"

Classifier classifier("resnet50.onnx", "cpu");
auto results = classifier.predict(image, 5);  // Top-5

for (const auto& cls : results.classifications) {
    std::cout << cls.label << ": " << cls.score * 100 << "%" << std::endl;
}
```

### C++ 語意分割

```cpp
#include "ivit/vision/segmentor.hpp"

Segmentor segmentor("deeplabv3.onnx", "npu");
auto results = segmentor.predict(image);

cv::Mat overlay = results.overlay_mask(image, 0.5);
cv::imwrite("segmentation.jpg", overlay);
```

### Python 範例

```python
import ivit

# 下載模型（首次使用）
ivit.zoo.download("yolox-s")

# 物件偵測
detector = ivit.Detector("~/.cache/ivit/models/yolox-s.onnx", device="npu")
results = detector.predict("image.jpg")

for det in results.detections:
    print(f"{det.label}: {det.confidence:.2%}")

# 影像分類
classifier = ivit.Classifier("~/.cache/ivit/models/resnet50.onnx", device="cpu")
results = classifier.predict("image.jpg", top_k=5)

for cls in results.classifications:
    print(f"{cls.label}: {cls.score:.2%}")
```

> **Windows 路徑**：`~/.cache/ivit/models/` 對應至 `C:\Users\<使用者名稱>\.cache\ivit\models\`。

---

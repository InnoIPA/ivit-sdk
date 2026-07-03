# iVIT-SDK Getting Started Guide（Windows）

iVIT-SDK (Innodisk Vision Intelligence Toolkit) 是一個統一的電腦視覺推理 SDK，支援多種 AI 加速器後端，包括 Intel OpenVINO、NVIDIA TensorRT，以及 Qualcomm QNN (IQ Series)（規劃中）。

## 目錄

- [系統需求](#系統需求)
- [環境安裝](#環境安裝)
- [編譯 SDK](#編譯-sdk)
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

| 套件                                    | 用途                                      | 安裝方式           |
| --------------------------------------- | ----------------------------------------- | ------------------ |
| Visual Studio 2026 Build Tools          | C++ 編譯器 (MSVC 19.4+)                   | 官網安裝程式       |
| CMake 3.18+                             | 建置系統                                  | winget / 官網      |
| Ninja                                   | Ninja 建置工具                            | winget / 官網      |
| OpenCV 4.5+                             | OpenCV（含 dnn 模組）                     | 官網預編譯版       |
| Intel OpenVINO Toolkit 2025.4.1         | OpenVINO C++ Runtime                     | Intel 官網安裝程式 |
| Intel GPU Driver（選用）                | Intel iGPU OpenCL Runtime                | Intel 官網         |
| CUDA Toolkit + TensorRT（選用）         | NVIDIA GPU 推論                           | NVIDIA 官網        |
| Python 3.9–3.11                         | Python 環境                               | python.org         |
| pybind11                                | Python/C++ 綁定（Python 綁定需要）        | pip                |

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

### Step 3：安裝 OpenCV

從 [OpenCV 官網](https://opencv.org/releases/) 下載 Windows 預編譯安裝程式（`opencv-4.x.x-windows.exe`）並執行，預設安裝至 `C:\opencv\`。

安裝後設定環境變數，讓 CMake 能找到 OpenCV：

```powershell
# 將以下路徑加入系統環境變數 OpenCV_DIR
[Environment]::SetEnvironmentVariable("OpenCV_DIR", "C:\opencv\build", "Machine")

# 將 OpenCV DLL 路徑加入 PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
[Environment]::SetEnvironmentVariable("PATH", "$currentPath;C:\opencv\build\x64\vc18\bin", "Machine")
```

> 安裝後需重新開啟 PowerShell 使環境變數生效。

### Step 4：安裝 Intel OpenVINO Toolkit

從 [Intel OpenVINO 下載頁面](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) 下載 Windows 封存版（Archive，ZIP）並解壓至目標目錄，例如 `C:\intel\openvino_2025\`。

> 安裝路徑會依下載方式不同而異，例如直接解壓下載的 ZIP 路徑會是：
> `C:\Users\<使用者名稱>\Downloads\openvino_toolkit_windows_2025.4.1.xxxxx_x86_64\`
> 請依實際路徑調整下方指令。

安裝後，每次開啟新 PowerShell 時需初始化 OpenVINO 環境。

**建議：使用 `.bat` 版本（無執行原則限制）**

```powershell
cmd /c "C:\intel\openvino_2025\setupvars.bat"
```

**若要用 `.ps1` 版本，需先解除 PowerShell 執行原則限制：**

```powershell
# 僅對目前使用者解除限制（執行一次即永久生效）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 之後每次開新 PowerShell 執行：
& "C:\intel\openvino_2025\setupvars.ps1"
```

或將初始化加入 PowerShell Profile 以自動套用：

```powershell
# 編輯 PowerShell Profile
notepad $PROFILE

# 在 Profile 中加入以下行（擇一）：
# cmd /c "C:\intel\openvino_2025\setupvars.bat"
```

> **iGPU 支援**：需另外安裝 [Intel Arc & Iris Xe Graphics Driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)。
>
> **NPU 支援**：Intel Core Ultra NPU 需安裝對應的 [NPU Driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)。

### Step 5：（選用）安裝 CUDA Toolkit + TensorRT（NVIDIA GPU）

#### 安裝 CUDA Toolkit

從 [NVIDIA CUDA 下載頁面](https://developer.nvidia.com/cuda-downloads) 選擇 Windows 版本，下載並執行安裝程式（`.exe`）。

安裝完成後確認：

```powershell
nvcc --version
```

#### 安裝 TensorRT

1. 前往 [NVIDIA TensorRT 下載頁面](https://developer.nvidia.com/tensorrt)（需登入 NVIDIA 帳號）
2. 下載 Windows ZIP 版本（`TensorRT-10.x.x.x.Windows.win10.cuda-12.x.zip`）
3. 解壓至 `C:\TensorRT\`

設定環境變數：

```powershell
[Environment]::SetEnvironmentVariable("TensorRT_DIR", "C:\TensorRT", "Machine")

$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
[Environment]::SetEnvironmentVariable("PATH", "$currentPath;C:\TensorRT\lib", "Machine")
```

> 安裝後需重新開啟 PowerShell 使環境變數生效。

### Step 6：安裝 Python 與 pybind11

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

# 確認 OpenVINO 環境（需先執行 setupvars.ps1）
python -c "import openvino; print(openvino.__version__)"

# （選用）確認 CUDA
nvcc --version

# （選用）確認 TensorRT
python -c "import tensorrt; print(tensorrt.__version__)"
```

---

## 編譯 SDK

### 取得原始碼

```powershell
git clone https://github.com/innodisk-mannywang/ivit-sdk.git
cd ivit-sdk
```

### 初始化 OpenVINO 環境

每次開啟新 PowerShell 都需執行（路徑依實際安裝位置調整）：

```powershell
cmd /c "C:\intel\openvino_2025\setupvars.bat"
```

### 開啟 Developer PowerShell

編譯前需在有 MSVC 環境的 PowerShell 中執行。開啟方式：

**開始選單 → 搜尋「Developer PowerShell for VS 2026」→ 以系統管理員開啟**

或在一般 PowerShell 中手動初始化：

```powershell
# 找到 vcvars64.bat 並初始化 MSVC 環境（路徑依實際安裝調整）
cmd /c "`"C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && powershell"
```

### 清除舊快取（切換 Generator 或重新編譯時需要）

若 `build` 目錄曾用不同 Generator 產生過快取，需先清除再重新執行 cmake：

```powershell
Remove-Item -Recurse -Force build\CMakeCache.txt, build\CMakeFiles
```

### C++ 編譯（不含 Python 綁定）

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

```powershell
cd ..
python -m build
pip install dist\ivit_sdk-1.0.0-cp311-cp311-win_amd64.whl
```

### 安裝 Python 套件

編譯完成後，安裝 Python 套件以使用 `ivit` CLI 和 Python API：

```powershell
pip install -e ".[zoo]"
```

若需要使用虛擬環境（推薦）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[zoo]"
```

> **注意**：首次執行 `.venv\Scripts\Activate.ps1` 若出現執行原則錯誤，請先執行：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

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
-- OpenVINO found: 2025.4.1
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

## 快速開始

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

## 核心概念

### 架構概覽

```
┌─────────────────────────────────────────────────────────────┐
│                      Application                             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Vision Tasks Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Classifier  │  │   Detector   │  │  Segmentor   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │    Model     │  │   Tensor     │  │   Results    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Runtime Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  OpenVINO    │  │  TensorRT    │  │     QNN      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Hardware Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Intel CPU   │  │ NVIDIA GPU   │  │ Qualcomm IQ  │       │
│  │  Intel GPU   │  │              │  │ (IQ9/IQ8/IQ6)│       │
│  │  Intel NPU   │  │              │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 主要類別

| 類別              | 說明                                |
| ----------------- | ----------------------------------- |
| `Detector`      | 物件偵測（YOLO, SSD, Faster R-CNN） |
| `Classifier`    | 影像分類（ResNet, EfficientNet）    |
| `Segmentor`     | 語意分割（DeepLabV3, U-Net）        |
| `Model`         | 底層模型抽象介面                    |
| `Results`       | 推理結果容器                        |
| `DeviceManager` | 裝置管理與發現                      |

---

## API 使用指南

### Detector API

```cpp
class Detector {
public:
    Detector(const std::string& model_path,
             const std::string& device = "auto",
             const LoadConfig& config = LoadConfig{});

    Results predict(const cv::Mat& image,
                    float conf_threshold = 0.5f,
                    float iou_threshold = 0.45f);

    Results predict(const std::string& image_path,
                    float conf_threshold = 0.5f,
                    float iou_threshold = 0.45f);

    std::vector<Results> predict_batch(const std::vector<cv::Mat>& images,
                                       const InferConfig& config = InferConfig{});

    void predict_video(const std::string& source,
                       std::function<void(const Results&, const cv::Mat&)> callback,
                       const InferConfig& config = InferConfig{});

    const std::vector<std::string>& classes() const;
    int num_classes() const;
    cv::Size input_size() const;
};
```

### Results API

```cpp
class Results {
public:
    std::vector<Detection> detections;
    std::vector<ClassificationResult> classifications;
    cv::Mat segmentation_mask;

    float inference_time_ms;
    std::string device_used;
    cv::Size image_size;

    const ClassificationResult& top1() const;
    std::vector<ClassificationResult> topk(int k) const;

    std::vector<Detection> filter_by_class(const std::vector<int>& class_ids) const;
    std::vector<Detection> filter_by_confidence(float min_conf) const;

    cv::Mat colorize_mask(const std::map<int, cv::Vec3b>& colormap = {}) const;
    cv::Mat overlay_mask(const cv::Mat& image, float alpha = 0.5) const;

    cv::Mat visualize(const cv::Mat& image,
                      bool show_labels = true,
                      bool show_confidence = true,
                      bool show_boxes = true,
                      bool show_masks = true) const;

    std::string to_json() const;
    void save(const std::string& path, const std::string& format = "json") const;
};
```

### Detection 結構

```cpp
struct Detection {
    BBox bbox;
    int class_id;
    std::string label;
    float confidence;
    std::optional<cv::Mat> mask;
};

struct BBox {
    float x1, y1, x2, y2;

    float width() const;
    float height() const;
    float area() const;
    float iou(const BBox& other) const;
    cv::Rect to_rect() const;
};
```

---

## 裝置選擇

### 支援的裝置

| 裝置字串   | 說明                   | 後端         |
| ---------- | ---------------------- | ------------ |
| `auto`   | 自動選擇最佳裝置       | 自動         |
| `cpu`    | CPU                    | OpenVINO     |
| `gpu:0`  | Intel 內顯             | OpenVINO     |
| `cuda:0` | NVIDIA GPU             | TensorRT     |
| `npu`    | Intel NPU              | OpenVINO     |
| `iq9`    | Qualcomm IQ9 (QCS9075) | QNN (規劃中) |
| `iq8`    | Qualcomm IQ8 (QCS8550) | QNN (規劃中) |
| `iq6`    | Qualcomm IQ6 (QCS6490) | QNN (規劃中) |

### 裝置發現

```cpp
#include "ivit/core/device.hpp"

auto& manager = DeviceManager::instance();
auto devices = manager.list_devices();

for (const auto& dev : devices) {
    std::cout << "Device: " << dev.id << std::endl;
    std::cout << "  Name:    " << dev.name << std::endl;
    std::cout << "  Backend: " << dev.backend << std::endl;
    std::cout << "  Type:    " << dev.type << std::endl;
}

auto best = manager.get_best_device("detection", "performance");
std::cout << "Best device: " << best.id << std::endl;
```

### 裝置選擇策略

```cpp
// 明確指定裝置
Detector detector("model.onnx", "npu");

// 自動選擇
Detector detector("model.onnx", "auto");

// 指定後端
LoadConfig config;
config.backend = "openvino";
config.device = "npu";
Detector detector("model.onnx", config.device, config);
```

---

## 模型格式

### 支援的格式

| 格式            | 副檔名                | 後端     | 說明                 |
| --------------- | --------------------- | -------- | -------------------- |
| ONNX            | `.onnx`             | 全部     | 通用交換格式（推薦） |
| OpenVINO IR     | `.xml`, `.bin`    | OpenVINO | Intel 優化格式       |
| TensorRT Engine | `.engine`, `.trt` | TensorRT | NVIDIA 優化格式      |
| PaddlePaddle    | `.pdmodel`          | OpenVINO | 百度框架格式         |

### 精度設定

| 精度     | 說明        | 適用場景                                                          |
| -------- | ----------- | ----------------------------------------------------------------- |
| `FP32` | 32 位元浮點 | 最高精度，NPU/CPU 推薦直接使用                                    |
| `FP16` | 16 位元浮點 | 縮小模型檔案（約 50%），適合儲存與部署                            |
| `INT8` | 8 位元整數  | 最高效能（需校準資料集），可透過 `ivit convert -p int8 -c` 轉換 |

```cpp
LoadConfig config;
config.precision = "fp16";
Detector detector("model.onnx", "npu", config);
```

### 模型轉換（`ivit convert`）

SDK 提供 CLI 工具將 ONNX 模型轉換為各後端的優化格式。

#### 基本用法

```powershell
# ONNX → OpenVINO IR（FP32）
ivit convert model.onnx --format openvino

# ONNX → OpenVINO IR（FP16 壓縮）
ivit convert model.onnx --format openvino --precision fp16

# ONNX → OpenVINO IR（INT8 量化，需指定校準圖片資料夾）
ivit convert model.onnx --format openvino --precision int8 -c .\calibration_images\

# ONNX → TensorRT Engine
ivit convert model.onnx --format tensorrt --precision fp16

# 指定輸出目錄
ivit convert model.onnx --format openvino -o .\output\
```

#### 完整轉換流程（含編譯）

首次使用或更新 SDK 後，需先編譯再轉換：

```powershell
# 1. 初始化 OpenVINO 環境（路徑依實際安裝位置調整）
cmd /c "C:\intel\openvino_2025\setupvars.bat"

# 2. 編譯 C++ 核心與 Python binding（需在 Developer PowerShell 中執行）
mkdir build
cd build
cmake .. -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release -DIVIT_BUILD_PYTHON=ON
cmake --build . --config Release
cd ..

# 3. 複製 DLL 至系統路徑（以系統管理員執行）
Copy-Item build\Release\ivit.dll "C:\Windows\System32\"

# 4. 安裝 Python 套件
pip install -e .

# 5. 執行轉換
ivit convert model.onnx --format openvino --precision fp16
```

#### 精度支援與限制

| 精度     | OpenVINO     | TensorRT     | 說明                                       |
| -------- | ------------ | ------------ | ------------------------------------------ |
| `fp32` | ✅           | ✅           | 預設精度                                   |
| `fp16` | ✅           | ✅           | 縮小模型檔案，不影響推論速度（見下方說明） |
| `int8` | ✅（需校準） | ✅（需校準） | 最高效能，需提供校準圖片資料夾             |

#### OpenVINO INT8 量化

**安裝量化依賴：**

```powershell
pip install -e ".[quantize]"
# 或單獨安裝
pip install nncf>=2.7 openvino>=2024.0
```

**使用方式：**

```powershell
# 準備校準圖片（建議 50~300 張）
dir .\calibration_images\
# image001.jpg  image002.jpg  image003.png  ...

# 執行 INT8 量化
ivit convert model.onnx -f openvino -p int8 -c .\calibration_images\
```

**注意事項：**

- 校準圖片資料夾只掃描頂層目錄，支援 jpg/jpeg/png/bmp/webp/tiff 格式
- 超過 300 張時會自動隨機取樣 300 張（固定 seed 確保可重現）
- 前處理方式：resize 到模型輸入尺寸 → float32 [0,1] 正規化
- 量化使用 NNCF `MIXED` preset，平衡精度與效能
- INT8 量化完全在 Python 端執行，不需要 C++ binding

#### FP16 轉換效能說明

FP16 壓縮**不會提升推論速度**，唯一好處是模型檔案大小減半。原因如下：

- **NPU**：硬體原生以 FP16 計算，不論輸入模型是 FP32 或 FP16，NPU 編譯器會自動處理精度轉換。使用 FP16 IR 反而因額外的 Convert 節點可能略慢。
- **CPU**：x86 CPU 原生以 FP32 計算，FP16 權重會在推論時解壓回 FP32，無效能增益。

以下為 YOLOX-Nano 在 Intel AIPC 平台的實測數據（100 次迭代）：

| 模型格式    | 裝置 | 平均延遲 | 吞吐量  | 檔案大小 |
| ----------- | ---- | -------- | ------- | -------- |
| ONNX (FP32) | NPU  | 8.74 ms  | 114 FPS | 13 MB    |
| IR FP32     | NPU  | 8.77 ms  | 114 FPS | ~12 MB   |
| IR FP16     | NPU  | 13.38 ms | 75 FPS  | ~6 MB    |
| ONNX (FP32) | CPU  | 33.25 ms | 30 FPS  | 13 MB    |
| IR FP32     | CPU  | 34.30 ms | 29 FPS  | ~12 MB   |
| IR FP16     | CPU  | 34.18 ms | 29 FPS  | ~6 MB    |

> **建議**：NPU 推論直接使用 ONNX 或 FP32 IR 即可，效能最佳。
> FP16 轉換適合需要縮小模型檔案的部署場景（如嵌入式裝置儲存空間有限）。

#### C++ / Python API 轉換

除 CLI 外，也可在程式中直接呼叫轉換 API：

```cpp
// C++
#include "ivit/ivit.hpp"

ivit::convert_model("yolox-s.onnx", "yolox-s.xml", "cpu", "fp16");
```

```python
# Python
import ivit
ivit.convert_model("yolox-s.onnx", "yolox-s.xml", "cpu", "fp16")
```

---

## Hailo 模型編譯（ONNX → HEF）

`scripts/compile/` 內提供針對各家骨幹的 Hailo 編譯腳本，可將 ONNX 編譯為 Hailo-8 上跑的 `.hef` 權重。

### 前置依賴

需先安裝 Hailo Dataflow Compiler（`hailo_sdk_client`）。請參考 Hailo 官方文件取得對應版本的 whl 並安裝：

```powershell
pip install hailo_dataflow_compiler-*.whl
```

> 校準資料集預設指向 `/home/dos/coco_val2017/val2017`，Windows 上請務必用 `--calib-dir` 指定實際路徑，例如 `--calib-dir C:\datasets\coco_val2017\val2017`。

### 執行位置

所有指令請在**專案根目錄** (`ivit-sdk\`) 執行，腳本內路徑（`models\onnx\`、`models\hailo\`）皆為相對於根目錄。

### 一鍵編譯（使用預設參數）

| 模型               | 指令                                                              |
| ------------------ | ----------------------------------------------------------------- |
| ResNet50           | `python scripts\compile\compile_resnet50.py`                    |
| EfficientNet-B0    | `python scripts\compile\compile_efficientnet_b0.py`             |
| MambaVision-T head | `python scripts\compile\compile_mambavision_t.py`               |
| DeepLabV3-ResNet50 | `python scripts\compile\compile_deeplabv3.py`                   |

輸出統一寫到 `models\hailo\{name}.har`（中間檔）和 `models\hailo\{name}.hef`（最終權重）。

### 常用覆寫參數

```powershell
python scripts\compile\compile_resnet50.py `
    --onnx models\onnx\resnet50.onnx `
    --hef  models\hailo\resnet50.hef `
    --calib-dir C:\path\to\calib_images `
    --calib-size 512 `
    --input-hw 224 `
    --hw-arch hailo8        # hailo8 | hailo8r | hailo8l
```

完整旗標可用 `-h` 查詢：

```powershell
python scripts\compile\compile_resnet50.py -h
```

### YOLOX → HAR

```powershell
python scripts\compile\onnx_to_hef.py    # 產出 models\hailo\yolox_s.har
```

### 輸出位置

| 類型           | 路徑                                       |
| -------------- | ------------------------------------------ |
| ONNX 原檔      | `models\onnx\`                           |
| 中間 HAR       | `models\hailo\{name}.har`                |
| 編譯後 HEF     | `models\hailo\{name}.hef`                |
| Hailo 設定檔   | `scripts\compile\configs\*.json`         |
| Hailo 編譯日誌 | `logs\compile\*.txt`、`logs\*.log`     |

---

## 效能優化

### 1. 選擇正確的裝置

```
效能排序（Intel 平台）:
NPU (效率最佳) > iGPU > CPU
```

### 2. 直接使用 ONNX 或 FP32 IR

NPU 硬體原生以 FP16 計算，不需要事先轉換模型精度。直接使用 ONNX 即可獲得最佳效能：

```cpp
// 直接載入 ONNX，NPU 會自動以 FP16 計算
Detector detector("model.onnx", "npu");
```

### 3. 模型預熱

```cpp
// 預熱 3 次，避免首次推理延遲（NPU 尤其重要）
detector.model()->warmup(3);
```

### 4. 啟用模型快取

```cpp
LoadConfig config;
config.use_cache = true;
config.cache_dir = "./cache";
Detector detector("model.onnx", "npu", config);
```

### 5. 效能測試

```powershell
.\bin\Release\simple_inference.exe benchmark yolox-s.onnx npu 100
```

---

## 範例程式

### C++ 範例

| 範例                    | 說明                                   |
| ----------------------- | -------------------------------------- |
| `simple_inference`    | 綜合範例（分類、偵測、分割、效能測試） |
| `classification_demo` | 影像分類範例                           |
| `detection_demo`      | 物件偵測範例                           |
| `video_demo`          | 即時影片偵測範例                       |

```powershell
# 列出可用裝置
.\bin\Release\simple_inference.exe devices

# 物件偵測（使用 NPU）
.\bin\Release\simple_inference.exe detect yolox-s.onnx image.jpg npu output.jpg

# 影像分類
.\bin\Release\simple_inference.exe classify resnet50.onnx cat.jpg cpu

# 語意分割
.\bin\Release\simple_inference.exe segment deeplabv3.onnx scene.jpg npu segmented.jpg

# 效能測試
.\bin\Release\simple_inference.exe benchmark yolox-s.onnx npu 100
```

### Python 範例

| 範例                     | 說明             |
| ------------------------ | ---------------- |
| `01_quickstart.py`     | 快速入門範例     |
| `02_detection.py`      | 物件偵測完整範例 |
| `02_classification.py` | 影像分類範例     |
| `02_segmentation.py`   | 語意分割範例     |

**快速入門（自動下載模型）：**

```powershell
# 直接執行，模型會自動從 Model Zoo 下載（預設 yolox-s）
python examples\python\01_quickstart.py

# 或使用自訂模型
$env:IVIT_MODEL_PATH = ".\my_model.onnx"
python examples\python\01_quickstart.py
```

**其他範例（需先下載模型）：**

```powershell
# 物件偵測（使用 NPU）
ivit zoo download yolox-s
python examples\python\02_detection.py `
    -m "$env:USERPROFILE\.cache\ivit\models\yolox-s.onnx" `
    -i examples\data\images\bus.jpg `
    -d npu

# 影像分類（使用 NPU）
ivit zoo download resnet50
python examples\python\02_classification.py `
    -m "$env:USERPROFILE\.cache\ivit\models\resnet50.onnx" `
    -i examples\data\images\bus.jpg `
    -d npu

# 語意分割（使用 NPU）
ivit zoo download deeplabv3-mobilenetv3
python examples\python\02_segmentation.py `
    -m "$env:USERPROFILE\.cache\ivit\models\deeplabv3-mobilenetv3.onnx" `
    -i examples\data\images\bus.jpg `
    -d npu
```

---

## 常見問題

### Q: CMake 找不到 OpenVINO？

確認已執行 OpenVINO 環境初始化，且安裝路徑正確：

```powershell
# 初始化 OpenVINO 環境（路徑依實際安裝位置調整）
cmd /c "C:\intel\openvino_2025\setupvars.bat"

# 手動指定 OpenVINO CMake 路徑（需在 Developer PowerShell 中執行）
cmake .. -G "Visual Studio 18 2026" -A x64 -DOpenVINO_DIR="C:\intel\openvino_2025\runtime\cmake"
```

> **注意**：`pip install openvino` 安裝的是 Python 版本，不提供 C++ CMake 設定檔。
> C++ 編譯必須使用 Intel 官網安裝程式版本。

### Q: 如何確認 NPU 是否可用？

```powershell
# 用 SDK 確認可用裝置
.\bin\Release\simple_inference.exe devices

# 在裝置管理員中確認 NPU
# 開始 → 搜尋「裝置管理員」→ 展開「神經處理器」
# 應可見到「Intel(R) AI Boost」

# 確認 NPU 驅動版本
Get-WmiObject Win32_PnPSignedDriver | Where-Object { $_.DeviceName -like "*NPU*" -or $_.DeviceName -like "*AI Boost*" } | Select-Object DeviceName, DriverVersion
```

### Q: 模型載入失敗怎麼辦？

1. 確認模型檔案存在且格式正確
2. 確認對應的後端已安裝，且 DLL 在 PATH 中可被找到
3. 確認已執行 OpenVINO setupvars.ps1
4. 檢查錯誤訊息

```cpp
try {
    Detector detector("model.onnx", "npu");
} catch (const ModelLoadError& e) {
    std::cerr << "Model load failed: " << e.what() << std::endl;
} catch (const DeviceNotFoundError& e) {
    std::cerr << "Device not found: " << e.what() << std::endl;
}
```

### Q: 執行時找不到 DLL？

Windows 上執行程式前需確認相關 DLL 在 PATH 中：

```powershell
# 確認 OpenVINO DLL 路徑已在 PATH 中（setupvars.ps1 會自動設定）
& "C:\Program Files (x86)\Intel\openvino_2025\setupvars.ps1"

# 確認 OpenCV DLL 路徑已在 PATH 中
$env:PATH += ";C:\opencv\build\x64\vc18\bin"

# 確認 TensorRT DLL 路徑已在 PATH 中（若使用 TensorRT）
$env:PATH += ";C:\TensorRT\lib"
```

### Q: 如何支援自訂類別？

在模型目錄下建立 `labels.txt`：

```
person
car
bicycle
```

模型會自動讀取同目錄下的 `labels.txt`。

### Q: PowerShell 執行原則錯誤怎麼辦？

```powershell
# 允許本機腳本執行（以系統管理員執行）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 聯絡資訊

- **GitHub**: https://github.com/innodisk-mannywang/ivit-sdk
- **Issues**: https://github.com/innodisk-mannywang/ivit-sdk/issues
- **Email**: support@innodisk.com

---

*iVIT-SDK v1.0.0 - Innodisk Corporation*


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

| 精度     | 說明        | 適用場景                                                         |
| -------- | ----------- | ---------------------------------------------------------------- |
| `FP32` | 32 位元浮點 | 最高精度，NPU/CPU 推薦直接使用                                   |
| `FP16` | 16 位元浮點 | 縮小模型檔案（約 50%），適合儲存與部署                           |
| `INT8` | 8 位元整數  | 最高效能（需校準資料集），可透過`ivit convert -p int8 -c` 轉換 |

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
# 1. 初始化 OpenVINO 環境（路徑依實際安裝位置調整；務必用 & 呼叫 .ps1，見 Step 5 說明）
& "C:\Users\<使用者名稱>\Documents\openvino_2026\setupvars.ps1"

# 2. 編譯 C++ 核心與 Python binding（需在 Developer PowerShell 中執行）
mkdir build
cd build
cmake .. -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release -DIVIT_BUILD_PYTHON=ON
cmake --build . --config Release
cd ..

# 3. 複製 DLL 至系統路徑（以系統管理員執行）
Copy-Item build\Release\ivit.dll "C:\Windows\System32\"

# 4. 安裝 Python 套件（-C cmake.define. 參數必要，見編譯 SDK 章節說明）
pip install -e . `
    -C cmake.define.OpenCV_DIR="C:\Users\<使用者名稱>\Documents\opencv\build_vc18\install" `
    -C cmake.define.OpenVINO_DIR="C:\Users\<使用者名稱>\Documents\openvino_2026\runtime\cmake"

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

> `pip install -e ".[quantize]"` 一樣會重新觸發 ivit-sdk 的 CMake 建置（見上方「安裝 Python 套件」的 `-C cmake.define.` 說明）。
> 若只是要跑 INT8 量化（純 Python，不需要 C++ binding），建議直接用下面「單獨安裝」那行，不用重新編譯整個 SDK。

```powershell
pip install -e ".[quantize]" `
    -C cmake.define.OpenCV_DIR="C:\Users\<你的帳號>\Documents\opencv\build_vc18\install" `
    -C cmake.define.OpenVINO_DIR="C:\Users\<你的帳號>\Documents\openvino_2026\runtime\cmake"
# 或單獨安裝（推薦，不會觸發 CMake 重新建置）
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

| 模型               | 指令                                                  |
| ------------------ | ----------------------------------------------------- |
| ResNet50           | `python scripts\compile\compile_resnet50.py`        |
| EfficientNet-B0    | `python scripts\compile\compile_efficientnet_b0.py` |
| MambaVision-T head | `python scripts\compile\compile_mambavision_t.py`   |
| DeepLabV3-ResNet50 | `python scripts\compile\compile_deeplabv3.py`       |

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

| 類型           | 路徑                                   |
| -------------- | -------------------------------------- |
| ONNX 原檔      | `models\onnx\`                       |
| 中間 HAR       | `models\hailo\{name}.har`            |
| 編譯後 HEF     | `models\hailo\{name}.hef`            |
| Hailo 設定檔   | `scripts\compile\configs\*.json`     |
| Hailo 編譯日誌 | `logs\compile\*.txt`、`logs\*.log` |

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
# 初始化 OpenVINO 環境（路徑依實際安裝位置調整；務必用 & 呼叫 .ps1，見 Step 5 說明）
& "C:\Users\<使用者名稱>\Documents\openvino_2026\setupvars.ps1"

# 手動指定 OpenVINO CMake 路徑（需在 Developer PowerShell 中執行）
cmake .. -G "Visual Studio 18 2026" -A x64 -DOpenVINO_DIR="C:\Users\<使用者名稱>\Documents\openvino_2026\runtime\cmake"
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

Windows 上執行程式前需確認相關 DLL 在 PATH 中（完整說明見 [執行期環境設定](#執行期環境設定dll-搜尋路徑)）：

```powershell
# 確認 OpenVINO DLL 路徑已在 PATH 中（務必用 & 呼叫 .ps1，cmd /c 呼叫 .bat 不會生效，見 Step 5 說明）
& "C:\Users\<你的帳號>\Documents\openvino_2026\setupvars.ps1"

# 確認 OpenCV DLL 路徑已在 PATH 中（路徑依實際安裝調整）
$env:PATH += ";C:\Users\<你的帳號>\Documents\opencv\build_vc18\install\x64\vc18\bin"

# 確認 SDK 核心 ivit.dll 目錄已在 PATH 中（editable 安裝需要）
$env:PATH += ";C:\Users\<你的帳號>\Documents\ivit-sdk\venv\Lib\site-packages\ivit\bin"

# 確認 TensorRT DLL 路徑已在 PATH 中（若使用 TensorRT）
$env:PATH += ";C:\TensorRT\lib"
```

> **症狀對照**：若 `import ivit` 沒報錯但 `ivit.is_cpp_available()` 為 `False`，多半就是上面某個 DLL 目錄不在 PATH（`_ivit_core.pyd` 載入失敗被靜默降級）。

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

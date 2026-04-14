# iVIT-SDK 程式碼架構完整說明

> 文件版本：2026-03-12
> 專案：iVIT-SDK (Innodisk Vision Intelligence Toolkit)

---

## 目錄

1. [專案總覽](#1-專案總覽)
2. [目錄樹狀結構](#2-目錄樹狀結構)
3. [架構分層圖](#3-架構分層圖)
4. [使用者主要調用入口](#4-使用者主要調用入口)
5. [Python 套件模組詳解](#5-python-套件模組詳解)
6. [C++ 核心模組詳解](#6-c-核心模組詳解)
7. [pybind11 綁定層詳解](#7-pybind11-綁定層詳解)
8. [資料流程圖](#8-資料流程圖)
9. [程式回追路線圖](#9-程式回追路線圖)
10. [後端推論引擎](#10-後端推論引擎)
11. [新增模型修改指南](#11-新增模型修改指南)
12. [設計模式總結](#12-設計模式總結)

---

## 1. 專案總覽

iVIT-SDK 是一個跨硬體平台的統一電腦視覺推論 SDK，核心設計理念是**一次開發，多平台部署**。

```
使用者程式碼
     │
     ▼
Python / C++ 統一 API
     │
     ▼
視覺任務層（分類 / 偵測 / 分割 / 姿態估計）
     │
     ▼
前後處理層（Preprocessor / Postprocessor）
     │
     ▼
後端推論引擎（OpenVINO / TensorRT）
     │
     ▼
硬體（Intel CPU/GPU/NPU/VPU、NVIDIA dGPU/Jetson）
```

---

## 2. 目錄樹狀結構

```
ivit-sdk/
├── CMakeLists.txt                  # 頂層 C++ 建置設定
├── pyproject.toml                  # Python 套件打包設定
│
├── include/ivit/                   # C++ 公開標頭（API 介面）
│   ├── ivit.hpp                    # 整合進入點（include all）
│   ├── core/
│   │   ├── common.hpp              # 全域型別、列舉、例外
│   │   ├── tensor.hpp              # Tensor 資料容器
│   │   ├── model.hpp               # 統一模型推論介面
│   │   ├── result.hpp              # 推論結果結構
│   │   ├── device.hpp              # 裝置資訊與管理
│   │   ├── callback.hpp            # 事件回呼系統
│   │   ├── runtime_config.hpp      # 執行期設定
│   │   └── video_source.hpp        # 影片串流抽象
│   ├── runtime/
│   │   ├── runtime.hpp             # IRuntime 抽象介面
│   │   ├── openvino_runtime.hpp    # Intel 後端
│   │   └── tensorrt_runtime.hpp    # NVIDIA 後端
│   ├── vision/
│   │   ├── classifier.hpp          # 分類器
│   │   ├── detector.hpp            # 偵測器
│   │   └── segmentor.hpp           # 分割器
│   └── utils/
│       ├── visualizer.hpp          # 視覺化工具
│       └── profiler.hpp            # 效能分析工具
│
├── src/                            # C++ 實作
│   ├── core/
│   │   └── model.cpp               # ★ 核心推論引擎（1094 行）
│   ├── runtime/
│   │   ├── runtime_factory.cpp     # ★ 後端選擇工廠（113 行）
│   │   ├── openvino_runtime.cpp    # OpenVINO 後端實作
│   │   └── tensorrt_runtime.cpp    # TensorRT 後端實作
│   ├── vision/                     # 視覺任務實作
│   └── utils/                      # 工具實作
│
├── python/ivit/                    # Python 套件（主要開發介面）
│   ├── __init__.py                 # ★ 套件進入點、load() API（286 行）
│   ├── cli.py                      # ★ CLI 命令列工具（1080 行）
│   ├── devices.py                  # 裝置探索與管理
│   ├── _binding.cpp                # ★ pybind11 C++↔Python 橋接（~800 行）
│   ├── core/
│   │   ├── types.py                # 資料型別、資料類別、列舉
│   │   ├── exceptions.py           # 自訂例外類別
│   │   ├── model.py                # Python 側模型載入
│   │   ├── result.py               # 推論結果容器
│   │   ├── processors.py           # ★ 前/後處理器（1176 行）
│   │   └── callbacks.py            # 回呼定義
│   ├── vision/
│   │   ├── classifier.py           # ★ 分類器 Python 封裝（124 行）
│   │   ├── detector.py             # ★ 偵測器 Python 封裝（176 行）
│   │   └── segmentor.py            # 分割器 Python 封裝
│   ├── runtime/
│   │   ├── openvino_runtime.py     # OpenVINO Python 封裝
│   │   └── tensorrt_runtime.py     # TensorRT Python 封裝
│   ├── train/
│   │   ├── trainer.py              # 訓練引擎（PyTorch-based）
│   │   ├── exporter.py             # 模型轉換匯出
│   │   ├── dataset.py              # 資料集載入
│   │   ├── augmentation.py         # 資料增強
│   │   └── callbacks.py            # 訓練回呼
│   ├── zoo/
│   │   ├── __init__.py             # Model Zoo 進入點
│   │   └── registry.py             # ★ 模型目錄與下載管理（1004 行）
│   └── utils/
│       ├── visualizer.py           # 結果視覺化
│       ├── profiler.py             # 效能分析
│       ├── metrics.py              # 評估指標
│       └── video.py                # 影片處理工具
│
├── models/                         # 本地 Model Zoo 暫存
├── examples/                       # 範例程式
├── tests/                          # 測試程式碼
└── docs/                           # 文件
```

---

## 3. 架構分層圖

```
┌─────────────────────────────────────────────────────────────────┐
│                      使用者 / 應用層                              │
│         Python script / C++ 應用程式 / CLI 工具                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                       統一 API 層                                 │
│                                                                   │
│   Python: ivit.load()  ivit.Detector()  ivit.Classifier()        │
│   C++:    ivit::Model  ivit::Detector   ivit::Classifier          │
│   CLI:    ivit predict / convert / benchmark / zoo               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      視覺任務層                                   │
│                                                                   │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │ Classifier  │  │  Detector   │  │  Segmentor  │  ...        │
│   │  分類器      │  │   偵測器     │  │   分割器     │            │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
└──────────┼────────────────┼────────────────┼────────────────────┘
           │                │                │
┌──────────▼────────────────▼────────────────▼────────────────────┐
│                   前後處理層 (processors.py)                       │
│                                                                   │
│  PreProcessors                  PostProcessors                    │
│  ┌──────────────────────┐      ┌──────────────────────────────┐  │
│  │ LetterboxPreProcessor│      │ YOLOPostProcessor            │  │
│  │ CenterCropPreProcess │      │ YOLOXPostProcessor           │  │
│  │ (可擴展新增)          │      │ RTDETRPostProcessor          │  │
│  └──────────────────────┘      │ RTMPosePostProcessor         │  │
│                                │ ClassificationPostProcessor   │  │
│                                │ TorchVisionDetPostProcessor   │  │
│                                └──────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    後端推論引擎層                                  │
│                                                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  IRuntime（抽象介面）                      │   │
│   │  load_model() / unload_model() / infer() / convert()    │   │
│   └──────────────────────────────────────────────────────────┘  │
│            │                              │                       │
│   ┌────────▼────────┐          ┌──────────▼──────────┐          │
│   │ OpenVINO Runtime│          │ TensorRT Runtime     │          │
│   │  (Intel 後端)   │          │  (NVIDIA 後端)        │          │
│   └────────┬────────┘          └──────────┬──────────┘          │
└────────────┼─────────────────────────────┼──────────────────────┘
             │                             │
┌────────────▼──────────┐    ┌────────────▼──────────────────────┐
│  Intel Hardware        │    │  NVIDIA Hardware                   │
│  CPU / iGPU / NPU     │    │  dGPU / Jetson                     │
│  VPU (Myriad)         │    │  CUDA Acceleration                  │
└───────────────────────┘    └────────────────────────────────────┘
```

---

## 4. 使用者主要調用入口

當你使用 iVIT-SDK 時，以下是**最常被呼叫的函數和類別**，按使用頻率排序：

### 4.1 最常用 API（日常開發 90% 會用到的）

| 使用方式 | 對應程式碼位置 | 說明 |
|----------|---------------|------|
| `import ivit` | `python/ivit/__init__.py:1-286` | 套件入口，import 後即可使用所有 API |
| `ivit.load(source, device, task)` | `python/ivit/__init__.py:168-205` | **最核心的一行程式**，載入模型 |
| `model.predict(image)` | `src/core/model.cpp:77-101` (C++) | 單張影像推論 |
| `model.predict_batch(images)` | `src/core/model.cpp:103-203` (C++) | 批次推論 |
| `results.detections` | `python/ivit/_binding.cpp:312` | 取得偵測結果列表 |
| `results.classifications` | `python/ivit/_binding.cpp:307` | 取得分類結果列表 |
| `results.to_json()` | `python/ivit/_binding.cpp:358` | 結果序列化 |
| `results.visualize(image)` | `python/ivit/_binding.cpp:347-355` | 結果視覺化 |

### 4.2 任務專用 API

| 使用方式 | 對應程式碼位置 | 說明 |
|----------|---------------|------|
| `ivit.Detector(model, device)` | `python/ivit/vision/detector.py:32-53` | 建立偵測器 |
| `detector.predict(image, conf, iou)` | `python/ivit/vision/detector.py:55-78` | 物件偵測 |
| `detector.predict_video(source, callback)` | `python/ivit/vision/detector.py:126-155` | 影片串流偵測 |
| `ivit.Classifier(model, device)` | `python/ivit/vision/classifier.py:27-48` | 建立分類器 |
| `classifier.predict(image, top_k)` | `python/ivit/vision/classifier.py:50-67` | 影像分類 |
| `ivit.Segmentor(model, device)` | `python/ivit/vision/segmentor.py` | 建立分割器 |

### 4.3 Model Zoo API

| 使用方式 | 對應程式碼位置 | 說明 |
|----------|---------------|------|
| `ivit.zoo.list_models(task)` | `python/ivit/zoo/registry.py:331-351` | 列出所有可用模型 |
| `ivit.zoo.search(query)` | `python/ivit/zoo/registry.py:354-388` | 搜尋模型 |
| `ivit.zoo.download(name, format)` | `python/ivit/zoo/registry.py:413-481` | 下載模型 |
| `ivit.zoo.load(name, device)` | `python/ivit/zoo/registry.py:933-971` | 下載 + 載入一步完成 |
| `ivit.zoo.get_model_info(name)` | `python/ivit/zoo/registry.py:391-410` | 查詢模型詳細資訊 |

### 4.4 裝置管理 API

| 使用方式 | 對應程式碼位置 | 說明 |
|----------|---------------|------|
| `ivit.devices.best()` | `python/ivit/devices.py` | 自動選取最佳裝置 |
| `ivit.list_devices()` | C++ binding → `DeviceManager::enumerate()` | 列出所有裝置 |

### 4.5 CLI 命令

| 命令 | 對應程式碼位置 | 說明 |
|------|---------------|------|
| `ivit predict <image> -m <model>` | `python/ivit/cli.py` → `cmd_predict()` | 命令列推論 |
| `ivit convert <model> --to openvino` | `python/ivit/cli.py` → `cmd_convert()` | 模型轉換 |
| `ivit benchmark -m <model>` | `python/ivit/cli.py` → `cmd_benchmark()` | 效能測試 |
| `ivit zoo list` | `python/ivit/cli.py` → `cmd_zoo_list()` | 列出 Zoo 模型 |
| `ivit zoo download <name>` | `python/ivit/cli.py` → `cmd_zoo_download()` | 下載模型 |

---

## 5. Python 套件模組詳解

### 5.1 `python/ivit/__init__.py` — 套件進入點（286 行）

**功能**：整個 Python 套件的對外 API 進入點，提供最高層次的簡化介面。

**關鍵程式碼解析**：

```python
# 行 29-36: 預載入 C++ 共享函式庫
_libivit_path = _os.path.join(_package_dir, "libivit.so")
if _os.path.exists(_libivit_path):
    _ctypes.CDLL(_libivit_path, mode=_ctypes.RTLD_GLOBAL)

# 行 42-101: 從 C++ binding 匯入所有核心類別
from ._ivit_core import (
    Model, load_model, Classifier, Detector, Segmentor,
    LoadConfig, InferConfig, Results, ...
)

# 行 168-205: load() 是使用者最常呼叫的函數
def load(source, device="auto", task=None, **kwargs):
    _require_cpp()           # 確保 C++ binding 可用
    config = LoadConfig()    # 建立載入設定
    config.device = device
    if task:
        config.task = task
    return load_model(source, config)  # 委派給 C++ ModelManager
```

| 函數 / 物件 | 行號 | 說明 |
|------------|------|------|
| `load(source, device, task, **kwargs)` | 168-205 | 核心函數。依據副檔名或 task 參數自動選擇 Classifier/Detector/Segmentor |
| `is_cpp_available()` | 109-111 | 檢測 C++ binding 是否可用 |
| `_require_cpp()` | 114-121 | C++ binding 檢查（內部用） |
| `get_runtime_info()` | 124-130 | 取得 SDK 執行期資訊 |
| `Classifier` | 由 C++ binding import (行 67) | 分類器 |
| `Detector` | 由 C++ binding import (行 68) | 偵測器 |
| `Segmentor` | 由 C++ binding import (行 69) | 分割器 |
| `devices` | 149-156 | `DeviceManager` 單例 |
| `zoo` | 159 | Model Zoo 模組 |

---

### 5.2 `python/ivit/cli.py` — 命令列工具（1080 行）

**功能**：提供 `ivit` CLI，讓使用者無需寫程式碼就能操作 SDK 全部功能。

| 子命令 | 說明 |
|--------|------|
| `ivit info` | 顯示 SDK 版本、Python 版本、已安裝後端 |
| `ivit devices` | 列出可用裝置，支援 `--json` 輸出 |
| `ivit benchmark` | 對指定模型進行效能測試 |
| `ivit predict` | 對圖片/影片執行推論 |
| `ivit convert` | 將 ONNX 轉換為 OpenVINO IR 或 TensorRT engine |
| `ivit export` | 匯出最佳化模型 |
| `ivit serve` | 啟動 HTTP 推論伺服器 |
| `ivit zoo list/search/download` | Model Zoo 操作 |

---

### 5.3 `python/ivit/core/processors.py` — 前後處理器（1176 行）

**功能**：模型推論前的影像前處理，以及推論後的輸出解碼。這是連接影像與模型的**關鍵橋樑**。

**前處理器**

| 類別 | 行號 | 說明 |
|------|------|------|
| `BasePreProcessor` | 18-58 | 抽象基礎類別，定義 `process()` 介面 |
| `LetterboxPreProcessor` | 116-176 | Letterbox 縮放：保持長寬比，填充灰色邊框（偵測模型用） |
| `CenterCropPreProcessor` | 179-251 | 中心裁切：裁切中央區域，ImageNet 標準化（分類模型用） |

**後處理器**

| 類別 | 行號 | 說明 |
|------|------|------|
| `BasePostProcessor` | 61-113 | 抽象基礎類別，定義 `process()` 介面 |
| `YOLOPostProcessor` | 254-420 | 解碼 YOLOv5/v8 輸出（含向量化 NMS） |
| `ClassificationPostProcessor` | 423-496 | 解碼分類輸出（Softmax + Top-K） |
| `YOLOXPostProcessor` | 499-647 | 解碼 YOLOX 輸出（含 obj_conf × class_conf） |
| `RTDETRPostProcessor` | 650-769 | 解碼 RT-DETR 輸出（Transformer，免 NMS） |
| `RTMPosePostProcessor` | 781-993 | 解碼姿態估計輸出（SimCC / Heatmap / Coordinates） |
| `TorchVisionDetectionPostProcessor` | 996-1117 | 解碼 torchvision 偵測模型（已內建 NMS） |

**處理器登錄表**（行 1120-1133）：

```python
_PREPROCESSORS = {
    "letterbox": LetterboxPreProcessor,
    "center_crop": CenterCropPreProcessor,
}
_POSTPROCESSORS = {
    "yolo": YOLOPostProcessor,
    "yolox": YOLOXPostProcessor,
    "rtdetr": RTDETRPostProcessor,
    "rtmpose": RTMPosePostProcessor,
    "classification": ClassificationPostProcessor,
    "torchvision_detection": TorchVisionDetectionPostProcessor,
}
```

**工具函數**：

| 函數 | 行號 | 說明 |
|------|------|------|
| `get_preprocessor(name)` | 1136-1149 | 依名稱取得前處理器實例 |
| `get_postprocessor(name)` | 1152-1165 | 依名稱取得後處理器實例 |
| `register_preprocessor(name, cls)` | 1168-1170 | 註冊自訂前處理器 |
| `register_postprocessor(name, cls)` | 1173-1175 | 註冊自訂後處理器 |

---

### 5.4 `python/ivit/core/types.py` — 核心型別定義

**功能**：定義整個 SDK 使用的標準型別，確保 Python 與 C++ 型別對齊。

| 型別 | 說明 |
|------|------|
| `DataType` (Enum) | Float32 / Float16 / Int8 / UInt8 / Int32 / Int64 / Bool |
| `TaskType` (Enum) | Classification / Detection / Segmentation / Pose / Face / OCR |
| `BackendType` (Enum) | OpenVINO / TensorRT / QNN / Auto |
| `Precision` (Enum) | FP32 / FP16 / INT8 / INT4 |
| `LoadConfig` (dataclass) | 模型載入設定：device、backend、task、precision、batch_size、cache_dir、use_cache |
| `InferConfig` (dataclass) | 推論設定：conf_threshold、iou_threshold、max_detections、class_filter |
| `TensorInfo` (dataclass) | Tensor 描述：name、shape、dtype |
| `DeviceInfo` (dataclass) | 裝置描述：id、type、backend、vendor、memory_mb、compute_units |
| `BBox` (dataclass) | 邊界框：x1、y1、x2、y2，附 `iou()` 計算方法 |
| `Detection` (dataclass) | 偵測結果：bbox、class_id、class_name、confidence |
| `ClassificationResult` (dataclass) | 分類結果：class_id、class_name、score |
| `Keypoint` (dataclass) | 關鍵點：x、y、score、name |
| `Pose` (dataclass) | 姿態估計結果：keypoints 列表、bbox |

---

### 5.5 `python/ivit/core/exceptions.py` — 例外處理

```
IVITError（基礎例外）
├── ModelLoadError      # 模型載入失敗（路徑錯誤、格式不支援）
├── DeviceNotFoundError # 找不到指定裝置
├── BackendNotAvailableError  # 後端未安裝（未裝 OpenVINO/TensorRT）
├── InferenceError      # 推論過程中發生錯誤
├── ConversionError     # 模型格式轉換失敗
└── ZooDownloadError    # Model Zoo 下載失敗
```

---

### 5.6 `python/ivit/vision/detector.py` — 偵測器（176 行）

**功能**：物件偵測任務的高階封裝。

**關鍵程式碼解析**：

```python
# 行 32-53: 建構子 - 建立 C++ 偵測器
class Detector:
    def __init__(self, model, device="auto", **kwargs):
        config = _CppLoadConfig()
        config.device = device
        self._cpp_detector = _CppDetector(model, device, config)  # 委派 C++

    # 行 55-78: predict - 對單張影像偵測
    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45, ...):
        return self._cpp_detector.predict(image, conf_threshold, iou_threshold)

    # 行 126-155: predict_video - 影片串流偵測
    def predict_video(self, source, callback, ...):
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            results = self.predict(frame, ...)
            callback(results, frame)  # 使用者回呼
```

| 方法 / 屬性 | 行號 | 說明 |
|------------|------|------|
| `__init__(model, device, **kwargs)` | 32-53 | 建立偵測器，建構 C++ 底層物件 |
| `predict(image, conf, iou)` | 55-78 | 對單張影像偵測 |
| `__call__(image, ...)` | 80-89 | 簡寫呼叫（等同 predict） |
| `predict_batch(images, ...)` | 91-124 | 批次偵測 |
| `predict_video(source, callback)` | 126-155 | 影片串流逐幀偵測 |
| `classes` | 157-160 | 類別名稱列表 |
| `num_classes` | 162-165 | 類別數量 |
| `input_size` | 167-170 | 模型輸入尺寸 |

---

### 5.7 `python/ivit/vision/classifier.py` — 分類器（124 行）

| 方法 / 屬性 | 行號 | 說明 |
|------------|------|------|
| `__init__(model, device, **kwargs)` | 27-48 | 建立分類器 |
| `predict(image, top_k)` | 50-67 | 對單張影像分類 |
| `predict_batch(images, top_k)` | 77-103 | 批次分類 |
| `classes` | 105-108 | 類別名稱列表 |
| `num_classes` | 110-113 | 類別數量 |
| `input_size` | 115-118 | 模型輸入尺寸 |

---

### 5.8 `python/ivit/zoo/registry.py` — Model Zoo 目錄（1004 行）

**功能**：維護所有預設模型的完整資訊、下載與轉換流程。

**核心資料結構**（行 17-31）：

```python
@dataclass
class ModelInfo:
    name: str               # 模型名稱
    task: str               # detect, classify, segment, pose
    description: str        # 描述
    input_size: tuple       # (H, W) 輸入尺寸
    num_classes: int        # 類別數
    formats: List[str]      # 可用格式：onnx, openvino, tensorrt
    url: Optional[str]      # 原始權重下載 URL
    onnx_url: Optional[str] # 預轉換 ONNX URL
    source: str             # torchvision, megvii, paddlepaddle, mmpose
    license: str            # 授權
    metrics: Dict           # 精度指標
    tags: List[str]         # 標籤（供搜尋用）
```

**模型登錄表 MODEL_REGISTRY**（行 36-318）：

| 模型系列 | 行號範圍 | 模型 | 任務 |
|---------|---------|------|------|
| YOLOX (Apache-2.0) | 40-109 | nano / tiny / s / m / l | 物件偵測 |
| RT-DETR (Apache-2.0) | 114-141 | l / x | 物件偵測 |
| torchvision 偵測 (BSD-3) | 146-205 | fasterrcnn, retinanet, ssdlite, fcos | 物件偵測 |
| torchvision 分類 (BSD-3) | 210-257 | resnet18/50, mobilenetv3, efficientnet-b0 | 分類 |
| torchvision 分割 (BSD-3) | 262-285 | deeplabv3-resnet50, deeplabv3-mobilenetv3 | 語意分割 |
| RTMPose (Apache-2.0) | 290-317 | rtmpose-s, rtmpose-m | 姿態估計 |

**模型名稱 → 原始框架 對照表**：

| 對照表 | 行號 | 說明 |
|--------|------|------|
| `_TORCHVISION_MODEL_MAP` | 484-499 | torchvision 模型名 → 函數名映射 |
| `_YOLOX_MODEL_MAP` | 744-750 | YOLOX 名稱 → (config, depth, width) 映射 |
| `_RTDETR_MODEL_MAP` | 834-837 | RT-DETR 名稱 → PaddleDetection config 映射 |
| `_RTMPOSE_MODEL_MAP` | 892-899 | RTMPose 名稱 → MMPose config 映射 |

**匯出函數**：

| 函數 | 行號 | 說明 |
|------|------|------|
| `_export_torchvision_onnx()` | 502-675 | torchvision → ONNX 匯出（含分類/偵測/分割特殊處理） |
| `_export_yolox_onnx()` | 753-831 | YOLOX .pth → ONNX 匯出 |
| `_export_rtdetr_onnx()` | 840-888 | RT-DETR → ONNX（需 PaddlePaddle） |
| `_export_mmpose_onnx()` | 902-930 | RTMPose → ONNX（需 MMPose） |
| `_download_onnx_direct()` | 706-740 | 直接從 URL 下載預轉換 ONNX |
| `_download_file()` | 678-699 | 通用檔案下載（含進度條） |

---

### 5.9 `python/ivit/train/` — 訓練模組

| 模組 | 類別/函數 | 說明 |
|------|-----------|------|
| `trainer.py` | `Trainer` | 訓練引擎：`fit()`、`evaluate()`、`resume()` |
| `exporter.py` | `Exporter` | PyTorch → ONNX → 目標格式匯出 |
| `dataset.py` | `IVITDataset` | 統一資料集介面 |
| `augmentation.py` | `build_augmentation()` | 資料增強管線 |
| `callbacks.py` | `EarlyStopping`, `ModelCheckpoint` | 訓練回呼 |

---

### 5.10 `python/ivit/utils/` — 工具模組

| 模組 | 主要類別/函數 | 說明 |
|------|-------------|------|
| `visualizer.py` | `Visualizer` | 繪製 bbox、mask、keypoint、標籤 |
| `profiler.py` | `Profiler` | 延遲/吞吐量/記憶體測量 |
| `metrics.py` | `mAP()`, `IoU()`, `Accuracy()` | 模型評估指標 |
| `video.py` | `VideoReader`, `VideoWriter` | 影片讀寫封裝 |

---

## 6. C++ 核心模組詳解

### 6.1 `include/ivit/core/common.hpp` — 全域型別（255 行）

**關鍵定義**（所有列舉是跨層溝通的基礎）：

| 定義 | 行號 | 說明 |
|------|------|------|
| `enum class DataType` | 36-45 | Float32, Float16, Int8, UInt8, Int32, Int64, Bool |
| `enum class Layout` | 50-57 | NCHW, NHWC, NC, CHW, HWC |
| `enum class Precision` | 62-68 | FP32, FP16, INT8, INT4 |
| `enum class TaskType` | 73-84 | Classification, Detection, Segmentation, InstanceSegmentation, PoseEstimation, FaceDetection, FaceRecognition, OCR, AnomalyDetection |
| `enum class BackendType` | 92-98 | OpenVINO, TensorRT, QNN, Auto |
| `struct LoadConfig` | 107-115 | 模型載入設定 |
| `struct InferConfig` | 120-126 | 推論設定 |
| `class IVITError` | 135-139 | 基礎例外 |
| `class ModelLoadError` | 153-157 | 模型載入錯誤 |
| `class DeviceNotFoundError` | 180-184 | 裝置找不到錯誤 |
| `class InferenceError` | 189-193 | 推論錯誤 |

---

### 6.2 `src/core/model.cpp` — 核心推論引擎（1094 行）

**這是整個 SDK 最核心的 C++ 檔案**，包含 `InferenceModel` 類別和 `ModelManager` 單例。

#### InferenceModel 類別（行 25-553）

| 方法 | 行號 | 說明 |
|------|------|------|
| `InferenceModel()` 建構子 | 27-42 | 接收 runtime、handle、task，取得輸入/輸出資訊 |
| `predict(cv::Mat, InferConfig)` | 77-101 | **核心推論方法**：前處理 → runtime 推論 → 後處理 → 計時 |
| `predict_batch(images, config)` | 103-203 | 批次推論（支援動態 batch 或逐張回退） |
| `infer_raw(inputs)` | 205-209 | 直接送 Tensor（繞過前後處理） |
| `warmup(iterations)` | 211-225 | 暖機推論（TensorRT 首次較慢） |
| `preprocess(cv::Mat)` | 228-277 | **影像前處理**：resize → normalize → BGR2RGB → NCHW |
| `postprocess(outputs, size, config, results)` | 279-300 | **任務路由 switch**：依 TaskType 分派到對應後處理 |
| `postprocess_classification()` | 302-352 | 分類後處理：softmax → top-K → 映射 label |
| `postprocess_detection()` | 354-437 | 偵測後處理：YOLO decode → conf filter → NMS |
| `postprocess_segmentation()` | 439-482 | 分割後處理：argmax → resize mask |
| `apply_nms()` | 484-543 | per-class NMS（使用 OpenCV cv::dnn::NMSBoxes） |

#### Model 基礎類別方法（行 555-920）

| 方法 | 行號 | 說明 |
|------|------|------|
| `predict(string image_path)` | 559-568 | 讀取圖片再呼叫 predict(Mat) |
| `predict_async(image, config)` | 574-584 | 非同步推論（std::async） |
| `submit_inference(image, callback)` | 602-621 | Fire-and-forget 推論（detached thread） |
| `predict_concurrent(images, max_concurrent)` | 635-680 | 限制並行數的併行推論 |
| `ensure_executor()` / `shutdown_async()` | 686-718 | 非同步執行器管理 |
| `on(event, callback)` | 724-730 | 註冊推論事件回呼 |
| `predict_tta(image, augmentations)` | 837-920 | TTA（Test-Time Augmentation）推論 |
| `stream(source, config)` | 932-941 | 影片串流迭代器 |

#### ModelManager 單例（行 980-1094）

| 方法 | 行號 | 說明 |
|------|------|------|
| `instance()` | 984-987 | 取得單例 |
| `load(path, config)` | 989-1063 | **模型載入主流程**（見下方回追路線） |
| `convert(src, dst, backend, precision)` | 1065-1082 | 模型格式轉換 |
| `clear_cache()` | 1084-1087 | 清除模型快取 |
| `set_cache_dir(path)` | 1089-1092 | 設定快取目錄 |

**ModelManager::load() 詳細步驟**（行 989-1063）：

```
行 993-1001: 快取查找 (thread-safe)
     ↓ cache miss
行 1005-1006: RuntimeFactory::get_best_runtime(path, device)
     ↓ 取得適合的後端
行 1012-1025: 建立 RuntimeConfig (device, precision, cache_dir)
     ↓
行 1028: runtime->load_model(path, rt_config)  ← 後端載入模型
     ↓
行 1031-1038: 解析 task type (config.task → TaskType enum)
     ↓
行 1041-1047: new InferenceModel(runtime, handle, path, task, device)
     ↓
行 1050-1060: 存入快取 (thread-safe, double-check)
```

---

### 6.3 `include/ivit/runtime/runtime.hpp` — 後端抽象介面（165 行）

**IRuntime 介面**（行 36-116）— 所有後端必須實作：

```cpp
class IRuntime {
    virtual BackendType type() const = 0;           // 行 43
    virtual string name() const = 0;                // 行 48
    virtual bool is_available() const = 0;          // 行 53
    virtual vector<string> supported_formats() = 0; // 行 58
    virtual vector<DeviceInfo> get_devices() = 0;   // 行 63
    virtual void* load_model(path, config) = 0;     // 行 72-75 ← 載入模型
    virtual void unload_model(handle) = 0;          // 行 80
    virtual vector<TensorInfo> get_input_info() = 0;  // 行 85
    virtual vector<TensorInfo> get_output_info() = 0; // 行 90
    virtual map<string, Tensor> infer(handle, inputs) = 0; // 行 99-102 ← 執行推論
    virtual void convert_model(src, dst, config) = 0; // 行 111-115
};
```

**RuntimeFactory 工廠**（行 121-161）：

| 方法 | 行號 | 說明 |
|------|------|------|
| `instance()` | 126 | 取得單例 |
| `get_runtime(BackendType)` | 131 | 依後端類型取得 runtime |
| `get_runtime_for_device(device)` | 136 | 依裝置選取 runtime |
| `get_best_runtime(model_path, device)` | 141-144 | 依模型格式+裝置自動選取最佳 runtime |
| `register_runtime(runtime)` | 149 | 註冊新後端 |

---

### 6.4 `src/runtime/runtime_factory.cpp` — 後端選擇工廠（113 行）

**`get_best_runtime()` 選擇邏輯**（行 61-99）：

```
解析模型副檔名 (ext)
    │
    ├─ .engine / .trt    → TensorRT
    ├─ .xml / .bin       → OpenVINO
    ├─ .dlc / .serialized → QNN
    ├─ .onnx             → 依裝置決定：
    │   ├─ device="cpu" / "npu" / "gpu"  → OpenVINO (行 84-86)
    │   ├─ device="cuda:*"                → TensorRT (行 90-91)
    │   └─ 預設優先序：TensorRT > OpenVINO (行 89-94)
    └─ 其他 → fallback: get_runtime_for_device() (行 98)
```

**`get_runtime_for_device()` 裝置選擇邏輯**（行 40-59）：

```
device → BackendType 對照
    │
    ├─ "cuda:*"  → TensorRT
    ├─ "cpu"     → OpenVINO
    ├─ "gpu:*"   → OpenVINO (Intel GPU)
    ├─ "npu"     → OpenVINO (Intel NPU)
    └─ "auto"    → 優先序：TensorRT > OpenVINO (行 46-55)
```

---

## 7. pybind11 綁定層詳解

### `python/ivit/_binding.cpp`（~800 行）

**功能**：C++ ↔ Python 的橋接層，透過 pybind11 實現 numpy/OpenCV 互操作。

**關鍵結構**：

| 區塊 | 行號 | 說明 |
|------|------|------|
| Mat ↔ NumPy 轉換 | 31-121 | `mat_to_numpy()` 和 `numpy_to_mat()` |
| 模組定義 `PYBIND11_MODULE` | 127 | 模組名稱 `_ivit_core` |
| 列舉綁定 | 137-163 | DataType, Precision, Layout |
| LoadConfig 綁定 | 169-178 | 模型載入設定 |
| InferConfig 綁定 | 180-189 | 推論設定 |
| TensorInfo 綁定 | 195-211 | Tensor 資訊 |
| DeviceInfo 綁定 | 213-225 | 裝置資訊 |
| BBox 綁定 | 231-252 | 邊界框（含 iou, to_xywh 等方法） |
| Detection 綁定 | 258-267 | 偵測結果 |
| ClassificationResult 綁定 | 273-281 | 分類結果 |
| Keypoint / Pose 綁定 | 287-297 | 姿態估計結果 |
| **Results 綁定** | 303-382 | 統一結果容器（含 visualize, to_json 等） |
| **Classifier 綁定** | 388-432 | 分類器（predict, predict_batch, __call__） |
| **Detector 綁定** | 438-496 | 偵測器（predict, predict_batch, predict_config, __call__） |
| **Segmentor 綁定** | 502-535 | 分割器 |
| Callback 綁定 | 541-565 | 回呼事件系統 |
| OpenVINOConfig 綁定 | 571-580 | OpenVINO 設定 |
| TensorRTConfig 綁定 | 582-593 | TensorRT 設定 |
| QNNConfig 綁定 | 595+ | QNN 設定 |

**Python 與 C++ 的資料轉換方式**：

```
Python numpy array → numpy_to_mat() (行 85-121) → cv::Mat → C++ 推論
C++ cv::Mat 結果 → mat_to_numpy() (行 31-80) → numpy array → Python
```

---

## 8. 資料流程圖

### 8.1 標準推論流程（以 YOLO 偵測為例）

```
輸入影像（numpy / PIL / 路徑）
        │
        ▼
  LetterboxPreProcessor                     ← processors.py:116-176
  ┌─────────────────────────────────┐
  │ 1. resize with padding          │
  │ 2. normalize (÷255, BGR→RGB)    │
  │ 3. HWC → NCHW, float32          │
  └─────────────────────────────────┘
        │ input_tensor (1, 3, H, W)
        ▼
  IRuntime.infer()                          ← runtime.hpp:99-102
  ┌─────────────────────────────────┐
  │ OpenVINO / TensorRT 推論         │
  │ 硬體加速（GPU/NPU/CPU）           │
  └─────────────────────────────────┘
        │ raw_output (1, N, 85) 或類似
        ▼
  YOLOPostProcessor / YOLOXPostProcessor    ← processors.py:254-647
  ┌─────────────────────────────────┐
  │ 1. decode bbox (xywh → xyxy)    │
  │ 2. filter by conf_threshold     │
  │ 3. NMS（去除重疊框）              │
  │ 4. 座標縮放回原始圖尺寸           │
  │ 5. 映射 class_id → class_name   │
  └─────────────────────────────────┘
        │
        ▼
  Results（Detection 列表）
        │
        ▼
  results.visualize() / results.save() / results.to_json()
```

### 8.2 模型載入流程（完整調用鏈）

```
使用者程式碼                                        檔案位置
──────────────────────────────────────────────────────────────
ivit.load("yolox-s.onnx", device="auto")
        │                                       __init__.py:168
        ▼
  _require_cpp()                                __init__.py:189
  devices.best() → device = "cuda:0"           __init__.py:193-195
  config = LoadConfig(device=device)            __init__.py:197-203
  load_model(source, config)                    __init__.py:205
        │                                       ↓ C++ binding
        ▼
  _ivit_core.load_model()                       _binding.cpp → ivit.hpp
        │
        ▼
  ModelManager::load(path, config)              model.cpp:989
        │
        ├→ cache lookup (thread-safe)           model.cpp:996-1001
        │    hit? → return cached model
        │
        ├→ RuntimeFactory::get_best_runtime()   runtime_factory.cpp:61
        │    ├ ext=".onnx" + device="cuda:0"
        │    └→ return TensorRT runtime          runtime_factory.cpp:90
        │
        ├→ runtime->load_model(path, config)    TensorRT: build engine
        │                                       OpenVINO: compile_model
        │
        ├→ 解析 task type                        model.cpp:1031-1038
        │    config.task="detection" → TaskType::Detection
        │
        ├→ new InferenceModel(runtime, handle,  model.cpp:1041-1047
        │                      path, task, device)
        │
        └→ cache_.insert(key, model)            model.cpp:1050-1060

  return model  ← 回到 Python
```

---

## 9. 程式回追路線圖

### 9.1 回追策略：從使用者 API 往下追蹤

**原則**：iVIT-SDK 的調用鏈是 **Python 薄封裝 → C++ binding → C++ 核心**，所以回追時遵循：

```
Step 1: Python API 入口（python/ivit/__init__.py 或 vision/*.py）
Step 2: C++ binding 轉接（python/ivit/_binding.cpp）
Step 3: C++ 核心邏輯（src/core/model.cpp）
Step 4: 後端推論（src/runtime/openvino_runtime.cpp 或 tensorrt_runtime.cpp）
```

### 9.2 回追路線 A：「模型載入失敗」

```
問題：ivit.load("model.onnx") 丟出 ModelLoadError

回追路線：
1. python/ivit/__init__.py:168-205     ← load() 入口
   → 檢查 device 是否正確
   → 檢查 config 參數

2. python/ivit/_binding.cpp:127+       ← C++ binding 的 load_model 綁定
   → 這裡會把 Python 參數轉成 C++ LoadConfig

3. src/core/model.cpp:989-1063         ← ModelManager::load()
   → 行 1006: get_best_runtime() 是否找到 runtime?
   → 行 1008-1010: "No suitable runtime found" = 後端未安裝
   → 行 1028: runtime->load_model() 失敗? = 模型格式問題

4. src/runtime/runtime_factory.cpp:61-99  ← 後端選擇
   → 行 84-94: .onnx 檔對應哪個後端？
   → 有沒有安裝對應後端？

5. src/runtime/openvino_runtime.cpp 或 tensorrt_runtime.cpp
   → 實際的模型載入錯誤
```

### 9.3 回追路線 B：「偵測結果座標錯誤」

```
問題：偵測到物件但 bbox 座標不正確

回追路線（C++ 路徑）：
1. src/core/model.cpp:77-101           ← predict() 入口
2. src/core/model.cpp:228-277          ← preprocess()
   → 行 244: resize 是否正確？
   → 行 261-272: NCHW 轉換是否正確？
3. src/core/model.cpp:354-437          ← postprocess_detection()
   → 行 380: confidence 解析
   → 行 409-417: 座標反算（center → corner, × image_size）
   → 行 431: NMS 是否過濾過多？

回追路線（Python processors.py 路徑）：
1. python/ivit/core/processors.py:116-176  ← LetterboxPreProcessor
   → 行 152-154: scale, pad_w, pad_h 計算是否正確？
2. python/ivit/core/processors.py:254-420  ← YOLOPostProcessor
   → 行 322-325: 座標反算（減去 padding，除以 scale）
   → 行 328-331: clip to image bounds
3. python/ivit/core/processors.py:499-647  ← YOLOXPostProcessor
   → 行 545-546: YOLOX 特殊格式 (obj_conf 和 class_scores 分開)
   → 行 567-570: 座標反算
```

### 9.4 回追路線 C：「分類結果顯示 class_XXX 而非實際名稱」

```
問題：分類結果的 label 是 "class_42" 不是 "cat"

回追路線：
1. src/core/model.cpp:346-348          ← postprocess_classification()
   → labels_[class_id] 是否有值？
   → 如果 labels_ 是空的，會回退到 "class_" + id

2. src/core/model.cpp:72-74            ← set_labels() 是否被呼叫？
   → labels_ 在哪裡被設定？

3. python/ivit/zoo/registry.py:664-673 ← _export_torchvision_onnx()
   → 行 665-671: 匯出時是否有儲存 labels .txt 檔？

4. 檢查模型目錄下是否有 <model_name>.txt 標籤檔
```

### 9.5 回追路線 D：「Model Zoo 下載失敗」

```
回追路線：
1. python/ivit/zoo/registry.py:413-481 ← download()
   → 行 437: get_model_info() 找不到模型？
   → 行 461-462: 有 onnx_url？走直接下載
   → 行 465-479: 依 source 路由到對應匯出函數

2. python/ivit/zoo/registry.py:706-740 ← _download_onnx_direct()
   → 行 728: 下載檔案
   → 行 731-737: 檔案太小？可能 URL 失效

3. python/ivit/zoo/registry.py:678-699 ← _download_file()
   → HTTP 請求是否成功？
```

### 9.6 回追路線 E：「推論速度太慢」

```
回追路線：
1. src/core/model.cpp:85-98            ← predict() 的計時區段
   → inference_time_ms 包含前處理+推論+後處理

2. 前處理：src/core/model.cpp:228-277
   → 行 244: cv::resize 可能是瓶頸
   → 行 262-272: channel split + memcpy

3. 推論：src/core/model.cpp:91
   → runtime->infer() 是實際推論時間
   → 用 model.warmup(5) 排除首次編譯延遲

4. 後處理：
   → NMS (model.cpp:484-543 或 processors.py:349-420)
   → 如果偵測結果過多（數千個），NMS 會變慢

5. 裝置選擇：
   → 確認 device 是否用到 GPU/NPU
   → ivit.devices.best() 回傳的裝置
```

---

## 10. 後端推論引擎

### 10.1 OpenVINO（Intel）

```
ONNX / PyTorch
      │
      ▼  mo (Model Optimizer)
OpenVINO IR (.xml + .bin)
      │
      ▼  compile_model(device="CPU"/"GPU"/"NPU")
Compiled Model
      │
      ▼  infer_request.infer()
Output Tensors
```

支援量化：NNCF（Neural Network Compression Framework）INT8 校準

### 10.2 TensorRT（NVIDIA）

```
ONNX
  │
  ▼  trt.OnnxParser → trt.Builder
TensorRT Engine (.engine)
  │  （快取到 cache_dir，避免重複編譯）
  ▼  trt.IExecutionContext
CUDA 推論（cudaMalloc / cudaMemcpy）
  │
  ▼
Output Tensors
```

支援量化：INT8 校準（需提供 calibration dataset）

---

## 11. 新增模型修改指南

### 11.1 情境 A：新增一個**已有任務類型**的模型（例如新的偵測模型 YOLOv9）

**只需改 Python，不需改 C++！**

#### Step 1: 註冊模型資訊

**檔案**：`python/ivit/zoo/registry.py`

在 `MODEL_REGISTRY` 字典中新增（行 36 之後，依類別分組）：

```python
# 在行 109 之後（YOLOX 區塊末尾）新增：
"yolov9-s": ModelInfo(
    name="yolov9-s",
    task="detect",                          # ← 已有的任務類型
    description="YOLOv9 Small - State-of-the-art detection",
    input_size=(640, 640),
    num_classes=80,
    formats=["onnx", "openvino", "tensorrt"],
    url="https://...",                      # 原始權重 URL
    onnx_url="https://...",                 # 預轉換 ONNX URL（推薦）
    source="custom",                        # 或對應的框架名
    license="GPL-3.0",
    metrics={"mAP50-95": 51.0, "params_m": 7.2},
    tags=["yolov9", "detection", "accurate"],
),
```

#### Step 2: 新增模型名稱對照表（如果是新的來源框架）

**檔案**：`python/ivit/zoo/registry.py`

在行 750 之後新增：

```python
_YOLOV9_MODEL_MAP = {
    "yolov9-s": ("yolov9-s", "gelan-s"),
    "yolov9-m": ("yolov9-m", "gelan-m"),
}
```

#### Step 3: 新增匯出函數（如果是新的來源框架）

**檔案**：`python/ivit/zoo/registry.py`

在行 930 之後新增：

```python
def _export_yolov9_onnx(name, onnx_path, info, force=False):
    # 下載權重 → 載入模型 → torch.onnx.export → 儲存標籤
    ...
```

#### Step 4: 在 download() 中新增路由

**檔案**：`python/ivit/zoo/registry.py:464-479`

```python
elif info.source == "yolov9":
    if format == "onnx":
        model_path = _export_yolov9_onnx(name, model_path, info, force)
```

#### Step 5: 新增後處理器（如果輸出格式不同於現有的）

**檔案**：`python/ivit/core/processors.py`

如果新模型的輸出格式與現有的 YOLOPostProcessor 不同：

1. 在行 1117 之後新增新類別：
```python
class YOLOv9PostProcessor(BasePostProcessor):
    def process(self, outputs, orig_size, preprocess_info, config, labels=None):
        # 解碼 YOLOv9 的特殊輸出格式
        ...
```

2. 在 `_POSTPROCESSORS` 登錄表（行 1126-1133）新增：
```python
"yolov9": YOLOv9PostProcessor,
```

---

### 11.2 情境 B：新增一個**全新任務類型**（例如 OCR 文字辨識）

需要同時修改 C++ 和 Python。

#### Step 1: C++ 列舉擴充（已內建 OCR）

**檔案**：`include/ivit/core/common.hpp:73-84`

確認 `TaskType` 已有所需值：

```cpp
enum class TaskType {
    Classification,
    Detection,
    Segmentation,
    InstanceSegmentation,
    PoseEstimation,
    FaceDetection,
    FaceRecognition,
    OCR,                    // ← 已存在
    AnomalyDetection,
    // 新增：
    TextDetection,          // ← 如果需要新的
    Unknown
};
```

#### Step 2: C++ 後處理擴充

**檔案**：`src/core/model.cpp`

1. 在 `postprocess()` switch（行 286-299）新增 case：

```cpp
case TaskType::OCR:
    postprocess_ocr(outputs, image_size, config, results);
    break;
```

2. 新增 `postprocess_ocr()` 方法（在行 482 之後）：

```cpp
void postprocess_ocr(
    const std::map<std::string, Tensor>& outputs,
    cv::Size image_size,
    const InferConfig& config,
    Results& results
) {
    // OCR 特殊後處理邏輯
}
```

#### Step 3: C++ 結果結構擴充

**檔案**：`include/ivit/core/result.hpp`

新增 OCR 結果欄位：

```cpp
struct Results {
    // ... 現有欄位 ...
    std::vector<TextResult> texts;  // 新增
};
```

#### Step 4: pybind11 綁定擴充

**檔案**：`python/ivit/_binding.cpp`

在 Results 綁定區塊（行 303-382）新增：

```cpp
.def_readwrite("texts", &ivit::Results::texts)
```

#### Step 5: Python 處理器

**檔案**：`python/ivit/core/processors.py`

新增 OCR 後處理器，並註冊到 `_POSTPROCESSORS`。

#### Step 6: Python 視覺任務封裝（可選）

新增 `python/ivit/vision/ocr.py`：

```python
class OCRReader:
    def __init__(self, model, device="auto", **kwargs): ...
    def predict(self, image): ...
```

#### Step 7: 在 `__init__.py` 匯出（可選）

**檔案**：`python/ivit/__init__.py`

在 import 區塊和 `__all__` 列表新增 `OCRReader`。

---

### 11.3 情境 C：新增一個**新的推論後端**（例如 ONNX Runtime）

#### Step 1: 實作 IRuntime 介面

**新增檔案**：`include/ivit/runtime/onnxruntime_runtime.hpp`

```cpp
class OnnxRuntimeRuntime : public IRuntime {
    BackendType type() const override { return BackendType::OnnxRuntime; }
    void* load_model(const string& path, const RuntimeConfig& config) override;
    map<string, Tensor> infer(void* handle, const map<string, Tensor>& inputs) override;
    void convert_model(const string& src, const string& dst, const RuntimeConfig& config) override;
    // ... 其他純虛函數
};
```

#### Step 2: 實作 .cpp 檔

**新增檔案**：`src/runtime/onnxruntime_runtime.cpp`

#### Step 3: 擴充 BackendType 列舉

**檔案**：`include/ivit/core/common.hpp:92-98`

```cpp
enum class BackendType {
    OpenVINO,
    TensorRT,
    QNN,
    OnnxRuntime,   // ← 新增
    Auto,
    Unknown
};
```

#### Step 4: 在 RuntimeFactory 中註冊

**檔案**：`src/runtime/runtime_factory.cpp`

在 `get_best_runtime()` 邏輯中新增 .onnx 的選擇支援。

#### Step 5: 更新 CMakeLists.txt

新增 ONNX Runtime 的 find_package 和連結設定。

---

### 11.4 各情境修改檔案速查表

| 修改項目 | 情境 A（新模型） | 情境 B（新任務） | 情境 C（新後端） |
|---------|:---:|:---:|:---:|
| `python/ivit/zoo/registry.py` (MODEL_REGISTRY) | ✅ 必改 | ✅ 必改 | - |
| `python/ivit/zoo/registry.py` (匯出函數) | ⚠️ 可能 | ⚠️ 可能 | - |
| `python/ivit/core/processors.py` (後處理器) | ⚠️ 可能 | ✅ 必改 | - |
| `include/ivit/core/common.hpp` (TaskType) | - | ⚠️ 可能 | - |
| `include/ivit/core/common.hpp` (BackendType) | - | - | ✅ 必改 |
| `src/core/model.cpp` (postprocess switch) | - | ✅ 必改 | - |
| `src/core/model.cpp` (postprocess 函數) | - | ✅ 必改 | - |
| `include/ivit/core/result.hpp` (結果結構) | - | ⚠️ 可能 | - |
| `python/ivit/_binding.cpp` (綁定) | - | ⚠️ 可能 | - |
| `include/ivit/runtime/runtime.hpp` | - | - | - |
| `src/runtime/runtime_factory.cpp` | - | - | ✅ 必改 |
| 新增 `src/runtime/*_runtime.cpp` | - | - | ✅ 必改 |
| 新增 `include/ivit/runtime/*_runtime.hpp` | - | - | ✅ 必改 |
| `python/ivit/vision/*.py` (視覺封裝) | - | ⚠️ 可選 | - |
| `python/ivit/__init__.py` (匯出) | - | ⚠️ 可選 | - |
| `CMakeLists.txt` | - | - | ✅ 必改 |

---

## 12. 設計模式總結

| 模式 | 應用位置 | 檔案:行號 | 說明 |
|------|----------|-----------|------|
| **Strategy（策略）** | `IRuntime` + 各後端 | `runtime.hpp:36-116` | 後端可替換，不影響上層呼叫者 |
| **Factory（工廠）** | `RuntimeFactory` | `runtime_factory.cpp:61-99` | 依模型格式+裝置自動建立 runtime |
| **Factory（工廠）** | `ivit.load()` | `__init__.py:168-205` | 依參數自動建立推論器 |
| **Adapter（適配器）** | Python vision classes | `detector.py`, `classifier.py` | 統一不同後端的 Python 介面 |
| **Template Method** | Pre/PostProcessor 基類 | `processors.py:18-113` | 定義處理流程骨架，子類別實作細節 |
| **Observer（觀察者）** | Callback 系統 | `model.cpp:724-760` | 在推論各階段注入自訂邏輯 |
| **Registry（登錄）** | Model Zoo | `registry.py:36-318` | 統一管理模型元資料 |
| **Registry（登錄）** | Processor 登錄 | `processors.py:1120-1175` | 依名稱查找前後處理器 |
| **PIMPL（橋接）** | `InferenceModel` 使用 `IRuntime` | `model.cpp:25-553` | 隱藏後端實作 |
| **Singleton（單例）** | `ModelManager`, `RuntimeFactory` | `model.cpp:984-987`, `runtime_factory.cpp:17-20` | 全域唯一管理器 |
| **Iterator（迭代器）** | `StreamIterator` | `model.cpp:926-978` | 影片串流逐幀推論 |

---

## 附錄 A：關鍵依賴版本

| 依賴 | 版本需求 | 用途 |
|------|----------|------|
| Python | ≥ 3.9 | 主要開發語言 |
| C++ | 17 | 核心引擎 |
| OpenCV | ≥ 4.5 | 影像讀寫/前處理 |
| OpenVINO | ≥ 2024.0 | Intel 推論後端 |
| TensorRT | ≥ 8.6 | NVIDIA 推論後端 |
| CUDA | ≥ 11.8 | GPU 運算 |
| pybind11 | ≥ 2.10 | C++↔Python 橋接 |
| PyTorch | ≥ 2.0 | 訓練模組（可選） |
| ONNX | ≥ 1.14 | 模型交換格式 |
| numpy | ≥ 1.21 | 數值運算 |
| albumentations | ≥ 1.3 | 資料增強（訓練用） |
| NNCF | ≥ 2.7 | INT8 量化（OpenVINO） |

---

## 附錄 B：關鍵檔案行號速查

| 元件 | 檔案 | 行號 | 功能 |
|------|------|------|------|
| **Python 入口** | `python/ivit/__init__.py` | 168-205 | `load()` 函數 |
| **C++ binding** | `python/ivit/_binding.cpp` | 127-600 | pybind11 模組定義 |
| **模型載入** | `src/core/model.cpp` | 989-1063 | `ModelManager::load()` |
| **任務偵測** | `src/core/model.cpp` | 1031-1038 | task string → TaskType |
| **推論入口** | `src/core/model.cpp` | 77-101 | `InferenceModel::predict()` |
| **前處理** | `src/core/model.cpp` | 228-277 | C++ 前處理 |
| **後處理路由** | `src/core/model.cpp` | 279-300 | switch(task_) |
| **分類後處理** | `src/core/model.cpp` | 302-352 | softmax + top-K |
| **偵測後處理** | `src/core/model.cpp` | 354-437 | YOLO decode + NMS |
| **分割後處理** | `src/core/model.cpp` | 439-482 | argmax + resize |
| **NMS** | `src/core/model.cpp` | 484-543 | per-class NMS |
| **後端選擇** | `src/runtime/runtime_factory.cpp` | 61-99 | 模型格式→後端映射 |
| **IRuntime 介面** | `include/ivit/runtime/runtime.hpp` | 36-116 | 純虛擬介面定義 |
| **全域列舉** | `include/ivit/core/common.hpp` | 36-98 | DataType, TaskType, BackendType |
| **Model Zoo** | `python/ivit/zoo/registry.py` | 36-318 | MODEL_REGISTRY 字典 |
| **Zoo 下載** | `python/ivit/zoo/registry.py` | 413-481 | download() 路由 |
| **Zoo 匯出** | `python/ivit/zoo/registry.py` | 502-930 | 各框架→ONNX 匯出 |
| **Python 偵測** | `python/ivit/vision/detector.py` | 32-176 | Detector wrapper |
| **Python 分類** | `python/ivit/vision/classifier.py` | 27-124 | Classifier wrapper |
| **前處理器** | `python/ivit/core/processors.py` | 116-251 | Letterbox / CenterCrop |
| **後處理器** | `python/ivit/core/processors.py` | 254-1117 | YOLO/YOLOX/RTDETR/RTMPose/Classification |
| **處理器登錄** | `python/ivit/core/processors.py` | 1120-1175 | 登錄表 + 工具函數 |

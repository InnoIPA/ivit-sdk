# iVIT-SDK

**Innodisk Vision Intelligence Toolkit** - 統一的電腦視覺推論與訓練 SDK

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org)

## 概述

iVIT-SDK 是宜鼎國際開發的統一電腦視覺 SDK，提供跨硬體平台的一致性 API 介面。無論您使用的是 Intel 還是 NVIDIA 的硬體，都可以使用相同的程式碼進行開發。

### 特色

- **統一 API** - 一套程式碼，支援多種硬體平台
- **多後端支援** - Intel OpenVINO、NVIDIA TensorRT、Qualcomm QNN [規劃中]
- **完整視覺任務** - 分類、物件偵測、語意分割、姿態估計
- **遷移式學習** - 支援模型微調和訓練
- **雙語言支援** - Python 和 C++ API

## 硬體支援

| 廠商 | 硬體類型 | 後端 |
|------|---------|------|
| Intel | CPU、iGPU、NPU | OpenVINO |
| NVIDIA | dGPU、Jetson | TensorRT |
| Qualcomm | IQ9/IQ8/IQ6 | QNN [規劃中] |

## 安裝

```bash
# pip 安裝
pip install ivit-sdk

# 從原始碼建置
git clone https://github.com/innodisk-mannywang/ivit-sdk.git
cd ivit-sdk
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

> 完整環境設定（OpenVINO APT 安裝、TensorRT、Python 綁定等）請參考 **[Getting Started Guide](docs/getting-started.md#環境安裝)**

## 快速開始

### Python

```python
import ivit

# 物件偵測
detector = ivit.Detector("yolox-s.onnx", device="npu")
results = detector.predict("image.jpg")

for det in results.detections:
    print(f"{det.label}: {det.confidence:.2%}")
```

### C++

```cpp
#include "ivit/vision/detector.hpp"

Detector detector("yolox-s.onnx", "npu");
auto results = detector.predict(image, 0.5f);

for (const auto& det : results.detections) {
    std::cout << det.label << ": " << det.confidence << std::endl;
}
```

> 更多範例請參考 **[Getting Started Guide](docs/getting-started.md#快速開始)**

## 支援的模型

| 任務 | 模型 |
|------|------|
| 分類 | ResNet、EfficientNet、MobileNet |
| 物件偵測 | YOLOv5/v8、SSD、Faster R-CNN |
| 語意分割 | DeepLabV3、U-Net、SegFormer |
| 姿態估計 | RTMPose、HRNet |
| 人臉分析 | RetinaFace、ArcFace |

## 文件

| 文件 | 說明 |
|------|------|
| **[Getting Started Guide](docs/getting-started.md)** | 環境安裝、編譯、API 使用、效能優化 |
| [API 規格](docs/api/api-spec.md) | 完整 API 參考 |
| [系統架構](docs/architecture/adr-001-system.md) | 架構設計文件 |
| [PRD](docs/development/prd.md) | 產品需求文件 |

## 開發

```bash
# 測試
pytest tests/python -v
ctest --test-dir build

# 程式碼風格
black python/
clang-format -i src/**/*.cpp include/**/*.hpp
```

## 授權

[Apache License 2.0](LICENSE)

---

Copyright (c) 2024 [宜鼎國際](https://www.innodisk.com). All rights reserved.

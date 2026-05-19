# Intel OpenVINO 部署指南

本指南說明如何在 Intel 平台上使用 iVIT-SDK 進行模型部署。

## 支援硬體

| 硬體類型 | 架構 | 說明 |
|----------|------|------|
| Intel CPU | x86_64 / ARM64 | Core, Xeon, Atom |
| Intel iGPU | x86_64 | Iris Xe, UHD Graphics |
| Intel NPU | x86_64 | Meteor Lake, Lunar Lake |

## 環境準備

### 1. 安裝 OpenVINO Runtime

```bash
# Ubuntu 22.04
pip install openvino

# 或安裝開發工具（含模型轉換）
pip install openvino-dev
```

### 2. 安裝 iVIT-SDK

```bash
pip install ivit-sdk[openvino]
```

### 3. 確認安裝

```bash
ivit info
```

預期輸出：
```
iVIT-SDK v1.0.0
============================================================
System: Linux (Ubuntu 22.04)
Python: 3.10.12

Available Backends:
  - OpenVINO 2024.0 ✓

Devices:
  - cpu: Intel Core i7-1365U (OpenVINO)
  - npu: Intel AI Boost (OpenVINO)
  - igpu: Intel Iris Xe (OpenVINO)
```

## 快速開始

### 基本推論

```python
import ivit

# 載入模型（自動選擇最佳裝置）
model = ivit.load("yolox-s.onnx")

# 執行推論
results = model("image.jpg")
results.show()
```

### 指定裝置

```python
# 使用 CPU
model = ivit.load("model.onnx", device="cpu")

# 使用 Intel iGPU
model = ivit.load("model.onnx", device="igpu")

# 使用 Intel NPU
model = ivit.load("model.onnx", device="npu")

# 自動選擇最佳裝置
model = ivit.load("model.onnx", device=ivit.devices.best())
```

## 模型轉換

### ONNX 轉 OpenVINO IR

```bash
# 使用 CLI
ivit convert model.onnx -f openvino -o ./output/

# 指定精度
ivit convert model.onnx -f openvino -o ./output/ --precision FP16
```

```python
# 使用 Python API
from ivit.convert import convert_to_openvino

convert_to_openvino(
    "model.onnx",
    output_dir="./output/",
    precision="FP16"
)
```

## 效能優化

### 配置 OpenVINO 參數

```python
import ivit

model = ivit.load("model.onnx", device="cpu")

# 配置 OpenVINO 特定參數
model.configure_openvino(
    performance_mode="LATENCY",      # 或 "THROUGHPUT"
    num_streams=4,                   # 並行推論流數
    inference_precision="FP16",      # 推論精度
    enable_cpu_pinning=True,         # CPU 核心綁定
    num_threads=8,                   # CPU 執行緒數
)

# 預熱
model.warmup(3)

# 執行推論
results = model("image.jpg")
```

### 效能模式說明

| 模式 | 說明 | 適用場景 |
|------|------|----------|
| LATENCY | 最低延遲 | 即時應用、邊緣裝置 |
| THROUGHPUT | 最高吞吐量 | 批次處理、伺服器 |
| CUMULATIVE_THROUGHPUT | 多裝置吞吐量 | 多裝置推論 |

### NPU 特定優化

```python
model = ivit.load("model.onnx", device="npu")

model.configure_openvino(
    npu_compilation_mode="DefaultHW",  # 硬體優化模式
    cache_dir="./cache/",              # 模型快取
)
```

## 串流處理

```python
import ivit

model = ivit.load("yolox-s.onnx", device="cpu")

# 處理影片串流
for results in model.stream("video.mp4"):
    # 處理每一幀
    frame = results.plot()
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 批次推論

```python
import ivit
import numpy as np

model = ivit.load("model.onnx", device="cpu")

# 配置高吞吐量模式
model.configure_openvino(
    performance_mode="THROUGHPUT",
    num_streams=8,
)

# 批次推論
images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
results = model.predict_batch(images, batch_size=4)

for r in results:
    print(f"Detected: {len(r)} objects")
```

## 效能監控

```python
import ivit
from ivit.core.callbacks import FPSCounter, LatencyLogger

model = ivit.load("model.onnx", device="cpu")

# 新增效能監控
fps = FPSCounter(window_size=30)
model.on("infer_end", fps)
model.on("infer_end", LatencyLogger())

# 執行推論
for i in range(100):
    results = model("image.jpg")

print(f"Average FPS: {fps.fps:.1f}")
```

## 常見問題

### Q: 如何確認 NPU 是否可用？

```bash
# 檢查裝置
ivit devices

# 或使用 Python
import ivit
devices = ivit.devices()
for d in devices:
    print(f"{d.id}: {d.name}")
```

### Q: OpenVINO IR 模型載入失敗？

確認 `.xml` 和 `.bin` 檔案在同一目錄：
```
model.xml  # 模型結構
model.bin  # 模型權重
```

### Q: 如何提升 CPU 推論速度？

1. 使用 FP16 精度
2. 開啟 CPU 核心綁定
3. 使用模型快取
4. 配置適當的執行緒數

```python
model.configure_openvino(
    inference_precision="FP16",
    enable_cpu_pinning=True,
    cache_dir="./cache/",
    num_threads=0,  # 0 = 自動
)
```

## 範例程式

完整範例程式位於：`examples/python/01_quickstart.py` 及 `examples/python/advanced/embedded_optimization.py`

```python
#!/usr/bin/env python3
"""Intel OpenVINO 部署範例"""

import ivit
from ivit.core.callbacks import FPSCounter

def main():
    # 列出可用裝置
    print("Available devices:")
    for d in ivit.devices():
        print(f"  - {d.id}: {d.name}")

    # 載入模型
    model = ivit.load("yolox-s.onnx", device="cpu")

    # 配置 OpenVINO
    model.configure_openvino(
        performance_mode="LATENCY",
        inference_precision="FP16",
    )

    # 新增 FPS 計數器
    fps = FPSCounter()
    model.on("infer_end", fps)

    # 預熱
    model.warmup(5)

    # 處理影片
    for results in model.stream("video.mp4"):
        print(f"FPS: {fps.fps:.1f}, Objects: {len(results)}")

if __name__ == "__main__":
    main()
```
